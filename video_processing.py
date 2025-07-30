import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import json
from video_processor.base_processor import AudioProcessor
from r2_storage import R2Storage
from runpod_queue_service import runpod_queue_service
from utils import cleanup_temp_files
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading

logger = logging.getLogger(__name__)

def process_video_background(
    video_source: str, audio_id: str, include_instruments: bool,
    generate_subtitles: bool, 
    target_language: str, language_code: Optional[str], speakers_expected: Optional[int], is_file_upload: bool,
    audio_processor=None, original_filename: Optional[str] = None, original_source_url: Optional[str] = None
):
    
    """
    Background processing function for video dubbing
    
    Args:
        video_source: URL or path to video file
        audio_id: Unique identifier for this processing job
        include_instruments: Whether to include background music in final output
        generate_subtitles: Whether to generate subtitle file
        target_language: Target language for translation (e.g., "English", "Spanish")
        language_code: Language code for transcription (e.g., "en", "es")
        speakers_expected: Expected number of speakers for transcription
        is_file_upload: Whether video_source is a file path (True) or URL (False)
        audio_processor: Shared AudioProcessor instance (optional)
        original_filename: Original filename if file upload
        original_source_url: Original source URL for tracking
    """
    from status_manager import status_manager, ProcessingStatus
    from config import settings
    r2_storage = R2Storage()
    
    try:
        # Create or get audio processor
        if audio_processor is None:
            from video_processor import get_audio_processor
            audio_processor = get_audio_processor()
        
        # Initialize status
        status_manager.update_status(audio_id, ProcessingStatus.PROCESSING, 10, {
            "message": "Starting video processing",
            "stage": "initialization"
        })
        
        # Handle video download if URL
        video_temp_path = None
        if not is_file_upload:
            status_manager.update_status(audio_id, ProcessingStatus.PROCESSING, 15, {
                "message": "Downloading video from URL",
                "stage": "download"
            })
            
            from video_processor.file_handler import FileHandler
            file_handler = FileHandler(temp_dir="./tmp/voice_cloning")
            download_result = file_handler.download_video(video_source, audio_id, original_filename)
            
            if not download_result["success"]:
                raise Exception(f"Failed to download video: {download_result['error']}")
            
            video_temp_path = download_result["file_path"]
            # For URLs, use the URL as the display name
            display_filename = original_filename or video_source.split('/')[-1]
        else:
            # For file uploads, use the provided path
            video_temp_path = video_source
            display_filename = original_filename or os.path.basename(video_source)
        
        # Get original audio metadata first
        original_audio_details = {
            "source": "file_upload" if is_file_upload else "url",
            "source_url": original_source_url or video_source,
            "filename": display_filename,
            "duration": None,
            "format": None,
            "size_mb": None
        }
        
        try:
            # Get audio metadata
            metadata = audio_processor.get_audio_metadata(video_temp_path)
            if metadata:
                original_audio_details.update({
                    "duration": metadata.get("duration"),
                    "format": metadata.get("format_name"),
                    "size_mb": round(metadata.get("size", 0) / (1024 * 1024), 2)
                })
        except Exception:
            pass
        
        # Process audio segments with language parameters
        processing_result = audio_processor.process_audio_segments(
            video_temp_path, 
            audio_id,
            include_instruments=include_instruments,
            generate_subtitles=generate_subtitles,
            target_language=target_language,
            language_code=language_code,
            speakers_expected=speakers_expected
        )
        
        if not processing_result["success"]:
            raise Exception(processing_result.get("error", "Audio processing failed"))
        
        # Get processing results
        cloning_result = processing_result.get("cloning_result", {})
        reconstruction_result = processing_result.get("audio_reconstruction", {})
        
        if not reconstruction_result.get("success"):
            raise Exception(f"Audio reconstruction failed: {reconstruction_result.get('error', 'Unknown error')}")
        
        # Get final audio path
        final_audio_path = reconstruction_result.get("output_path")
        if not final_audio_path or not os.path.exists(final_audio_path):
            raise Exception("Final audio file not found after reconstruction")
        
        logger.info(f"Using final audio path: {final_audio_path}")
        
        # Process separation with optimized RunPod queue
        status_manager.update_status(audio_id, ProcessingStatus.PROCESSING, 20, {
            "message": "Separating vocals and instruments using RunPod service...",
            "stage": "separation"
        })
        
        separation_result = runpod_queue_service.process_audio_separation_sync(video_temp_path, caller_info="process_video_background")
        
        if not separation_result["success"]:
            raise Exception(f"Audio separation failed: {separation_result.get('error', 'Unknown error')}")
        
        # Download separated files locally
        status_manager.update_status(audio_id, ProcessingStatus.PROCESSING, 30, {
            "message": "Downloading separated audio tracks...",
            "stage": "download_separation"
        })
        
        # Use optimized local storage
        from utils import LocalStorageManager
        local_storage = LocalStorageManager()
        
        vocal_path = local_storage.save_separated_audio(
            audio_id, 
            separation_result["vocal_url"], 
            "vocal"
        )
        
        instruments_path = local_storage.save_separated_audio(
            audio_id,
            separation_result["instruments_url"],
            "instruments"
        )
        
        if not vocal_path or not instruments_path:
            raise Exception("Failed to download separated audio files")
        
        # Create video with subtitles or without
        video_result = None
        if generate_subtitles:
            # Create video with subtitles
            instruments_for_video = instruments_path if include_instruments else None
            video_result = audio_processor.create_video_with_subtitles(
                video_temp_path,
                final_audio_path,
                processing_result["segments_dir"],
                audio_id,
                instruments_for_video
            )
        else:
            # Create video without subtitles
            instruments_for_video = instruments_path if include_instruments else None
            video_result = audio_processor.create_video_with_audio(
                video_temp_path,
                final_audio_path,
                audio_id,
                instruments_for_video,
                processing_result["segments_dir"]
            )
        
        if not video_result or not video_result["success"]:
            error_msg = video_result.get('error', 'Unknown error') if video_result else 'No result'
            raise Exception(f"Video creation failed: {error_msg}")
        
        status_manager.update_status(audio_id, ProcessingStatus.PROCESSING, 90, {
            "message": "Uploading files to cloud storage..."
        })
        
        # Upload files to R2
        try:
            r2_segments_result = r2_storage.upload_audio_segments(audio_id, processing_result["segments_dir"])
            r2_final_result = r2_storage.upload_final_audio(audio_id, final_audio_path)
            
            # Upload vocal and instruments files
            vocal_filename = f"vocal_separated_{audio_id}.wav"
            vocal_r2_key = r2_storage.generate_file_path(audio_id, "vocal", vocal_filename)
            r2_vocal_result = r2_storage.upload_file(vocal_path, vocal_r2_key, "audio/wav")
            
            instruments_filename = f"instruments_separated_{audio_id}.wav"
            instruments_r2_key = r2_storage.generate_file_path(audio_id, "instruments", instruments_filename)
            r2_instruments_result = r2_storage.upload_file(instruments_path, instruments_r2_key, "audio/wav")
            
            # Upload original video file for reference (especially for file uploads)
            r2_original_video_result = None
            if is_file_upload:
                original_video_filename = f"original_{display_filename}"
                original_video_r2_key = r2_storage.generate_file_path(audio_id, "original", original_video_filename)
                r2_original_video_result = r2_storage.upload_file(video_temp_path, original_video_r2_key, "video/mp4")
                
                # Update source_url with uploaded original video URL
                if r2_original_video_result and r2_original_video_result.get("success"):
                    original_audio_details["source_url"] = r2_original_video_result.get("url")
            
            r2_video_result = None
            r2_subtitle_result = None
            
            if video_result and video_result["success"]:
                # Upload video
                video_filename = f"video_processed_{audio_id}.mp4"
                r2_key = r2_storage.generate_file_path(audio_id, "video", video_filename)
                r2_video_result = r2_storage.upload_file(
                    video_result["video_path"], 
                    r2_key, 
                    "video/mp4"
                )
                
                # Upload subtitle file if it exists
                if "subtitle_path" in video_result and os.path.exists(video_result["subtitle_path"]):
                    subtitle_filename = f"subtitles_{audio_id}.srt"
                    r2_subtitle_key = r2_storage.generate_file_path(audio_id, "subtitles", subtitle_filename)
                    r2_subtitle_result = r2_storage.upload_file(
                        video_result["subtitle_path"],
                        r2_subtitle_key,
                        "text/srt"
                    )
        
        except Exception as e:
            logger.error(f"File upload failed: {str(e)}")
            raise Exception(f"File upload failed: {str(e)}")
        
        # Prepare completion data
        completion_data = {
            "final_audio_url": r2_final_result.get("url"),
            "video_url": r2_video_result.get("url") if r2_video_result else None,
            "subtitles_url": r2_subtitle_result.get("url") if r2_subtitle_result else None,
            "processing_stats": {
                "total_segments": processing_result.get("total_segments", 0),
                "cloned_segments": cloning_result.get("successful_clones", 0),
                "speakers": processing_result.get("speakers", []),
                "duration": processing_result.get("total_duration", 0)
            },
            "uploaded_files": {
                "segments": {
                    "count": r2_segments_result.get("files_uploaded", 0) if r2_segments_result else 0,
                    "uploaded": r2_segments_result.get("success", False) if r2_segments_result else False
                },
                "final_audio": {
                    "url": r2_final_result.get("url"),
                    "size_mb": round(r2_final_result.get("size", 0) / (1024 * 1024), 2),
                    "uploaded": r2_final_result.get("success", False)
                },
                "vocal_audio": {
                    "url": r2_vocal_result.get("url"),
                    "size_mb": round(r2_vocal_result.get("size", 0) / (1024 * 1024), 2),
                    "uploaded": r2_vocal_result.get("success", False)
                },
                "instruments_audio": {
                    "url": r2_instruments_result.get("url"),
                    "size_mb": round(r2_instruments_result.get("size", 0) / (1024 * 1024), 2),
                    "uploaded": r2_instruments_result.get("success", False)
                },
                "video": {
                    "url": r2_video_result.get("url"),
                    "size_mb": round(r2_video_result.get("size", 0) / (1024 * 1024), 2),
                    "uploaded": r2_video_result.get("success", False)
                } if r2_video_result else None,
                "subtitles": {
                    "url": r2_subtitle_result.get("url"),
                    "size_kb": round(r2_subtitle_result.get("size", 0) / 1024, 2),
                    "uploaded": r2_subtitle_result.get("success", False)
                } if r2_subtitle_result else None
            },
            "processing_details": {
                "original_audio": original_audio_details,
                "audio_processing_info": {
                    "processing_completed": True,
                    "segments_processed": cloning_result.get("total_segments", 0),
                    "seed_used": cloning_result.get("seed_used", settings.DEFAULT_SEED)
                },
                "features_used": {
                    "vocal_separation": True,
                    "audio_processing": True,
                    "subtitle_generation": generate_subtitles,
                    "instrument_mixing": include_instruments
                },
                "processing_timeline": {
                    "transcription_source": "AssemblyAI",
                    "audio_processing_model": "Disabled",
                    "separation_service": "RunPod"
                }
            },
            "segment_details": audio_processor.get_processing_stats(processing_result["segments_dir"]),
            "r2_storage": {
                "bucket": r2_storage.bucket_name,
                "base_path": r2_storage.get_storage_info(audio_id)["base_path"],
                "segments_uploaded": r2_segments_result.get("success", False),
                "total_files": (r2_segments_result.get("files_uploaded", 0) if r2_segments_result and r2_segments_result.get("success") else 0) + 
                              (1 if r2_final_result.get("success") else 0) + 
                              (1 if r2_vocal_result.get("success") else 0) + 
                              (1 if r2_instruments_result.get("success") else 0) + 
                              (1 if r2_video_result and r2_video_result.get("success") else 0) + 
                              (1 if r2_subtitle_result and r2_subtitle_result.get("success") else 0),
                "segment_urls": r2_storage.get_segment_urls(audio_id, r2_segments_result) if r2_segments_result else {}
            }
        }
        
        # Mark as completed
        status_manager.complete_processing(audio_id, completion_data)
        
        # Create processing summary
        processing_data = {
            "audio_id": audio_id,
            "original_audio": original_audio_details,
            "seeds_used": cloning_result.get("seeds_used", {}),
            "parameters": {
                "temperature": 1.8,
                "cfg_scale": 3.0,
                "top_p": 0.95,
                "target_language": target_language,
                "include_instruments": include_instruments,
                "generate_subtitles": generate_subtitles,
                "video_provided": True,
                "video_source": "url" if not is_file_upload else "file_upload",
                "video_url": video_source if not is_file_upload else None,
                "runpod_separation": True
            },
            "processing_stats": audio_processor.get_processing_stats(processing_result["segments_dir"]),
            "reconstruction_results": reconstruction_result,
            "video_generated": video_result is not None and video_result["success"],
            "subtitles_generated": generate_subtitles and video_result is not None and video_result.get("subtitle_count", 0) > 0,
            "separation_used": True
        }
        
        r2_storage.create_processing_summary(audio_id, processing_data)
        
        # Clean up all temp files after successful upload
        try:
            # Now we can clean up everything including final output files
            from video_processor.file_manager import FileManager
            file_manager = FileManager(temp_dir="./tmp/voice_cloning")
            file_manager.cleanup_temp_files(audio_id, keep_final_output=False)
            
            # Also clean up the reconstructed audio file
            if final_audio_path and os.path.exists(final_audio_path):
                os.unlink(final_audio_path)
        except Exception as e:
            logger.warning(f"Post-upload cleanup failed: {str(e)}")

        
    except Exception as e:
        status_manager.fail_processing(audio_id, str(e))
        logger.error(f"Error processing video for {audio_id}: {str(e)}")
    
    finally:
        # Don't cleanup here - only cleanup after successful upload
        pass


def process_video_with_queue(queue_request) -> Dict[str, Any]:
    """
    SIMPLIFIED: Process video and return audio path ONLY
    No upload, no video creation, just clone and reconstruct
    """
    from config import settings
    from status_manager import status_manager, ProcessingStatus
    from video_processor.base_processor import AudioProcessor
    from video_processor.file_handler import FileHandler
    from video_processor.video_queue_manager import VideoQueueStatus
    
    audio_id = queue_request.audio_id
    video_source = queue_request.video_source
    is_file_upload = queue_request.is_file_upload
    parameters = queue_request.parameters
    
    # Extract minimal parameters
    target_language = parameters.get("target_language", "English")
    
    # Initialize services
    from video_processor import get_audio_processor
    audio_processor = get_audio_processor()
    
    # Initialize file handler
    file_handler = FileHandler(temp_dir="./tmp/voice_cloning")
    
    video_temp_path = None
    
    try:
        # Check if still processing
        if queue_request.status != VideoQueueStatus.PROCESSING:
            return {"success": False, "error": "Request was cancelled or timed out"}
        
        # Initialize status
        status_manager.update_status(audio_id, ProcessingStatus.PROCESSING, 10, {
            "message": "Starting audio processing",
            "stage": "initialization"
        })
        
        # Handle video download if URL
        if not is_file_upload:
            logger.info(f"Downloading video from URL: {video_source}")
            download_result = file_handler.download_video(
                video_source, 
                audio_id, 
                parameters.get("original_filename")
            )
            
            if not download_result["success"]:
                return {"success": False, "error": download_result["error"]}
            
            video_temp_path = download_result["file_path"]
        else:
            video_temp_path = video_source
        
        # Check if still processing
        if queue_request.status != VideoQueueStatus.PROCESSING:
            return {"success": False, "error": "Request was cancelled or timed out"}
        
        # Process audio segments - SIMPLIFIED
        logger.info(f"Processing audio segments for {audio_id}")
        processing_result = audio_processor.process_audio_segments(
            video_temp_path, 
            audio_id,
            include_instruments=False,  # Keep it simple
            generate_subtitles=False,   # No subtitles
            target_language=target_language
        )
        
        if not processing_result["success"]:
            return {"success": False, "error": processing_result.get("error", "Audio processing failed")}
        
        # Get reconstruction result
        reconstruction_result = processing_result.get("audio_reconstruction", {"success": False, "error": "No reconstruction result from base processor"})
        
        if not reconstruction_result["success"]:
            return {"success": False, "error": f"Audio reconstruction failed: {reconstruction_result['error']}"}
        
        status_manager.set_progress(audio_id, 80)
        
        # Get the output path directly - NO FALLBACK
        final_audio_path = reconstruction_result.get("output_path")
        logger.info(f"Got reconstruction output path: {final_audio_path}")
        
        # Upload to R2
        status_manager.update_status(audio_id, ProcessingStatus.UPLOADING, 90, {
            "message": "Uploading to cloud storage..."
        })
        
        try:
            from r2_storage import R2Storage
            r2_storage = R2Storage()
            
            # Upload final audio
            r2_result = r2_storage.upload_final_audio(audio_id, final_audio_path)
            
            if not r2_result.get("success"):
                raise Exception(f"R2 upload failed: {r2_result.get('error')}")
            
            final_audio_url = r2_result.get("url")
            logger.info(f"Uploaded to R2: {final_audio_url}")
            
        except Exception as e:
            logger.error(f"R2 upload failed: {str(e)}")
            return {"success": False, "error": f"Upload failed: {str(e)}"}
        
        # Mark as completed with R2 URL
        status_manager.update_status(audio_id, ProcessingStatus.COMPLETED, 100, {
            "message": "Processing completed successfully",
            "output_path": final_audio_path,
            "download_url": final_audio_url
        })
        
        # Clean up local file after successful upload
        try:
            if os.path.exists(final_audio_path):
                os.unlink(final_audio_path)
                logger.info(f"Cleaned up local file: {final_audio_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup local file: {str(e)}")
        
        return {
            "success": True,
            "audio_id": audio_id,
            "download_url": final_audio_url,
            "duration": reconstruction_result.get("duration", 0),
            "stats": reconstruction_result.get("reconstruction_stats", {})
        }
        
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        logger.error(error_msg)
        status_manager.fail_processing(audio_id, error_msg)
        return {"success": False, "error": error_msg}