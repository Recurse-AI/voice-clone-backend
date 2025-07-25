"""
Video Background Processing
Contains the main video processing function moved from main.py for better organization
"""

import os
import urllib.parse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import soundfile as sf
import requests

logger = logging.getLogger(__name__)

def process_video_background(
    video_source: str, audio_id: str, include_instruments: bool,
    generate_subtitles: bool, temperature: float, cfg_scale: float, top_p: float,
    target_language: str, language_code: Optional[str], speakers_expected: Optional[int], is_file_upload: bool,
    audio_processor=None, original_filename: Optional[str] = None, original_source_url: Optional[str] = None
):
    """Background video processing - handles both URL and file inputs"""
    from config import settings
    from status_manager import status_manager, ProcessingStatus
    from r2_storage import R2Storage
    from video_processor.base_processor import AudioProcessor
    from utils import cleanup_temp_files
    
    # Initialize required services
    r2_storage = R2Storage()
    
    # Use passed audio_processor or create new one
    if audio_processor is None:
        from video_processor import get_audio_processor
        audio_processor = get_audio_processor()
    
    final_language_code = language_code if language_code and language_code.strip() else None
    video_temp_path = None
    
    try:
        # Handle video source
        status_manager.update_status(audio_id, ProcessingStatus.DOWNLOADING, 5)
        
        if is_file_upload:
            # File upload: video_source is already a local file path
            video_temp_path = video_source
            filename = original_filename if original_filename else Path(video_source).name
        else:
            # URL download: download the video with progress tracking
            video_temp_path = os.path.join(settings.TEMP_DIR, f"{audio_id}_video.mp4")
            
            parsed_url = urllib.parse.urlparse(video_source)
            filename = os.path.basename(parsed_url.path)
            if not filename or '.' not in filename:
                filename = "video.mp4"
            
            try:
                response = requests.get(video_source, stream=True, timeout=300)
                response.raise_for_status()
                
                # Get content length for progress tracking
                content_length = response.headers.get('content-length')
                if content_length:
                    content_length = int(content_length)
                
                downloaded = 0
                chunk_size = 8192
                progress_updated = False
                
                with open(video_temp_path, "wb") as buffer:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            buffer.write(chunk)
                            downloaded += len(chunk)
                            
                            # Simple progress - only once at 50% completion
                            if content_length and not progress_updated:
                                if downloaded >= content_length * 0.5:
                                    status_manager.set_progress(audio_id, 15)
                                    logger.info("Download 50% completed")
                                    progress_updated = True
                
            except requests.exceptions.RequestException as e:
                status_manager.fail_processing(audio_id, f"Download failed: {str(e)}")
                return
        
        status_manager.set_progress(audio_id, 20)
        
        # Process video
        status_manager.update_status(audio_id, ProcessingStatus.PROCESSING, 30)
        
        processing_result = audio_processor.process_video_with_separation(
            video_temp_path,
            audio_id,
            target_language,
            language_code=final_language_code,
            speakers_expected=speakers_expected
        )
        
        if not processing_result["success"]:
            status_manager.fail_processing(audio_id, f"Processing failed: {processing_result.get('error', 'Unknown error')}")
            return
        
        status_manager.set_progress(audio_id, 50)
        
        # Get original audio details
        vocal_path = processing_result["vocal_path"]
        separated_instruments_path = processing_result["instruments_path"]
        
        original_audio, original_sr = sf.read(vocal_path)
        original_duration = len(original_audio) / original_sr
        file_size = os.path.getsize(video_temp_path)
        
        # Set proper source URL for consistency
        if is_file_upload:
            source_url = original_filename if original_filename else filename
        else:
            source_url = original_source_url if original_source_url else video_source
        
        original_audio_details = {
            "filename": filename,
            "source_url": source_url,
            "source_type": "url" if not is_file_upload else "file_upload",
            "duration": original_duration,
            "sample_rate": original_sr,
            "channels": len(original_audio.shape) if len(original_audio.shape) > 1 else 1,
            "size_mb": file_size / (1024 * 1024),
            "processing_type": "video_with_separation",
            "language_code": final_language_code,
            "speakers_expected": speakers_expected,
            "detected_speakers": processing_result.get("detected_speakers", len(processing_result.get("speakers", [])))
        }
        
        # Clone voices at natural speed
        from status_manager import ProcessingStatus
        status_manager.update_status(
            audio_id, 
            ProcessingStatus.PROCESSING, 
            progress=60, 
            details={"message": "Starting voice cloning process..."}
        )
        
        cloning_result = audio_processor.clone_voice_segments(
            processing_result["segments_dir"],
            audio_id,
            temperature=temperature,
            cfg_scale=cfg_scale,
            top_p=top_p,
            seed=settings.DEFAULT_SEED
        )
        
        if not cloning_result["success"]:
            status_manager.fail_processing(audio_id, f"Voice cloning failed: {cloning_result.get('error', 'Unknown error')}")
            return
        
        audio_processor.voice_cloning_service.clear_cache()
        
        # Update status after voice cloning completion
        from status_manager import ProcessingStatus
        status_manager.update_status(
            audio_id, 
            ProcessingStatus.PROCESSING, 
            progress=90, 
            details={"message": "Voice cloning completed, reconstructing final audio..."}
        )
        
        # Reconstruct audio
        if include_instruments:
            reconstruction_result = audio_processor.reconstruct_final_audio(
                processing_result["segments_dir"],
                audio_id,
                include_instruments=True,
                instruments_path=separated_instruments_path
            )
        else:
            reconstruction_result = audio_processor.reconstruct_final_audio(
                processing_result["segments_dir"],
                audio_id,
                include_instruments=False,
                instruments_path=None
            )
        
        if not reconstruction_result["success"]:
            status_manager.fail_processing(audio_id, f"Audio reconstruction failed: {reconstruction_result['error']}")
            return
        
        status_manager.set_progress(audio_id, 80)
        final_audio_path = reconstruction_result["final_audio_path"]
        
        # Create video
        video_result = None
        if generate_subtitles:
            instruments_for_video = separated_instruments_path if include_instruments else None
            video_result = audio_processor.create_video_with_subtitles(
                video_temp_path,
                final_audio_path,
                processing_result["segments_dir"],
                audio_id,
                instruments_for_video
            )
        else:
            instruments_for_video = separated_instruments_path if include_instruments else None
            video_result = audio_processor.create_video_with_audio(
                video_temp_path,
                final_audio_path,
                audio_id,
                instruments_for_video,
                processing_result["segments_dir"]
            )
        
        if not video_result or not video_result["success"]:
            error_msg = video_result.get('error', 'Unknown error') if video_result else 'No result'
            status_manager.fail_processing(audio_id, f"Video processing failed: {error_msg}")
            return
        
        # Upload files
        status_manager.update_status(audio_id, ProcessingStatus.UPLOADING, 90)
        
        try:
            r2_segments_result = r2_storage.upload_audio_segments(audio_id, processing_result["segments_dir"])
            r2_final_result = r2_storage.upload_final_audio(audio_id, final_audio_path)
            
            # Upload vocal and instruments files
            vocal_filename = f"vocal_separated_{audio_id}.wav"
            vocal_r2_key = r2_storage.generate_file_path(audio_id, "vocal", vocal_filename)
            r2_vocal_result = r2_storage.upload_file(vocal_path, vocal_r2_key, "audio/wav")
            
            instruments_filename = f"instruments_separated_{audio_id}.wav"
            instruments_r2_key = r2_storage.generate_file_path(audio_id, "instruments", instruments_filename)
            r2_instruments_result = r2_storage.upload_file(separated_instruments_path, instruments_r2_key, "audio/wav")
            
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
            status_manager.fail_processing(audio_id, f"File upload failed: {str(e)}")
            return
        
        # Mark as completed
        status_manager.complete_processing(audio_id, {
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
                "cloning_parameters": {
                    "temperature": temperature,
                    "cfg_scale": cfg_scale,
                    "top_p": top_p,
                    "seed_used": cloning_result.get("seed_used", settings.DEFAULT_SEED),
                    "seeds_used": cloning_result.get("seeds_used", {})
                },
                "features_used": {
                    "vocal_separation": True,
                    "voice_cloning": True,
                    "subtitle_generation": generate_subtitles,
                    "instrument_mixing": include_instruments
                },
                "processing_timeline": {
                    "transcription_source": "AssemblyAI",
                    "voice_cloning_model": "Dia",
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
        })
        
        # Create processing summary
        processing_data = {
            "audio_id": audio_id,
            "original_audio": original_audio_details,
            "seeds_used": cloning_result.get("seeds_used", {}),
            "parameters": {
                "temperature": temperature,
                "cfg_scale": cfg_scale,
                "top_p": top_p,
                "target_language": target_language,
                "include_instruments": include_instruments,
                "generate_subtitles": generate_subtitles,
                "video_provided": True,
                "video_source": "url" if not is_file_upload else "file_upload",
                "video_url": video_source if not is_file_upload else None,
                "runpod_separation": True
            },
            "processing_stats": audio_processor.get_processing_stats(processing_result["segments_dir"]),
            "cloning_results": cloning_result,
            "reconstruction_results": reconstruction_result,
            "video_generated": video_result is not None and video_result["success"],
            "subtitles_generated": generate_subtitles and video_result is not None and video_result.get("subtitle_count", 0) > 0,
            "separation_used": True
        }
        
        r2_storage.create_processing_summary(audio_id, processing_data)

        
    except Exception as e:
        status_manager.fail_processing(audio_id, f"Unexpected error: {str(e)}")
        
    finally:
        # Clean up temp files
        try:
            cleanup_temp_files(audio_id, None, None, video_temp_path)
        except Exception:
            pass


def process_video_with_queue(queue_request) -> Dict[str, Any]:
    """
    Process video using queue system with timeout monitoring
    
    Args:
        queue_request: VideoQueueRequest object containing all processing parameters
        
    Returns:
        Result dictionary with success/error information
    """
    from config import settings
    from status_manager import status_manager
    from r2_storage import R2Storage
    from video_processor.base_processor import AudioProcessor
    from video_processor.file_handler import FileHandler
    from video_processor.video_queue_manager import VideoQueueStatus
    from utils import cleanup_temp_files
    
    audio_id = queue_request.audio_id
    video_source = queue_request.video_source
    is_file_upload = queue_request.is_file_upload
    parameters = queue_request.parameters
    
    # Extract parameters
    include_instruments = parameters.get("include_instruments", True)
    generate_subtitles = parameters.get("generate_subtitles", True)
    temperature = parameters.get("temperature", settings.DIA_TEMPERATURE)
    cfg_scale = parameters.get("cfg_scale", settings.DIA_CFG_SCALE)
    top_p = parameters.get("top_p", settings.DIA_TOP_P)
    target_language = parameters.get("target_language", "English")
    language_code = parameters.get("language_code")
    speakers_expected = parameters.get("speakers_expected", 1)
    original_filename = parameters.get("original_filename")
    original_source_url = parameters.get("original_source_url")
    
    # Initialize services
    r2_storage = R2Storage()
    from video_processor import get_audio_processor
    audio_processor = get_audio_processor()
    file_handler = FileHandler(settings.TEMP_DIR)
    
    video_temp_path = None
    
    try:
        # Check if request is still valid (not cancelled/timed out)
        if queue_request.status != VideoQueueStatus.PROCESSING:
            return {"success": False, "error": "Request was cancelled or timed out"}
        
        # Phase 1: Handle file source (0% → 10%)
        success, local_file_path, error = file_handler.handle_video_source(
            video_source, audio_id, is_file_upload, status_manager
        )
        
        if not success:
            return {"success": False, "error": error}
        
        video_temp_path = local_file_path
        
        # Check if still processing
        if queue_request.status != VideoQueueStatus.PROCESSING:
            return {"success": False, "error": "Request was cancelled or timed out"}
        
        # Phase 2: Video processing (10% → 100%)
        status_manager.set_progress(audio_id, 20)
        
        processing_result = audio_processor.process_video_with_separation(
            video_temp_path,
            audio_id,
            target_language,
            language_code=language_code,
            speakers_expected=speakers_expected
        )
        
        if not processing_result["success"]:
            return {"success": False, "error": f"Processing failed: {processing_result.get('error', 'Unknown error')}"}
        
        # Check if still processing
        if queue_request.status != VideoQueueStatus.PROCESSING:
            return {"success": False, "error": "Request was cancelled or timed out"}
        
        status_manager.set_progress(audio_id, 50)
        
        # Get original audio details
        vocal_path = processing_result["vocal_path"]
        separated_instruments_path = processing_result["instruments_path"]
        
        import soundfile as sf
        original_audio, original_sr = sf.read(vocal_path)
        original_duration = len(original_audio) / original_sr
        file_size = os.path.getsize(video_temp_path)
         
        # Set proper filename and source URL (will be updated after R2 upload)
        if is_file_upload:
            display_filename = original_filename if original_filename else os.path.basename(video_temp_path)
            source_url = original_filename  # Temporary - will be updated with R2 URL
        else:
            display_filename = os.path.basename(video_temp_path)
            source_url = original_source_url if original_source_url else video_source
        
        original_audio_details = {
            "filename": display_filename,
            "source_url": source_url,
            "source_type": "url" if not is_file_upload else "file_upload",
            "duration": original_duration,
            "sample_rate": original_sr,
            "channels": len(original_audio.shape) if len(original_audio.shape) > 1 else 1,
            "size_mb": file_size / (1024 * 1024),
            "processing_type": "video_with_separation",
            "language_code": language_code,
            "speakers_expected": speakers_expected,
            "detected_speakers": processing_result.get("detected_speakers", len(processing_result.get("speakers", [])))
        }
        
        # Check if still processing
        if queue_request.status != VideoQueueStatus.PROCESSING:
            return {"success": False, "error": "Request was cancelled or timed out"}
        
        # Clone voices at natural speed
        from status_manager import ProcessingStatus
        status_manager.update_status(
            audio_id, 
            ProcessingStatus.PROCESSING, 
            progress=60, 
            details={"message": "Starting voice cloning process..."}
        )
        
        cloning_result = audio_processor.clone_voice_segments(
            processing_result["segments_dir"],
            audio_id,
            temperature=temperature,
            cfg_scale=cfg_scale,
            top_p=top_p,
            seed=settings.DEFAULT_SEED
        )
        
        if not cloning_result["success"]:
            return {"success": False, "error": f"Voice cloning failed: {cloning_result.get('error', 'Unknown error')}"}
        
        audio_processor.voice_cloning_service.clear_cache()
        
        # Check if still processing
        if queue_request.status != VideoQueueStatus.PROCESSING:
            return {"success": False, "error": "Request was cancelled or timed out"}
        
        # Update status after voice cloning completion
        from status_manager import ProcessingStatus
        status_manager.update_status(
            audio_id, 
            ProcessingStatus.PROCESSING, 
            progress=90, 
            details={"message": "Voice cloning completed, reconstructing final audio..."}
        )
        
        # Reconstruct audio
        if include_instruments:
            reconstruction_result = audio_processor.reconstruct_final_audio(
                processing_result["segments_dir"],
                audio_id,
                include_instruments=True,
                instruments_path=separated_instruments_path
            )
        else:
            reconstruction_result = audio_processor.reconstruct_final_audio(
                processing_result["segments_dir"],
                audio_id,
                include_instruments=False,
                instruments_path=None
            )
        
        if not reconstruction_result["success"]:
            return {"success": False, "error": f"Audio reconstruction failed: {reconstruction_result['error']}"}
        
        status_manager.set_progress(audio_id, 80)
        final_audio_path = reconstruction_result["final_audio_path"]
        
        # Check if still processing
        if queue_request.status != VideoQueueStatus.PROCESSING:
            return {"success": False, "error": "Request was cancelled or timed out"}
        
        # Create video
        video_result = None
        if generate_subtitles:
            instruments_for_video = separated_instruments_path if include_instruments else None
            video_result = audio_processor.create_video_with_subtitles(
                video_temp_path,
                final_audio_path,
                processing_result["segments_dir"],
                audio_id,
                instruments_for_video
            )
        else:
            instruments_for_video = separated_instruments_path if include_instruments else None
            video_result = audio_processor.create_video_with_audio(
                video_temp_path,
                final_audio_path,
                audio_id,
                instruments_for_video,
                processing_result["segments_dir"]
            )
        
        if not video_result or not video_result["success"]:
            error_msg = video_result.get('error', 'Unknown error') if video_result else 'No result'
            return {"success": False, "error": f"Video processing failed: {error_msg}"}
        
        # Upload files
        from status_manager import ProcessingStatus
        status_manager.update_status(audio_id, ProcessingStatus.UPLOADING, 90)
        
        try:
            r2_segments_result = r2_storage.upload_audio_segments(audio_id, processing_result["segments_dir"])
            r2_final_result = r2_storage.upload_final_audio(audio_id, final_audio_path)
            
            # Upload vocal and instruments files
            vocal_filename = f"vocal_separated_{audio_id}.wav"
            vocal_r2_key = r2_storage.generate_file_path(audio_id, "vocal", vocal_filename)
            r2_vocal_result = r2_storage.upload_file(vocal_path, vocal_r2_key, "audio/wav")
            
            instruments_filename = f"instruments_separated_{audio_id}.wav"
            instruments_r2_key = r2_storage.generate_file_path(audio_id, "instruments", instruments_filename)
            r2_instruments_result = r2_storage.upload_file(separated_instruments_path, instruments_r2_key, "audio/wav")
            
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
            return {"success": False, "error": f"File upload failed: {str(e)}"}
        
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
                "cloning_parameters": {
                    "temperature": temperature,
                    "cfg_scale": cfg_scale,
                    "top_p": top_p,
                    "seed_used": cloning_result.get("seed_used", settings.DEFAULT_SEED),
                    "seeds_used": cloning_result.get("seeds_used", {})
                },
                "features_used": {
                    "vocal_separation": True,
                    "voice_cloning": True,
                    "subtitle_generation": generate_subtitles,
                    "instrument_mixing": include_instruments
                },
                "processing_timeline": {
                    "transcription_source": "AssemblyAI",
                    "voice_cloning_model": "Dia",
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
                "temperature": temperature,
                "cfg_scale": cfg_scale,
                "top_p": top_p,
                "target_language": target_language,
                "include_instruments": include_instruments,
                "generate_subtitles": generate_subtitles,
                "video_provided": True,
                "video_source": "url" if not is_file_upload else "file_upload",
                "video_url": video_source if not is_file_upload else None,
                "runpod_separation": True
            },
            "processing_stats": audio_processor.get_processing_stats(processing_result["segments_dir"]),
            "cloning_results": cloning_result,
            "reconstruction_results": reconstruction_result,
            "video_generated": video_result is not None and video_result["success"],
            "subtitles_generated": generate_subtitles and video_result is not None and video_result.get("subtitle_count", 0) > 0,
            "separation_used": True
        }
        
        r2_storage.create_processing_summary(audio_id, processing_data)
        
        return {"success": True, "result": completion_data}
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        status_manager.fail_processing(audio_id, error_msg)
        return {"success": False, "error": error_msg}
        
    finally:
        # Clean up temp files
        try:
            cleanup_temp_files(audio_id, None, None, video_temp_path)
        except Exception:
            pass