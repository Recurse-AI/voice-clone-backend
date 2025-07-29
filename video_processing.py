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
import torch
import gc
import time

from status_manager import ProcessingStatus

logger = logging.getLogger(__name__)

def process_video_background(
    video_source: str, audio_id: str, include_instruments: bool,
    generate_subtitles: bool, 
    # OpenVoice specific parameters
    max_length: int, temperature: float, top_p: float, repetition_penalty: float,
    speed_factor: float, target_language: str, language_code: Optional[str], 
    speakers_expected: Optional[int], is_file_upload: bool,
    audio_processor=None, original_filename: Optional[str] = None, 
    original_source_url: Optional[str] = None,
    emotion: Optional[str] = None,
    # Legacy compatibility parameters (ignored)
    max_tokens: Optional[int] = None, cfg_scale: Optional[float] = None, 
    cfg_filter_top_k: Optional[int] = None, use_torch_compile: Optional[bool] = None
):
    """Background video processing with OpenVoice voice cloning (MIT Licensed)"""
    from config import settings
    from status_manager import status_manager, ProcessingStatus
    from r2_storage import R2Storage
    from video_processor.base_processor import AudioProcessor  # Keep for other functions
    from video_processor.clean_processor import clean_audio_processor  # Use global instance
    from video_processor import get_audio_processor  # Get shared AudioProcessor instance
    from utils import cleanup_temp_files, local_storage
    
    # Initialize required services
    r2_storage = R2Storage()
    
    # Initialize audio_processor at the beginning to avoid reference errors
    if audio_processor is None:
        # Use shared AudioProcessor instance for video/audio processing (not voice cloning)
        audio_processor = get_audio_processor()

    # Use global clean audio processor with OpenVoice
    
    status_manager.update_status(audio_id, ProcessingStatus.PENDING, "Starting video processing with OpenVoice...")
    logger.info(f"🎙️ Starting background video processing with OpenVoice for audio_id: {audio_id}")
    
    logger.info(f"📊 OpenVoice Parameters - Temperature: {temperature}, Max Length: {max_length}, Emotion: {emotion}")
    
    total_start_time = time.time()
    
    # Handle video source
    status_manager.update_status(audio_id, ProcessingStatus.DOWNLOADING, 5)
    
    if is_file_upload:
        # File upload: video_source is already a local file path
        video_temp_path = video_source
        filename = original_filename if original_filename else Path(video_source).name
    else:
        # Check if video is available locally first
        # Extract file_id from R2 URL if possible
        file_id = None
        if "/uploads/" in video_source:
            # Extract file_id from R2 URL pattern: .../uploads/{file_id}/{filename}
            try:
                url_parts = video_source.split("/uploads/")[1].split("/")
                if len(url_parts) >= 2:
                    file_id = url_parts[0]
                    logger.info(f"Extracted file_id from upload URL: {file_id}")
            except:
                pass
        elif "/downloaded_videos/" in video_source:
            # Extract file_id from old download pattern: .../downloaded_videos/{date}/{download_id}_{title}.ext
            try:
                # Extract download_id which starts with "video_"
                url_parts = video_source.split("/")
                for part in url_parts:
                    if part.startswith("video_"):
                        # Remove title part if exists (download_id_title format)
                        if "_" in part[6:]:  # Skip "video_" prefix
                            file_id = part.split("_")[0] + "_" + part.split("_")[1] + "_" + part.split("_")[2]  # video_date_uuid
                        else:
                            file_id = part
                        logger.info(f"Extracted file_id from download URL: {file_id}")
                        break
            except:
                pass
        
        # Try to get from local storage first
        local_video_path = None
        if file_id:
            logger.info(f"Checking local storage for file_id: {file_id}")
            local_video_path = local_storage.get_video_path(file_id)
            
        if local_video_path:
            # Use local video - move to processing temp directory
            logger.info(f"Using local video from: {local_video_path}")
            video_temp_path = os.path.join(settings.TEMP_DIR, f"{audio_id}_video.mp4")
            
            # Move from local storage to processing temp
            target_path = local_storage.move_to_processing(file_id, settings.TEMP_DIR)
            if target_path:
                video_temp_path = target_path
                filename = Path(target_path).name
                status_manager.set_progress(audio_id, 15)
                logger.info("Video retrieved from local storage - skipping download")
            else:
                # Fallback to download if move fails
                logger.warning("Failed to move from local storage, falling back to download")
                local_video_path = None
                
        if not local_video_path:
            # Download from URL as fallback
            logger.info(f"Downloading video from URL: {video_source}")
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
        language_code=language_code,
        speakers_expected=speakers_expected,
        include_instruments=include_instruments
    )
    
    if not processing_result["success"]:
        status_manager.fail_processing(audio_id, f"Processing failed: {processing_result.get('error', 'Unknown error')}")
        return
    
    status_manager.set_progress(audio_id, 50)
    
    # Get original audio details from processing result
    audio_path = processing_result["audio_path"]  # Vocal track (for voice cloning)
    instruments_path = processing_result.get("instruments_path")  # Instruments track (for mixing)
    segments_dir = processing_result["segments_dir"]  # Directory with segments
    total_segments = processing_result.get("total_segments", 0)
    separation_performed = processing_result.get("separation_performed", False)
    
    # Load original audio details
    original_audio, original_sr = sf.read(audio_path)
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
        "language_code": language_code,
        "speakers_expected": speakers_expected,
        "detected_speakers": processing_result.get("detected_speakers", total_segments)
    }
    
    # Clone voices with enhanced parameters
    status_manager.update_status(
        audio_id, 
        ProcessingStatus.PROCESSING, 
        progress=60, 
        details={"message": "Starting enhanced voice cloning process..."}
    )
    
    cloning_result = clean_audio_processor.process_voice_cloning_only(
        segments_dir=processing_result["segments_dir"],
        audio_id=audio_id,
        max_length=max_tokens,
        temperature=temperature,
        top_p=top_p,
        speed_factor=speed_factor,
        seed=settings.OPENVOICE_DEFAULT_SEED
    )
    
    if not cloning_result["success"]:
        status_manager.fail_processing(audio_id, f"Voice cloning failed: {cloning_result.get('error', 'Unknown error')}")
        return
    
    # Clean up memory after voice cloning
    clean_audio_processor.clear_model()
    
    # Update status after voice cloning completion
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
            instruments_path=instruments_path # Use the instruments_path from processing_result
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
        instruments_for_video = None # No instruments separated here
        video_result = audio_processor.create_video_with_subtitles(
            video_temp_path,
            final_audio_path,
            processing_result["segments_dir"],
            audio_id,
            instruments_for_video
        )
    else:
        instruments_for_video = None # No instruments separated here
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
        r2_vocal_result = r2_storage.upload_file(audio_path, vocal_r2_key, "audio/wav") # Use audio_path here
        
        # Upload instruments file if separation was performed
        if separation_performed and instruments_path and os.path.exists(instruments_path):
            instruments_filename = f"instruments_separated_{audio_id}.wav"
            instruments_r2_key = r2_storage.generate_file_path(audio_id, "instruments", instruments_filename)
            r2_instruments_result = r2_storage.upload_file(instruments_path, instruments_r2_key, "audio/wav")
            logger.info(f"✅ Uploaded separated instruments to R2")
        else:
            r2_instruments_result = {"success": True, "message": "No instrument separation performed", "url": None}
            logger.info(f"ℹ️ No instruments to upload (separation_performed={separation_performed})")
        
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
                "vocal_separation": separation_performed,
                "voice_cloning": True,
                "subtitle_generation": generate_subtitles,
                "instrument_mixing": separation_performed and include_instruments
            },
            "processing_timeline": {
                "transcription_source": "AssemblyAI",
                "voice_cloning_model": "OpenVoice",
                "separation_service": "RunPod"
            }
        },
        "segment_details": audio_processor.get_processing_stats(segments_dir),
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
        "processing_stats": audio_processor.get_processing_stats(segments_dir),
        "cloning_results": cloning_result,
        "reconstruction_results": reconstruction_result,
        "video_generated": video_result is not None and video_result["success"],
        "subtitles_generated": generate_subtitles and video_result is not None and video_result.get("subtitle_count", 0) > 0,
        "separation_used": True
    }
    
    r2_storage.create_processing_summary(audio_id, processing_data)

    


def process_video_with_queue(queue_request) -> Dict[str, Any]:
    """Process video with queue management - using global OpenVoice processor"""
    import time
    import logging
    from config import settings
    from status_manager import status_manager, ProcessingStatus
    from r2_storage import R2Storage
    from video_processor.base_processor import AudioProcessor  # Keep for other functions
    from video_processor.clean_processor import clean_audio_processor  # Use global instance
    from video_processor.file_handler import FileHandler
    from video_processor.video_queue_manager import VideoQueueStatus
    from utils import cleanup_temp_files
    
    logger = logging.getLogger(__name__)
    
    # Initialize variables to avoid UnboundLocalError
    video_temp_path = None
    
    # Extract parameters from queue request
    audio_id = queue_request.audio_id
    video_source = queue_request.video_source
    is_file_upload = queue_request.is_file_upload
    parameters = queue_request.parameters
    
    include_instruments = parameters.get("include_instruments", True)
    generate_subtitles = parameters.get("generate_subtitles", True)
    # OpenVoice parameters with defaults
    max_length = parameters.get("max_length", parameters.get("max_tokens", settings.OPENVOICE_MAX_LENGTH))
    max_tokens = parameters.get("max_tokens", settings.OPENVOICE_MAX_LENGTH)
    cfg_scale = parameters.get("cfg_scale", 3.0)  # Legacy parameter
    temperature = parameters.get("temperature", settings.OPENVOICE_TEMPERATURE)
    top_p = parameters.get("top_p", settings.OPENVOICE_TOP_P)
    cfg_filter_top_k = parameters.get("cfg_filter_top_k", 50)  # Legacy parameter - use default
    speed_factor = parameters.get("speed_factor", 1.0)  # Default speed
    use_torch_compile = parameters.get("use_torch_compile", True)  # Legacy parameter
    target_language = parameters.get("target_language", "English")
    language_code = parameters.get("language_code")
    speakers_expected = parameters.get("speakers_expected", 1)
    original_filename = parameters.get("original_filename")
    original_source_url = parameters.get("original_source_url")
    # New OpenVoice specific parameters
    repetition_penalty = parameters.get("repetition_penalty", settings.OPENVOICE_REPETITION_PENALTY)
    emotion = parameters.get("emotion", settings.OPENVOICE_DEFAULT_EMOTION)
    
    # Initialize services
    r2_storage = R2Storage()
    
    try:
        # Use global clean audio processor with OpenVoice
        
        status_manager.update_status(audio_id, ProcessingStatus.PENDING, "Starting video processing with OpenVoice...")
        logger.info(f"🎙️ Starting background video processing with OpenVoice for audio_id: {audio_id}")
        
        logger.info(f"📊 OpenVoice Parameters - Temperature: {temperature}, Max Length: {max_length}, Emotion: {emotion}")
        
        total_start_time = time.time()
        
        # Initialize file handler
        file_handler = FileHandler(settings.TEMP_DIR)
        
        # Phase 1: Handle file source (0% → 10%)
        success, local_file_path, error = file_handler.handle_video_source(
            video_source, audio_id, is_file_upload, status_manager
        )
        
        if not success:
            status_manager.fail_processing(audio_id, f"Video source handling failed: {error}")
            return {"success": False, "error": f"Video source handling failed: {error}"}
        
        video_temp_path = local_file_path
        logger.info(f"✅ Video source handled successfully: {video_temp_path}")
        
        # Phase 2: Audio separation and processing (10% → 90%)
        logger.info(f"🎤 Starting audio processing with OpenVoice for {audio_id}")
        
        # Use clean audio processor with OpenVoice for voice cloning
        processing_result = clean_audio_processor.process_audio_complete(
            audio_path=video_temp_path,
            audio_id=audio_id,
            target_language=target_language,
            language_code=language_code,
            speakers_expected=speakers_expected,
            # OpenVoice parameters
            max_length=max_length or max_tokens,  # Map legacy parameter
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            speed_factor=speed_factor,
            seed=settings.OPENVOICE_DEFAULT_SEED,  # Use consistent seed from config
            emotion=emotion
        )
        
        if not processing_result.get("success", False):
            error_msg = f"OpenVoice audio processing failed: {processing_result.get('error', 'Unknown error')}"
            status_manager.fail_processing(audio_id, error_msg)
            logger.error(f"❌ {error_msg}")
            return {"success": False, "error": error_msg, "details": processing_result}
        
        logger.info(f"✅ OpenVoice audio processing completed successfully")
        
        # Phase 3: Final audio reconstruction and upload (90% → 100%)
        status_manager.update_status(audio_id, ProcessingStatus.UPLOADING, 90, "Uploading processed audio...")
        
        # Validate processing_result before accessing its content
        if not isinstance(processing_result, dict):
            error_msg = f"Processing result is not a dictionary: {type(processing_result)} - {processing_result}"
            logger.error(f"❌ {error_msg}")
            status_manager.fail_processing(audio_id, error_msg)
            return {"success": False, "error": error_msg}
        
        # Check if output section exists
        if "output" not in processing_result:
            error_msg = "Output section missing from processing result"
            logger.error(f"❌ {error_msg}")
            status_manager.fail_processing(audio_id, error_msg)
            return {"success": False, "error": error_msg}
        
        # Check if final_audio_path exists
        if "final_audio_path" not in processing_result["output"]:
            error_msg = "Final audio path missing from processing result output"
            logger.error(f"❌ {error_msg}")
            status_manager.fail_processing(audio_id, error_msg)
            return {"success": False, "error": error_msg}
        
        final_audio_path = processing_result["output"]["final_audio_path"]
        
        if not os.path.exists(final_audio_path):
            error_msg = "Final audio file not found after processing"
            status_manager.fail_processing(audio_id, error_msg)
            return {"success": False, "error": error_msg}
        
        # Upload final audio to R2
        audio_key = f"processed-audio/{audio_id}/final_audio.wav"
        upload_result = r2_storage.upload_file(final_audio_path, audio_key, "audio/wav")
        
        if not upload_result.get("success"):
            error_msg = f"Failed to upload final audio: {upload_result.get('error', 'Unknown error')}"
            status_manager.fail_processing(audio_id, error_msg)
            return {"success": False, "error": error_msg}
        
        final_audio_url = upload_result["url"]
        
        # Generate video with subtitles if requested
        final_video_url = None
        if generate_subtitles:
            # Implementation for video generation would go here
            pass
        
        # Extract voice cloning stats (processing_result is already validated as dict)
        voice_cloning_data = processing_result.get("voice_cloning", {})
        if isinstance(voice_cloning_data, dict):
            total_segments = voice_cloning_data.get("total_segments", 0)
            successful_segments = voice_cloning_data.get("successful_segments", 0)
        else:
            # Fallback if voice_cloning section is not a dict
            total_segments = 0
            successful_segments = 0
            logger.warning(f"⚠️ voice_cloning data is not a dict: {type(voice_cloning_data)}")

        status_manager.complete_processing(audio_id, {
            "final_audio_url": final_audio_url,
            "final_video_url": final_video_url,
            "processing_stats": {
                "total_segments": total_segments,
                "successful_segments": successful_segments,
                "model_used": "OpenVoice"
            },
            "metadata": {
                "original_filename": original_filename,
                "original_source_url": original_source_url,
                "processing_timeline": {
                    "transcription_source": "AssemblyAI",
                    "voice_cloning_model": "OpenVoice",
                    "separation_service": "RunPod"
                }
            }
        })
        
        logger.info(f"🎉 Queue processing completed successfully for {audio_id}")
        
        return {
            "success": True,
            "audio_id": audio_id,
            "final_audio_url": final_audio_url,
            "final_video_url": final_video_url,
            "model_used": "OpenVoice",
            "processing_result": processing_result
        }
        
    except Exception as e:
        error_msg = f"Queue processing failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        status_manager.fail_processing(audio_id, error_msg)
        return {"success": False, "error": error_msg}
    
    finally:
        # Clean up temporary files
        try:
            cleanup_temp_files(audio_id, None, None, video_temp_path)
        except Exception:
            pass