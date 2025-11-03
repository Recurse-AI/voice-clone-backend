"""
Dub Worker - Clean background task processing  
Handles video dubbing workflow with separation of concerns
"""
import logging
import gc
import time
import os
from app.services.dub_service import dub_service
from app.services.simple_status_service import status_service, JobStatus
from app.utils.runpod_service import runpod_service
from app.utils.runpod_monitor import monitor_runpod_job
from app.utils.separation_utils import separation_utils

logger = logging.getLogger(__name__)


def process_dub_task(request_dict: dict, user_id: str):
    job_id = request_dict.get("job_id")
    target_language = request_dict.get("target_language")
    source_video_language = request_dict.get("source_video_language")
    human_review = request_dict.get("humanReview", False)
    video_subtitle = request_dict.get("video_subtitle", False)
    model_type = request_dict.get("model_type", "normal")
    add_subtitle_to_video = request_dict.get("add_subtitle_to_video", False)
    voice_type = request_dict.get("voice_type")
    reference_ids = request_dict.get("reference_ids", [])
    num_of_speakers = request_dict.get("num_of_speakers", 1)
    
    from app.utils.pipeline_utils import mark_dub_job_active, mark_dub_job_inactive, update_dub_job_stage
    
    # No artificial job limits - start immediately
    mark_dub_job_active(job_id, "initialization")
    
    try:
        # Step 1: Validate uploaded files
        validation_result = dub_service.validate_uploaded_files(job_id)
        if not validation_result["success"]:
            dub_service.fail_job(job_id, validation_result["error"], "file_validation_failed")
            return
        
        audio_path = validation_result["audio_path"]
        job_dir = validation_result["job_dir"]
        
        # Step 2: Upload audio to R2
        status_service.update_status(
            job_id, "dub", JobStatus.PROCESSING, 10,
            {"message": "Uploading audio for processing", "phase": "upload"}
        )
        
        upload_result = dub_service.upload_audio_to_r2(job_id, audio_path)
        if not upload_result["success"]:
            dub_service.fail_job(job_id, upload_result["error"], "upload_failed")
            return
        
        audio_url = upload_result["audio_url"]
        r2_key = upload_result["r2_key"]
        
        # Step 3: Start audio separation
        separation_result = _process_audio_separation(job_id, audio_url, job_dir)
        if not separation_result["success"]:
            _cleanup_r2_file(r2_key)
            return
        
        # Step 4: Process dubbing pipeline with separation URLs
        update_dub_job_stage(job_id, "dubbing")
        
        if model_type == "excellent":
            success = _process_elevenlabs_dubbing_api(
                job_id, target_language, source_video_language,
                job_dir, audio_path, num_of_speakers,
                separation_result["runpod_urls"]
            )
        else:
            success = _process_dubbing_pipeline(
                job_id, target_language, source_video_language, 
                job_dir, human_review, separation_result["runpod_urls"], video_subtitle, model_type,
                voice_type, reference_ids, add_subtitle_to_video, num_of_speakers
            )
        
        if not success:
            _cleanup_r2_file(r2_key)
            return
        
        # Step 5: Cleanup
        _cleanup_r2_file(r2_key)
        _cleanup_temp_files(job_id)
        
    except Exception as e:
        try:
            dub_service.fail_job(job_id, f"Worker error: {str(e)}", "worker_error")
        except:
            pass
    finally:
        mark_dub_job_inactive(job_id)


def _process_audio_separation(job_id: str, audio_url: str, job_dir: str) -> dict:
    try:
        from app.utils.pipeline_utils import update_dub_job_stage
        
        # No waiting logic needed - separation workers handle queuing
        update_dub_job_stage(job_id, "separation")
        
        status_service.update_status(
            job_id, "dub", JobStatus.SEPARATING, 25,
            {"message": "Starting audio separation", "phase": "separation"}
        )
        
        logger.info(f"Submitting RunPod separation for {job_id}")
        runpod_request_id = runpod_service.submit_separation_request(audio_url, job_id)
        def on_progress(status: str, progress: int):
            separation_progress = 25 + int((progress / 100.0) * 20)
            status_service.update_status(
                job_id, "dub", JobStatus.SEPARATING, separation_progress,
                {"message": f"Audio separation in progress ({progress}%)", "phase": "separation"}
            )
        
        def on_failed():
            dub_service.fail_job(job_id, "Audio separation failed by RunPod", "separation_failed")
        
        monitor_result = monitor_runpod_job(
            runpod_request_id=runpod_request_id,
            job_id=job_id,
            timeout_seconds=600,
            on_progress=on_progress,
            on_failed=on_failed
        )
        
        if not monitor_result["success"]:
            error_msg = monitor_result.get("error", "Audio separation failed")
            dub_service.fail_job(job_id, error_msg, "separation_failed")
            return {"success": False}
        
        output = monitor_result.get("output", {})
        runpod_urls = separation_utils.extract_urls_from_clearvocals_response(output)
        
        download_success, file_paths = separation_utils.download_separation_files(
            job_id=job_id,
            job_dir=job_dir,
            runpod_urls=runpod_urls,
            on_error_callback=lambda msg, err: dub_service.fail_job(job_id, err, "download_failed")
        )
        
        if not download_success:
            return {"success": False}
        
        # 🚀 CRITICAL FIX: Update pipeline stage tracking to free up separation slot
        update_dub_job_stage(job_id, "transcription")
        
        status_service.update_status(
            job_id, "dub", JobStatus.TRANSCRIBING, 45,
            {
                "message": "Separation completed - starting transcription",
                "phase": "transcription",
                "runpod_urls": runpod_urls
            }
        )
        
        logger.info(f"Separation completed for {job_id}")
        return {"success": True, "runpod_urls": runpod_urls}
        
    except Exception as e:
        logger.error(f"Separation failed for {job_id}: {e}")
        dub_service.fail_job(job_id, f"Separation error: {str(e)}", "separation_error")
        return {"success": False}


def _process_dubbing_pipeline(job_id: str, target_language: str, 
                            source_video_language: str, job_dir: str,
                            human_review: bool, runpod_urls: dict = None, video_subtitle: bool = False, model_type: str = "normal",
                            voice_type: str = None, reference_ids: list = None, add_subtitle_to_video: bool = False, num_of_speakers: int = 1) -> bool:
    try:
        from app.utils.pipeline_utils import update_dub_job_stage
        
        # No waiting logic needed - service workers handle VRAM queuing
        update_dub_job_stage(job_id, "dubbing")
        
        status_service.update_status(
            job_id, "dub", JobStatus.TRANSCRIBING, 46,
            {"message": "Starting dubbing pipeline", "phase": "transcription"}
        )
        
        from app.services.dub.simple_dubbed_api import get_simple_dubbed_api
        api = get_simple_dubbed_api()
        
        pipeline_result = api.process_dubbed_audio(
            job_id=job_id,
            target_language=target_language,
            source_video_language=source_video_language,
            output_dir=job_dir,
            review_mode=human_review,
            separation_urls=runpod_urls,
            video_subtitle=video_subtitle,
            model_type=model_type,
            voice_type=voice_type,
            reference_ids=reference_ids,
            add_subtitle_to_video=add_subtitle_to_video,
            num_of_speakers=num_of_speakers
        )
        
        if not pipeline_result["success"]:
            error = pipeline_result.get("error", "Dubbing pipeline failed")
            dub_service.fail_job(job_id, error, "pipeline_failed")
            return False
        
        # Handle review mode vs completion
        if human_review:
            logger.info(f"Dub job {job_id} ready for review")
            return True
        
        # Complete the job
        result_url = pipeline_result.get("result_url")

        details = {
            "folder_upload": pipeline_result.get("folder_upload", {}),
            "result_urls": pipeline_result.get("result_urls", {}),
            "video_upload": pipeline_result.get("video_upload", {}),
            "details": pipeline_result.get("details", {})
        }

        # Extract manifest URL from pipeline result
        manifest_url = pipeline_result.get("manifest_url")
        manifest_key = pipeline_result.get("manifest_key")
        if manifest_url:
            details["segments_manifest_url"] = manifest_url
        if manifest_key:
            details["segments_manifest_key"] = manifest_key

        success = dub_service.complete_job(job_id, result_url, details)
        if not success:
            dub_service.fail_job(job_id, "Failed to complete job", "completion_failed")
            return False
        
        logger.info(f"Dubbing pipeline completed for {job_id}")
        _cleanup_temp_files(job_id)
        return True
        
    except Exception as e:
        logger.error(f"Dubbing pipeline failed for {job_id}: {e}")
        dub_service.fail_job(job_id, f"Pipeline error: {str(e)}", "pipeline_error")
        return False


def _process_elevenlabs_dubbing_api(
    job_id: str,
    target_language: str,
    source_video_language: str,
    job_dir: str,
    audio_path: str,
    num_of_speakers: int,
    separation_urls: dict
) -> bool:
    """Process dubbing using ElevenLabs Dubbing API"""
    try:
        from app.services.elevenlabs_dubbing_service import get_elevenlabs_dubbing_service
        
        elevenlabs_service = get_elevenlabs_dubbing_service()
        
        result = elevenlabs_service.process_dubbing(
            job_id=job_id,
            target_language=target_language,
            source_video_language=source_video_language,
            job_dir=job_dir,
            audio_path=audio_path,
            num_of_speakers=num_of_speakers,
            separation_urls=separation_urls
        )
        
        if not result.get("success"):
            error = result.get("error", "Unknown error")
            dub_service.fail_job(job_id, error, "elevenlabs_failed")
            return False
        
        # Extract the permanent R2 URLs for separated tracks
        separation_r2_urls = result.get("separation_r2_urls", {})
        result_urls = result.get("result_urls", {})
        
        # Build folder_upload structure to match normal dubbing flow
        folder_upload = {}
        
        # Add vocal audio if available
        if separation_r2_urls.get("vocal_url"):
            folder_upload[f"vocal_{job_id}.wav"] = {
                "success": True,
                "url": separation_r2_urls.get("vocal_url"),
                "size": None
            }
        
        # Add instrumental audio if available
        if separation_r2_urls.get("instrumental_url"):
            folder_upload[f"instrument_{job_id}.wav"] = {
                "success": True,
                "url": separation_r2_urls.get("instrumental_url"),
                "size": None
            }
        
        # Add final dubbed audio
        if result_urls.get("audio_url"):
            folder_upload[f"final_{job_id}.wav"] = {
                "success": True,
                "url": result_urls.get("audio_url"),
                "size": None
            }
        
        # Add final video if available
        if result_urls.get("video_url"):
            folder_upload[f"final_video_{job_id}.mp4"] = {
                "success": True,
                "url": result_urls.get("video_url"),
                "size": None
            }
        
        details = {
            "folder_upload": folder_upload,
            "result_urls": result_urls,
            "dubbing_id": result.get("dubbing_id"),
            "provider": "elevenlabs",
            "has_watermark": True,
            "vocal_audio_url": separation_r2_urls.get("vocal_url"),
            "instrumental_audio_url": separation_r2_urls.get("instrumental_url")
        }
        
        logger.info(f"🎵 Saved permanent separation URLs - Vocal: {separation_r2_urls.get('vocal_url')}, Instrumental: {separation_r2_urls.get('instrumental_url')}")
        logger.info(f"📦 Folder upload structure created with {len(folder_upload)} files")
        
        audio_url = result_urls.get("audio_url")
        video_url = result_urls.get("video_url")
        
        logger.info(f"📦 Result URLs - Audio: {audio_url}, Video: {video_url}")
        
        # Use video URL as primary result if available, otherwise audio
        primary_result_url = video_url if video_url else audio_url
        logger.info(f"✅ Primary result URL: {primary_result_url}")
        
        success = dub_service.complete_job(job_id, primary_result_url, details)
        if not success:
            dub_service.fail_job(job_id, "Failed to complete job", "completion_failed")
            return False
        
        logger.info(f"✅ ElevenLabs dubbing completed for job {job_id}")
        _cleanup_temp_files(job_id)
        return True
        
    except Exception as e:
        logger.error(f"ElevenLabs dubbing failed for {job_id}: {e}")
        dub_service.fail_job(job_id, f"ElevenLabs error: {str(e)}", "elevenlabs_failed")
        return False


def _cleanup_r2_file(r2_key: str):
    """Cleanup R2 file"""
    try:
        if r2_key:
            from app.services.r2_service import R2Service
            r2_service = R2Service()
            r2_service.delete_file(r2_key)
            logger.info(f"Cleaned up R2 file: {r2_key}")
    except Exception as e:
        logger.warning(f"Failed to cleanup R2 file {r2_key}: {e}")


def _cleanup_temp_files(job_id: str):
    """Cleanup temporary files"""
    try:
        from app.utils.cleanup_utils import cleanup_utils
        cleanup_utils.cleanup_job_comprehensive(job_id, "dub")
        logger.info(f"Cleaned up temp files for job: {job_id}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp files for {job_id}: {e}")


# Task function for RQ
def enqueue_dub_task(request_dict: dict, user_id: str):
    """RQ task wrapper with memory management"""
    gc.collect()
    
    try:
        process_dub_task(request_dict, user_id)
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# Redub task processing
def process_redub_task(redub_job_id: str, target_language: str, 
                      source_video_language: str, redub_job_dir: str, 
                      manifest: dict, human_review: bool):
    """Process redub task with existing manifest"""
    logger.info(f"REDUB WORKER: Processing job {redub_job_id}")
    logger.info(f"🔧 DEBUG: Redub worker using manifest model_type = {manifest.get('model_type', 'normal')}")
    
    num_of_speakers = manifest.get('num_of_speakers', 1)
    
    try:
        from app.utils.pipeline_utils import mark_dub_job_active, mark_dub_job_inactive, update_dub_job_stage
        
        # Simplified redub processing - service workers handle VRAM queuing
        mark_dub_job_active(redub_job_id, "redub")
        update_dub_job_stage(redub_job_id, "redub")
        
        try:
            status_service.update_status(
                redub_job_id, "dub", JobStatus.PROCESSING, 60,
                {"message": f"Redubbing to {target_language}", "phase": "dubbing"}
            )
            
            # Use simplified API with manifest override
            from app.services.dub.simple_dubbed_api import get_simple_dubbed_api
            api = get_simple_dubbed_api()
            
            logger.info(f"🔧 DEBUG: Redub using manifest model_type = {manifest.get('model_type', 'normal')}")
            result = api.process_dubbed_audio(
                job_id=redub_job_id,
                target_language=target_language,
                source_video_language=source_video_language,
                output_dir=redub_job_dir,
                review_mode=human_review,
                manifest_override=manifest,
                video_subtitle=False,  # Redubs use existing transcription
                model_type=manifest.get('model_type', 'normal'),
                voice_type=manifest.get('voice_type'),
                reference_ids=manifest.get('reference_ids', []),
                add_subtitle_to_video=manifest.get('add_subtitle_to_video', False),
                num_of_speakers=num_of_speakers
            )
        finally:
            mark_dub_job_inactive(redub_job_id)
        
        if not result["success"]:
            dub_service.fail_job(redub_job_id, result.get("error", "Redub failed"), "redub_failed")
            return
        
        # Handle review vs completion
        if human_review and result.get("review"):
            logger.info(f"Redub job {redub_job_id} ready for review")
            return
        
        # Complete redub
        result_url = result.get("result_url")

        details = {
            "folder_upload": result.get("folder_upload", {}),
            "result_urls": result.get("result_urls", {}),
            "details": result.get("details", {})
        }

        # Extract manifest URL from pipeline result
        manifest_url = result.get("manifest_url")
        manifest_key = result.get("manifest_key")
        if manifest_url:
            details["segments_manifest_url"] = manifest_url
        if manifest_key:
            details["segments_manifest_key"] = manifest_key

        success = dub_service.complete_job(redub_job_id, result_url, details)
        if not success:
            dub_service.fail_job(redub_job_id, "Failed to complete redub", "completion_failed")
            return
        
        logger.info(f"Redub completed: {redub_job_id}")
        _cleanup_temp_files(redub_job_id)
        
    except Exception as e:
        logger.error(f"Redub failed for {redub_job_id}: {e}")
        try:
            dub_service.fail_job(redub_job_id, f"Redub error: {str(e)}", "redub_error")
        except:
            pass
