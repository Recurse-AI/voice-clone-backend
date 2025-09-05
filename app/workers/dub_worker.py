"""
Dub Worker - Clean background task processing
Handles video dubbing workflow with separation of concerns
"""
import logging
import gc
from app.services.dub_service import dub_service
from app.services.simple_status_service import status_service, JobStatus
from app.utils.runpod_service import runpod_service
from app.utils.runpod_monitor import monitor_runpod_job
from app.utils.separation_utils import separation_utils

logger = logging.getLogger(__name__)


def process_dub_task(request_dict: dict, user_id: str):
    """
    Clean dub task processing
    Single responsibility: Process video dubbing workflow
    """
    job_id = request_dict.get("job_id")
    target_language = request_dict.get("target_language")
    source_video_language = request_dict.get("source_video_language")
    human_review = request_dict.get("humanReview", False)
    
    logger.info(f"DUB WORKER: Processing job {job_id}")
    
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
        success = _process_audio_separation(job_id, audio_url, job_dir)
        if not success:
            _cleanup_r2_file(r2_key)
            return
        
        # Step 4: Process dubbing pipeline
        success = _process_dubbing_pipeline(
            job_id, target_language, source_video_language, 
            job_dir, human_review
        )
        
        if not success:
            _cleanup_r2_file(r2_key)
            return
        
        # Step 5: Cleanup
        _cleanup_r2_file(r2_key)
        
        logger.info(f"DUB WORKER: Completed job {job_id}")
        
    except Exception as e:
        logger.error(f"DUB WORKER: Failed job {job_id}: {e}")
        try:
            dub_service.fail_job(job_id, f"Worker error: {str(e)}", "worker_error")
        except:
            pass  # Avoid double error


def _process_audio_separation(job_id: str, audio_url: str, job_dir: str) -> bool:
    """Process audio separation step"""
    try:
        # Update status to separation
        status_service.update_status(
            job_id, "dub", JobStatus.SEPARATING, 25,
            {"message": "Starting audio separation", "phase": "separation"}
        )
        
        # Submit to RunPod
        logger.info(f"Submitting RunPod separation for {job_id}")
        runpod_request_id = runpod_service.submit_separation_request(audio_url, job_id)
        
        # Monitor separation progress
        def on_progress(status: str, progress: int):
            # Map RunPod progress (0-100) to separation range (25-45)
            separation_progress = 25 + int((progress / 100.0) * 20)
            status_service.update_status(
                job_id, "dub", JobStatus.SEPARATING, separation_progress,
                {"message": f"Audio separation in progress ({progress}%)", "phase": "separation"}
            )
        
        def on_failed():
            dub_service.fail_job(job_id, "Audio separation failed by RunPod", "separation_failed")
        
        # Monitor RunPod job
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
            return False
        
        # Download separation files
        output = monitor_result.get("output", {})
        runpod_urls = separation_utils.extract_urls_from_clearvocals_response(output)
        
        download_success, file_paths = separation_utils.download_separation_files(
            job_id=job_id,
            job_dir=job_dir,
            runpod_urls=runpod_urls,
            on_error_callback=lambda msg, err: dub_service.fail_job(job_id, err, "download_failed")
        )
        
        if not download_success:
            return False
        
        # Update status with separation complete
        status_service.update_status(
            job_id, "dub", JobStatus.TRANSCRIBING, 45,
            {
                "message": "Separation completed - starting transcription",
                "phase": "transcription",
                "runpod_urls": runpod_urls
            }
        )
        
        logger.info(f"Separation completed for {job_id}")
        return True
        
    except Exception as e:
        logger.error(f"Separation failed for {job_id}: {e}")
        dub_service.fail_job(job_id, f"Separation error: {str(e)}", "separation_error")
        return False


def _process_dubbing_pipeline(job_id: str, target_language: str, 
                            source_video_language: str, job_dir: str,
                            human_review: bool) -> bool:
    """Process the dubbing pipeline using simplified API"""
    try:
        # Update status to processing
        status_service.update_status(
            job_id, "dub", JobStatus.PROCESSING, 60,
            {"message": "Starting dubbing pipeline", "phase": "dubbing"}
        )
        
        # Use simplified dubbing API
        from app.services.dub.simple_dubbed_api import get_simple_dubbed_api
        api = get_simple_dubbed_api()
        
        pipeline_result = api.process_dubbed_audio(
            job_id=job_id,
            target_language=target_language,
            source_video_language=source_video_language,
            output_dir=job_dir,
            review_mode=human_review
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
        
        success = dub_service.complete_job(job_id, result_url, details)
        if not success:
            dub_service.fail_job(job_id, "Failed to complete job", "completion_failed")
            return False
        
        logger.info(f"Dubbing pipeline completed for {job_id}")
        return True
        
    except Exception as e:
        logger.error(f"Dubbing pipeline failed for {job_id}: {e}")
        dub_service.fail_job(job_id, f"Pipeline error: {str(e)}", "pipeline_error")
        return False


def _cleanup_r2_file(r2_key: str):
    """Cleanup R2 file"""
    try:
        if r2_key:
            from app.services.r2_service import get_r2_service
            r2_service = get_r2_service()
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
    """RQ task wrapper"""
    # Memory cleanup before processing
    gc.collect()
    
    try:
        process_dub_task(request_dict, user_id)
    finally:
        # Memory cleanup after processing
        gc.collect()


# Redub task processing
def process_redub_task(redub_job_id: str, target_language: str, 
                      source_video_language: str, redub_job_dir: str, 
                      manifest: dict, human_review: bool):
    """Process redub task with existing manifest"""
    logger.info(f"REDUB WORKER: Processing job {redub_job_id}")
    
    try:
        status_service.update_status(
            redub_job_id, "dub", JobStatus.PROCESSING, 45,
            {"message": f"Redubbing to {target_language}", "phase": "initialization"}
        )
        
        # Use simplified API with manifest override
        from app.services.dub.simple_dubbed_api import get_simple_dubbed_api
        api = get_simple_dubbed_api()
        
        result = api.process_dubbed_audio(
            job_id=redub_job_id,
            target_language=target_language,
            source_video_language=source_video_language,
            output_dir=redub_job_dir,
            review_mode=human_review,
            manifest_override=manifest
        )
        
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
        
        success = dub_service.complete_job(redub_job_id, result_url, details)
        if not success:
            dub_service.fail_job(redub_job_id, "Failed to complete redub", "completion_failed")
            return
        
        logger.info(f"Redub completed: {redub_job_id}")
        
    except Exception as e:
        logger.error(f"Redub failed for {redub_job_id}: {e}")
        try:
            dub_service.fail_job(redub_job_id, f"Redub error: {str(e)}", "redub_error")
        except:
            pass
