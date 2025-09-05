"""
Separation Worker - Clean background task processing
Handles separation job monitoring and completion
"""
import asyncio
import logging
import time
from app.services.separation_service import separation_service
from app.services.simple_status_service import status_service, JobStatus
from app.utils.runpod_service import runpod_service
from app.config.constants import MAX_ATTEMPTS_DEFAULT, POLLING_INTERVAL_SECONDS

logger = logging.getLogger(__name__)


def process_separation_task(job_id: str, runpod_request_id: str, user_id: str, duration_seconds: float):
    """
    Clean separation task processing
    Single responsibility: Monitor RunPod job and update status
    """
    logger.info(f"SEPARATION WORKER: Processing job {job_id} (RunPod: {runpod_request_id})")
    
    try:
        # Update status to processing
        status_service.update_status(
            job_id, "separation", JobStatus.PROCESSING, 5,
            {"message": "Worker started processing", "phase": "initialization"}
        )
        
        # Monitor RunPod job
        max_attempts = MAX_ATTEMPTS_DEFAULT
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Get RunPod status
                runpod_status = runpod_service.get_separation_status(runpod_request_id)
                
                if not runpod_status:
                    logger.warning(f"No RunPod status for {runpod_request_id} (attempt {attempt + 1})")
                    attempt += 1
                    time.sleep(POLLING_INTERVAL_SECONDS)
                    continue
                
                job_status = runpod_status.get("status", "unknown")
                progress = runpod_status.get("progress", 0)
                
                logger.info(f"RunPod status for {job_id}: {job_status} ({progress}%)")
                
                # Handle different RunPod statuses
                if job_status.upper() == "CANCELLED":
                    logger.info(f"RunPod job {runpod_request_id} was cancelled")
                    separation_service.fail_separation_job(job_id, "Job cancelled by RunPod", "job_cancelled")
                    break
                
                elif job_status == "processing" and progress > 0:
                    # Update progress (cap at 90% until completion)
                    capped_progress = min(progress, 90)
                    status_service.update_status(
                        job_id, "separation", JobStatus.PROCESSING, capped_progress,
                        {"message": f"Audio separation in progress... ({progress}%)", "phase": "separating"}
                    )
                
                elif job_status == "completed":
                    logger.info(f"RunPod job {runpod_request_id} completed")
                    
                    # Process results
                    output = runpod_status.get("result", {})
                    success = asyncio.run(separation_service.process_separation_results(job_id, output))
                    
                    if success:
                        # Complete credit billing
                        from app.utils.job_utils import job_utils
                        job_utils.complete_job_billing_sync(job_id, "separation", user_id)
                        
                        # Cleanup temp files
                        from app.utils.cleanup_utils import cleanup_utils
                        cleanup_utils.cleanup_job_comprehensive(job_id, "separation")
                        
                        logger.info(f"Separation job {job_id} completed successfully")
                    else:
                        separation_service.fail_separation_job(job_id, "Failed to process results", "processing_failed")
                    
                    break
                
                elif job_status == "failed":
                    error_msg = runpod_status.get("error", "Audio separation failed")
                    logger.error(f"RunPod job {runpod_request_id} failed: {error_msg}")
                    separation_service.fail_separation_job(job_id, error_msg, "job_failed")
                    break
                
                # Wait before next check
                time.sleep(POLLING_INTERVAL_SECONDS)
                attempt += 1
                
            except Exception as e:
                logger.error(f"Error monitoring separation job {job_id}: {e}")
                time.sleep(POLLING_INTERVAL_SECONDS)
                attempt += 1
        
        # Handle timeout
        if attempt >= max_attempts:
            logger.warning(f"Separation job {job_id} monitoring timed out")
            separation_service.fail_separation_job(job_id, "Job monitoring timed out", "job_timeout")
        
    except Exception as e:
        logger.error(f"Separation worker failed for {job_id}: {e}")
        try:
            separation_service.fail_separation_job(job_id, f"Worker error: {str(e)}", "worker_error")
        except:
            pass  # Avoid double error
    
    finally:
        logger.info(f"SEPARATION WORKER: Finished job {job_id}")


# Task function for RQ
def enqueue_separation_task(job_id: str, runpod_request_id: str, user_id: str, duration_seconds: float):
    """RQ task wrapper"""
    process_separation_task(job_id, runpod_request_id, user_id, duration_seconds)
