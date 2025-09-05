import logging

logger = logging.getLogger(__name__)

def process_audio_separation_task(job_id: str, runpod_request_id: str, user_id: str, duration_seconds: float):
    """Process audio separation task using clean worker"""
    from app.workers.separation_worker import process_separation_task
    
    logger.info(f"SEPARATION WORKER: Starting job {job_id}")
    process_separation_task(job_id, runpod_request_id, user_id, duration_seconds)
    logger.info(f"SEPARATION WORKER: Completed job {job_id}")
