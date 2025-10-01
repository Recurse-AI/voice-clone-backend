import logging

logger = logging.getLogger(__name__)

def process_clip_generation_task(job_id: str, user_id: str):
    from app.workers.clip_worker import process_clip_job
    logger.info(f"CLIP WORKER: Starting job {job_id}")
    process_clip_job(job_id, user_id)
    logger.info(f"CLIP WORKER: Completed job {job_id}")
