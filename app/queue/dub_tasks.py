import logging

logger = logging.getLogger(__name__)

def process_video_dub_task(request_dict: dict, user_id: str):
    """Process video dub task using clean worker"""
    from app.workers.dub_worker import process_dub_task
    
    logger.info(f"DUB WORKER: Starting job {request_dict.get('job_id')}")
    process_dub_task(request_dict, user_id)
    logger.info(f"DUB WORKER: Completed job {request_dict.get('job_id')}")

def process_redub_task(redub_job_id: str, target_language: str, source_video_language: str, 
                      redub_job_dir: str, manifest: dict, human_review: bool, voice_premium_model: bool = False):
    """Process redub task"""
    from app.workers.dub_worker import process_redub_task as worker_redub
    
    logger.info(f"REDUB: Starting {redub_job_id}")
    worker_redub(redub_job_id, target_language, source_video_language, redub_job_dir, manifest, human_review, voice_premium_model)
    logger.info(f"REDUB: Completed {redub_job_id}")

