import logging

logger = logging.getLogger(__name__)

def process_video_dub_task(request_dict: dict, user_id: str):
    """Process video dub task using clean worker"""
    from app.workers.dub_worker import process_dub_task
    
    logger.info(f"DUB WORKER: Starting job {request_dict.get('job_id')}")
    process_dub_task(request_dict, user_id)
    logger.info(f"DUB WORKER: Completed job {request_dict.get('job_id')}")

def process_redub_task(redub_job_id: str, target_language: str, source_video_language: str, 
                      redub_job_dir: str, manifest: dict, human_review: bool):
    """Process redub task using clean worker"""
    from app.workers.dub_worker import process_redub_task as worker_process_redub
    
    logger.info(f"REDUB WORKER: Starting job {redub_job_id}")
    worker_process_redub(redub_job_id, target_language, source_video_language, 
                        redub_job_dir, manifest, human_review)
    logger.info(f"REDUB WORKER: Completed job {redub_job_id}")

