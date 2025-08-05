"""
Job management helper functions
"""
import logging
from app.services.dub_job_service import dub_job_service

logger = logging.getLogger(__name__)

async def update_dub_job_completion(job_id: str, result_url: str, pipeline_result: dict):
    """Update dub job completion status in MongoDB"""
    try:
        await dub_job_service.update_job_status(
            job_id,
            "completed",
            100,
            result_url=result_url,
            result_urls=pipeline_result.get("result_urls"),
            details=pipeline_result.get("details")
        )
        logger.info(f"Updated MongoDB completion status for job {job_id}")
    except Exception as e:
        logger.error(f"Failed to update MongoDB for job {job_id}: {e}")

async def update_dub_job_failure(job_id: str, error_message: str):
    """Update dub job failure status in MongoDB"""
    try:
        await dub_job_service.update_job_status(
            job_id,
            "failed",
            0,
            error=error_message
        )
        logger.info(f"Updated MongoDB failure status for job {job_id}")
    except Exception as e:
        logger.error(f"Failed to update MongoDB failure for job {job_id}: {e}")