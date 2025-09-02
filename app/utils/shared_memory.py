"""
Simplified shared memory for job cancellation tracking only.
No upload status persistence - removed all upload status functionality.
"""

import logging

import os

logger = logging.getLogger(__name__)


class StatusManager:
    """
    Simple status manager for job cancellation tracking.
    No database persistence - only in-memory tracking.
    """

    def __init__(self):
        # Track cancelled jobs to signal background threads
        self.cancelled_jobs: set = set()

    def exists(self, job_id: str) -> bool:
        """Check if job directory exists"""
        try:
            from app.config.settings import settings
            job_dir = os.path.join(settings.TEMP_DIR, job_id)
            return os.path.exists(job_dir)
        except:
            return False

    async def exists_async(self, job_id: str) -> bool:
        """Async check if job directory exists"""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(None, self.exists, job_id)


# Global instance for application use
_status_manager = StatusManager()



# Job Cancellation Management
def mark_job_cancelled(job_id: str) -> None:
    """Mark a job as cancelled to signal background threads"""
    _status_manager.cancelled_jobs.add(job_id)
    logger.info(f"🛑 Marked job {job_id} as cancelled | Total cancelled: {len(_status_manager.cancelled_jobs)} | List: {_status_manager.cancelled_jobs}")

def is_job_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled"""
    is_cancelled = job_id in _status_manager.cancelled_jobs

    return is_cancelled

def unmark_job_cancelled(job_id: str) -> None:
    """Remove job from cancelled list"""
    _status_manager.cancelled_jobs.discard(job_id)
    logger.info(f"Unmarked job {job_id} as cancelled")