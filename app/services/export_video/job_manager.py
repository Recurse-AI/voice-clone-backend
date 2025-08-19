"""
Export Job Manager
Handles export job lifecycle and status tracking
"""

from typing import Dict, Optional
import logging
from datetime import datetime, timezone

from .models import ExportJob

logger = logging.getLogger(__name__)

class ExportJobManager:
    """
    Manages export jobs with database storage for production safety.
    Uses cache for performance + database for persistence.
    """
    
    def __init__(self):
        self._jobs_cache: Dict[str, ExportJob] = {}
        from app.config.database import db
        self.collection = db.export_jobs
    
    async def create_job(self, export_data: Dict) -> ExportJob:
        """Create a new export job with database persistence"""
        job = ExportJob.create_new(export_data)
        
        # Cache for fast access
        self._jobs_cache[job.job_id] = job
        
        # Persist to database for cross-worker access
        try:
            job_dict = {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "created_at": job.created_at,
                "export_data": job.export_data,
                "processing_logs": job.processing_logs,
                "download_url": job.download_url,
                "error": job.error
            }
            await self.collection.insert_one(job_dict)
        except Exception as e:
            logger.warning(f"Failed to persist export job to database: {e}")
        
        logger.info(f"Created export job {job.job_id}")
        return job
    
    async def get_job(self, job_id: str) -> Optional[ExportJob]:
        """Get job by ID with cache + database fallback"""
        # Check cache first
        if job_id in self._jobs_cache:
            return self._jobs_cache[job_id]
        
        # Fallback to database
        try:
            job_data = await self.collection.find_one({"job_id": job_id})
            if job_data:
                job = ExportJob(
                    job_id=job_data["job_id"],
                    status=job_data["status"],
                    progress=job_data["progress"],
                    created_at=job_data["created_at"],
                    export_data=job_data["export_data"],
                    processing_logs=job_data["processing_logs"],
                    download_url=job_data.get("download_url"),
                    error=job_data.get("error")
                )
                # Cache for future access
                self._jobs_cache[job_id] = job
                return job
        except Exception as e:
            logger.error(f"Failed to load export job from database: {e}")
        
        return None
    
    def update_status(self, job_id: str, status: str, progress: int = None):
        """Update job status and progress"""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        job.status = status
        if progress is not None:
            job.progress = progress
        
        logger.info(f"Job {job_id}: {status} ({progress}%)")
        return True
    
    def add_log(self, job_id: str, message: str):
        """Add processing log to job"""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        job.processing_logs.append(log_message)
        logger.info(f"Job {job_id}: {message}")
        return True
    
    def set_download_url(self, job_id: str, url: str):
        """Set download URL for completed job"""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        job.download_url = url
        return True
    
    def fail_job(self, job_id: str, error: str):
        """Mark job as failed with error message"""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        job.status = "FAILED"
        job.error = error
        self.add_log(job_id, f"Export failed: {error}")
        return True
    
    def complete_job(self, job_id: str, download_url: str):
        """Mark job as completed with download URL"""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        job.status = "COMPLETED"
        job.progress = 100
        job.download_url = download_url
        self.add_log(job_id, "Export completed successfully")
        return True
    
    def cancel_job(self, job_id: str):
        """Cancel job if it's not already completed or failed"""
        if job_id not in self._jobs:
            return False
        
        job = self._jobs[job_id]
        if job.status in ["COMPLETED", "FAILED"]:
            return False
        
        job.status = "CANCELLED"
        self.add_log(job_id, "Export cancelled by user")
        return True
    
    def estimate_duration(self, timeline_duration_ms: float) -> int:
        """
        Estimate processing time based on video duration
        """
        duration_seconds = timeline_duration_ms / 1000
        # Rough estimate: 1 second of video = 2-3 seconds processing
        return int(duration_seconds * 2.5)
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove old jobs to prevent memory leaks"""
        current_time = datetime.now(timezone.utc)
        jobs_to_remove = []
        
        for job_id, job in self._jobs.items():
            age_hours = (current_time - job.created_at).total_seconds() / 3600
            if age_hours > max_age_hours:
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self._jobs[job_id]
            logger.info(f"Cleaned up old job {job_id}")
        
        return len(jobs_to_remove)

# Global job manager instance
export_job_manager = ExportJobManager() 