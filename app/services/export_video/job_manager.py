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
    Manages export jobs with in-memory storage
    In production, this should use Redis or a database
    """
    
    def __init__(self):
        self._jobs: Dict[str, ExportJob] = {}
    
    def create_job(self, export_data: Dict) -> ExportJob:
        """Create a new export job"""
        job = ExportJob.create_new(export_data)
        self._jobs[job.job_id] = job
        logger.info(f"Created export job {job.job_id}")
        return job
    
    def get_job(self, job_id: str) -> Optional[ExportJob]:
        """Get job by ID"""
        return self._jobs.get(job_id)
    
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