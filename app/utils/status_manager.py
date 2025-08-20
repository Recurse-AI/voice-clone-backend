"""
Status Manager - Using Centralized MongoDB Only (No Local Storage)
"""
import logging
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime, timezone
from app.services.dub_job_service import dub_job_service
from app.services.separation_job_service import separation_job_service

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Processing status enum"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    SEPARATING = "separating"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_REVIEW = "awaiting_review"
    REVIEWING = "reviewing"


class StatusManager:
    """Hybrid Status Manager: Local cache for processing states + MongoDB for final states"""
    
    def __init__(self):
        # Local cache for processing states (fast access)
        self.processing_cache = {}
        
        # Final states that go directly to MongoDB
        self.final_states = {"completed", "failed", "cancelled"}
        
        # Processing states that use local cache
        self.processing_states = {
            "pending", "downloading", "separating", "transcribing",
            "processing", "uploading", "awaiting_review", "reviewing"
        }
    
    def _get_status_message(self, status: ProcessingStatus) -> str:
        """Get status message"""
        messages = {
            ProcessingStatus.PENDING: "Processing queued",
            ProcessingStatus.DOWNLOADING: "Downloading video...",
            ProcessingStatus.SEPARATING: "Separating audio tracks...",
            ProcessingStatus.TRANSCRIBING: "Transcribing audio...",
            ProcessingStatus.CANCELLED: "Job cancelled by user",
            ProcessingStatus.PROCESSING: "Processing audio and video...",
            ProcessingStatus.UPLOADING: "Uploading results...",
            ProcessingStatus.COMPLETED: "Processing completed successfully",
            ProcessingStatus.FAILED: "Processing failed",
            ProcessingStatus.AWAITING_REVIEW: "Awaiting human review...",
            ProcessingStatus.REVIEWING: "Applying human edits..."
        }
        return messages.get(status, "Unknown status")
    
    async def initialize_status(self, job_id: str, user_id: Optional[str] = None, job_type: str = "dub") -> None:
        """Initialize status for new processing job - delegates to appropriate service"""
        try:
            if job_type == "dub":
                # Job should already exist in dub_jobs collection
                job = await dub_job_service.get_job(job_id)
                if job:
                    await dub_job_service.update_job_status(
                        job_id=job_id,
                        status=ProcessingStatus.PENDING.value,
                        progress=0
                    )
                    logger.info(f"Initialized dub job status: {job_id}")
                else:
                    logger.warning(f"Dub job not found for status initialization: {job_id}")
                    
            elif job_type == "separation":
                # Job should already exist in separation_jobs collection
                job = await separation_job_service.get_job(job_id)
                if job:
                    await separation_job_service.update_job_status(
                        job_id=job_id,
                        status=ProcessingStatus.PENDING.value,
                        progress=0
                    )
                    logger.info(f"Initialized separation job status: {job_id}")
                else:
                    logger.warning(f"Separation job not found for status initialization: {job_id}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize status for {job_id}: {e}")
    
    async def update_status(self, job_id: str, status: ProcessingStatus, progress: int = None, 
                           details: Optional[Dict[str, Any]] = None, job_type: str = "dub"):
        """Hybrid status update: Local cache for processing, MongoDB for final states"""
        try:
            status_value = status.value if isinstance(status, ProcessingStatus) else status
            
            if status_value in self.processing_states:
                # Use local cache for processing states (super fast)
                self.processing_cache[job_id] = {
                    "job_id": job_id,
                    "status": status_value,
                    "progress": progress or 0,
                    "details": details or {},
                    "job_type": job_type,
                    "message": self._get_status_message(ProcessingStatus(status_value)),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                logger.info(f"Cached {job_type} job {job_id} status: {status_value} (progress: {progress}%)")
                
            elif status_value in self.final_states:
                # Final states go to MongoDB and clear cache
                if job_type == "dub":
                    await dub_job_service.update_job_status(
                        job_id=job_id,
                        status=status_value,
                        progress=progress or 0,
                        details=details
                    )
                elif job_type == "separation":
                    await separation_job_service.update_job_status(
                        job_id=job_id,
                        status=status_value,
                        progress=progress or 0,
                        details=details
                    )
                
                # Clear from local cache since job is finished
                self.processing_cache.pop(job_id, None)
                logger.info(f"Final state {job_type} job {job_id}: {status_value} (saved to MongoDB, cleared cache)")
            
        except Exception as e:
            logger.error(f"Failed to update status for {job_id}: {e}")
    
    async def set_progress(self, job_id: str, progress: int, job_type: str = "dub") -> None:
        """Update progress percentage (hybrid approach)"""
        try:
            progress = min(100, max(0, progress))
            
            # If job is in local cache, update progress there (fast)
            if job_id in self.processing_cache:
                self.processing_cache[job_id]["progress"] = progress
                self.processing_cache[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
                logger.debug(f"Updated progress in cache for {job_id}: {progress}%")
                return
            
            # Fall back to MongoDB update if not in cache
            if job_type == "dub":
                job = await dub_job_service.get_job(job_id)
                if job:
                    await dub_job_service.update_job_status(
                        job_id=job_id,
                        status=job.status,
                        progress=progress
                    )
                    
            elif job_type == "separation":
                job = await separation_job_service.get_job(job_id)
                if job:
                    await separation_job_service.update_job_status(
                        job_id=job_id,
                        status=job.status,
                        progress=progress
                    )
                    
        except Exception as e:
            logger.error(f"Failed to set progress for {job_id}: {e}")
    
    def clear_cache(self, job_id: str) -> None:
        """Clear job from local cache"""
        removed = self.processing_cache.pop(job_id, None)
        if removed:
            logger.info(f"Cleared cache for job {job_id}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            "total_cached_jobs": len(self.processing_cache),
            "cached_job_ids": list(self.processing_cache.keys()),
            "cache_size_bytes": len(str(self.processing_cache))
        }
    
    def cleanup_stale_cache(self, max_age_hours: int = 24) -> int:
        """Clean up stale cache entries older than max_age_hours"""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        stale_jobs = []
        
        for job_id, data in self.processing_cache.items():
            try:
                updated_at = datetime.fromisoformat(data["updated_at"])
                if updated_at < cutoff_time:
                    stale_jobs.append(job_id)
            except (KeyError, ValueError):
                stale_jobs.append(job_id)  # Remove malformed entries
        
        for job_id in stale_jobs:
            self.processing_cache.pop(job_id, None)
        
        if stale_jobs:
            logger.info(f"Cleaned up {len(stale_jobs)} stale cache entries")
        
        return len(stale_jobs)
    
    async def get_status(self, job_id: str, job_type: str = "dub") -> Optional[Dict[str, Any]]:
        """Hybrid status lookup: Check local cache first, then MongoDB"""
        try:
            # First check local cache (super fast for processing states)
            if job_id in self.processing_cache:
                cached_status = self.processing_cache[job_id]
                logger.debug(f"Status cache hit for {job_id}: {cached_status['status']}")
                return cached_status
            
            # Fall back to MongoDB for final states or if not in cache
            if job_type == "dub":
                job = await dub_job_service.get_job(job_id)
                if job:
                    return {
                        "status": job.status,
                        "progress": job.progress,
                        "message": self._get_status_message(ProcessingStatus(job.status)),
                        "job_id": job.job_id,
                        "user_id": job.user_id,
                        "created_at": job.created_at.isoformat(),
                        "updated_at": job.updated_at.isoformat(),
                        "details": job.details or {},
                        "result_url": job.result_url,
                        "error": job.error,
                        "job_type": job_type
                    }
                    
            elif job_type == "separation":
                job = await separation_job_service.get_job(job_id)
                if job:
                    return {
                        "status": job.status,
                        "progress": job.progress,
                        "message": self._get_status_message(ProcessingStatus(job.status)),
                        "job_id": job.job_id,
                        "user_id": job.user_id,
                        "created_at": job.created_at.isoformat(),
                        "updated_at": job.updated_at.isoformat(),
                        "details": job.details or {},
                        "result_url": job.result_url,
                        "error": job.error,
                        "job_type": job_type
                    }
                    
            logger.debug(f"Status not found for {job_id} (checked cache + MongoDB)")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get status for {job_id}: {e}")
            return None
    
    def initialize_status_sync(self, job_id: str, user_id: Optional[str] = None):
        """Sync version for backward compatibility - DEPRECATED"""
        logger.warning(f"Using deprecated sync initialize_status for {job_id}")
        # This is kept for backward compatibility but should be replaced with async version
        pass
        
    def update_status_sync(self, job_id: str, status: ProcessingStatus, progress: int = None, 
                          details: Optional[Dict[str, Any]] = None):
        """Sync version for backward compatibility - DEPRECATED"""
        logger.warning(f"Using deprecated sync update_status for {job_id}")
        # This is kept for backward compatibility but should be replaced with async version
        pass

# For backward compatibility - maintain the same interface
def initialize_status(job_id: str, user_id: Optional[str] = None):
    """Backward compatibility function - DEPRECATED"""
    logger.warning(f"Using deprecated global initialize_status for {job_id}")
    
def update_status(job_id: str, status: ProcessingStatus, progress: int = None, 
                 details: Optional[Dict[str, Any]] = None):
    """Backward compatibility function - DEPRECATED"""
    logger.warning(f"Using deprecated global update_status for {job_id}")

# Global instance for backward compatibility
status_manager = StatusManager()