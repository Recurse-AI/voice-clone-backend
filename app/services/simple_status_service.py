import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum
from app.config.database import sync_client, db, get_async_db
from app.config.settings import settings

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Clean job status enum"""
    PENDING = "pending"
    PROCESSING = "processing"
    SEPARATING = "separating"  # For dub jobs only
    TRANSCRIBING = "transcribing"  # For dub jobs only
    UPLOADING = "uploading"  # For dub jobs only
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_REVIEW = "awaiting_review"  # For dub jobs only
    REVIEWING = "reviewing"  # For dub jobs only


class SimpleStatusService:
    def __init__(self):
        self._progress_floors = {
            JobStatus.PENDING: 0,
            JobStatus.PROCESSING: 5,
            JobStatus.SEPARATING: 25,
            JobStatus.TRANSCRIBING: 45,
            JobStatus.UPLOADING: 90,
            JobStatus.COMPLETED: 100,
            JobStatus.AWAITING_REVIEW: 80,
            JobStatus.REVIEWING: 80,
            JobStatus.FAILED: 0,
        }
        
        self._persist_statuses = {
            JobStatus.PROCESSING,
            JobStatus.AWAITING_REVIEW,
            JobStatus.COMPLETED,
            JobStatus.FAILED
        }
        
        self._sync_lock = threading.Lock()
        self._init_redis()
        self._init_sync_service()
    
    def _init_redis(self):
        try:
            from app.services.redis_status_service import redis_status_service
            self.redis_service = redis_status_service
            self.redis_available = self.redis_service.is_available()
        except Exception as e:
            logger.warning(f"Redis not available, falling back to MongoDB only: {e}")
            self.redis_service = None
            self.redis_available = False
    
    def _init_sync_service(self):
        try:
            from app.services.status_sync_service import sync_service
            self.sync_service = sync_service
            self.sync_service.start()
        except Exception as e:
            logger.warning(f"Sync service not available: {e}")
            self.sync_service = None
    
    def update_status(self, job_id: str, job_type: str, status: JobStatus, 
                     progress: int = None, details: Dict[str, Any] = None) -> bool:
        current_progress = self._get_current_progress(job_id, job_type)
        
        if progress is None:
            validated_progress = self._progress_floors.get(status, 0)
        else:
            validated_progress = max(progress, current_progress, self._progress_floors.get(status, 0))
        
        status_data = {
            "job_id": job_id,
            "status": status.value,
            "progress": validated_progress,
            "details": details or {}
        }
        
        redis_success = self._update_redis(job_id, job_type, status_data)
        
        mongo_success = True
        if status in self._persist_statuses or not redis_success:
            mongo_success = self._update_mongodb(job_id, job_type, status, validated_progress, details, current_progress)
        
        if redis_success or mongo_success:
            if status in self._persist_statuses and self.sync_service:
                self.sync_service.schedule_sync(job_id, job_type)
            
            logger.info(f"✅ Status updated: {job_id} → {status.value} ({validated_progress}%)")
            return True
        
        logger.error(f"❌ Failed to update status for {job_id}")
        return False
    
    def get_status(self, job_id: str, job_type: str) -> Optional[Dict[str, Any]]:
        if self.redis_available:
            redis_data = self.redis_service.get_status(job_id, job_type)
            if redis_data:
                return {
                    "job_id": redis_data.get("job_id", job_id),
                    "status": redis_data.get("status", "unknown"),
                    "progress": redis_data.get("progress", 0),
                    "details": redis_data.get("details", {}),
                    "updated_at": redis_data.get("updated_at"),
                    "created_at": redis_data.get("created_at")
                }
        
        return self._get_mongodb_status(job_id, job_type)
    
    def _get_current_progress(self, job_id: str, job_type: str) -> int:
        if self.redis_available:
            redis_data = self.redis_service.get_status(job_id, job_type)
            if redis_data and "progress" in redis_data:
                return int(redis_data["progress"])
        
        mongo_data = self._get_mongodb_status(job_id, job_type)
        return mongo_data.get("progress", 0) if mongo_data else 0
    
    def _update_redis(self, job_id: str, job_type: str, status_data: Dict[str, Any]) -> bool:
        if not self.redis_available:
            return False
        
        try:
            return self.redis_service.set_status(job_id, job_type, status_data)
        except Exception as e:
            logger.warning(f"Redis update failed for {job_id}: {e}")
            self.redis_available = False
            return False
    
    def _update_mongodb(self, job_id: str, job_type: str, status: JobStatus,
                       validated_progress: int, details: Dict[str, Any], current_progress: int) -> bool:
        try:
            # Use global sync client for connection pooling
            collection = sync_client[settings.DB_NAME][f"{job_type}_jobs"]

            # Get current job details for merging
            current_job = collection.find_one({"job_id": job_id}, {"details": 1})

            update_data = {
                "status": status.value,
                "progress": validated_progress,
                "updated_at": datetime.now(timezone.utc)
            }

            # Add timestamps based on status
            if status == JobStatus.PROCESSING and current_progress == 0:
                update_data["started_at"] = datetime.now(timezone.utc)
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                update_data["completed_at"] = datetime.now(timezone.utc)

            # Merge details if provided
            if details:
                existing_details = {}
                if current_job and current_job.get("details"):
                    existing_details = current_job["details"]

                merged_details = {**existing_details, **details}
                update_data["details"] = merged_details

            # Perform atomic update
            result = collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )

            success = result.modified_count > 0
            if success:
                logger.debug(f"MongoDB updated: {job_id} → {status.value}")
            return success

        except Exception as e:
            logger.error(f"MongoDB update failed for {job_id}: {e}")
            return False
    
    def _get_mongodb_status(self, job_id: str, job_type: str) -> Optional[Dict[str, Any]]:
        try:
            # Use global sync client for connection pooling
            collection = sync_client[settings.DB_NAME][f"{job_type}_jobs"]
            job = collection.find_one({"job_id": job_id})

            if job:
                return {
                    "job_id": job["job_id"],
                    "status": job.get("status", "unknown"),
                    "progress": job.get("progress", 0),
                    "details": job.get("details", {}),
                    "updated_at": job.get("updated_at"),
                    "created_at": job.get("created_at")
                }
            return None

        except Exception as e:
            logger.error(f"MongoDB get failed for {job_id}: {e}")
            return None


# Global service instance
status_service = SimpleStatusService()


