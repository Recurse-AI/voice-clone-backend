"""
Simple Status Service - Clean replacement for UnifiedStatusManager
Single responsibility: Job status management with clean, simple logic
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum
from pymongo import MongoClient
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
    """
    Simple, clean status service with single responsibility
    - Updates job status in database
    - Validates progress (monotonic only)
    - No caching, no complex logic, just simple updates
    """
    
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
    
    def update_status(self, job_id: str, job_type: str, status: JobStatus, 
                     progress: int = None, details: Dict[str, Any] = None) -> bool:
        """
        Simple status update - sync operation for thread safety
        """
        try:
            client = MongoClient(settings.MONGODB_URI)
            db = client[settings.DB_NAME]
            collection = db[f"{job_type}_jobs"]
            
            # Get current progress and details for monotonic validation and merging
            current_job = collection.find_one({"job_id": job_id}, {"progress": 1, "details": 1})
            current_progress = current_job.get("progress", 0) if current_job else 0
            
            # Simple progress validation - never go backwards
            if progress is None:
                validated_progress = self._progress_floors.get(status, 0)
            else:
                validated_progress = max(progress, current_progress, self._progress_floors.get(status, 0))
            
            # Build update data
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
            
            # Merge details if provided (preserve existing details)
            if details:
                # Get existing details first
                existing_details = {}
                if current_job and current_job.get("details"):
                    existing_details = current_job["details"]
                
                # Merge new details with existing ones
                merged_details = {**existing_details, **details}
                update_data["details"] = merged_details
            
            # Update database
            result = collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            
            client.close()
            
            if result.modified_count > 0:
                logger.info(f"✅ Status updated: {job_id} → {status.value} ({validated_progress}%)")
                return True
            else:
                logger.warning(f"❌ No update for job {job_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update status for {job_id}: {e}")
            return False
    
    def get_status(self, job_id: str, job_type: str) -> Optional[Dict[str, Any]]:
        """Get job status from database"""
        try:
            client = MongoClient(settings.MONGODB_URI)
            db = client[settings.DB_NAME]
            collection = db[f"{job_type}_jobs"]
            
            job = collection.find_one({"job_id": job_id})
            client.close()
            
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
            logger.error(f"Failed to get status for {job_id}: {e}")
            return None


# Global service instance
status_service = SimpleStatusService()


def update_job_status_simple(job_id: str, job_type: str, status: str, 
                           progress: int = None, details: Dict[str, Any] = None) -> bool:
    """Convenience function for status updates"""
    try:
        status_enum = JobStatus(status)
        return status_service.update_status(job_id, job_type, status_enum, progress, details)
    except ValueError:
        logger.error(f"Invalid status: {status}")
        return False


def get_job_status_simple(job_id: str, job_type: str) -> Optional[Dict[str, Any]]:
    """Convenience function for getting status"""
    return status_service.get_status(job_id, job_type)
