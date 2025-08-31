"""
Synchronous Database Operations for Thread-Safe Background Tasks
"""
import logging
import math
import threading
import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
from pymongo import MongoClient
from bson import ObjectId
from app.config.settings import settings
from app.config.credit_constants import CreditRates

logger = logging.getLogger(__name__)

class SyncDBOperations:
    """Thread-safe synchronous database operations"""
    
    @staticmethod
    def _get_sync_client():
        """Get synchronous MongoDB client"""
        return MongoClient(settings.MONGODB_URI)
    
    @staticmethod
    def update_separation_job_status(job_id: str, status: str, progress: int = None, **kwargs) -> bool:
        """Update separation job status synchronously"""
        try:
            sync_client = SyncDBOperations._get_sync_client()
            sync_db = sync_client[settings.DB_NAME]
            sync_collection = sync_db.separation_jobs
            
            update_data = {
                "status": status,
                "updated_at": datetime.now(timezone.utc)
            }
            
            if progress is not None:
                update_data["progress"] = progress
                
            # Set timestamps based on status
            if status in ['downloading', 'processing']:
                # Check if already started
                existing_job = sync_collection.find_one({"job_id": job_id}, {"started_at": 1})
                if existing_job and not existing_job.get("started_at"):
                    update_data["started_at"] = datetime.now(timezone.utc)
            elif status in ['completed', 'failed', 'cancelled']:
                update_data["completed_at"] = datetime.now(timezone.utc)
            
            # Add additional fields
            update_data.update(kwargs)
            
            result = sync_collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            
            sync_client.close()
            
            if result.modified_count > 0:
                logger.info(f"Updated separation job {job_id}: {status} ({progress}%)")
                return True
            else:
                logger.warning(f"No documents updated for separation job {job_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update separation status for {job_id}: {e}")
            return False
    
    @staticmethod
    def update_dub_job_status(job_id: str, status: str, progress: int, details: Dict[str, Any] = None) -> bool:
        """Update dub job status synchronously"""
        try:
            sync_client = SyncDBOperations._get_sync_client()
            sync_db = sync_client[settings.DB_NAME]
            sync_collection = sync_db.dub_jobs
            
            # Read current job to apply monotonic progress clamp
            current_doc = sync_collection.find_one({"job_id": job_id}, {"progress": 1, "status": 1}) or {}
            current_progress = current_doc.get("progress")
            current_status = current_doc.get("status")
            
            # 🛑 CRITICAL: Prevent overriding cancelled jobs (soft delete protection)
            if current_status == "cancelled" and status != "cancelled":
                logger.warning(f"🛑 Prevented status override for cancelled job {job_id}: cancelled -> {status}")
                sync_client.close()
                return False

            # Monotonic progress: never decrease progress, with sensible floors per status
            adjusted_progress = progress
            try:
                if status == 'completed':
                    adjusted_progress = 100
                elif status == 'awaiting_review':
                    # Ensure at least 77, but never less than current (review files ready)
                    min_review_progress = 77
                    adjusted_progress = max(min_review_progress, (progress if progress is not None else min_review_progress))
                    if isinstance(current_progress, int):
                        adjusted_progress = max(current_progress, adjusted_progress)
                elif status == 'reviewing':
                    # Ensure at least 79, but never less than current (applying human edits)
                    min_reviewing_progress = 79
                    adjusted_progress = max(min_reviewing_progress, (progress if progress is not None else min_reviewing_progress))
                    if isinstance(current_progress, int):
                        adjusted_progress = max(current_progress, adjusted_progress)
                elif status in ['failed']:
                    # Keep current progress to avoid backward jumps; status conveys failure
                    if isinstance(current_progress, int):
                        adjusted_progress = max(current_progress, (progress if progress is not None else current_progress))
                else:
                    # Default: clamp to never go backwards
                    if isinstance(current_progress, int) and progress is not None:
                        adjusted_progress = max(current_progress, progress)
            except Exception:
                # If any issue in adjustment, fall back to provided progress
                adjusted_progress = progress

            update_data = {
                "status": status,
                "progress": adjusted_progress,
                "updated_at": datetime.now(timezone.utc)
            }
            
            # Set timestamps based on status
            if status in ['downloading', 'processing']:
                # Check if already started
                existing_job = sync_collection.find_one({"job_id": job_id}, {"started_at": 1})
                if existing_job and not existing_job.get("started_at"):
                    update_data["started_at"] = datetime.now(timezone.utc)
            elif status in ['completed', 'failed', 'cancelled']:
                update_data["completed_at"] = datetime.now(timezone.utc)
            
            # Add details
            if details:
                update_data["details"] = details
                
                # Extract manifest fields from details to top-level fields for easier access
                if "segments_manifest_url" in details:
                    update_data["segments_manifest_url"] = details["segments_manifest_url"]
                if "segments_manifest_key" in details:
                    update_data["segments_manifest_key"] = details["segments_manifest_key"]
                if "segments_count" in details:
                    update_data["segments_count"] = details["segments_count"]
                if "transcript_id" in details:
                    update_data["transcript_id"] = details["transcript_id"]
                if "review_required" in details:
                    update_data["review_required"] = details["review_required"]
                if "review_status" in details:
                    update_data["review_status"] = details["review_status"]
                if "edited_segments_version" in details:
                    update_data["edited_segments_version"] = details["edited_segments_version"]
            
            # Use $set with $currentDate for atomic timestamp update
            # and add version check to prevent race conditions
            update_query = {"job_id": job_id}
            
            # For critical status transitions, ensure we're not overwriting newer status
            if status in ['awaiting_review', 'reviewing', 'completed', 'failed', 'cancelled']:
                current_status = current_doc.get("status")
                
                # Prevent backwards status transitions (except explicit overrides)
                status_hierarchy = {
                    'pending': 0, 'downloading': 1, 'separating': 2, 'transcribing': 3,
                    'processing': 4, 'awaiting_review': 5, 'reviewing': 6, 
                    'completed': 7, 'failed': 7, 'cancelled': 7  # terminal states
                }
                
                current_level = status_hierarchy.get(current_status, 0)
                new_level = status_hierarchy.get(status, 0)
                
                # Allow backwards transition only for specific valid cases
                if current_level > new_level and not (
                    current_status == 'reviewing' and status == 'awaiting_review'  # Allow re-review
                ):
                    logger.warning(f"Prevented backwards status transition for {job_id}: {current_status} -> {status}")
                    sync_client.close()
                    return False
            
            result = sync_collection.update_one(
                update_query,
                {"$set": update_data}
            )
            
            sync_client.close()
            
            if result.modified_count > 0:
                logger.info(f"Updated dub job {job_id}: {status} ({adjusted_progress}%)")
                return True
            else:
                logger.warning(f"No documents updated for dub job {job_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update dub job status for {job_id}: {e}")
            return False
    






def update_job_status_sync(job_id: str, job_type: str, status: str, progress: int = None, details: Dict[str, Any] = None) -> bool:
    """
    Generic sync function to update job status for any job type
    Used by unified_status_manager to avoid event loop issues
    """
    try:
        if job_type.lower() == "dub":
            return SyncDBOperations.update_dub_job_status(job_id, status, progress, details)
        elif job_type.lower() == "separation":
            kwargs = details or {}
            return SyncDBOperations.update_separation_job_status(job_id, status, progress, **kwargs)
        else:
            logger.warning(f"Unknown job type: {job_type}")
            return False
    except Exception as e:
        logger.error(f"Failed to update job status (sync) for {job_id}: {e}")
        return False

class CleanupManager:
    def __init__(self):
        self._locks = {}
        self._lock = threading.Lock()

    def cleanup_job(self, job_id: str):
        lock_key = job_id
        with self._lock:
            if lock_key not in self._locks:
                self._locks[lock_key] = threading.Lock()
            job_lock = self._locks[lock_key]

        with job_lock:
            try:
                self._cleanup_files(job_id)
                self._cleanup_temp_dirs(job_id)
            finally:
                with self._lock:
                    if lock_key in self._locks:
                        del self._locks[lock_key]

    def _cleanup_files(self, job_id: str):
        sync_client = SyncDBOperations._get_sync_client()
        try:
            db = sync_client[settings.DB_NAME]
            job_data = db.separation_jobs.find_one({"job_id": job_id})

            if job_data and job_data.get("details"):
                local_path = job_data["details"].get("local_audio_path")
                if local_path and os.path.exists(local_path):
                    os.remove(local_path)
        finally:
            sync_client.close()

    def _cleanup_temp_dirs(self, job_id: str):
        from pathlib import Path
        from app.services.dub.audio_utils import AudioUtils

        temp_dir = Path(settings.TEMP_DIR)
        patterns = [f"dub_{job_id}", f"separation_{job_id}", f"audio_{job_id}"]

        for pattern in patterns:
            temp_path = temp_dir / pattern
            if temp_path.exists():
                AudioUtils.remove_temp_dir(str(temp_path))

_cleanup_manager = CleanupManager()

def cleanup_separation_files(job_id: str):
    _cleanup_manager.cleanup_job(job_id)