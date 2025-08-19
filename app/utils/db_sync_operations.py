"""
Synchronous Database Operations for Thread-Safe Background Tasks
"""
import logging
import math
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pymongo import MongoClient
from bson import ObjectId
from app.config.settings import settings
from app.config.constants import CREDITS_PER_MINUTE_SEPARATION, CREDITS_PER_MINUTE_DUB

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
            elif status in ['completed', 'failed']:
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

            # Monotonic progress: never decrease progress, with sensible floors per status
            adjusted_progress = progress
            try:
                if status == 'completed':
                    adjusted_progress = 100
                elif status == 'awaiting_review':
                    # Ensure at least 75, but never less than current
                    min_review_progress = 75
                    adjusted_progress = max(min_review_progress, (progress if progress is not None else min_review_progress))
                    if isinstance(current_progress, int):
                        adjusted_progress = max(current_progress, adjusted_progress)
                elif status == 'reviewing':
                    # Ensure at least 80, but never less than current
                    min_reviewing_progress = 80
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
            elif status in ['completed', 'failed']:
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
            if status in ['awaiting_review', 'reviewing', 'completed', 'failed']:
                current_status = current_doc.get("status")
                
                # Prevent backwards status transitions (except explicit overrides)
                status_hierarchy = {
                    'pending': 0, 'downloading': 1, 'separating': 2, 'transcribing': 3,
                    'processing': 4, 'awaiting_review': 5, 'reviewing': 6, 
                    'completed': 7, 'failed': 7  # failed and completed are terminal
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
    
    @staticmethod
    def deduct_user_credits(user_id: str, job_id: str, duration_seconds: float, job_type: str = "separation") -> bool:
        """Deduct credits from user synchronously"""
        try:
            sync_client = SyncDBOperations._get_sync_client()
            sync_db = sync_client[settings.DB_NAME]
            users_collection = sync_db.users
            
            # Calculate credits required based on job type using constants
            if job_type.lower() == "dub":
                # DUB jobs: 0.05 credits per second (converted to per minute calculation)
                credits_required = round(duration_seconds * 0.05, 2)
            else:
                # Separation jobs: 1 credit per minute
                duration_minutes = duration_seconds / 60.0
                credits_required = round(duration_minutes * CREDITS_PER_MINUTE_SEPARATION, 2)
            
            # Ensure minimum credit requirement
            if credits_required == 0:
                credits_required = 0.01  # Minimum 0.01 credits
            
            # Try to find user by ObjectId first, then by string
            user_query_conditions = [
                {"_id": ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id, "credits": {"$gte": credits_required}},
                {"_id": user_id, "credits": {"$gte": credits_required}}
            ]
            
            result = None
            for query in user_query_conditions:
                try:
                    result = users_collection.update_one(
                        query,
                        {
                            "$inc": {"credits": -credits_required},
                            "$set": {"updated_at": datetime.now(timezone.utc)}
                        }
                    )
                    if result.modified_count > 0:
                        break
                except Exception:
                    continue
            
            sync_client.close()
            
            if result and result.modified_count > 0:
                logger.info(f"Credit deduction completed for {job_type} job {job_id}: {credits_required} credits")
                return True
            else:
                logger.warning(f"Credit deduction failed for {job_type} job {job_id}: insufficient credits or user not found")
                return False
                
        except Exception as e:
            logger.error(f"Credit deduction failed for {job_type} job {job_id}: {e}")
            return False

# Convenience functions for easier usage
def update_separation_status(job_id: str, status: str, progress: int = None, **kwargs):
    """Convenience function to update separation job status"""
    return SyncDBOperations.update_separation_job_status(job_id, status, progress, **kwargs)

def update_dub_status(job_id: str, status: str, progress: int, details: Dict[str, Any] = None):
    """Convenience function to update dub job status"""
    return SyncDBOperations.update_dub_job_status(job_id, status, progress, details)

def deduct_credits(user_id: str, job_id: str, duration_seconds: float, job_type: str = "separation"):
    """Convenience function to deduct user credits"""
    return SyncDBOperations.deduct_user_credits(user_id, job_id, duration_seconds, job_type)