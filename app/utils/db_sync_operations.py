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
                "updated_at": datetime.now()
            }
            
            if progress is not None:
                update_data["progress"] = progress
                
            # Set timestamps based on status
            if status in ['downloading', 'processing']:
                # Check if already started
                existing_job = sync_collection.find_one({"job_id": job_id}, {"started_at": 1})
                if existing_job and not existing_job.get("started_at"):
                    update_data["started_at"] = datetime.now()
            elif status in ['completed', 'failed']:
                update_data["completed_at"] = datetime.now()
            
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
            
            update_data = {
                "status": status,
                "progress": progress,
                "updated_at": datetime.now()
            }
            
            # Set timestamps based on status
            if status in ['downloading', 'processing']:
                # Check if already started
                existing_job = sync_collection.find_one({"job_id": job_id}, {"started_at": 1})
                if existing_job and not existing_job.get("started_at"):
                    update_data["started_at"] = datetime.now()
            elif status in ['completed', 'failed']:
                update_data["completed_at"] = datetime.now()
            
            # Add details
            if details:
                update_data["details"] = details
            
            result = sync_collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            
            sync_client.close()
            
            if result.modified_count > 0:
                logger.info(f"Updated dub job {job_id}: {status} ({progress}%)")
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
                            "$set": {"updated_at": datetime.now()}
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