import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from bson import ObjectId

from app.models.dub_job import DubJob
from app.config.database import dub_jobs_collection
from app.config.constants import DEFAULT_QUERY_LIMIT, MAX_QUERY_LIMIT
from app.services.base_job_service import BaseJobService

logger = logging.getLogger(__name__)


class DubJobService(BaseJobService[DubJob]):
    """
    Service for managing dub jobs in MongoDB.
    Inherits duplicate prevention and common operations from BaseJobService.
    """
    
    def __init__(self):
        super().__init__(dub_jobs_collection)
    
    def _create_job_model(self, job_data: Dict[str, Any]) -> DubJob:
        """Create DubJob model instance from data"""
        return DubJob(**job_data)
    
    async def create_job(self, job_data: Dict[str, Any]) -> Optional[DubJob]:
        """
        Create a new dub job with duplicate prevention.
        Uses BaseJobService for comprehensive safety.
        """
        return await self.create_job_safe(job_data)
    
    async def get_job(self, job_id: str) -> Optional[DubJob]:
        """Get dub job by job_id"""
        try:
            job_data = await self.collection.find_one({"job_id": job_id})
            if job_data:
                job_data['id'] = str(job_data['_id'])
                del job_data['_id']
                return DubJob(**job_data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get dub job {job_id}: {e}")
            return None
    
    async def update_job_status(self, job_id: str, status: str, progress: int = None, **kwargs) -> bool:
        """Update job status and progress with automatic timestamp handling"""
        # Add timestamp fields based on status
        if status in ['downloading', 'processing'] and not await self._has_started(job_id):
            kwargs["started_at"] = datetime.now(timezone.utc)
        elif status in ['completed', 'failed']:
            kwargs["completed_at"] = datetime.now(timezone.utc)
        
        # Use base class method for consistent handling
        return await self.update_job_status_safe(job_id, status, progress, **kwargs)
    
    async def _has_started(self, job_id: str) -> bool:
        """Check if job has started_at timestamp"""
        try:
            job_data = await self.collection.find_one(
                {"job_id": job_id}, 
                {"started_at": 1}
            )
            return job_data and job_data.get("started_at") is not None
        except:
            return False
    
    async def get_user_jobs(self, user_id: str, page: int = 1, limit: int = None) -> Tuple[List[DubJob], int]:
        """Get paginated dub jobs for a user"""
        try:
            # Ensure limit doesn't exceed maximum
            limit = min(limit, MAX_QUERY_LIMIT) if limit else MAX_QUERY_LIMIT
            
            # Calculate skip value for pagination
            skip = (page - 1) * limit
            
            # Base query filter
            query_filter = {"user_id": user_id}
            
            # Get total count
            total_count = await self.collection.count_documents(query_filter)
            
            # Get paginated results
            cursor = self.collection.find(query_filter).sort("created_at", -1).skip(skip).limit(limit)
            
            jobs = []
            async for job_data in cursor:
                try:
                    job_data['id'] = str(job_data['_id'])
                    del job_data['_id']
                    
                    # Debug log for validation issues
                    logger.debug(f"Processing job_data: {job_data.get('job_id')} with keys: {list(job_data.keys())}")
                    
                    job = DubJob(**job_data)
                    jobs.append(job)
                except Exception as validation_error:
                    logger.error(f"DubJob validation failed for job {job_data.get('job_id', 'unknown')}: {validation_error}")
                    logger.error(f"Problematic job_data: {job_data}")
                    # Continue with next job instead of failing completely
                    continue
            
            return jobs, total_count
            
        except Exception as e:
            logger.error(f"Failed to get user dub jobs: {e}")
            return [], 0
    
    async def get_user_job_statistics(self, user_id: str) -> dict:
        """Get job statistics for a user"""
        try:
            query_filter = {"user_id": user_id}
            
            # Get total count
            total_count = await self.collection.count_documents(query_filter)
            
            # Get completed count
            completed_count = await self.collection.count_documents({
                **query_filter,
                "status": "completed"
            })
            
            # Get processing count (pending, downloading, separating, transcribing, processing, uploading, awaiting_review, reviewing statuses)
            processing_count = await self.collection.count_documents({
                **query_filter,
                "status": {"$in": ["pending", "downloading", "separating", "transcribing", "processing", "uploading", "awaiting_review", "reviewing"]}
            })
            
            return {
                "total": total_count,
                "completed": completed_count,
                "processing": processing_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get user dub job statistics: {e}")
            return {"total": 0, "completed": 0, "processing": 0}
    
    async def get_jobs_by_status(self, status: str, limit: int = 100) -> List[DubJob]:
        """Get jobs by status (for monitoring/cleanup)"""
        try:
            cursor = self.collection.find(
                {"status": status}
            ).sort("created_at", -1).limit(limit)
            
            jobs = []
            async for job_data in cursor:
                job_data['id'] = str(job_data['_id'])
                del job_data['_id']
                jobs.append(DubJob(**job_data))
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to get jobs by status {status}: {e}")
            return []
    
    async def update_details(self, job_id: str, details: Dict[str, Any]) -> bool:
        """Update job details"""
        try:
            result = await self.collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "details": details,
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update dub job details {job_id}: {e}")
            return False
    
    async def delete_job(self, job_id: str, user_id: str) -> bool:
        """Delete a dub job (with user ownership check)"""
        try:
            # First check if job exists and belongs to user
            job = await self.get_job(job_id)
            if not job:
                logger.warning(f"Dub job {job_id} not found for deletion")
                return False
            
            if job.user_id != user_id:
                logger.warning(f"User {user_id} attempted to delete dub job {job_id} they don't own")
                return False
            
            # Delete the job
            result = await self.collection.delete_one({"job_id": job_id})
            
            success = result.deleted_count > 0
            if success:
                logger.info(f"Deleted dub job {job_id} for user {user_id}")
            else:
                logger.warning(f"Failed to delete dub job {job_id} - no documents deleted")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete dub job {job_id}: {e}")
            return False

# Global service instance
dub_job_service = DubJobService()