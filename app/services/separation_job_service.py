import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from bson import ObjectId

from app.models.separation_job import SeparationJob
from app.config.database import separation_jobs_collection
from app.config.constants import DEFAULT_QUERY_LIMIT, MAX_QUERY_LIMIT
from app.services.base_job_service import BaseJobService

logger = logging.getLogger(__name__)


class SeparationJobService(BaseJobService[SeparationJob]):
    """
    Service for managing separation jobs in MongoDB.
    Inherits duplicate prevention and common operations from BaseJobService.
    """
    
    def __init__(self):
        super().__init__(separation_jobs_collection)
    
    def _create_job_model(self, job_data: Dict[str, Any]) -> SeparationJob:
        """Create SeparationJob model instance from data"""
        return SeparationJob(**job_data)
    
    async def create_job(self, job_data: Dict[str, Any]) -> Optional[SeparationJob]:
        """
        Create a new separation job with duplicate prevention.
        Uses BaseJobService for comprehensive safety.
        """
        return await self.create_job_safe(job_data)
    
    async def get_job(self, job_id: str) -> Optional[SeparationJob]:
        """Get separation job by job_id"""
        try:
            job_data = await self.collection.find_one({"job_id": job_id})
            if job_data:
                job_data['id'] = str(job_data['_id'])
                del job_data['_id']
                return SeparationJob(**job_data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get separation job {job_id}: {e}")
            return None
    

    
    async def update_job_status(self, job_id: str, status: str, progress: int = None, **kwargs) -> bool:
        """Update job status and progress with automatic timestamp handling"""
        # Add timestamp fields based on status
        if status == 'processing':
            kwargs["started_at"] = datetime.now(timezone.utc)
        elif status in ['completed', 'failed']:
            kwargs["completed_at"] = datetime.now(timezone.utc)
        
        # Use base class method for consistent handling
        return await self.update_job_status_safe(job_id, status, progress, **kwargs)
    
    async def get_user_jobs(self, user_id: str, page: int = 1, limit: int = None) -> Tuple[List[SeparationJob], int]:
        """Get paginated separation jobs for a user"""
        try:
            # Use default limit if not provided
            if limit is None:
                limit = DEFAULT_QUERY_LIMIT
            
            # Ensure limit doesn't exceed maximum
            limit = min(limit, MAX_QUERY_LIMIT)
            
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
                job_data['id'] = str(job_data['_id'])
                del job_data['_id']
                jobs.append(SeparationJob(**job_data))
            
            return jobs, total_count
            
        except Exception as e:
            logger.error(f"Failed to get user separation jobs: {e}")
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
            
            # Get processing count (pending, processing, uploading statuses)
            processing_count = await self.collection.count_documents({
                **query_filter,
                "status": {"$in": ["pending", "processing", "uploading"]}
            })
            
            return {
                "total": total_count,
                "completed": completed_count,
                "processing": processing_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get user separation job statistics: {e}")
            return {"total": 0, "completed": 0, "processing": 0}
    
    async def get_jobs_by_status(self, status: str, limit: int = 100) -> List[SeparationJob]:
        """Get jobs by status (for monitoring/cleanup)"""
        try:
            cursor = self.collection.find(
                {"status": status}
            ).sort("created_at", -1).limit(limit)
            
            jobs = []
            async for job_data in cursor:
                job_data['id'] = str(job_data['_id'])
                del job_data['_id']
                jobs.append(SeparationJob(**job_data))
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to get jobs by status {status}: {e}")
            return []
    
    async def update_job_field(self, job_id: str, field_name: str, field_value: Any) -> bool:
        """Update a specific job field"""
        try:
            result = await self.collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        field_name: field_value,
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update separation job field {field_name} for {job_id}: {e}")
            return False
    
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
            logger.error(f"Failed to update separation job details {job_id}: {e}")
            return False
    
    async def delete_job(self, job_id: str, user_id: str) -> bool:
        """Delete a separation job (with user ownership check)"""
        try:
            # First check if job exists and belongs to user
            job = await self.get_job(job_id)
            if not job:
                logger.warning(f"Separation job {job_id} not found for deletion")
                return False
            
            if job.user_id != user_id:
                logger.warning(f"User {user_id} attempted to delete separation job {job_id} they don't own")
                return False
            
            # Delete the job
            result = await self.collection.delete_one({"job_id": job_id})
            
            success = result.deleted_count > 0
            if success:
                logger.info(f"Deleted separation job {job_id} for user {user_id}")
            else:
                logger.warning(f"Failed to delete separation job {job_id} - no documents deleted")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete separation job {job_id}: {e}")
            return False

# Global service instance
separation_job_service = SeparationJobService()