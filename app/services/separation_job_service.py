import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
from app.models.separation_job import SeparationJob
from app.config.database import separation_jobs_collection

logger = logging.getLogger(__name__)

class SeparationJobService:
    """Service for managing separation jobs in MongoDB"""
    
    def __init__(self):
        self.collection = separation_jobs_collection
    
    async def create_job(self, job_data: Dict[str, Any]) -> Optional[SeparationJob]:
        """Create a new separation job"""
        try:
            separation_job = SeparationJob(**job_data)
            
            # Convert to dict for MongoDB
            job_dict = separation_job.dict(exclude={'id'})
            
            # Insert into database
            result = await self.collection.insert_one(job_dict)
            separation_job.id = str(result.inserted_id)
            
            logger.info(f"Created separation job: {separation_job.job_id}")
            return separation_job
            
        except Exception as e:
            logger.error(f"Failed to create separation job: {e}")
            return None
    
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
        """Update job status and progress"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.now()
            }
            
            if progress is not None:
                update_data["progress"] = progress
                
            # Set timestamps based on status
            if status == 'processing':
                update_data["started_at"] = datetime.now()
            elif status in ['completed', 'failed']:
                update_data["completed_at"] = datetime.now()
            
            # Add additional fields
            update_data.update(kwargs)
            
            result = await self.collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            
            success = result.modified_count > 0
            if success:
                logger.info(f"Updated separation job {job_id}: {status} ({progress}%)")
            return success
            
        except Exception as e:
            logger.error(f"Failed to update separation job {job_id}: {e}")
            return False
    
    async def get_user_jobs(self, user_id: str, limit: int = 50) -> List[SeparationJob]:
        """Get all separation jobs for a user"""
        try:
            cursor = self.collection.find(
                {"user_id": user_id}
            ).sort("created_at", -1).limit(limit)
            
            jobs = []
            async for job_data in cursor:
                job_data['id'] = str(job_data['_id'])
                del job_data['_id']
                jobs.append(SeparationJob(**job_data))
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to get user separation jobs: {e}")
            return []
    
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
    
    async def update_details(self, job_id: str, details: Dict[str, Any]) -> bool:
        """Update job details"""
        try:
            result = await self.collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "details": details,
                        "updated_at": datetime.now()
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