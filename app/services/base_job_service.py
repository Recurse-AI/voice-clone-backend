"""
Base Job Service
Provides common job management functionality to prevent code duplication
"""

import logging
from typing import Dict, Any, Optional, TypeVar, Generic
from abc import ABC, abstractmethod
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError

logger = logging.getLogger(__name__)

# Generic type for job models
JobModel = TypeVar('JobModel')


class BaseJobService(Generic[JobModel], ABC):
    """
    Base class for job services with common duplicate prevention logic
    """
    
    def __init__(self, collection: AsyncIOMotorCollection):
        self.collection = collection
    
    @abstractmethod
    def _create_job_model(self, job_data: Dict[str, Any]) -> JobModel:
        """Create job model instance from data"""
        pass
    
    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[JobModel]:
        """Get job by job_id - must be implemented by subclass"""
        pass
    
    async def create_job_safe(self, job_data: Dict[str, Any]) -> Optional[JobModel]:
        """
        Create a new job with comprehensive duplicate prevention.
        
        Args:
            job_data: Job data dictionary
            
        Returns:
            Created job model or existing job if duplicate
        """
        try:
            # Create job model instance
            job_model = self._create_job_model(job_data)
            job_id = getattr(job_model, 'job_id')
            
            # Pre-check for existing job (application level)
            existing_job = await self.get_job(job_id)
            if existing_job:
                logger.info(f"Job {job_id} already exists, returning existing")
                return existing_job
            
            # Attempt database insertion
            job_dict = job_model.dict(exclude={'id'}) if hasattr(job_model, 'dict') else job_model.__dict__
            result = await self.collection.insert_one(job_dict)
            
            # Update model with MongoDB ID
            if hasattr(job_model, 'id'):
                job_model.id = str(result.inserted_id)
            
            logger.info(f"Created job: {job_id}")
            return job_model
            
        except DuplicateKeyError:
            # Database level duplicate - fetch existing
            job_id = job_data.get('job_id')
            logger.warning(f"Database duplicate for job {job_id}, fetching existing")
            return await self.get_job(job_id)
            
        except Exception as e:
            job_id = job_data.get('job_id', 'unknown')
            logger.error(f"Failed to create job {job_id}: {type(e).__name__}: {e}")
            return None
    
    async def update_job_status_safe(
        self, 
        job_id: str, 
        status: str, 
        progress: Optional[int] = None,
        **extra_fields
    ) -> bool:
        """
        Safely update job status with error handling.
        
        Args:
            job_id: Job identifier
            status: New status
            progress: Optional progress percentage
            **extra_fields: Additional fields to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            from datetime import datetime, timezone
            
            update_data = {
                "status": status,
                "updated_at": datetime.now(timezone.utc)
            }
            
            if progress is not None:
                update_data["progress"] = max(0, min(100, progress))  # Clamp 0-100
            
            # Add extra fields
            update_data.update(extra_fields)
            
            result = await self.collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            
            if result.matched_count == 0:
                logger.warning(f"Job {job_id} not found for status update")
                return False
                

            return True
            
        except Exception as e:
            logger.error(f"Failed to update job {job_id} status: {type(e).__name__}: {e}")
            return False
