"""
Separation Repository - Clean database operations for separation jobs
"""
import logging
from typing import Dict, Any, Optional
from app.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class SeparationRepository(BaseRepository):
    """Repository for separation job operations"""
    
    def __init__(self):
        super().__init__("separation_jobs")
    
    async def create_separation_job(self, job_data: Dict[str, Any]) -> Optional[str]:
        """Create a new separation job"""
        # Set default values for separation jobs
        job_data.setdefault("status", "pending")
        job_data.setdefault("progress", 0)
        job_data.setdefault("vocal_url", None)
        job_data.setdefault("instrument_url", None)
        job_data.setdefault("error", None)
        
        return await self.create(job_data)
    
    async def update_status(self, job_id: str, status: str, progress: int = None, **kwargs) -> bool:
        """Update separation job status with additional fields"""
        update_data = {
            "status": status
        }
        
        if progress is not None:
            update_data["progress"] = progress
        
        # Add additional fields from kwargs
        update_data.update(kwargs)
        
        return await self.update(job_id, update_data)
    
    async def update_results(self, job_id: str, vocal_url: str = None, 
                           instrument_url: str = None, details: Dict[str, Any] = None) -> bool:
        """Update separation job with result URLs"""
        update_data = {}
        
        if vocal_url:
            update_data["vocal_url"] = vocal_url
        if instrument_url:
            update_data["instrument_url"] = instrument_url
        if details:
            update_data["details"] = details
        
        return await self.update(job_id, update_data)
    
    async def set_runpod_request_id(self, job_id: str, runpod_request_id: str) -> bool:
        """Set RunPod request ID for tracking"""
        return await self.update(job_id, {"runpod_request_id": runpod_request_id})


# Global repository instance
separation_repo = SeparationRepository()
