"""
Dub Repository - Clean database operations for dub jobs
"""
import logging
from typing import Dict, Any, Optional
from app.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class DubRepository(BaseRepository):
    """Repository for dub job operations"""
    
    def __init__(self):
        super().__init__("dub_jobs")
    
    async def create_dub_job(self, job_data: Dict[str, Any]) -> Optional[str]:
        """Create a new dub job"""
        # Set default values for dub jobs
        job_data.setdefault("status", "pending")
        job_data.setdefault("progress", 0)
        job_data.setdefault("result_url", None)
        job_data.setdefault("error", None)
        job_data.setdefault("review_required", False)
        job_data.setdefault("review_status", None)
        
        return await self.create(job_data)
    
    async def update_status(self, job_id: str, status: str, progress: int = None, 
                          details: Dict[str, Any] = None) -> bool:
        """Update dub job status with progress and details"""
        update_data = {
            "status": status
        }
        
        if progress is not None:
            update_data["progress"] = progress
        
        if details:
            update_data["details"] = details
            
            # Extract important fields to top level for easy querying
            if "result_url" in details:
                update_data["result_url"] = details["result_url"]
            if "error" in details:
                update_data["error"] = details["error"]
            if "review_status" in details:
                update_data["review_status"] = details["review_status"]
            if "segments_manifest_url" in details:
                update_data["segments_manifest_url"] = details["segments_manifest_url"]
        
        return await self.update(job_id, update_data)
    
    async def update_review_status(self, job_id: str, review_status: str, 
                                 manifest_url: str = None) -> bool:
        """Update review status and manifest URL"""
        update_data = {
            "review_status": review_status,
            "review_required": True
        }
        
        if manifest_url:
            update_data["segments_manifest_url"] = manifest_url
        
        return await self.update(job_id, update_data)
    
    async def complete_job(self, job_id: str, result_url: str = None, 
                         details: Dict[str, Any] = None) -> bool:
        """Mark job as completed with results"""
        update_data = {
            "status": "completed",
            "progress": 100
        }
        
        if result_url:
            update_data["result_url"] = result_url
        if details:
            update_data["details"] = details
        
        return await self.update(job_id, update_data)
    
    async def fail_job(self, job_id: str, error: str, details: Dict[str, Any] = None) -> bool:
        """Mark job as failed with error"""
        update_data = {
            "status": "failed",
            "error": error
        }
        
        if details:
            update_data["details"] = details
        
        return await self.update(job_id, update_data)


# Global repository instance
dub_repo = DubRepository()
