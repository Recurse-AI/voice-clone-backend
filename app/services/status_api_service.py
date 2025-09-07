import logging
from typing import Dict, Any, Optional, List
from app.services.simple_status_service import status_service
from app.services.redis_status_service import redis_status_service

logger = logging.getLogger(__name__)


class StatusAPIService:
    def __init__(self):
        self.redis_service = redis_status_service
        self.status_service = status_service
    
    def get_job_status(self, job_id: str, job_type: str) -> Optional[Dict[str, Any]]:
        try:
            status_data = self.status_service.get_status(job_id, job_type)
            
            if not status_data:
                return None
            
            return {
                "job_id": status_data["job_id"],
                "status": status_data["status"],
                "progress": status_data["progress"],
                "message": status_data.get("details", {}).get("message", ""),
                "phase": status_data.get("details", {}).get("phase", ""),
                "details": status_data.get("details", {}),
                "updated_at": status_data.get("updated_at"),
                "created_at": status_data.get("created_at")
            }
        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {e}")
            return None
    
    def get_all_jobs(self, job_type: str, include_completed: bool = False) -> List[Dict[str, Any]]:
        try:
            if self.redis_service.is_available():
                redis_jobs = self.redis_service.get_all_jobs(job_type)
                
                if not include_completed:
                    redis_jobs = [
                        job for job in redis_jobs 
                        if job.get("status") not in ["completed", "failed"]
                    ]
                
                return redis_jobs
            
            return []
        except Exception as e:
            logger.error(f"Failed to get all jobs for {job_type}: {e}")
            return []
    
    def get_job_progress(self, job_id: str, job_type: str) -> Dict[str, Any]:
        try:
            status_data = self.get_job_status(job_id, job_type)
            
            if not status_data:
                return {"progress": 0, "status": "unknown", "message": "Job not found"}
            
            return {
                "progress": status_data["progress"],
                "status": status_data["status"],
                "message": status_data["message"],
                "phase": status_data["phase"]
            }
        except Exception as e:
            logger.error(f"Failed to get job progress for {job_id}: {e}")
            return {"progress": 0, "status": "error", "message": str(e)}
    
    def is_job_active(self, job_id: str, job_type: str) -> bool:
        try:
            status_data = self.get_job_status(job_id, job_type)
            
            if not status_data:
                return False
            
            active_statuses = [
                "pending", "processing", "separating", 
                "transcribing", "uploading", "awaiting_review"
            ]
            
            return status_data["status"] in active_statuses
        except Exception as e:
            logger.error(f"Failed to check if job is active for {job_id}: {e}")
            return False
    
    def get_latest_jobs(self, job_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            all_jobs = self.get_all_jobs(job_type, include_completed=True)
            
            sorted_jobs = sorted(
                all_jobs,
                key=lambda x: x.get("updated_at", ""),
                reverse=True
            )
            
            return sorted_jobs[:limit]
        except Exception as e:
            logger.error(f"Failed to get latest jobs for {job_type}: {e}")
            return []
    
    def get_jobs_by_status(self, job_type: str, status: str) -> List[Dict[str, Any]]:
        try:
            all_jobs = self.get_all_jobs(job_type, include_completed=True)
            
            return [
                job for job in all_jobs 
                if job.get("status") == status
            ]
        except Exception as e:
            logger.error(f"Failed to get jobs by status for {job_type}: {e}")
            return []


api_status_service = StatusAPIService()
