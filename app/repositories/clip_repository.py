from typing import Dict, Any, Optional, List
from app.repositories.base_repository import BaseRepository

class ClipRepository(BaseRepository):
    def __init__(self):
        super().__init__("clip_jobs")
    
    async def create_clip_job(self, job_data: Dict[str, Any]) -> Optional[str]:
        job_data.setdefault("status", "pending")
        job_data.setdefault("progress", 0)
        job_data.setdefault("segments", [])
        return await self.create(job_data)
    
    async def update_status(self, job_id: str, status: str, progress: int = None) -> bool:
        update_data = {"status": status}
        if progress is not None:
            update_data["progress"] = progress
        return await self.update(job_id, update_data)
    
    async def add_segment(self, job_id: str, segment: Dict[str, Any]) -> bool:
        job = await self.get_by_id(job_id)
        if not job:
            return False
        segments = job.get("segments", [])
        segments.append(segment)
        return await self.update(job_id, {"segments": segments})
    
    async def update_segments(self, job_id: str, segments: List[Dict[str, Any]]) -> bool:
        return await self.update(job_id, {"segments": segments})