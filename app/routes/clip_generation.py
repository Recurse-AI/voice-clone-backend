from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import Optional, List
import uuid
import os
import tempfile
import logging
from app.services.r2_service import R2Service
from app.repositories.clip_repository import ClipRepository
from app.dependencies.auth import get_current_user
from app.queue.queue_manager import QueueManager
from app.models import ClipJob
from app.schemas import (
    ClipJobListResponse, 
    ClipJobDetailResponse, 
    GenerateClipsRequest
)
from app.config.constants import DEFAULT_QUERY_LIMIT

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload-clip-video")
async def upload_clip_video(video_file: UploadFile = File(...), user=Depends(get_current_user)):
    try:
        r2_service = R2Service()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await video_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            r2_key = f"clips/{user.id}/uploads/{uuid.uuid4()}.mp4"
            result = r2_service.upload_file(tmp_path, r2_key, "video/mp4")
            
            if not result["success"]:
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return {"success": True, "video_url": result["url"]}
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-clip-srt")
async def upload_clip_srt(srt_file: UploadFile = File(...), user=Depends(get_current_user)):
    try:
        r2_service = R2Service()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp:
            content = await srt_file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            r2_key = f"clips/{user.id}/uploads/{uuid.uuid4()}.srt"
            result = r2_service.upload_file(tmp_path, r2_key, "text/plain")
            
            if not result["success"]:
                raise HTTPException(status_code=500, detail=result.get("error"))
            
            return {"success": True, "srt_url": result["url"]}
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-clips")
async def generate_clips(request: GenerateClipsRequest, user=Depends(get_current_user)):
    try:
        from app.services.credit_service import credit_service
        from app.config.credit_constants import JobType as CreditJobType
        
        job_id = str(uuid.uuid4())
        duration = request.end_time - request.start_time
        
        if duration <= 0:
            raise HTTPException(status_code=400, detail="Invalid time range")
        
        job_data = {
            "job_id": job_id,
            "user_id": str(user.id),
            "video_url": request.video_url,
            "srt_url": request.srt_url,
            "start_time": request.start_time,
            "end_time": request.end_time,
            "expected_duration": request.expected_duration,
            "subtitle_style": request.subtitle_style,
            "subtitle_preset": request.subtitle_preset,
            "subtitle_font": request.subtitle_font,
            "subtitle_font_size": request.subtitle_font_size,
            "subtitle_wpl": request.subtitle_wpl,
        }
        
        result = await credit_service.reserve_credits_and_create_job(
            user_id=str(user.id),
            job_data=job_data,
            job_type=CreditJobType.CLIP,
            duration_seconds=duration
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Credit reservation failed"))
        
        queue_manager = QueueManager()
        from app.queue.clip_tasks import process_clip_generation_task
        queue_manager.enqueue(process_clip_generation_task, job_id, str(user.id))
        
        return {"success": True, "job_id": job_id}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/clip-job-status/{job_id}")
async def get_clip_job_status(job_id: str, user=Depends(get_current_user)):
    try:
        repo = ClipRepository()
        job = await repo.get_by_id(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job["user_id"] != str(user.id):
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "error_message": job.get("error_message"),
            "segments": job.get("segments", []),
            "overall_rating": job.get("overall_rating"),
            "created_at": job.get("created_at"),
            "completed_at": job.get("completed_at")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list", response_model=ClipJobListResponse)
async def get_user_clips(
    page: int = 1,
    limit: int = None,
    current_user = Depends(get_current_user)
):
    """Get paginated clip jobs for current user"""
    try:
        user_id = str(current_user.id)
        page = max(1, page)
        actual_limit = limit or DEFAULT_QUERY_LIMIT
        
        repo = ClipRepository()
        jobs, total_count = await repo.get_user_jobs(user_id, page, actual_limit)
        
        clip_jobs = [ClipJob(**job) for job in jobs]
        
        return ClipJobListResponse(
            success=True,
            message=f"Found {len(clip_jobs)} clip jobs",
            jobs=clip_jobs,
            total=total_count,
            page=page,
            limit=actual_limit,
            total_pages=(total_count + actual_limit - 1) // actual_limit
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user clips: {e}")
        raise HTTPException(status_code=500, detail="Failed to get clip jobs")

@router.get("/clip/{job_id}", response_model=ClipJobDetailResponse)
async def get_clip_job_detail(
    job_id: str,
    current_user = Depends(get_current_user)
):
    """Get detailed clip job information"""
    try:
        user_id = str(current_user.id)
        repo = ClipRepository()
        job = await repo.get_by_id(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Clip job not found")
        
        if job["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        clip_job = ClipJob(**job)
        return ClipJobDetailResponse(success=True, job=clip_job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get clip job detail: {e}")
        raise HTTPException(status_code=500, detail="Failed to get clip job details")

@router.delete("/clip/{job_id}")
async def delete_clip_job(
    job_id: str,
    current_user = Depends(get_current_user)
):
    """Delete clip job"""
    try:
        user_id = str(current_user.id)
        repo = ClipRepository()
        
        deleted = await repo.delete(job_id, user_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Clip job not found")
        
        return {"success": True, "message": "Clip job deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete clip job: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete clip job")
