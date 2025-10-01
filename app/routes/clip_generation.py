from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from typing import Optional
import uuid
import os
import tempfile
from app.services.r2_service import R2Service
from app.repositories.clip_repository import ClipRepository
from app.dependencies.auth import get_current_user
from app.queue.queue_manager import QueueManager

router = APIRouter()

@router.get("/test-auth")
async def test_auth(user=Depends(get_current_user)):
    return {"success": True, "user_id": user.id, "email": user.email}

class GenerateClipsRequest(BaseModel):
    video_url: str
    srt_url: Optional[str] = None
    start_time: float
    end_time: float
    expected_duration: float
    subtitle_style: Optional[str] = None
    subtitle_preset: Optional[str] = "reels"
    subtitle_font: Optional[str] = None
    subtitle_font_size: Optional[int] = None
    subtitle_wpl: Optional[int] = None

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
