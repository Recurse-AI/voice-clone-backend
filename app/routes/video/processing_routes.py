from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import logging
import os
from app.schemas import VideoProcessingResponse
from typing import Optional
import uuid
import json
import tempfile
from pathlib import Path

router = APIRouter()

logger = logging.getLogger(__name__)

@router.post("/process-video-complete", response_model=VideoProcessingResponse)
async def process_video_complete(
    video_url: Optional[str] = Form(None),
    dubbed_audio_url: Optional[str] = Form(None),
    instrument_audio_url: Optional[str] = Form(None),
    timeline_audio: Optional[str] = Form(None),
    subtitle_url: Optional[str] = Form(None),
    options: str = Form("{}")
):
    """
    Complete video/audio processing API - Uses background queue processing.
    
    ALL INPUTS ARE OPTIONAL - But at least ONE must be provided:
    - video_url: Video URL (optional - for video output)
    - dubbed_audio_url: Dubbed audio URL (for audio-only processing)  
    - instrument_audio_url: Background music URL (optional)
    - timeline_audio: JSON array of audio segments (for timeline reconstruction)
    - subtitle_url: SRT file URL (optional)
    - options: Processing options (resolution, format, audio_only, etc.)
    
    Use Cases:
    - Audio-only: provide dubbed_audio_url and/or instrument_audio_url
    - Video with audio: provide video_url + audio URLs
    - Timeline reconstruction: provide timeline_audio
    
    Response:
    - job_id: Task identifier for status checking
    - message: Status message
    """
    from app.queue.queue_manager import queue_manager
    from app.services.simple_status_service import status_service, JobStatus
    
    job_id = str(uuid.uuid4())
    logger.info(f"🎬 Enqueueing video processing job {job_id}")
    
    try:
        # 1. Quick validation: At least one input required
        has_input = any([
            video_url,
            dubbed_audio_url,
            instrument_audio_url, 
            timeline_audio
        ])
        
        if not has_input:
            return VideoProcessingResponse(
                success=False,
                message="No input provided",
                job_id=job_id,
                error="At least one input required: video_url (for video), dubbed_audio_url (for audio), instrument_audio_url, or timeline_audio",
                error_code="NO_INPUT"
            )
        
        # 2. Prepare task data for queue
        task_data = {
            "job_id": job_id,
            "video_url": video_url,
            "dubbed_audio_url": dubbed_audio_url,
            "instrument_audio_url": instrument_audio_url,
            "timeline_audio": timeline_audio,
            "subtitle_url": subtitle_url,
            "options": options
        }
        # 3. Initialize job status
        status_service.update_status(
            job_id, "video_processing", JobStatus.PENDING, 0,
            {"message": "Video processing job queued"}
        )
        
        # 4. Enqueue the task
        success = queue_manager.enqueue_video_processing_task(task_data)
        if not success:
            return VideoProcessingResponse(
                success=False,
                message="Failed to queue video processing task",
                job_id=job_id,
                error="Queue system unavailable",
                error_code="QUEUE_ERROR"
            )
        
        # 5. Return immediate response with job ID
        return VideoProcessingResponse(
            success=True,
            message="Video processing task queued successfully",
            job_id=job_id
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to enqueue video processing task: {e}")
        return VideoProcessingResponse(
            success=False,
            message="Failed to queue video processing task",
            job_id=job_id,
            error=str(e),
            error_code="QUEUE_ERROR"
        )


@router.get("/process-video-status/{job_id}")
async def get_video_processing_status(job_id: str):
    """Get status of video processing job"""
    try:
        from app.services.simple_status_service import status_service
        
        status = status_service.get_status(job_id, "video_processing")
        if not status:
            return {"error": "Job not found", "job_id": job_id}
        
        return {
            "job_id": job_id,
            "status": status.get("status", "unknown"),
            "progress": status.get("progress", 0),
            "message": status.get("details", {}).get("message", ""),
            "details": status.get("details", {})
        }
    except Exception as e:
        logger.error(f"❌ Failed to get video processing status: {e}")
        return {"error": str(e), "job_id": job_id}


@router.get("/download/{job_id}/{filename}")
async def download_processed_file(job_id: str, filename: str):
    """Download processed video/audio file directly"""
    try:
        # Validate job_id format (UUID)
        try:
            uuid.UUID(job_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid job ID format")
        
        # Construct file path
        file_path = Path("tmp") / "processed" / job_id / filename
        
        # Security check: ensure file is within expected directory
        if not str(file_path.resolve()).startswith(str(Path("tmp/processed").resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine media type based on extension
        media_type = "application/octet-stream"
        if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            media_type = "video/mp4"
        elif filename.lower().endswith(('.mp3', '.wav', '.aac', '.m4a')):
            media_type = "audio/mpeg"
        elif filename.lower().endswith('.webm'):
            media_type = "video/webm"
        
        logger.info(f"📥 Serving file: {filename} for job {job_id}")
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to serve file {filename} for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
