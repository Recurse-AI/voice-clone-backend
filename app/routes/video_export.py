from fastapi import APIRouter, HTTPException, BackgroundTasks
import logging
from app.schemas import ExportVideoRequest, ExportJobResponse, ExportStatusResponse, ProcessingLogs

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/export-video", response_model=ExportJobResponse)
async def start_video_export(request: ExportVideoRequest, background_tasks: BackgroundTasks):
    """Start video export job"""
    try:
        from app.services.export_video.job_manager import export_job_manager
        from app.services.export_video.background_processor import BackgroundProcessor
        from app.config.settings import settings
        from app.utils.r2_storage import R2Storage
        
        r2_storage = R2Storage()
        
        job = export_job_manager.create_job(request.dict())
        timeline_duration = request.timeline.get("duration", 0)
        estimated_duration = export_job_manager.estimate_duration(timeline_duration)
        
        background_processor = BackgroundProcessor(settings, r2_storage)
        background_tasks.add_task(background_processor.process_video_export_background, job.job_id, request.dict())
        
        return ExportJobResponse(
            jobId=job.job_id,
            status=job.status,
            message="Video export started successfully",
            estimatedDuration=estimated_duration
        )
        
    except Exception as e:
        logger.error(f"Failed to start video export: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start export: {str(e)}")

@router.get("/export-status/{job_id}", response_model=ExportStatusResponse)
async def get_export_status(job_id: str):
    """Get export job status"""
    try:
        from app.services.export_video.job_manager import export_job_manager
        
        job = export_job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Export job not found")
        
        return ExportStatusResponse(
            jobId=job.job_id,
            status=job.status,
            progress=job.progress,
            downloadUrl=job.download_url,
            error=job.error,
            processingLogs=ProcessingLogs(logs=job.processing_logs) if job.processing_logs else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get export status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")