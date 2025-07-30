from fastapi import FastAPI, Form, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import os
import shutil
from pathlib import Path
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from config import settings
from r2_storage import R2Storage
from status_manager import status_manager, ProcessingStatus
from utils import cleanup_temp_files, local_storage
from video_downloader import video_download_service
from schemas import StatusResponse, StartProcessingResponse, ExportVideoRequest, ExportJobResponse, ExportStatusResponse, ProcessingLogs, AudioSeparationRequest, AudioSeparationResponse, SeparationStatusResponse, QueueStatsResponse, VideoDownloadRequest, VideoDownloadResponse, FileUploadResponse, UploadStatusResponse

# Configure logging
os.makedirs(settings.LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(settings.LOGS_DIR, 'api.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Local memory storage for uploaded files
uploaded_files_memory = {}
upload_status_memory = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    logger.info("Starting API initialization...")
    
    # Create temp directory
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    
    # Initialize Fish Speech models on startup
    logger.info("🚀 Initializing Fish Speech for voice cloning...")
    try:
        from video_processor.voice_cloning import get_fish_speech_service
        
        # Get Fish Speech service and force complete initialization
        fish_service = get_fish_speech_service()
        fish_service._initialize_models()
        
        logger.info("✅ OpenAudio S1-mini ready for voice cloning!")
        
    except Exception as e:
        logger.error(f"❌ Fish Speech initialization failed: {str(e)}")
        logger.error("Make sure to run: ./runpod_setup.sh to setup the complete environment")
        raise e  # Exit if models can't load
    
    logger.info(f"API started successfully on {settings.HOST}:{settings.PORT}")
    yield
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down API...")
    try:
        from video_processor.voice_cloning import get_fish_speech_service
        fish_service = get_fish_speech_service()
        fish_service.cleanup()
        logger.info("✅ Fish Speech service cleaned up")
    except:
        pass

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
r2_storage = R2Storage()

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint - API status"""
    return StatusResponse(
        status="active",
        message=f"Voice Cloning API {settings.API_VERSION} is running"
    )



@app.post("/video-dub", response_model=StartProcessingResponse)
async def video_dub(
    video_url: str = Form(..., description="Video URL for dubbing"),
    instructment: bool = Form(True, description="Whether to include instruments in final audio"),
    generate_subtitles: bool = Form(True, description="Whether to generate subtitles"),
    target_language: str = Form("English", description="Target language for translation"),
    language_code: Optional[str] = Form(None, description="Language code for transcription"),
    speakers_expected: Optional[str] = Form(None, description="Expected number of speakers (1-10)")
):
    """Start video dubbing process - downloads video, extracts audio, and processes with queue system"""
    
    # Clean up inputs
    video_url = video_url.strip()
    language_code = language_code.strip() if language_code else None
    speakers_expected = int(speakers_expected) if speakers_expected else 1
    
    # Validate URL format
    if not video_url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid video URL format")
        
    # Generate unique audio ID
    audio_id = r2_storage.generate_audio_id()
    
    # Initialize status tracking
    status_manager.initialize_status(audio_id)
    
    # Submit to video processing queue
    from video_processor.video_queue_manager import video_queue_manager
    
    queue_parameters = {
        "include_instruments": instructment,
        "generate_subtitles": generate_subtitles,
        "target_language": target_language,
        "language_code": language_code,
        "speakers_expected": speakers_expected,
        "original_source_url": video_url
    }
    
    video_queue_manager.submit_request(
        video_source=video_url,
        audio_id=audio_id,
        is_file_upload=False,
        parameters=queue_parameters
    )
    
    return StartProcessingResponse(
        success=True,
        audio_id=audio_id,
        message="Video dubbing started successfully - added to queue",
        status="queued",
        estimated_time="10-20 minutes",
        status_check_url=f"/status/{audio_id}",
    )

# Export Video Endpoints
@app.post("/api/export-video", response_model=ExportJobResponse)
async def start_video_export(request: ExportVideoRequest, background_tasks: BackgroundTasks):
    """Start video export job"""
    try:
        from export_video.job_manager import export_job_manager
        
        job = export_job_manager.create_job(request.dict())
        timeline_duration = request.timeline.get("duration", 0)
        estimated_duration = export_job_manager.estimate_duration(timeline_duration)
        
        from export_video.background_processor import BackgroundProcessor
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

@app.get("/api/export-status/{job_id}", response_model=ExportStatusResponse)
async def get_export_status(job_id: str):
    """Get export job status"""
    try:
        from export_video.job_manager import export_job_manager
        
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

# Audio Separation API Endpoints
@app.post("/api/audio-separation", response_model=AudioSeparationResponse)
async def start_audio_separation(request: AudioSeparationRequest):
    """Start audio separation job using RunPod queue"""
    try:
        from runpod_queue_service import runpod_queue_service
        
        request_id = runpod_queue_service.submit_separation_request(
            request.audioUrl,
            caller_info=request.callerInfo or "audio_separation_api"
        )
        
        status = runpod_queue_service.get_separation_status(request_id)
        queue_position = status.get("queue_position") if status else None
        
        return AudioSeparationResponse(
            success=True,
            requestId=request_id,
            message="Audio separation request submitted successfully",
            estimatedTime="5-15 minutes",
            statusCheckUrl=f"/api/audio-separation-status/{request_id}",
            queuePosition=queue_position
        )
        
    except Exception as e:
        logger.error(f"Failed to start audio separation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start separation: {str(e)}")

@app.get("/api/audio-separation-status/{request_id}", response_model=SeparationStatusResponse)
async def get_audio_separation_status(request_id: str):
    """Get status of audio separation request"""
    try:
        from runpod_queue_service import runpod_queue_service
        
        status = runpod_queue_service.get_separation_status(request_id)
        if not status:
            raise HTTPException(status_code=404, detail="Separation request not found")
        
        vocal_url = None
        instrument_url = None
        
        if status["status"] == "completed" and status.get("result"):
            result = status["result"]
            if result.get("output"):
                vocal_url = result["output"].get("vocal_audio")
                instrument_url = result["output"].get("instrument_audio")
        
        return SeparationStatusResponse(
            requestId=request_id,
            status=status["status"],
            progress=status["progress"],
            queuePosition=status.get("queue_position"),
            vocalUrl=vocal_url,
            instrumentUrl=instrument_url,
            error=status.get("error"),
            createdAt=status["created_at"],
            startedAt=status.get("started_at"),
            completedAt=status.get("completed_at"),
            callerInfo=status.get("caller_info")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get separation status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.get("/api/audio-separation-queue-stats", response_model=QueueStatsResponse)
async def get_separation_queue_stats():
    """Get current queue statistics"""
    try:
        from runpod_queue_service import runpod_queue_service
        
        stats = runpod_queue_service.get_queue_stats()
        
        return QueueStatsResponse(
            totalRequests=stats["total_requests"],
            pending=stats["pending"],
            processing=stats["processing"],
            completed=stats["completed"],
            failed=stats["failed"],
            maxConcurrent=stats["max_concurrent"],
            queueLength=stats["queue_length"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue stats: {str(e)}")

@app.get("/status/{audio_id}")
async def get_status(audio_id: str):
    """Get processing status for a specific audio ID"""
    try:
        status = status_manager.get_status(audio_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-file")
async def upload_file(video_file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Upload video file to Cloudflare with background processing"""
    file_id = r2_storage.generate_audio_id()
    original_filename = video_file.filename
    
    try:
        temp_file_path = os.path.join(settings.TEMP_DIR, f"upload_{file_id}_{original_filename}")
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        
        upload_status_memory[file_id] = {
            "status": "uploading",
            "progress": 5,
            "message": "Saving uploaded file...",
            "original_filename": original_filename,
            "started_at": datetime.now().isoformat()
        }
        
        total_size = 0
        with open(temp_file_path, "wb") as buffer:
            while chunk := await video_file.read(8192):
                buffer.write(chunk)
                total_size += len(chunk)
        
        file_size = os.path.getsize(temp_file_path)
        upload_status_memory[file_id].update({
            "progress": 15,
            "message": f"File saved ({file_size // (1024*1024)} MB), starting background processing..."
        })
        
        background_tasks.add_task(process_file_background_only, file_id, temp_file_path, original_filename, file_size)
        
        return {
            "success": True,
            "message": "File upload started successfully",
            "file_id": file_id,
            "status_check_url": f"/upload-status/{file_id}",
            "original_filename": original_filename,
            "file_size_mb": file_size // (1024*1024),
            "estimated_time": "2-10 minutes"
        }
        
    except Exception as e:
        temp_file_path = os.path.join(settings.TEMP_DIR, f"upload_{file_id}_{original_filename}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        if file_id in upload_status_memory:
            upload_status_memory[file_id].update({
                "status": "failed",
                "progress": 0,
                "message": f"File save failed: {str(e)}"
            })
        raise HTTPException(status_code=500, detail=f"Failed to start upload: {str(e)}")

async def process_file_background_only(file_id: str, temp_file_path: str, filename: str, file_size: int):
    """Background processing for uploaded file"""
    try:
        upload_status_memory[file_id].update({
            "progress": 20, 
            "message": "Validating uploaded file..."
        })
        
        if not os.path.exists(temp_file_path):
            raise Exception("Temporary file not found")
            
        if file_size == 0:
            raise Exception("Uploaded file is empty")
        
        upload_status_memory[file_id].update({"progress": 30, "message": "Checking file format..."})
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            upload_status_memory[file_id].update({
                "status": "failed", 
                "progress": 0, 
                "message": f"Unsupported video format. Allowed: {', '.join(allowed_extensions)}"
            })
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return
        
        upload_status_memory[file_id].update({"progress": 60, "message": "Storing locally..."})
        
        with open(temp_file_path, 'rb') as f:
            video_file_content = f.read()
        
        local_result = local_storage.store_video(file_id, video_file_content, filename)
        
        if not local_result.get("success"):
            upload_status_memory[file_id].update({
                "status": "failed",
                "progress": 0,
                "message": f"Local storage failed: {local_result.get('error', 'Unknown error')}"
            })
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return
        
        upload_status_memory[file_id].update({"progress": 85, "message": "Uploading to cloud storage..."})
        file_key = f"uploads/{file_id}/{filename}"
        upload_result = r2_storage.upload_file(temp_file_path, file_key)
        
        if not upload_result.get("success"):
            upload_status_memory[file_id].update({
                "status": "failed",
                "progress": 0,
                "message": f"Cloud upload failed: {upload_result.get('error', 'Unknown error')}"
            })
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return
        
        uploaded_url = upload_result["url"]
        
        upload_status_memory[file_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Upload completed successfully",
            "file_url": uploaded_url,
            "local_path": local_result["local_path"],
            "expires_at": local_result["expires_at"]
        })
        
        uploaded_files_memory[file_id] = {
            "url": uploaded_url,
            "filename": filename,
            "size": file_size,
            "upload_time": datetime.now().isoformat(),
            "local_path": local_result["local_path"],
            "expires_at": local_result["expires_at"]
        }
        
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
    except Exception as e:
        upload_status_memory[file_id].update({
            "status": "failed",
            "progress": 0,
            "message": f"Upload processing failed: {str(e)}"
        })
        
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/upload-status/{file_id}", response_model=UploadStatusResponse)
async def get_upload_status(file_id: str):
    """Get upload progress status"""
    try:
        if file_id not in upload_status_memory:
            raise HTTPException(status_code=404, detail="Upload ID not found")
        
        data = upload_status_memory[file_id]
        
        return UploadStatusResponse(
            file_id=file_id,
            status=data["status"],
            progress=data["progress"],
            message=data["message"],
            original_filename=data.get("original_filename"),
            file_url=data.get("file_url")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/terminate/{audio_id}")
async def terminate_process(audio_id: str, reason: Optional[str] = None):
    """Terminate a processing request"""
    try:
        if not audio_id:
            raise HTTPException(status_code=400, detail="Audio ID is required")
        
        status = status_manager.get_status(audio_id)
        if not status:
            raise HTTPException(status_code=404, detail="Process not found")
        
        if status.get("status") in ["completed", "failed"]:
            return {
                "success": False, 
                "message": f"Process already {status.get('status')}",
                "current_status": status.get("status")
            }
        
        termination_reason = reason or "Manual termination via API"
        success = status_manager.force_terminate_process(audio_id, termination_reason)
        
        if success:
            logger.info(f"Successfully terminated process {audio_id}: {termination_reason}")
            return {
                "success": True,
                "message": "Process terminated successfully",
                "audio_id": audio_id,
                "reason": termination_reason,
                "terminated_at": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "Failed to terminate process - it may have already completed",
                "audio_id": audio_id
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to terminate process {audio_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Termination failed: {str(e)}")

@app.post("/api/download-video", response_model=VideoDownloadResponse)
async def download_video(request: VideoDownloadRequest):
    """Download video from URL and upload to Cloudflare R2"""
    try:
        logger.info(f"Video download request: {request.url}")
        
        video_download_service.cleanup_old_files()
        
        result = await video_download_service.download_video(
            url=request.url,
            quality=request.quality
        )
        
        if result["success"]:
            logger.info(f"Video download successful: {result['download_id']}")
            return VideoDownloadResponse(
                success=True,
                message=result["message"],
                download_id=result["download_id"],
                video_info=result["video_info"],
                cloudflare=result["cloudflare"]
            )
        else:
            logger.error(f"Video download failed: {result['error']}")
            return VideoDownloadResponse(
                success=False,
                message="Video download failed",
                error=result["error"]
            )
            
    except Exception as e:
        logger.error(f"Video download endpoint error: {str(e)}")
        return VideoDownloadResponse(
            success=False,
            message="Internal server error",
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=False
    ) 