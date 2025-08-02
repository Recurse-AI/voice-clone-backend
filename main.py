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
import asyncio

from config import settings
from r2_storage import R2Storage
from status_manager import status_manager, ProcessingStatus
from video_downloader import video_download_service
from schemas import StatusResponse, ExportVideoRequest, ExportJobResponse, ExportStatusResponse, ProcessingLogs, AudioSeparationRequest, AudioSeparationResponse, SeparationStatusResponse, QueueStatsResponse, VideoDownloadRequest, VideoDownloadResponse, UploadStatusResponse, VideoDubRequest, VideoDubResponse, VideoDubStatusResponse, VoiceCloneRequest, VoiceCloneResponse

# Configure logging
os.makedirs(settings.LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(settings.LOGS_DIR, 'api.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=10*1024*1024,  # 10 MB per file
            backupCount=5,
            encoding='utf-8'
        )
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
    
    # Initialize Fish Speech service
    try:
        from dub.fish_speech_service import initialize_fish_speech
        if initialize_fish_speech():
            logger.info("✅ Fish Speech service initialized successfully")
        else:
            logger.warning("⚠️ Fish Speech service initialization failed")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Fish Speech: {e}")
    
    logger.info(f"API started successfully on {settings.HOST}:{settings.PORT}")
    
    # Start periodic cleanup task (runs every hour)
    async def _cleanup_loop():
        while True:
            video_download_service.cleanup_old_files()
            await asyncio.sleep(3600)
    asyncio.create_task(_cleanup_loop())
    
    yield
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down API...")
    # Cleanup Fish Speech service
    try:
        from dub.fish_speech_service import cleanup_fish_speech
        cleanup_fish_speech()
        logger.info("✅ Fish Speech service cleaned up")
    except Exception as e:
        logger.error(f"❌ Failed to cleanup Fish Speech: {e}")

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

# Video-dub status endpoint removed as requested

@app.post("/upload-file")
async def upload_file(video_file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Upload video file to Cloudflare with background processing"""
    job_id = r2_storage.generate_job_id()
    original_filename = video_file.filename
    try:
        # Change: save as tmp/voice_cloning/dub_{job_id}/{original_filename}
        job_dir = os.path.join(settings.TEMP_DIR, f"dub_{job_id}")
        os.makedirs(job_dir, exist_ok=True)
        temp_file_path = os.path.join(job_dir, original_filename)
        upload_status_memory[job_id] = {
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
        upload_status_memory[job_id].update({
            "progress": 15,
            "message": f"File saved ({file_size // (1024*1024)} MB), starting background processing..."
        })
        background_tasks.add_task(process_file_background_only, job_id, temp_file_path, original_filename, file_size)
        return {
            "success": True,
            "message": "File upload started successfully",
            "job_id": job_id,
            "status_check_url": f"/upload-status/{job_id}",
            "original_filename": original_filename,
            "file_size_mb": file_size // (1024*1024),
            "estimated_time": "2-10 minutes"
        }
    except Exception as e:
        job_dir = os.path.join(settings.TEMP_DIR, f"dub_{job_id}")
        temp_file_path = os.path.join(job_dir, original_filename)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if job_id in upload_status_memory:
            upload_status_memory[job_id].update({
                "status": "failed",
                "progress": 0,
                "message": f"File save failed: {str(e)}"
            })
        from dub.audio_utils import AudioUtils
        AudioUtils.remove_temp_dir(folder_path=job_dir)
        raise HTTPException(status_code=500, detail=f"Failed to start upload: {str(e)}")

# --- Background process for upload ---
async def process_file_background_only(job_id: str, temp_file_path: str, filename: str, file_size: int):
    from dub.audio_utils import AudioUtils
    job_dir = os.path.dirname(temp_file_path)
    try:
        upload_status_memory[job_id].update({
            "progress": 20, 
            "message": "Validating uploaded file...",
            "status": "uploading"
        })
        if not os.path.exists(temp_file_path):
            raise Exception("Temporary file not found")
        if file_size == 0:
            raise Exception("Uploaded file is empty")
        upload_status_memory[job_id].update({"progress": 30, "message": "Checking file format...", "status": "uploading"})
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            upload_status_memory[job_id].update({
                "status": "failed", 
                "progress": 0, 
                "message": f"Unsupported video format. Allowed: {', '.join(allowed_extensions)}"
            })
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            AudioUtils.remove_temp_dir(folder_path=job_dir)
            return
        upload_status_memory[job_id].update({
            "progress": 100,
            "message": "File saved locally.",
            "status": "done",
            "file_url": temp_file_path
        })
    except Exception as e:
        upload_status_memory[job_id].update({
            "status": "failed",
            "progress": 0,
            "message": f"Processing failed: {str(e)}"
        })
        AudioUtils.remove_temp_dir(folder_path=job_dir)

@app.get("/upload-status/{job_id}", response_model=UploadStatusResponse)
async def get_upload_status(job_id: str):
    """Get upload progress status"""
    try:
        if job_id not in upload_status_memory:
            raise HTTPException(status_code=404, detail="Upload ID not found")
        data = upload_status_memory[job_id]
        # status: pending, uploading, done, failed
        status = data.get("status", "pending")
        progress = data.get("progress", 0)
        message = data.get("message", "")
        original_filename = data.get("original_filename")
        file_url = data.get("file_url")
        return {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "message": message,
            "original_filename": original_filename,
            "file_url": file_url
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Video-dub terminate endpoint removed as requested

@app.post("/api/video-dub", response_model=VideoDubResponse)
async def start_video_dub(request: VideoDubRequest, background_tasks: BackgroundTasks):
    """
    Start video dubbing job (background).
    ভিডিও upload-file API দিয়ে upload করতে হবে, তারপর এখানে সেই job_id দিতে হবে।
    video_url ফিল্ড নেই, কারণ ভিডিও local-এ সংরক্ষিত থাকবে।
    """
    # Immediately initialize status so that the status endpoint can return data right away
    from status_manager import status_manager, ProcessingStatus
    status_manager.initialize_status(request.job_id)
    status_manager.update_status(request.job_id, ProcessingStatus.PENDING, progress=0)

    background_tasks.add_task(process_video_dub_background, request)
    return VideoDubResponse(
        success=True,
        message="Video dub started successfully",
        job_id=request.job_id,
        status_check_url=f"/api/video-dub-status/{request.job_id}"
    )

@app.get("/api/video-dub-status/{job_id}", response_model=VideoDubStatusResponse)
async def get_video_dub_status(job_id: str):
    if not status_manager.get_status(job_id):
        raise HTTPException(status_code=404, detail="Job ID not found")
    data = status_manager.get_status(job_id)
    return VideoDubStatusResponse(
        job_id=job_id,
        status=data["status"],
        progress=data["progress"],
        message=data["message"],
        result_url=data.get("result_url"),
        error=data.get("error"),
        details=data.get("details")
    )

# Background processing function (runs in threadpool)
def process_video_dub_background(request: VideoDubRequest):
    from dub.audio_utils import AudioUtils
    job_id = request.job_id
    job_dir = os.path.join(settings.TEMP_DIR, f"dub_{job_id}")
    try:
        status_manager.update_status(job_id, ProcessingStatus.PROCESSING, progress=10, details={"message": "Finding uploaded video..."})
        job_dir = os.path.join(settings.TEMP_DIR, f"dub_{job_id}")
        if not os.path.exists(job_dir):
            status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=0, details={"message": "Uploaded video not found.", "error": "No such directory"})
            return
        video_files = [f for f in os.listdir(job_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'))]
        if not video_files:
            status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=0, details={"message": "No video file found in upload folder.", "error": "No video file"})
            return
        local_video_path = os.path.join(job_dir, video_files[0])
        status_manager.update_status(job_id, ProcessingStatus.PROCESSING, progress=20, details={"message": "Extracting audio..."})
        from dub.audio_utils import AudioUtils
        audio_utils = AudioUtils()
        audio_path = os.path.join(job_dir, f"{job_id}.wav")
        extract_result = audio_utils.extract_audio_from_video(local_video_path, audio_path)
        if not extract_result["success"]:
            status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=0, details={"message": "Audio extraction failed.", "error": extract_result.get("error")})
            return
        status_manager.update_status(job_id, ProcessingStatus.PROCESSING, progress=30, details={"message": "Separating vocals and instruments..."})
        from dub.runpod_service import RunPodService
        runpod = RunPodService()
        r2_audio_path = r2_storage.upload_file(audio_path, f"temp/{job_id}/{job_id}.wav")
        if not r2_audio_path["success"]:
            status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=0, details={"message": "Audio upload failed.", "error": r2_audio_path.get("error")})
            return
        sep_result = runpod.process_audio_separation(r2_audio_path["url"])
        if not sep_result or not sep_result.get('id'):
            status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=0, details={"message": "Audio separation job submission failed.", "error": sep_result.get("error") if sep_result else "Unknown error"})
            return
        status = runpod.wait_for_completion(sep_result['id'])
        if status.get('status') != 'COMPLETED' or not status.get('output'):
            status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=0, details={"message": "Audio separation failed.", "error": status.get("error")})
            return
        vocal_url = status['output'].get('vocal_audio')
        instrument_url = status['output'].get('instrument_audio')
        vocal_path = os.path.join(job_dir, f"{job_id}_vocal.wav")
        instrument_path = os.path.join(job_dir, f"{job_id}_instrument.wav")
        if vocal_url:
            download_result = audio_utils.download_audio_file(vocal_url, vocal_path)
            if not download_result["success"]:
                status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=0, details={"message": "Vocal audio download failed.", "error": download_result.get("error")})
                return
        if instrument_url:
            download_result = audio_utils.download_audio_file(instrument_url, instrument_path)
            if not download_result["success"]:
                status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=0, details={"message": "Instrument audio download failed.", "error": download_result.get("error")})
                return
        status_manager.update_status(job_id, ProcessingStatus.PROCESSING, progress=40, details={"message": "Cloning voice and reconstructing audio..."})
        from dub.simple_dubbed_api import SimpleDubbedAPI
        simple_dubbed_api = SimpleDubbedAPI()
        upload_response = r2_storage.upload_file(audio_path, f"temp/{job_id}/{job_id}.wav")
        if not upload_response["success"]:
            status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=0, details={"message": "Audio upload failed.", "error": upload_response.get("error")})
            return
        audio_url = upload_response["url"]
        pipeline_result = simple_dubbed_api.process_dubbed_audio(
            job_id=job_id,
            audio_url=audio_url,
            video_path=local_video_path,
            instrument_path=instrument_path,
            target_language=request.target_language,
            speakers_count=int(request.expected_speaker) if request.expected_speaker else 1,
            source_video_language=request.source_video_language,
            subtitle=request.subtitle,
            instrument=request.instrument,
            output_dir=job_dir
        )
        if not pipeline_result["success"]:
            status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=0, details={"message": "Dubbing pipeline failed.", "error": pipeline_result.get("error"), "details": pipeline_result.get("details")})
            return
        # Clean up temporary R2 files
        r2_storage.delete_file(upload_response["r2_key"])
        r2_storage.delete_file(r2_audio_path["r2_key"])
        status_manager.update_status(job_id, ProcessingStatus.COMPLETED, progress=100, details={"message": "Video dubbing completed.", "result_url": pipeline_result.get("result_url") or (pipeline_result.get("result_urls", {}) or {}).get("final_video"), "details": pipeline_result.get("details")})
    except Exception as e:
        status_manager.update_status(job_id, ProcessingStatus.FAILED, progress=0, details={"message": f"Processing failed: {str(e)}", "error": str(e)})
    finally:
        # Ensure temp directory is removed in any case
        AudioUtils.remove_temp_dir(folder_path=job_dir)

@app.post("/api/download-video", response_model=VideoDownloadResponse)
async def download_video(request: VideoDownloadRequest):
    """Download video from URL and store locally"""
    try:
        logger.info(f"Video download request: {request.url}")

        result = await video_download_service.download_video(
            url=request.url,
            quality=request.quality
        )

        if result["success"]:
            logger.info(f"Video download successful: {result['download_id']}")
            return VideoDownloadResponse(
                success=True,
                message=result["message"],
                job_id=result["job_id"],
                video_info=result["video_info"]
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

# Voice Clone Segment API
@app.post("/api/voice-clone-segment", response_model=VoiceCloneResponse)
async def voice_clone_segment(request: VoiceCloneRequest):
    """
    Voice clone a single text segment using reference audio.
    1. Downloads reference audio from URL.
    2. Generates cloned voice with FishSpeech.
    3. Uploads generated audio to R2 bucket.
    4. Returns public URL of cloned audio.
    """
    try:
        job_id = r2_storage.generate_job_id()
        job_dir = os.path.join(settings.TEMP_DIR, f"voice_clone_{job_id}")
        os.makedirs(job_dir, exist_ok=True)

        # Download reference audio
        from dub.audio_utils import AudioUtils
        audio_utils = AudioUtils()
        reference_path = os.path.join(job_dir, "reference.wav")
        download_res = audio_utils.download_audio_file(request.referenceAudioUrl, reference_path)
        if not download_res["success"]:
            raise HTTPException(status_code=400, detail=f"Reference audio download failed: {download_res.get('error')}")

        # Read reference audio bytes
        with open(reference_path, "rb") as f:
            reference_bytes = f.read()

        # Generate cloned audio
        from dub.fish_speech_service import get_fish_speech_service
        fish_service = get_fish_speech_service()

        generation_kwargs = {}
        if request.speakerLabel:
            generation_kwargs["seed"] = abs(hash(request.speakerLabel)) % (2 ** 32)

        result = fish_service.generate_with_reference_audio(
            text=request.text,
            reference_audio_bytes=reference_bytes,
            reference_text=request.referenceText,
            **generation_kwargs
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Voice generation failed"))

        cloned_path = os.path.join(job_dir, f"{job_id}.wav")
        with open(cloned_path, "wb") as f:
            f.write(result["audio_data"])

        # Upload to R2
        r2_key = r2_storage.generate_file_path(job_id, "", f"{job_id}.wav")
        upload_res = r2_storage.upload_file(cloned_path, r2_key, content_type="audio/wav")

        if not upload_res.get("success"):
            raise HTTPException(status_code=500, detail=upload_res.get("error", "Upload failed"))

        duration_seconds = None
        if result.get("sample_rate"):
            duration_seconds = round(len(result["audio_data"]) / (result["sample_rate"] * 2), 2)

        return VoiceCloneResponse(
            success=True,
            message="Voice cloned successfully",
            jobId=job_id,
            audioUrl=upload_res["url"],
            duration=duration_seconds
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        from dub.audio_utils import AudioUtils
        AudioUtils.remove_temp_dir(folder_path=locals().get("job_dir"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=False 
    ) 