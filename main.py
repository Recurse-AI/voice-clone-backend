from fastapi import FastAPI, Form, HTTPException, BackgroundTasks, UploadFile, File, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union
import os
import shutil
from pathlib import Path
import json
import random
import requests
import urllib.parse
import time
import logging
import soundfile as sf
from datetime import datetime

from config import settings
from r2_storage import R2Storage
from video_processor.base_processor import AudioProcessor
from video_processor.voice_cloning import set_seed

from status_manager import status_manager, ProcessingStatus
from utils import cleanup_temp_files

from contextlib import asynccontextmanager

from schemas import StatusResponse, StartProcessingResponse, RegenerateSegmentRequest, RegenerateSegmentResponse, ExportVideoRequest, ExportJobResponse, ExportStatusResponse, ProcessingLogs, AudioSeparationRequest, AudioSeparationResponse, SeparationStatusResponse, QueueStatsResponse

# Configure logging with UTF-8 support
os.makedirs(settings.LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(settings.LOGS_DIR, 'api.log')

# Configure UTF-8 encoding for both console and file handlers
import sys
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# File handler with UTF-8 encoding for Unicode support
file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

# Custom dependency to handle video_file parameter properly
async def get_video_file(video_file: Union[UploadFile, str, None] = File(None)) -> Optional[UploadFile]:
    """Handle video_file parameter - convert empty strings to None"""
    if isinstance(video_file, str):
        # If it's an empty string, return None
        if not video_file.strip():
            return None
        # If it's a non-empty string, this shouldn't happen but handle gracefully
        return None
    return video_file

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    # Load Dia model
    print("Starting Dia model loading...")
    try:
        success = audio_processor.load_dia_model(
            repo_id=settings.DIA_MODEL_REPO
        )
        if not success:
            print("ERROR: Failed to load Dia model on startup")
            logger.warning("Failed to load Dia model on startup")
        else:
            print("Dia model loaded successfully on startup")
    except Exception as e:
        print(f"EXCEPTION during model loading: {e}")
        logger.error(f"Exception during model loading: {e}")
    
    # Create temp directory
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    logger.info(f"API started successfully on {settings.HOST}:{settings.PORT}")
    yield

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
audio_processor = AudioProcessor(settings.TEMP_DIR)

# Response models

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint - API status"""
    return StatusResponse(
        status="active",
        message=f"Voice Cloning API {settings.API_VERSION} is running",
        details={
            "version": settings.API_VERSION,
            "title": settings.API_TITLE,
            "description": settings.API_DESCRIPTION,
            "endpoints": {
                "process": "/process-video",
                "status": "/status/{audio_id}",
                "regenerate": "/regenerate-segment",
                "export": "/api/export-video",
                "export_status": "/api/export-status/{job_id}",
                "export_cancel": "/api/export-cancel/{job_id}",
                "audio_separation": "/api/audio-separation",
                "separation_status": "/api/audio-separation-status/{request_id}",
                "separation_cancel": "/api/audio-separation-cancel/{request_id}",
                "queue_stats": "/api/audio-separation-queue-stats"
            }
        }
    )


@app.post("/process-video", response_model=StartProcessingResponse)
async def process_video(
    video_url: Optional[str] = Form(None, description="Video URL (HTTP/HTTPS) for processing with automatic separation"),
    video_file: Optional[UploadFile] = Depends(get_video_file),
    include_instruments: bool = Form(True, description="Whether to include instruments in final audio"),
    generate_subtitles: bool = Form(True, description="Whether to generate subtitles"),
    temperature: float = Form(settings.DIA_TEMPERATURE, description="Voice cloning temperature"),
    cfg_scale: float = Form(settings.DIA_CFG_SCALE, description="CFG scale for voice cloning"),
    top_p: float = Form(settings.DIA_TOP_P, description="Top-p for voice cloning"),
    target_language: str = Form("English", description="Target language for translation"),
    language_code: Optional[str] = Form(None, description="Language code for transcription (e.g., en, es, fr, de, hi, ja, zh) - leave empty/None for auto-detection"),
    speakers_expected: Optional[str] = Form(None, description="Expected number of speakers (1-10)")
):
    """Start video processing with immediate response - accepts either URL or file upload"""
    
    # Clean up inputs
    video_url = video_url.strip() if video_url else None
    language_code = language_code.strip() if language_code else None
    
    # Handle speakers_expected: convert empty string to default value 1
    if speakers_expected is None or speakers_expected == "":
        speakers_expected_int = 1
    else:
        try:
            speakers_expected_int = int(speakers_expected)
        except ValueError:
            raise HTTPException(status_code=400, detail="speakers_expected must be a valid integer")
    
    speakers_expected = speakers_expected_int
    
    # Validate input: either video_url or video_file, not both, not both empty
    has_url = bool(video_url)
    has_file = bool(video_file and hasattr(video_file, 'filename') and video_file.filename)
    
    if not has_url and not has_file:
        raise HTTPException(status_code=400, detail="Either video_url or video_file must be provided")
    
    if has_url and has_file:
        raise HTTPException(status_code=400, detail="Provide either video_url or video_file, not both")
    
    # Validate URL format if provided
    if has_url and not video_url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid video URL format")
    
    # Validate file if provided
    if has_file:
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
        file_ext = Path(video_file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported video format. Allowed: {', '.join(allowed_extensions)}")
    
    if speakers_expected is not None and (speakers_expected < 1 or speakers_expected > 10):
        raise HTTPException(status_code=400, detail="speakers_expected must be between 1 and 10")
    
    # Generate unique audio ID
    audio_id = r2_storage.generate_audio_id()
    
    # Prepare video source for queue processing
    video_source = None
    if has_file:
        try:
            # Save uploaded file temporarily for queue processing
            upload_temp_path = os.path.join(settings.TEMP_DIR, f"{audio_id}_uploaded_video{Path(video_file.filename).suffix}")
            
            with open(upload_temp_path, "wb") as buffer:
                content = await video_file.read()
                buffer.write(content)
            
            # FileHandler will validate this file in queue processing
            video_source = upload_temp_path
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    else:
        # FileHandler will download this URL in queue processing  
        video_source = video_url
    
    # Initialize status tracking
    status_manager.initialize_status(audio_id)
    
    # Submit to video processing queue
    from video_processor.video_queue_manager import video_queue_manager
    
    # Prepare parameters for queue processing
    queue_parameters = {
        "include_instruments": include_instruments,
        "generate_subtitles": generate_subtitles,
        "temperature": temperature,
        "cfg_scale": cfg_scale,
        "top_p": top_p,
        "target_language": target_language,
        "language_code": language_code,
        "speakers_expected": speakers_expected
    }
    
    # Submit to queue (this will handle the processing automatically)
    request_id = video_queue_manager.submit_request(
        video_source=video_source,
        audio_id=audio_id,
        is_file_upload=has_file,
        parameters=queue_parameters
    )
    
    # Get queue information for response
    queue_stats = video_queue_manager.get_queue_stats()
    queue_request_status = video_queue_manager.get_request_status(request_id)
    
    # Determine initial status message
    if queue_request_status and queue_request_status.get("queue_position", 0) > 0:
        message = f"Video processing queued successfully (position: {queue_request_status['queue_position']})"
        estimated_time = queue_request_status.get("estimated_time", "15-30 minutes")
    else:
        message = "Video processing started successfully"
        estimated_time = "10-20 minutes"
    
    # Return immediate response
    return StartProcessingResponse(
        success=True,
        audio_id=audio_id,
        message=message,
        status="processing",
        estimated_time=estimated_time,
        status_check_url=f"/status/{audio_id}",
    )

@app.post("/regenerate-segment", response_model=RegenerateSegmentResponse)
async def regenerate_segment(request: RegenerateSegmentRequest):
    """Regenerate a single audio segment with custom parameters"""
    try:
        import tempfile
        import time
        from datetime import datetime
        
        # Validate inputs
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if not request.reference_audio_url.strip():
            raise HTTPException(status_code=400, detail="Reference audio URL cannot be empty")
        
        if request.duration <= 0:
            raise HTTPException(status_code=400, detail="Duration must be positive")
        
        # Set parameters with defaults
        seed = request.seed or settings.DEFAULT_SEED
        temperature = request.temperature or 1.3
        cfg_scale = request.cfg_scale or 3.0
        top_p = request.top_p or 0.95
        
        generation_start = time.time()
        
        # Set seed for reproducibility
        set_seed(seed)
        
        # Download reference audio to temp file
        response = requests.get(request.reference_audio_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download reference audio")
        
        # Create temporary file for reference audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_ref:
            temp_ref.write(response.content)
            temp_ref_path = temp_ref.name
        
        try:
            # Load voice cloning service
            voice_service = audio_processor.voice_cloning_service
            if not voice_service.is_model_loaded():
                if not voice_service.load_dia_model():
                    raise HTTPException(status_code=500, detail="Failed to load Dia model")
            
            # Generate audio using Dia format (text + "\n" + text)
            combined_text = request.text + "\n" + request.text
            
            import torch
            with torch.inference_mode():
                cloned_audio = voice_service.dia_model.generate(
                    text=combined_text,
                    audio_prompt=temp_ref_path,
                    use_torch_compile=False,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=settings.DIA_CFG_FILTER_TOP_K,
                    max_tokens=settings.DIA_MAX_TOKENS,
                    verbose=False
                )
            
            if cloned_audio is None:
                raise HTTPException(status_code=500, detail="Audio generation failed")
            
            # Adjust audio length
            adjusted_audio = voice_service._adjust_audio_length(
                cloned_audio, 
                request.duration, 
                use_speed_adjustment=settings.USE_SPEED_ADJUSTMENT
            )
            
            # Create output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"regenerated_segment_{timestamp}_{seed}.wav"
            output_path = os.path.join(settings.TEMP_DIR, output_filename)
            
            # Save the audio
            sf.write(output_path, adjusted_audio, voice_service.sample_rate)
            
            generation_time = time.time() - generation_start
            
            # Upload to R2 if available
            audio_url = None
            try:
                r2_key = f"regenerated-segments/{timestamp}/{output_filename}"
                upload_result = r2_storage.upload_file(output_path, r2_key, "audio/wav")
                if upload_result.get("success"):
                    audio_url = upload_result.get("url")
            except Exception as e:
                logger.warning(f"R2 upload failed: {e}")
            
            return RegenerateSegmentResponse(
                success=True,
                message="Segment regenerated successfully",
                audio_url=audio_url,
                duration=request.duration,
                generation_time=generation_time,
                parameters_used={
                    "seed": seed,
                    "temperature": temperature,
                    "cfg_scale": cfg_scale,
                    "top_p": top_p,
                    "text": request.text,
                    "reference_audio_url": request.reference_audio_url
                }
            )
            
        finally:
            # Clean up temp reference file
            if os.path.exists(temp_ref_path):
                os.unlink(temp_ref_path)
            
            # Clean up generated output file after R2 upload
            if 'output_path' in locals() and os.path.exists(output_path):
                os.unlink(output_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in regenerate_segment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Export Video Endpoints
@app.post("/api/export-video", response_model=ExportJobResponse)
async def start_video_export(request: ExportVideoRequest, background_tasks: BackgroundTasks):
    """Start video export job"""
    try:
        from export_video.job_manager import export_job_manager
        
        # Create new export job
        job = export_job_manager.create_job(request.dict())
        
        # Estimate processing duration
        timeline_duration = request.timeline.get("duration", 0)
        estimated_duration = export_job_manager.estimate_duration(timeline_duration)
        
        # Start background processing
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

@app.post("/api/export-cancel/{job_id}")
async def cancel_export(job_id: str):
    """Cancel ongoing export job"""
    try:
        from export_video.job_manager import export_job_manager
        
        job = export_job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Export job not found")
        
        success = export_job_manager.cancel_job(job_id)
        if not success:
            return {"success": False, "message": "Cannot cancel completed or failed job"}
        
        return {"success": True, "message": "Export cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel export: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel export: {str(e)}")

# Audio Separation API Endpoints
@app.post("/api/audio-separation", response_model=AudioSeparationResponse)
async def start_audio_separation(request: AudioSeparationRequest):
    """Start audio separation job using RunPod queue"""
    try:
        from runpod_queue_service import runpod_queue_service
        from schemas import AudioSeparationResponse
        
        # Submit request to queue
        request_id = runpod_queue_service.submit_separation_request(
            request.audioUrl,
            caller_info=request.callerInfo or "audio_separation_api"
        )
        
        # Get initial status for queue position
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
        from schemas import SeparationStatusResponse
        
        status = runpod_queue_service.get_separation_status(request_id)
        if not status:
            raise HTTPException(status_code=404, detail="Separation request not found")
        
        # Extract vocal and instrument URLs if completed
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

@app.post("/api/audio-separation-cancel/{request_id}")
async def cancel_audio_separation(request_id: str):
    """Cancel audio separation request"""
    try:
        from runpod_queue_service import runpod_queue_service
        
        success = runpod_queue_service.cancel_separation_request(request_id)
        if not success:
            return {"success": False, "message": "Cannot cancel completed or failed request"}
        
        return {"success": True, "message": "Audio separation cancelled successfully"}
        
    except Exception as e:
        logger.error(f"Failed to cancel separation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel separation: {str(e)}")

@app.get("/api/audio-separation-queue-stats", response_model=QueueStatsResponse)
async def get_separation_queue_stats():
    """Get current queue statistics"""
    try:
        from runpod_queue_service import runpod_queue_service
        from schemas import QueueStatsResponse
        
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





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=False
    ) 