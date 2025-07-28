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
from video_processor.voice_cloning import set_seed

from status_manager import status_manager, ProcessingStatus
from utils import cleanup_temp_files, local_storage
from video_downloader import video_download_service

from contextlib import asynccontextmanager

from schemas import StatusResponse, StartProcessingResponse, RegenerateSegmentRequest, RegenerateSegmentResponse, ExportVideoRequest, ExportJobResponse, ExportStatusResponse, ProcessingLogs, AudioSeparationRequest, AudioSeparationResponse, SeparationStatusResponse, QueueStatsResponse, VideoDownloadRequest, VideoDownloadResponse, FileUploadResponse, UploadStatusResponse

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

# Local memory storage for uploaded files
uploaded_files_memory = {}  # {file_id: {url, filename, size, upload_time}}
upload_status_memory = {}   # {file_id: {status, progress, message, started_at}}

# File upload utilities
async def cleanup_uploaded_file(file_id: str):
    """Remove uploaded file from local memory after processing completion"""
    if file_id in uploaded_files_memory:
        del uploaded_files_memory[file_id]
        logger.info(f"Cleaned up uploaded file: {file_id}")

async def periodic_cleanup():
    """Periodic cleanup of expired local storage files"""
    import asyncio
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            local_storage.cleanup_expired()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    # Initialize shared audio processor with model loading
    print("Starting API initialization...")
    try:
        # This will create the shared instance and load the model if not already loaded
        audio_processor = get_audio_processor(load_model=True)
        from video_processor import is_model_loaded
        
        if not is_model_loaded():
            print("WARNING: AudioProcessor initialized but Dia model not loaded")
            logger.warning("AudioProcessor initialized but Dia model not loaded")
            
    except Exception as e:
        print(f"EXCEPTION during initialization: {e}")
        logger.error(f"Exception during initialization: {e}")
    
    # Create temp directory
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    
    # Start cleanup task
    import asyncio
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    logger.info(f"API started successfully on {settings.HOST}:{settings.PORT}")
    yield
    
    # Cancel cleanup task on shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
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
from video_processor import get_audio_processor
# Audio processor will be initialized in lifespan event

# Response models

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint - API status"""
    return StatusResponse(
        status="active",
        message=f"Voice Cloning API {settings.API_VERSION} is running"
    )


@app.post("/process-video", response_model=StartProcessingResponse)
async def process_video(
    video_url: str = Form(..., description="Video URL (already uploaded to Cloudflare) for processing"),
    include_instruments: bool = Form(True, description="Whether to include instruments in final audio"),
    generate_subtitles: bool = Form(True, description="Whether to generate subtitles"),
    # Enhanced Dia Parameters (Colab optimized defaults)
    max_tokens: int = Form(3072, description="Maximum tokens for generation (~36 seconds audio)"),
    cfg_scale: float = Form(3.0, description="CFG scale for voice cloning (1.0-15.0)"),
    temperature: float = Form(1.8, description="Voice cloning temperature (0.5-2.0) - Reference optimized"),
    top_p: float = Form(0.95, description="Top-p for voice cloning (0.5-1.0)"),
    cfg_filter_top_k: int = Form(45, description="CFG filter top K (15-100)"),
    speed_factor: float = Form(0.9, description="Audio speed factor (0.5-1.5) - Reference optimal"),
    use_torch_compile: bool = Form(False, description="Use torch.compile for optimization - Reference stable"),
    target_language: str = Form("English", description="Target language for translation"),
    language_code: Optional[str] = Form(None, description="Language code for transcription (e.g., en, es, fr, de, hi, ja, zh) - leave empty/None for auto-detection"),
    speakers_expected: Optional[str] = Form(None, description="Expected number of speakers (1-10)")
):
    """Start video processing with Cloudflare URL"""
    
    # Clean up inputs
    video_url = video_url.strip()
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
    
    # Validate URL format
    if not video_url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid video URL format")
    
    if speakers_expected is not None and (speakers_expected < 1 or speakers_expected > 10):
        raise HTTPException(status_code=400, detail="speakers_expected must be between 1 and 10")
    
    # Generate unique audio ID
    audio_id = r2_storage.generate_audio_id()
    
    # Use Cloudflare URL directly (no download/upload needed)
    video_source = video_url
    original_source_url = video_url
    
    # Initialize status tracking
    status_manager.initialize_status(audio_id)
    
    # Submit to video processing queue
    from video_processor.video_queue_manager import video_queue_manager
    
    # Prepare enhanced parameters for queue processing
    queue_parameters = {
        "include_instruments": include_instruments,
        "generate_subtitles": generate_subtitles,
        # Enhanced Dia Parameters
        "max_tokens": max_tokens,
        "cfg_scale": cfg_scale,
        "temperature": temperature,
        "top_p": top_p,
        "cfg_filter_top_k": cfg_filter_top_k,
        "speed_factor": speed_factor,
        "use_torch_compile": use_torch_compile,
        "target_language": target_language,
        "language_code": language_code,
        "speakers_expected": speakers_expected,
        "original_source_url": original_source_url
    }
    
    # Submit to queue (this will handle the processing automatically)
    request_id = video_queue_manager.submit_request(
        video_source=video_source,
        audio_id=audio_id,
        is_file_upload=False,
        parameters=queue_parameters
    )
    
    # Get queue information for response
    try:
        queue_stats = video_queue_manager.get_queue_stats()
        queue_request_status = video_queue_manager.get_request_status(request_id)
        
        # Determine initial status message
        queue_position = queue_request_status.get("queue_position") if queue_request_status else None
        if queue_request_status and queue_position is not None and queue_position > 0:
            message = f"Video processing queued successfully (position: {queue_position})"
            estimated_time = queue_request_status.get("estimated_time", "15-30 minutes")
        else:
            message = "Video processing started successfully"
            estimated_time = "10-20 minutes"
    except Exception as e:
        logger.warning(f"Failed to get queue info: {e}")
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
        import random
        
        # Validate inputs
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if not request.reference_audio_url.strip():
            raise HTTPException(status_code=400, detail="Reference audio URL cannot be empty")
        
        if request.duration <= 0:
            raise HTTPException(status_code=400, detail="Duration must be positive")
        
        # Set parameters with defaults
        seed = request.seed or random.randint(1, 1000000)
        temperature = request.temperature or 1.0
        cfg_scale = request.cfg_scale or 3.5
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
            # Get shared audio processor
            from video_processor import is_model_loaded, get_audio_processor
            audio_processor = get_audio_processor(load_model=True)
            voice_service = audio_processor.voice_cloning_service
            
            if not voice_service.is_model_loaded():
                raise HTTPException(status_code=500, detail="Dia model not available")
            
            # For voice cloning, use reference transcript + generation text
            combined_text = request.text + " " + request.text

            print(f"Combined text: {combined_text}")
            print(f"Reference audio URL: {request.reference_audio_url}")
         
            
            
            import torch
            with torch.inference_mode():
                cloned_audio = voice_service.dia_model.generate(
                    text=combined_text,
                    audio_prompt=temp_ref_path,
                    use_torch_compile=settings.DIA_ENHANCED_USE_TORCH_COMPILE,
                    cfg_scale=settings.DIA_ENHANCED_CFG_SCALE,
                    temperature=settings.DIA_ENHANCED_TEMPERATURE,
                    top_p=settings.DIA_ENHANCED_TOP_P,
                    cfg_filter_top_k=settings.DIA_ENHANCED_CFG_FILTER_TOP_K,
                    max_tokens=settings.DIA_ENHANCED_MAX_TOKENS,
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



@app.post("/upload-file")
async def upload_file(video_file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Upload video file to Cloudflare with immediate response and background processing.
    
    This endpoint saves the file first, then processes in background.
    
    Returns:
        - file_id: Unique identifier for tracking upload progress
        - status_check_url: URL to check upload status (/upload-status/{file_id})
        - estimated_time: Expected time for upload completion
    
    Example usage:
        1. POST /upload-file with video file
        2. Get immediate response with file_id  
        3. Periodically GET /upload-status/{file_id} to check progress
        4. When status="completed", use file_url for video processing
    """
    file_id = r2_storage.generate_audio_id()
    original_filename = video_file.filename
    
    try:
        # Generate temp file path
        temp_file_path = os.path.join(settings.TEMP_DIR, f"upload_{file_id}_{original_filename}")
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        
        # Initialize upload status
        upload_status_memory[file_id] = {
            "status": "uploading",
            "progress": 5,
            "message": "Saving uploaded file...",
            "original_filename": original_filename,
            "started_at": datetime.now().isoformat()
        }
        
        # Save file immediately to temp location (this ensures file is accessible)
        total_size = 0
        with open(temp_file_path, "wb") as buffer:
            while chunk := await video_file.read(8192):
                buffer.write(chunk)
                total_size += len(chunk)
        
        # Update progress after file save
        file_size = os.path.getsize(temp_file_path)
        upload_status_memory[file_id].update({
            "progress": 15,
            "message": f"File saved ({file_size // (1024*1024)} MB), starting background processing..."
        })
        
        # Start background processing with saved file (no UploadFile object needed)
        background_tasks.add_task(process_file_background_only, file_id, temp_file_path, original_filename, file_size)
        
        # Return immediate response
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
        # Clean up temp file if it was created
        temp_file_path = os.path.join(settings.TEMP_DIR, f"upload_{file_id}_{original_filename}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        # Update status if exists
        if file_id in upload_status_memory:
            upload_status_memory[file_id].update({
                "status": "failed",
                "progress": 0,
                "message": f"File save failed: {str(e)}"
            })
        raise HTTPException(status_code=500, detail=f"Failed to start upload: {str(e)}")

async def process_file_background_only(file_id: str, temp_file_path: str, filename: str, file_size: int):
    """Background processing for already saved file (no UploadFile dependency)"""
    try:
        # Validate file exists and is readable - 20%
        upload_status_memory[file_id].update({
            "progress": 20, 
            "message": "Validating uploaded file..."
        })
        
        if not os.path.exists(temp_file_path):
            raise Exception("Temporary file not found")
            
        if file_size == 0:
            raise Exception("Uploaded file is empty")
        
        # Validate file format - 30%
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
        
        # Read file content for local storage - 40%
        upload_status_memory[file_id].update({"progress": 40, "message": "Reading file content..."})
        
        with open(temp_file_path, 'rb') as f:
            video_file_content = f.read()
        
        if len(video_file_content) == 0:
            raise Exception("File content is empty after reading")
        
        # Store locally - 60%
        upload_status_memory[file_id].update({"progress": 60, "message": "Storing locally..."})
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
        
        # Upload to Cloudflare - 85%
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
        
        # Complete - 100%
        upload_status_memory[file_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Upload completed successfully",
            "file_url": uploaded_url,
            "local_path": local_result["local_path"],
            "expires_at": local_result["expires_at"]
        })
        
        # Store in memory
        uploaded_files_memory[file_id] = {
            "url": uploaded_url,
            "filename": filename,
            "size": file_size,
            "upload_time": datetime.now().isoformat(),
            "local_path": local_result["local_path"],
            "expires_at": local_result["expires_at"]
        }
        
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
    except Exception as e:
        # Update status to failed
        upload_status_memory[file_id].update({
            "status": "failed",
            "progress": 0,
            "message": f"Upload processing failed: {str(e)}"
        })
        
        # Clean up temp file if exists
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



@app.get("/upload-status/{file_id}", response_model=UploadStatusResponse)
async def get_upload_status(file_id: str):
    """Get upload progress status with detailed information"""
    try:
        if file_id not in upload_status_memory:
            raise HTTPException(status_code=404, detail="Upload ID not found")
        
        data = upload_status_memory[file_id]
        
        # Calculate elapsed time and estimated remaining time
        started_at = data.get("started_at")
        elapsed_time = None
        estimated_remaining = None
        
        if started_at:
            from datetime import datetime
            start_time = datetime.fromisoformat(started_at)
            elapsed_seconds = (datetime.now() - start_time).total_seconds()
            elapsed_time = f"{int(elapsed_seconds // 60)}m {int(elapsed_seconds % 60)}s"
            
            # Estimate remaining time based on progress
            progress = data.get("progress", 0)
            if progress > 0 and progress < 100 and data["status"] == "uploading":
                total_estimated = elapsed_seconds * (100 / progress)
                remaining_seconds = total_estimated - elapsed_seconds
                estimated_remaining = f"{int(remaining_seconds // 60)}m {int(remaining_seconds % 60)}s"
        
        # Enhanced message with timing info
        enhanced_message = data["message"]
        if elapsed_time and data["status"] == "uploading":
            enhanced_message += f" (elapsed: {elapsed_time}"
            if estimated_remaining:
                enhanced_message += f", remaining: ~{estimated_remaining}"
            enhanced_message += ")"
        
        # Clean up completed/failed uploads after 5 minutes
        if data["status"] in ["completed", "failed"]:
            import asyncio
            async def delayed_cleanup():
                await asyncio.sleep(300)  # 5 minutes
                if file_id in upload_status_memory:
                    del upload_status_memory[file_id]
            asyncio.create_task(delayed_cleanup())
        
        return UploadStatusResponse(
            file_id=file_id,
            status=data["status"],
            progress=data["progress"],
            message=enhanced_message,
            original_filename=data.get("original_filename"),
            file_url=data.get("file_url")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/terminate/{audio_id}")
async def terminate_process(audio_id: str, reason: Optional[str] = None):
    """Force terminate a processing request immediately"""
    try:
        if not audio_id:
            raise HTTPException(status_code=400, detail="Audio ID is required")
        
        # Check if process exists
        status = status_manager.get_status(audio_id)
        if not status:
            raise HTTPException(status_code=404, detail="Process not found")
        
        # Check if already completed or failed
        if status.get("status") in ["completed", "failed"]:
            return {
                "success": False, 
                "message": f"Process already {status.get('status')}",
                "current_status": status.get("status")
            }
        
        # Force terminate the process
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

@app.get("/api/queue-health")
async def get_queue_health():
    """Get health status of the processing queue system"""
    try:
        from video_processor.video_queue_manager import video_queue_manager
        
        health_status = video_queue_manager.get_health_status()
        
        # Add timestamp
        health_status["checked_at"] = datetime.now().isoformat()
        
        return health_status
        
    except Exception as e:
        logger.error(f"Failed to get queue health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/api/cleanup-stuck-requests")
async def cleanup_stuck_requests():
    """Force cleanup of stuck processing requests"""
    try:
        from video_processor.video_queue_manager import video_queue_manager
        
        # Get health status first
        health_status = video_queue_manager.get_health_status()
        stuck_count = len(health_status.get("stuck_requests", []))
        
        if stuck_count == 0:
            return {
                "success": True,
                "message": "No stuck requests found",
                "cleaned_count": 0,
                "health_status": health_status
            }
        
        # Force cleanup stuck requests
        cleaned_count = video_queue_manager.force_cleanup_stuck_requests()
        
        # Get updated health status
        updated_health = video_queue_manager.get_health_status()
        
        logger.info(f"Cleaned up {cleaned_count} stuck requests")
        
        return {
            "success": True,
            "message": f"Successfully cleaned up {cleaned_count} stuck requests",
            "cleaned_count": cleaned_count,
            "stuck_before": stuck_count,
            "health_status": updated_health,
            "cleaned_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup stuck requests: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.post("/api/download-video", response_model=VideoDownloadResponse)
async def download_video(request: VideoDownloadRequest):
    """Download video from URL and upload to Cloudflare R2"""
    try:
        logger.info(f"Video download request: {request.url}")
        
        # Clean up old temporary files before processing
        video_download_service.cleanup_old_files()
        
        # Download and upload video
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