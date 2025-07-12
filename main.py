from fastapi import FastAPI, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import shutil
from pathlib import Path
from datetime import datetime
import json
import random
import requests
import urllib.parse
import time
import logging

from config import settings
from r2_storage import R2Storage
from video_processor.base_processor import AudioProcessor
from video_processor.voice_cloning import set_seed

from status_manager import status_manager, ProcessingStatus

from contextlib import asynccontextmanager

# Configure logging
os.makedirs(settings.LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(settings.LOGS_DIR, 'api.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Custom dependency for handling empty seed values
def parse_seed(seed: str = Form(None, description="Optional seed for reproducible results")) -> Optional[int]:
    """Parse seed parameter, converting empty strings to None"""
    if seed is None or seed == "" or seed.strip() == "":
        return None
    try:
        return int(seed)
    except ValueError:
        raise HTTPException(status_code=400, detail="Seed must be a valid integer or empty")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    # Load Dia model
    success = audio_processor.load_dia_model(
        repo_id=settings.DIA_MODEL_REPO
    )
    if not success:
        logger.warning("Failed to load Dia model on startup")
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
class ProcessingResponse(BaseModel):
    success: bool
    audio_id: str
    message: str
    processing_details: Optional[Dict[str, Any]] = None
    r2_storage: Optional[Dict[str, Any]] = None
    final_audio_url: Optional[str] = None
    subtitles_url: Optional[str] = None
    video_url: Optional[str] = None
    original_audio_details: Optional[Dict[str, Any]] = None
    seed_used: Optional[int] = None

class StatusResponse(BaseModel):
    status: str
    message: str
    audio_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint - API status"""
    return StatusResponse(
        status="active",
        message="Voice Cloning API is running",
        details={
            "version": settings.API_VERSION,
            "features": {
                "voice_cloning": True,
                "video_processing": True,
                "subtitle_generation": settings.ENABLE_SUBTITLES,
                "instrument_mixing": settings.ENABLE_INSTRUMENTS,
                "r2_storage": bool(settings.R2_BUCKET_NAME)
            },
            "endpoints": {
                "process_video": "/process-video",
                "status": "/status/{audio_id}",
                "download": "/download/{audio_id}/{file_type}"
            }
        }
    )

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Health check endpoint"""
    # Clean up old statuses periodically
    status_manager.cleanup_old_statuses()
    
    return StatusResponse(
        status="healthy",
        message="API is healthy and ready to process requests",
        details={
            "dia_model_loaded": audio_processor.voice_cloning_service.is_model_loaded(),
            "r2_configured": bool(settings.R2_BUCKET_NAME),
            "temp_dir": str(settings.TEMP_DIR),
            "active_processing_count": len([s for s in status_manager.get_all_statuses().values() if s["status"] == "processing"])
        }
    )


class StartProcessingResponse(BaseModel):
    success: bool
    audio_id: str
    message: str
    status: str
    estimated_time: str
    status_check_url: str

@app.post("/process-video", response_model=StartProcessingResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    video_url: str = Form(..., description="Video URL (HTTP/HTTPS) for processing with automatic separation"),
    seed: Optional[int] = Depends(parse_seed),
    include_instruments: bool = Form(True, description="Whether to include instruments in final audio"),
    generate_subtitles: bool = Form(True, description="Whether to generate subtitles"),
    temperature: float = Form(settings.DIA_TEMPERATURE, description="Voice cloning temperature"),
    cfg_scale: float = Form(settings.DIA_CFG_SCALE, description="CFG scale for voice cloning"),
    top_p: float = Form(settings.DIA_TOP_P, description="Top-p for voice cloning"),
    target_language: str = Form("English", description="Target language for translation"),
    language_code: Optional[str] = Form(None, description="Language code for transcription (e.g., en, es, fr, de, hi, ja, zh) - leave empty/None for auto-detection"),
    speakers_expected: Optional[int] = Form(None, description="Expected number of speakers (1-10)")
):
    """Start video processing with immediate response. Processing happens in background - use status_check_url to monitor progress."""
    
    # Validate URL format
    if not video_url.startswith(('http://', 'https://')):
        raise HTTPException(
            status_code=400,
            detail="Invalid video URL. Must be a valid HTTP/HTTPS URL"
        )
    
    # Validate URL is not empty
    if not video_url.strip():
        raise HTTPException(
            status_code=400,
            detail="Video URL cannot be empty"
        )
    
    # Validate speakers_expected
    if speakers_expected is not None and (speakers_expected < 1 or speakers_expected > 10):
        raise HTTPException(
            status_code=400,
            detail="speakers_expected must be between 1 and 10"
        )
    
    # Generate unique audio ID
    audio_id = r2_storage.generate_audio_id()
    
    # Start background processing
    background_tasks.add_task(
        process_video_background,
        video_url, audio_id, seed, include_instruments, generate_subtitles,
        temperature, cfg_scale, top_p, target_language, language_code, speakers_expected
    )
    
    # Return immediate response
    return StartProcessingResponse(
        success=True,
        audio_id=audio_id,
        message="Video processing started successfully in background. Use status_check_url to monitor progress.",
        status="processing",
        estimated_time="10-20 minutes",
        status_check_url=f"/status/{audio_id}",
    )

def process_video_background(
    video_url: str, audio_id: str, seed: Optional[int], include_instruments: bool,
    generate_subtitles: bool, temperature: float, cfg_scale: float, top_p: float,
    target_language: str, language_code: Optional[str], speakers_expected: Optional[int]
):
    """Background processing function"""
    actual_seed = seed if seed is not None else random.randint(1, 1000000)
    set_seed(actual_seed)
    
    # Convert empty string to None for proper handling
    final_language_code = language_code if language_code and language_code.strip() else None
    
    # Start processing tracking
    status_manager.start_processing(audio_id)
    

    
    video_temp_path = None
    try:
        # Download video
        status_manager.update_status(audio_id, ProcessingStatus.DOWNLOADING, 10)
        video_temp_path = os.path.join(settings.TEMP_DIR, f"{audio_id}_video.mp4")
        
        parsed_url = urllib.parse.urlparse(video_url)
        filename = os.path.basename(parsed_url.path)
        if not filename or '.' not in filename:
            filename = "video.mp4"
        
        try:
            response = requests.get(video_url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(video_temp_path, "wb") as buffer:
                for chunk in response.iter_content(chunk_size=8192):
                    buffer.write(chunk)
            
            status_manager.set_progress(audio_id, 20)
            
        except requests.exceptions.RequestException as e:
            status_manager.fail_processing(audio_id, f"Download failed: {str(e)}")
            return
        
        # Validate downloaded video file
        if not os.path.exists(video_temp_path) or os.path.getsize(video_temp_path) == 0:
            status_manager.fail_processing(audio_id, "Downloaded file is empty or missing")
            return
        
        # Validate video file format
        if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            content_type = response.headers.get('content-type', '')
            if 'video/mp4' in content_type:
                filename = "video.mp4"
            elif 'video/avi' in content_type:
                filename = "video.avi"
            elif 'video/mov' in content_type:
                filename = "video.mov"
            elif 'video/webm' in content_type:
                filename = "video.webm"
            else:
                status_manager.fail_processing(audio_id, "Invalid video format")
                return
        
        # Process video with RunPod separation
        status_manager.update_status(audio_id, ProcessingStatus.PROCESSING, 30)
        
        video_result = audio_processor.process_video_with_separation(
            video_temp_path, 
            audio_id, 
            target_language,
            language_code=final_language_code,
            speakers_expected=speakers_expected
        )
        
        if not video_result["success"]:
            status_manager.fail_processing(audio_id, f"Audio processing failed: {video_result['error']}")
            return
        
        status_manager.set_progress(audio_id, 50)
        
        segments_dir = video_result["segments_dir"]
        vocal_path = video_result["vocal_path"]
        separated_instruments_path = video_result["instruments_path"]
        
        # Get original audio details
        import soundfile as sf
        original_audio, original_sr = sf.read(vocal_path)
        original_duration = len(original_audio) / original_sr
        file_size = os.path.getsize(video_temp_path)
        
        original_audio_details = {
            "filename": filename,
            "source_url": video_url,
            "duration": original_duration,
            "sample_rate": original_sr,
            "channels": len(original_audio.shape) if len(original_audio.shape) > 1 else 1,
            "size_mb": file_size / (1024 * 1024),
            "processing_type": "video_with_separation",
            "language_code": final_language_code,
            "language_detection_used": final_language_code is None,
            "speakers_expected": speakers_expected,
            "detected_speakers": video_result.get("detected_speakers", len(video_result.get("speakers", [])))
        }
        
        # Clone voice segments
        status_manager.set_progress(audio_id, 60)
        cloning_result = audio_processor.clone_voice_segments(
            segments_dir,
            audio_id,
            temperature=temperature,
            cfg_scale=cfg_scale,
            top_p=top_p,
            seed=actual_seed
        )
        
        if not cloning_result["success"]:
            status_manager.fail_processing(audio_id, f"Voice cloning failed: {cloning_result['error']}")
            return
        
        status_manager.set_progress(audio_id, 70)
        
        # Reconstruct final audio
        if include_instruments:
            reconstruction_result = audio_processor.reconstruct_final_audio(
                segments_dir,
                audio_id,
                include_instruments=True,
                instruments_path=separated_instruments_path
            )
        else:
            reconstruction_result = audio_processor.reconstruct_final_audio(
                segments_dir,
                audio_id,
                include_instruments=False,
                instruments_path=None
            )
        
        if not reconstruction_result["success"]:
            status_manager.fail_processing(audio_id, f"Audio reconstruction failed: {reconstruction_result['error']}")
            return
        
        status_manager.set_progress(audio_id, 80)
        final_audio_path = reconstruction_result["final_audio_path"]
        
        # Handle video processing
        video_result = None
        if generate_subtitles:
            instruments_for_video = separated_instruments_path if include_instruments else None
            video_result = audio_processor.create_video_with_subtitles(
                video_temp_path,
                final_audio_path,
                segments_dir,
                audio_id,
                instruments_for_video
            )
        else:
            video_result = audio_processor.create_video_with_audio(
                video_temp_path,
                final_audio_path,
                audio_id
            )
        
        if not video_result or not video_result["success"]:
            error_msg = video_result.get('error', 'Unknown error') if video_result else 'No result'
            status_manager.fail_processing(audio_id, f"Video processing failed: {error_msg}")
            return
        
        # Upload files
        status_manager.update_status(audio_id, ProcessingStatus.UPLOADING, 90)
        
        try:
            r2_segments_result = r2_storage.upload_audio_segments(audio_id, segments_dir)
            r2_final_result = r2_storage.upload_final_audio(audio_id, final_audio_path)
            
            r2_video_result = None
            if video_result and video_result["success"]:
                video_filename = f"video_processed_{audio_id}.mp4"
                r2_key = r2_storage.generate_file_path(audio_id, "video", video_filename)
                r2_video_result = r2_storage.upload_file(
                    video_result["video_path"], 
                    r2_key, 
                    "video/mp4"
                )
            
            r2_subtitles_result = None
            if video_result and video_result.get("subtitle_path"):
                subtitle_filename = f"subtitles_{audio_id}.srt"
                r2_key = r2_storage.generate_file_path(audio_id, "subtitles", subtitle_filename)
                r2_subtitles_result = r2_storage.upload_file(
                    video_result["subtitle_path"], 
                    r2_key, 
                    "text/srt"
                )
            
        except Exception as e:
            status_manager.fail_processing(audio_id, f"File upload failed: {str(e)}")
            return
        
        # Complete processing
        completion_details = {
            "final_audio_url": r2_final_result.get("url") if r2_final_result and r2_final_result["success"] else None,
            "video_url": r2_video_result.get("url") if r2_video_result and r2_video_result["success"] else None,
            "subtitles_url": r2_subtitles_result.get("url") if r2_subtitles_result and r2_subtitles_result["success"] else None,
            "segments_processed": cloning_result.get("cloned_segments", 0),
            "speakers_detected": len(video_result.get("speakers", [])),
            "total_duration": original_duration
        }
        
        status_manager.complete_processing(audio_id, completion_details)
        
        # Create processing summary for R2
        processing_data = {
            "audio_id": audio_id,
            "original_audio": original_audio_details,
            "seed_used": actual_seed,
            "parameters": {
                "temperature": temperature,
                "cfg_scale": cfg_scale,
                "top_p": top_p,
                "target_language": target_language,
                "include_instruments": include_instruments,
                "generate_subtitles": generate_subtitles,
                "video_provided": True,
                "video_source": "url",
                "video_url": video_url,
                "runpod_separation": True
            },
            "processing_stats": audio_processor.get_processing_stats(segments_dir),
            "cloning_results": cloning_result,
            "reconstruction_results": reconstruction_result,
            "video_generated": video_result is not None and video_result["success"],
            "subtitles_generated": generate_subtitles and video_result is not None and video_result.get("subtitle_path"),
            "separation_used": True
        }
        
        r2_storage.create_processing_summary(audio_id, processing_data)
        

        
    except Exception as e:
        # Handle any unexpected errors
        error_msg = f"Unexpected error: {str(e)}"
        status_manager.fail_processing(audio_id, error_msg)
        
    finally:
        # Always cleanup temp files
        try:
            cleanup_temp_files(audio_id, None, None, video_temp_path)
        except Exception as cleanup_error:
            pass

@app.get("/download/{audio_id}/{file_type}")
async def download_file(audio_id: str, file_type: str):
    """Download processed files (for development/testing)"""
    try:
        if file_type == "final":
            file_path = os.path.join(settings.TEMP_DIR, f"final_output_{audio_id}.wav")
            media_type = "audio/wav"
            extension = "wav"
        elif file_type == "subtitles":
            file_path = os.path.join(settings.TEMP_DIR, f"subtitles_{audio_id}.srt")
            media_type = "text/plain"
            extension = "srt"
        elif file_type == "video":
            file_path = os.path.join(settings.TEMP_DIR, f"video_with_subtitles_{audio_id}.mp4")
            media_type = "video/mp4"
            extension = "mp4"
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            file_path,
            media_type=media_type,
            filename=f"{audio_id}_{file_type}.{extension}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{audio_id}")
async def get_status(audio_id: str):
    """Get processing status for a specific audio ID"""
    try:
        status = status_manager.get_status(audio_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_temp_files(audio_id: str, audio_temp_path: Optional[str] = None, instruments_temp_path: Optional[str] = None, video_temp_path: Optional[str] = None):
    """Clean up temporary files comprehensively"""
    try:
        # Clean up processor temp files (this will call all sub-modules)
        audio_processor.cleanup_temp_files(audio_id)
        
        # Clean up uploaded files
        if audio_temp_path and os.path.exists(audio_temp_path):
            os.remove(audio_temp_path)
        
        if instruments_temp_path and os.path.exists(instruments_temp_path):
            os.remove(instruments_temp_path)
        
        if video_temp_path and os.path.exists(video_temp_path):
            os.remove(video_temp_path)
        
        # Clean up any remaining temp files in temp directory
        temp_dir = Path(settings.TEMP_DIR)
        if temp_dir.exists():
            # Remove all files containing the audio_id
            for temp_file in temp_dir.glob(f"*{audio_id}*"):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
                except Exception:
                    pass
        
        # Clean up R2 temp files (if any)
        try:
            r2_storage.cleanup_temp_files(audio_id)
        except Exception:
            pass
            
    except Exception:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=False
    ) 