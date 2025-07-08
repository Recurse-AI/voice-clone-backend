from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json
import random

from config import settings
from r2_storage import R2Storage
from audio_processor.base_processor import AudioProcessor
from audio_processor.voice_cloning import set_seed

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

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
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

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    # Load Dia model
    success = audio_processor.load_dia_model(
        repo_id=settings.DIA_MODEL_REPO
    )
    if not success:
        print("Warning: Failed to load Dia model on startup")
    
    # Create temp directory
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    
    print(f"API started successfully on {settings.HOST}:{settings.PORT}")

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
            }
        }
    )

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Health check endpoint"""
    return StatusResponse(
        status="healthy",
        message="API is healthy and ready to process requests",
        details={
            "dia_model_loaded": audio_processor.voice_cloning_service.is_model_loaded(),
            "r2_configured": bool(settings.R2_BUCKET_NAME),
            "temp_dir": str(settings.TEMP_DIR)
        }
    )


@app.post("/process-video", response_model=ProcessingResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(..., description="Video file for processing with automatic separation"),
    seed: Optional[int] = Form(None, description="Optional seed for reproducible results"),
    include_instruments: bool = Form(True, description="Whether to include instruments in final audio"),
    generate_subtitles: bool = Form(True, description="Whether to generate subtitles"),
    temperature: float = Form(settings.DEFAULT_TEMPERATURE, description="Voice cloning temperature"),
    cfg_scale: float = Form(settings.DEFAULT_CFG_SCALE, description="CFG scale for voice cloning"),
    top_p: float = Form(settings.DEFAULT_TOP_P, description="Top-p for voice cloning"),
    target_language: str = Form("English", description="Target language for translation"),
    language_code: str = Form("en", description="Language code for transcription (e.g., en, es, fr, de, hi, ja, zh)"),
    speakers_expected: Optional[int] = Form(None, description="Expected number of speakers (1-10)")
):
    """
    Video processing endpoint with automatic vocal/instrument separation
    
    This endpoint:
    1. Accepts video file only
    2. Automatically separates vocal and instruments using RunPod
    3. Processes vocal audio with voice cloning using Assembly AI Universal model
    4. Creates video with subtitles using separated audio
    5. Stores all results in R2 bucket
    6. Returns comprehensive response with URLs
    """
    
    # Validate file format
    if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400,
            detail="Invalid video format. Supported formats: mp4, avi, mov, mkv"
        )
    
    # Validate speakers_expected
    if speakers_expected is not None and (speakers_expected < 1 or speakers_expected > 10):
        raise HTTPException(
            status_code=400,
            detail="speakers_expected must be between 1 and 10"
        )
    
    # Check file size
    if video_file.size > settings.MAX_FILE_SIZE * 3:  # Allow larger video files
        raise HTTPException(
            status_code=400,
            detail=f"Video file too large. Maximum size: {settings.MAX_FILE_SIZE * 3 / (1024*1024):.1f}MB"
        )
    
    # Generate unique audio ID
    audio_id = r2_storage.generate_audio_id()
    
    # Set seed for reproducible results
    actual_seed = seed if seed is not None else random.randint(1, 1000000)
    set_seed(actual_seed)
    
    try:
        # Save video file
        video_temp_path = os.path.join(settings.TEMP_DIR, f"{audio_id}_video.mp4")
        with open(video_temp_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        # Process video with RunPod separation
        video_result = audio_processor.process_video_with_separation(
            video_temp_path, 
            audio_id, 
            target_language,
            language_code=language_code,
            speakers_expected=speakers_expected
        )
        
        if not video_result["success"]:
            raise HTTPException(status_code=500, detail=video_result["error"])
        
        segments_dir = video_result["segments_dir"]
        vocal_path = video_result["vocal_path"]
        separated_instruments_path = video_result["instruments_path"]
        
        # Get original audio details from extracted vocal
        import soundfile as sf
        original_audio, original_sr = sf.read(vocal_path)
        original_duration = len(original_audio) / original_sr
        
        original_audio_details = {
            "filename": video_file.filename,
            "duration": original_duration,
            "sample_rate": original_sr,
            "channels": len(original_audio.shape) if len(original_audio.shape) > 1 else 1,
            "size_mb": video_file.size / (1024 * 1024),
            "processing_type": "video_with_separation",
            "language_code": language_code,
            "speakers_expected": speakers_expected,
            "detected_speakers": video_result.get("detected_speakers", len(video_result.get("speakers", [])))
        }
        
        # Step 2: Clone voice segments
        cloning_result = audio_processor.clone_voice_segments(
            segments_dir,
            audio_id,
            temperature=temperature,
            cfg_scale=cfg_scale,
            top_p=top_p,
            seed=actual_seed
        )
        
        if not cloning_result["success"]:
            raise HTTPException(status_code=500, detail=cloning_result["error"])
        
        # Step 3: Reconstruct final audio
        if include_instruments:
            # Use separated instruments from RunPod
            reconstruction_result = audio_processor.reconstruct_final_audio(
                segments_dir,
                audio_id,
                include_instruments=True,
                instruments_path=separated_instruments_path
            )
        else:
            # Vocal only
            reconstruction_result = audio_processor.reconstruct_final_audio(
                segments_dir,
                audio_id,
                include_instruments=False,
                instruments_path=None
            )
        
        if not reconstruction_result["success"]:
            raise HTTPException(status_code=500, detail=reconstruction_result["error"])
        
        final_audio_path = reconstruction_result["final_audio_path"]
        
        # Step 4: Handle video processing
        video_result = None
        
        if generate_subtitles:
            # Create video with subtitles using original video
            instruments_for_video = separated_instruments_path if include_instruments else None
            video_result = audio_processor.create_video_with_subtitles(
                video_temp_path,
                final_audio_path,
                segments_dir,
                audio_id,
                instruments_for_video
            )
        else:
            # Create video with new audio only (no subtitles)
            video_result = audio_processor.create_video_with_audio(
                video_temp_path,
                final_audio_path,
                audio_id
            )
        
        # Step 5: Upload to R2 bucket
        # Upload segments and metadata
        r2_segments_result = r2_storage.upload_audio_segments(audio_id, segments_dir)
        
        # Upload final audio
        r2_final_result = r2_storage.upload_final_audio(audio_id, final_audio_path)
        
        # Upload video
        r2_video_result = None
        if video_result and video_result["success"]:
            video_filename = f"video_processed_{audio_id}.mp4"
            r2_key = r2_storage.generate_file_path(audio_id, "video", video_filename)
            r2_video_result = r2_storage.upload_file(
                video_result["video_path"], 
                r2_key, 
                "video/mp4"
            )
        
        # Upload subtitles if generated
        r2_subtitles_result = None
        if video_result and video_result.get("subtitle_path"):
            subtitle_filename = f"subtitles_{audio_id}.srt"
            r2_key = r2_storage.generate_file_path(audio_id, "subtitles", subtitle_filename)
            r2_subtitles_result = r2_storage.upload_file(
                video_result["subtitle_path"], 
                r2_key, 
                "text/srt"
            )
        
        # Step 6: Create processing summary
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
                "runpod_separation": True
            },
            "processing_stats": audio_processor.get_processing_stats(segments_dir),
            "cloning_results": cloning_result,
            "reconstruction_results": reconstruction_result,
            "video_generated": video_result is not None and video_result["success"],
            "subtitles_generated": generate_subtitles and video_result is not None and video_result.get("subtitle_path"),
            "separation_used": True
        }
        
        summary_result = r2_storage.create_processing_summary(audio_id, processing_data)
        
        # Step 7: Get storage info
        storage_info = r2_storage.get_storage_info(audio_id)
        
        # Step 8: Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, audio_id, None, None, video_temp_path)
        
        # Prepare response
        processing_type = "Video processing with RunPod separation"
        message = f"Video processed successfully with vocal/instrument separation"
        
        response = ProcessingResponse(
            success=True,
            audio_id=audio_id,
            message=message,
            processing_details={
                "segments_processed": len(cloning_result.get("cloned_segments", {})),
                "total_duration": reconstruction_result.get("duration", 0),
                "speakers_detected": list(cloning_result.get("cloned_segments", {}).keys()),
                "processing_stats": audio_processor.get_processing_stats(segments_dir),
                "video_created": video_result is not None and video_result["success"],
                "subtitle_count": video_result.get("subtitle_count", 0) if video_result else 0,
                "processing_type": "video_processing_with_runpod_separation",
                "runpod_separation": True,
                "separated_files": {
                    "vocal_audio": vocal_path,
                    "instruments_audio": separated_instruments_path
                }
            },
            r2_storage={
                "bucket": storage_info["bucket"],
                "base_path": storage_info["base_path"],
                "segments_upload": r2_segments_result,
                "final_audio_upload": r2_final_result,
                "video_upload": r2_video_result,
                "subtitles_upload": r2_subtitles_result,
                "summary_upload": summary_result,
                "access_url": storage_info["access_url"]
            },
            final_audio_url=r2_final_result.get("url") if r2_final_result and r2_final_result["success"] else None,
            video_url=r2_video_result.get("url") if r2_video_result and r2_video_result["success"] else None,
            subtitles_url=r2_subtitles_result.get("url") if r2_subtitles_result and r2_subtitles_result["success"] else None,
            original_audio_details=original_audio_details,
            seed_used=actual_seed
        )
        
        return response
        
    except Exception as e:
        # Clean up on error
        cleanup_temp_files(audio_id, None, None, video_temp_path)
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/status/{audio_id}", response_model=StatusResponse)
async def get_processing_status(audio_id: str):
    """Get processing status for a specific audio ID"""
    try:
        # Check if processing is complete by looking for final output
        final_file = os.path.join(settings.TEMP_DIR, f"final_output_{audio_id}.wav")
        
        if os.path.exists(final_file):
            return StatusResponse(
                status="completed",
                message="Processing completed successfully",
                audio_id=audio_id,
                details={
                    "final_audio_available": True,
                    "r2_storage_info": r2_storage.get_storage_info(audio_id)
                }
            )
        else:
            return StatusResponse(
                status="processing",
                message="Processing in progress",
                audio_id=audio_id
            )
            
    except Exception as e:
        return StatusResponse(
            status="error",
            message=str(e),
            audio_id=audio_id
        )

def cleanup_temp_files(audio_id: str, audio_temp_path: Optional[str] = None, instruments_temp_path: Optional[str] = None, video_temp_path: Optional[str] = None):
    """Clean up temporary files"""
    try:
        # Clean up processor temp files
        audio_processor.cleanup_temp_files(audio_id)
        
        # Clean up uploaded files
        if audio_temp_path and os.path.exists(audio_temp_path):
            os.remove(audio_temp_path)
        
        if instruments_temp_path and os.path.exists(instruments_temp_path):
            os.remove(instruments_temp_path)
        
        if video_temp_path and os.path.exists(video_temp_path):
            os.remove(video_temp_path)
            
    except Exception as e:
        print(f"Error cleaning up temp files: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=False
    ) 