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

# Response models
class ProcessingResponse(BaseModel):
    success: bool
    audio_id: str
    message: str
    processing_details: Optional[Dict[str, Any]] = None
    r2_storage: Optional[Dict[str, Any]] = None
    final_audio_url: Optional[str] = None
    subtitles_url: Optional[str] = None
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

@app.post("/process-audio", response_model=ProcessingResponse)
async def process_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Main audio file to process"),
    instruments_file: Optional[UploadFile] = File(None, description="Optional instruments file to mix"),
    seed: Optional[int] = Form(None, description="Optional seed for reproducible results"),
    include_instruments: bool = Form(False, description="Whether to include instruments in final audio"),
    generate_subtitles: bool = Form(True, description="Whether to generate subtitles"),
    temperature: float = Form(settings.DEFAULT_TEMPERATURE, description="Voice cloning temperature"),
    cfg_scale: float = Form(settings.DEFAULT_CFG_SCALE, description="CFG scale for voice cloning"),
    top_p: float = Form(settings.DEFAULT_TOP_P, description="Top-p for voice cloning"),
    target_language: str = Form("English", description="Target language for translation")
):
    """
    Main endpoint for processing audio with voice cloning
    
    This endpoint:
    1. Accepts audio file and optional parameters
    2. Processes audio into segments
    3. Performs voice cloning using Dia model
    4. Reconstructs final audio with optional instruments
    5. Generates subtitles if requested
    6. Stores all results in R2 bucket
    7. Returns comprehensive response with URLs and metadata
    """
    
    # Validate file format
    if not audio_file.filename.lower().endswith(tuple(settings.ALLOWED_AUDIO_FORMATS)):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format. Supported formats: {settings.ALLOWED_AUDIO_FORMATS}"
        )
    
    # Check file size
    if audio_file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    # Generate unique audio ID
    audio_id = r2_storage.generate_audio_id()
    
    # Set seed for reproducible results
    actual_seed = seed if seed is not None else random.randint(1, 1000000)
    random.seed(actual_seed)
    
    try:
        # Save uploaded files temporarily
        audio_temp_path = None
        instruments_temp_path = None
        
        # Save main audio file
        audio_temp_path = os.path.join(settings.TEMP_DIR, f"{audio_id}_input.wav")
        with open(audio_temp_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Save instruments file if provided
        if instruments_file and include_instruments:
            instruments_temp_path = os.path.join(settings.TEMP_DIR, f"{audio_id}_instruments.wav")
            with open(instruments_temp_path, "wb") as buffer:
                shutil.copyfileobj(instruments_file.file, buffer)
        
        # Get original audio details
        import soundfile as sf
        original_audio, original_sr = sf.read(audio_temp_path)
        original_duration = len(original_audio) / original_sr
        
        original_audio_details = {
            "filename": audio_file.filename,
            "duration": original_duration,
            "sample_rate": original_sr,
            "channels": len(original_audio.shape) if len(original_audio.shape) > 1 else 1,
            "size_mb": audio_file.size / (1024 * 1024)
        }
        
        # Step 1: Process audio into segments
        segment_result = audio_processor.process_audio_segments(
            audio_temp_path, 
            audio_id, 
            target_language
        )
        
        if not segment_result["success"]:
            raise HTTPException(status_code=500, detail=segment_result["error"])
        
        segments_dir = segment_result["segments_dir"]
        
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
        reconstruction_result = audio_processor.reconstruct_final_audio(
            segments_dir,
            audio_id,
            include_instruments=include_instruments,
            instruments_path=instruments_temp_path
        )
        
        if not reconstruction_result["success"]:
            raise HTTPException(status_code=500, detail=reconstruction_result["error"])
        
        final_audio_path = reconstruction_result["final_audio_path"]
        
        # Step 4: Generate subtitles if requested
        subtitles_result = None
        if generate_subtitles:
            subtitles_result = audio_processor.generate_subtitles(segments_dir, audio_id)
        
        # Step 5: Upload to R2 bucket
        # Upload segments and metadata
        r2_segments_result = r2_storage.upload_audio_segments(audio_id, segments_dir)
        
        # Upload final audio
        r2_final_result = r2_storage.upload_final_audio(audio_id, final_audio_path)
        
        # Upload subtitles if generated
        r2_subtitles_result = None
        if subtitles_result and subtitles_result["success"]:
            subtitle_filename = f"subtitles_{audio_id}.srt"
            r2_key = r2_storage.generate_file_path(audio_id, "subtitles", subtitle_filename)
            r2_subtitles_result = r2_storage.upload_file(
                subtitles_result["subtitle_path"], 
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
                "generate_subtitles": generate_subtitles
            },
            "processing_stats": audio_processor.get_processing_stats(segments_dir),
            "cloning_results": cloning_result,
            "reconstruction_results": reconstruction_result,
            "subtitles_generated": subtitles_result is not None and subtitles_result["success"]
        }
        
        summary_result = r2_storage.create_processing_summary(audio_id, processing_data)
        
        # Step 7: Get storage info
        storage_info = r2_storage.get_storage_info(audio_id)
        
        # Step 8: Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, audio_id, audio_temp_path, instruments_temp_path)
        
        # Prepare response
        response = ProcessingResponse(
            success=True,
            audio_id=audio_id,
            message="Audio processing completed successfully",
            processing_details={
                "segments_processed": len(cloning_result.get("cloned_segments", {})),
                "total_duration": reconstruction_result.get("duration", 0),
                "speakers_detected": list(cloning_result.get("cloned_segments", {}).keys()),
                "processing_stats": audio_processor.get_processing_stats(segments_dir)
            },
            r2_storage={
                "bucket": storage_info["bucket"],
                "base_path": storage_info["base_path"],
                "segments_upload": r2_segments_result,
                "final_audio_upload": r2_final_result,
                "subtitles_upload": r2_subtitles_result,
                "summary_upload": summary_result,
                "access_url": storage_info["access_url"]
            },
            final_audio_url=r2_final_result.get("url") if r2_final_result and r2_final_result["success"] else None,
            subtitles_url=r2_subtitles_result.get("url") if r2_subtitles_result and r2_subtitles_result["success"] else None,
            original_audio_details=original_audio_details,
            seed_used=actual_seed
        )
        
        return response
        
    except Exception as e:
        # Clean up on error
        cleanup_temp_files(audio_id, audio_temp_path, instruments_temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{audio_id}/{file_type}")
async def download_file(audio_id: str, file_type: str):
    """Download processed files (for development/testing)"""
    try:
        if file_type == "final":
            file_path = os.path.join(settings.TEMP_DIR, f"final_output_{audio_id}.wav")
        elif file_type == "subtitles":
            file_path = os.path.join(settings.TEMP_DIR, f"subtitles_{audio_id}.srt")
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            file_path,
            media_type="audio/wav" if file_type == "final" else "text/plain",
            filename=f"{audio_id}_{file_type}.{'wav' if file_type == 'final' else 'srt'}"
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

def cleanup_temp_files(audio_id: str, audio_temp_path: Optional[str] = None, instruments_temp_path: Optional[str] = None):
    """Clean up temporary files"""
    try:
        # Clean up processor temp files
        audio_processor.cleanup_temp_files(audio_id)
        
        # Clean up uploaded files
        if audio_temp_path and os.path.exists(audio_temp_path):
            os.remove(audio_temp_path)
        
        if instruments_temp_path and os.path.exists(instruments_temp_path):
            os.remove(instruments_temp_path)
            
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