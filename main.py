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

from config import settings
from r2_storage import R2Storage
from video_processor.base_processor import AudioProcessor
from video_processor.voice_cloning import set_seed

from status_manager import status_manager, ProcessingStatus

from contextlib import asynccontextmanager

from schemas import StatusResponse, StartProcessingResponse, RegenerateSegmentRequest, RegenerateSegmentResponse, ReconstructVideoRequest, ReconstructVideoResponse

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
                "reconstruct": "/reconstruct-video"
            }
        }
    )


@app.post("/process-video", response_model=StartProcessingResponse)
async def process_video(
    background_tasks: BackgroundTasks,
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
    
    # Handle file upload if provided
    video_source = None
    if has_file:
        try:
            # Save uploaded file
            upload_temp_path = os.path.join(settings.TEMP_DIR, f"{audio_id}_uploaded_video{Path(video_file.filename).suffix}")
            
            with open(upload_temp_path, "wb") as buffer:
                content = await video_file.read()
                buffer.write(content)
            
            video_source = upload_temp_path
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    else:
        video_source = video_url
    
    # Initialize status tracking
    status_manager.initialize_status(audio_id)
    
    # Start background processing
    background_tasks.add_task(
        process_video_background,
        video_source, audio_id, include_instruments, generate_subtitles,
        temperature, cfg_scale, top_p, target_language, 
        language_code, speakers_expected, has_file
    )
    
    # Return immediate response
    return StartProcessingResponse(
        success=True,
        audio_id=audio_id,
        message="Video processing started successfully",
        status="processing",
        estimated_time="10-20 minutes",
        status_check_url=f"/status/{audio_id}",
    )

def process_video_background(
    video_source: str, audio_id: str, include_instruments: bool,
    generate_subtitles: bool, temperature: float, cfg_scale: float, top_p: float,
    target_language: str, language_code: Optional[str], speakers_expected: Optional[int], is_file_upload: bool
):
    """Background video processing - handles both URL and file inputs"""
    final_language_code = language_code if language_code and language_code.strip() else None
    video_temp_path = None
    
    try:
        # Handle video source
        status_manager.update_status(audio_id, ProcessingStatus.DOWNLOADING, 10)
        
        if is_file_upload:
            # File upload: video_source is already a local file path
            video_temp_path = video_source
            filename = Path(video_source).name
        else:
            # URL download: download the video
            video_temp_path = os.path.join(settings.TEMP_DIR, f"{audio_id}_video.mp4")
            
            parsed_url = urllib.parse.urlparse(video_source)
            filename = os.path.basename(parsed_url.path)
            if not filename or '.' not in filename:
                filename = "video.mp4"
            
            try:
                response = requests.get(video_source, stream=True, timeout=300)
                response.raise_for_status()
                
                with open(video_temp_path, "wb") as buffer:
                    for chunk in response.iter_content(chunk_size=8192):
                        buffer.write(chunk)
                
            except requests.exceptions.RequestException as e:
                status_manager.fail_processing(audio_id, f"Download failed: {str(e)}")
                return
        
        status_manager.set_progress(audio_id, 20)
        
        # Process video
        status_manager.update_status(audio_id, ProcessingStatus.PROCESSING, 30)
        
        processing_result = audio_processor.process_video_with_separation(
            video_temp_path,
            audio_id,
            target_language,
            language_code=final_language_code,
            speakers_expected=speakers_expected
        )
        
        if not processing_result["success"]:
            status_manager.fail_processing(audio_id, f"Processing failed: {processing_result.get('error', 'Unknown error')}")
            return
        
        status_manager.set_progress(audio_id, 50)
        
        # Get original audio details
        vocal_path = processing_result["vocal_path"]
        separated_instruments_path = processing_result["instruments_path"]
        
        original_audio, original_sr = sf.read(vocal_path)
        original_duration = len(original_audio) / original_sr
        file_size = os.path.getsize(video_temp_path)
        
        original_audio_details = {
            "filename": filename,
            "source_url": video_source if not is_file_upload else None,
            "source_type": "url" if not is_file_upload else "file_upload",
            "duration": original_duration,
            "sample_rate": original_sr,
            "channels": len(original_audio.shape) if len(original_audio.shape) > 1 else 1,
            "size_mb": file_size / (1024 * 1024),
            "processing_type": "video_with_separation",
            "language_code": final_language_code,
            "speakers_expected": speakers_expected,
            "detected_speakers": processing_result.get("detected_speakers", len(processing_result.get("speakers", [])))
        }
        
        # Clone voices at natural speed
        status_manager.set_progress(audio_id, 60)
        
        cloning_result = audio_processor.clone_voice_segments(
            processing_result["segments_dir"],
            audio_id,
            temperature=temperature,
            cfg_scale=cfg_scale,
            top_p=top_p,
            seed=settings.DEFAULT_SEED
        )
        
        if not cloning_result["success"]:
            status_manager.fail_processing(audio_id, f"Voice cloning failed: {cloning_result.get('error', 'Unknown error')}")
            return
        
        audio_processor.voice_cloning_service.clear_cache()
        status_manager.set_progress(audio_id, 70)
        
        # Reconstruct audio
        if include_instruments:
            reconstruction_result = audio_processor.reconstruct_final_audio(
                processing_result["segments_dir"],
                audio_id,
                include_instruments=True,
                instruments_path=separated_instruments_path
            )
        else:
            reconstruction_result = audio_processor.reconstruct_final_audio(
                processing_result["segments_dir"],
                audio_id,
                include_instruments=False,
                instruments_path=None
            )
        
        if not reconstruction_result["success"]:
            status_manager.fail_processing(audio_id, f"Audio reconstruction failed: {reconstruction_result['error']}")
            return
        
        status_manager.set_progress(audio_id, 80)
        final_audio_path = reconstruction_result["final_audio_path"]
        
        # Create video
        video_result = None
        if generate_subtitles:
            instruments_for_video = separated_instruments_path if include_instruments else None
            video_result = audio_processor.create_video_with_subtitles(
                video_temp_path,
                final_audio_path,
                processing_result["segments_dir"],
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
            r2_segments_result = r2_storage.upload_audio_segments(audio_id, processing_result["segments_dir"])
            r2_final_result = r2_storage.upload_final_audio(audio_id, final_audio_path)
            
            # Upload vocal and instruments files
            vocal_filename = f"vocal_separated_{audio_id}.wav"
            vocal_r2_key = r2_storage.generate_file_path(audio_id, "vocal", vocal_filename)
            r2_vocal_result = r2_storage.upload_file(vocal_path, vocal_r2_key, "audio/wav")
            
            instruments_filename = f"instruments_separated_{audio_id}.wav"
            instruments_r2_key = r2_storage.generate_file_path(audio_id, "instruments", instruments_filename)
            r2_instruments_result = r2_storage.upload_file(separated_instruments_path, instruments_r2_key, "audio/wav")
            
            r2_video_result = None
            r2_subtitle_result = None
            
            if video_result and video_result["success"]:
                # Upload video
                video_filename = f"video_processed_{audio_id}.mp4"
                r2_key = r2_storage.generate_file_path(audio_id, "video", video_filename)
                r2_video_result = r2_storage.upload_file(
                    video_result["video_path"], 
                    r2_key, 
                    "video/mp4"
                )
                
                # Upload subtitle file if it exists
                if "subtitle_path" in video_result and os.path.exists(video_result["subtitle_path"]):
                    subtitle_filename = f"subtitles_{audio_id}.srt"
                    r2_subtitle_key = r2_storage.generate_file_path(audio_id, "subtitles", subtitle_filename)
                    r2_subtitle_result = r2_storage.upload_file(
                        video_result["subtitle_path"],
                        r2_subtitle_key,
                        "text/srt"
                    )
        
        except Exception as e:
            status_manager.fail_processing(audio_id, f"File upload failed: {str(e)}")
            return
        
        # Mark as completed
        status_manager.complete_processing(audio_id, {
            "final_audio_url": r2_final_result.get("url"),
            "video_url": r2_video_result.get("url") if r2_video_result else None,
            "subtitles_url": r2_subtitle_result.get("url") if r2_subtitle_result else None,
            "processing_stats": {
                "total_segments": processing_result.get("total_segments", 0),
                "cloned_segments": cloning_result.get("cloned_segments_count", 0),
                "speakers": processing_result.get("speakers", []),
                "duration": processing_result.get("total_duration", 0)
            },
            "uploaded_files": {
                "segments": {
                    "count": r2_segments_result.get("files_uploaded", 0) if r2_segments_result else 0,
                    "uploaded": r2_segments_result.get("success", False) if r2_segments_result else False
                },
                "final_audio": {
                    "url": r2_final_result.get("url"),
                    "size_mb": round(r2_final_result.get("size", 0) / (1024 * 1024), 2),
                    "uploaded": r2_final_result.get("success", False)
                },
                "vocal_audio": {
                    "url": r2_vocal_result.get("url"),
                    "size_mb": round(r2_vocal_result.get("size", 0) / (1024 * 1024), 2),
                    "uploaded": r2_vocal_result.get("success", False)
                },
                "instruments_audio": {
                    "url": r2_instruments_result.get("url"),
                    "size_mb": round(r2_instruments_result.get("size", 0) / (1024 * 1024), 2),
                    "uploaded": r2_instruments_result.get("success", False)
                },
                "video": {
                    "url": r2_video_result.get("url"),
                    "size_mb": round(r2_video_result.get("size", 0) / (1024 * 1024), 2),
                    "uploaded": r2_video_result.get("success", False)
                } if r2_video_result else None,
                "subtitles": {
                    "url": r2_subtitle_result.get("url"),
                    "size_kb": round(r2_subtitle_result.get("size", 0) / 1024, 2),
                    "uploaded": r2_subtitle_result.get("success", False)
                } if r2_subtitle_result else None
            },
            "processing_details": {
                "original_audio": original_audio_details,
                "cloning_parameters": {
                    "temperature": temperature,
                    "cfg_scale": cfg_scale,
                    "top_p": top_p,
                    "seed_used": cloning_result.get("seed_used", settings.DEFAULT_SEED),
                    "seeds_used": cloning_result.get("seeds_used", {})
                },
                "features_used": {
                    "vocal_separation": True,
                    "voice_cloning": True,
                    "subtitle_generation": generate_subtitles,
                    "instrument_mixing": include_instruments
                },
                "processing_timeline": {
                    "transcription_source": "AssemblyAI",
                    "voice_cloning_model": "Dia",
                    "separation_service": "RunPod"
                }
            },
            "segment_details": audio_processor.get_processing_stats(processing_result["segments_dir"]),
            "r2_storage": {
                "bucket": r2_storage.bucket_name,
                "base_path": r2_storage.get_storage_info(audio_id)["base_path"],
                "segments_uploaded": r2_segments_result.get("success", False),
                "total_files": (r2_segments_result.get("files_uploaded", 0) if r2_segments_result and r2_segments_result.get("success") else 0) + 
                              (1 if r2_final_result.get("success") else 0) + 
                              (1 if r2_vocal_result.get("success") else 0) + 
                              (1 if r2_instruments_result.get("success") else 0) + 
                              (1 if r2_video_result and r2_video_result.get("success") else 0) + 
                              (1 if r2_subtitle_result and r2_subtitle_result.get("success") else 0),
                "segment_urls": r2_storage.get_segment_urls(audio_id, r2_segments_result) if r2_segments_result else {}
            }
        })
        
        # Create processing summary
        processing_data = {
            "audio_id": audio_id,
            "original_audio": original_audio_details,
            "seeds_used": cloning_result.get("seeds_used", {}),
            "parameters": {
                "temperature": temperature,
                "cfg_scale": cfg_scale,
                "top_p": top_p,
                "target_language": target_language,
                "include_instruments": include_instruments,
                "generate_subtitles": generate_subtitles,
                "video_provided": True,
                "video_source": "url" if not is_file_upload else "file_upload",
                "video_url": video_source if not is_file_upload else None,
                "runpod_separation": True
            },
            "processing_stats": audio_processor.get_processing_stats(processing_result["segments_dir"]),
            "cloning_results": cloning_result,
            "reconstruction_results": reconstruction_result,
            "video_generated": video_result is not None and video_result["success"],
            "subtitles_generated": generate_subtitles and video_result is not None and video_result.get("subtitle_count", 0) > 0,
            "separation_used": True
        }
        
        r2_storage.create_processing_summary(audio_id, processing_data)

        
    except Exception as e:
        status_manager.fail_processing(audio_id, f"Unexpected error: {str(e)}")
        
    finally:
        # Clean up temp files
        try:
            cleanup_temp_files(audio_id, None, None, video_temp_path)
        except Exception:
            pass

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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in regenerate_segment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/reconstruct-video", response_model=ReconstructVideoResponse)
async def reconstruct_video(request: ReconstructVideoRequest):
    """Reconstruct video with edited segments and custom timeline"""
    try:
        import tempfile
        import time
        from datetime import datetime
        import uuid
        
        # Validate inputs
        if not request.segments:
            raise HTTPException(status_code=400, detail="At least one segment is required")
        
        reconstruction_start = time.time()
        reconstruction_id = str(uuid.uuid4())[:8]
        
        # Create temporary directory for this reconstruction
        temp_dir = Path(settings.TEMP_DIR) / f"reconstruction_{reconstruction_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download all segment audio files
            segment_files = []
            for i, segment in enumerate(request.segments):
                response = requests.get(segment.segment_url)
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download segment {i+1}")
                
                segment_file = temp_dir / f"segment_{i+1:03d}.wav"
                with open(segment_file, 'wb') as f:
                    f.write(response.content)
                
                segment_files.append({
                    'file': segment_file,
                    'start_time': segment.start_time,
                    'duration': segment.duration,
                    'speaker': segment.speaker
                })
            
            # Sort segments by start time
            segment_files.sort(key=lambda x: x['start_time'])
            
            # Calculate total duration
            total_duration = max(seg['start_time'] + seg['duration'] for seg in segment_files)
            
            # Create final audio timeline
            import numpy as np
            sample_rate = 44100
            final_audio = np.zeros(int(total_duration * sample_rate), dtype=np.float32)
            
            # Mix segments into timeline
            for seg_info in segment_files:
                # Load segment audio
                import soundfile as sf
                segment_audio, sr = sf.read(seg_info['file'])
                
                # Ensure correct sample rate
                if sr != sample_rate:
                    import librosa
                    segment_audio = librosa.resample(segment_audio, orig_sr=sr, target_sr=sample_rate)
                
                # Ensure mono
                if len(segment_audio.shape) > 1:
                    segment_audio = np.mean(segment_audio, axis=1)
                
                # Adjust segment duration
                target_samples = int(seg_info['duration'] * sample_rate)
                if len(segment_audio) != target_samples:
                    if len(segment_audio) > target_samples:
                        segment_audio = segment_audio[:target_samples]
                    else:
                        segment_audio = np.pad(segment_audio, (0, target_samples - len(segment_audio)))
                
                # Place in timeline
                start_sample = int(seg_info['start_time'] * sample_rate)
                end_sample = start_sample + len(segment_audio)
                
                if end_sample <= len(final_audio):
                    final_audio[start_sample:end_sample] = segment_audio
            
            # Save final audio
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = request.output_name or f"reconstructed_{timestamp}_{reconstruction_id}"
            final_audio_file = temp_dir / f"{output_name}.wav"
            sf.write(final_audio_file, final_audio, sample_rate)
            
            # Download and mix instruments if requested
            if request.include_instruments and request.instruments_url:
                try:
                    response = requests.get(request.instruments_url)
                    if response.status_code == 200:
                        instruments_file = temp_dir / "instruments.wav"
                        with open(instruments_file, 'wb') as f:
                            f.write(response.content)
                        
                        # Mix with instruments
                        instruments_audio, inst_sr = sf.read(instruments_file)
                        if inst_sr != sample_rate:
                            instruments_audio = librosa.resample(instruments_audio, orig_sr=inst_sr, target_sr=sample_rate)
                        
                        if len(instruments_audio.shape) > 1:
                            instruments_audio = np.mean(instruments_audio, axis=1)
                        
                        # Adjust length to match final audio
                        if len(instruments_audio) > len(final_audio):
                            instruments_audio = instruments_audio[:len(final_audio)]
                        elif len(instruments_audio) < len(final_audio):
                            instruments_audio = np.pad(instruments_audio, (0, len(final_audio) - len(instruments_audio)))
                        
                        # Mix (70% vocals, 30% instruments)
                        mixed_audio = final_audio * 0.7 + instruments_audio * 0.3
                        sf.write(final_audio_file, mixed_audio, sample_rate)
                        
                except Exception as e:
                    logger.warning(f"Failed to include instruments: {e}")
            
            # Generate video if video URL is provided
            video_url_result = None
            if request.video_url:
                try:
                    # Download original video
                    response = requests.get(request.video_url)
                    if response.status_code == 200:
                        video_file = temp_dir / "original_video.mp4"
                        with open(video_file, 'wb') as f:
                            f.write(response.content)
                        
                        # Replace audio in video using ffmpeg
                        output_video_file = temp_dir / f"{output_name}.mp4"
                        import subprocess
                        
                        cmd = [
                            'ffmpeg', '-y',
                            '-i', str(video_file),
                            '-i', str(final_audio_file),
                            '-c:v', 'copy',
                            '-c:a', 'aac',
                            '-map', '0:v:0',
                            '-map', '1:a:0',
                            '-shortest',
                            str(output_video_file)
                        ]
                        
                        subprocess.run(cmd, check=True, capture_output=True)
                        
                        # Upload video to R2
                        r2_key = f"reconstructed-videos/{timestamp}/{output_name}.mp4"
                        upload_result = r2_storage.upload_file(str(output_video_file), r2_key, "video/mp4")
                        if upload_result.get("success"):
                            video_url_result = upload_result.get("url")
                            
                except Exception as e:
                    logger.warning(f"Failed to generate video: {e}")
            
            # Handle subtitles if requested
            subtitles_url_result = None
            if request.include_subtitles and request.subtitles_url:
                try:
                    # For now, just return the original subtitles URL
                    # In a full implementation, you'd adjust subtitle timings based on segment edits
                    subtitles_url_result = request.subtitles_url
                except Exception as e:
                    logger.warning(f"Failed to process subtitles: {e}")
            
            # Upload final audio to R2
            audio_url_result = None
            try:
                r2_key = f"reconstructed-audio/{timestamp}/{output_name}.wav"
                upload_result = r2_storage.upload_file(str(final_audio_file), r2_key, "audio/wav")
                if upload_result.get("success"):
                    audio_url_result = upload_result.get("url")
            except Exception as e:
                logger.warning(f"R2 upload failed: {e}")
            
            reconstruction_time = time.time() - reconstruction_start
            
            return ReconstructVideoResponse(
                success=True,
                message="Video reconstructed successfully",
                video_url=video_url_result,
                audio_url=audio_url_result,
                subtitles_url=subtitles_url_result,
                processing_time=reconstruction_time,
                reconstruction_id=reconstruction_id
            )
            
        finally:
            # Clean up temp files
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in reconstruct_video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/status/{audio_id}")
async def get_status(audio_id: str):
    """Get processing status for a specific audio ID"""
    try:
        status = status_manager.get_status(audio_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_temp_files(audio_id: str, audio_temp_path: Optional[str] = None, instruments_temp_path: Optional[str] = None, video_temp_path: Optional[str] = None):
    """Clean up temporary files and log cache statistics"""
    try:
        # Log cache statistics before cleanup
        cache_stats = audio_processor.get_cache_stats()
        logger.info(f"Cache stats before cleanup for {audio_id}: {cache_stats}")
        
        # Clean up processor files and caches
        audio_processor.cleanup_temp_files(audio_id)
        
        # Clean up specific temp files
        temp_files_to_clean = []
        if audio_temp_path:
            temp_files_to_clean.append(audio_temp_path)
        if instruments_temp_path:
            temp_files_to_clean.append(instruments_temp_path)
        if video_temp_path:
            temp_files_to_clean.append(video_temp_path)
        
        for temp_file in temp_files_to_clean:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
        
        # Clean up any remaining temp files for this audio_id
        temp_dir = Path(settings.TEMP_DIR)
        for temp_file in temp_dir.glob(f"*{audio_id}*"):
            if temp_file.is_file():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
                    
        logger.info(f"Cleanup completed for {audio_id}")
        
    except Exception as e:
        logger.error(f"Error during cleanup for {audio_id}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=False
    ) 