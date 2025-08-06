from fastapi import APIRouter, HTTPException, Depends, Query
import asyncio
import threading
from fastapi.responses import JSONResponse
import logging
import os
from app.schemas import AudioSeparationRequest, AudioSeparationResponse, SeparationStatusResponse, VoiceCloneRequest, VoiceCloneResponse
from app.dependencies.auth import get_current_user
from app.services.separation_job_service import separation_job_service
from app.services.credit_service import credit_service, JobType
from app.config.constants import MAX_ATTEMPTS_DEFAULT, POLLING_INTERVAL_SECONDS, MSG_PROCESSING_STARTED, ERROR_PROCESSING_FAILED
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

def _update_separation_status_non_blocking(job_id: str, status: str, progress: int = None, **kwargs):
    """Update separation status using common utility"""
    from app.utils.db_sync_operations import update_separation_status
    update_separation_status(job_id, status, progress, **kwargs)

def _deduct_separation_credits_non_blocking(user_id: str, job_id: str, duration_seconds: float):
    """Deduct credits using common utility"""
    from app.utils.db_sync_operations import deduct_credits
    deduct_credits(user_id, job_id, duration_seconds, "separation")

# Background Task Functions
def process_audio_separation_background(job_id: str, runpod_request_id: str, user_id: str, duration_seconds: float):
    """Background task to monitor audio separation progress and auto-deduct credits on completion (sync - runs in separate thread)"""
    try:
        from app.utils.runpod_service import runpod_service
        import asyncio
        
        logger.info(f"Starting background monitoring for separation job {job_id} (RunPod: {runpod_request_id})")
        
        max_attempts = MAX_ATTEMPTS_DEFAULT
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Check job status from RunPod using RunPod request ID
                status = runpod_service.get_separation_status(runpod_request_id)
                
                if not status:
                    logger.warning(f"No status found for RunPod job {runpod_request_id} (attempt {attempt + 1})")
                    # Don't break immediately - might be temporary RunPod issue
                    attempt += 1
                    time.sleep(POLLING_INTERVAL_SECONDS)
                    continue
                
                job_status = status.get("status", "unknown")
                progress = status.get("progress", 0)
                

                
                # Update MongoDB status (non-blocking) using our job_id
                _update_separation_status_non_blocking(
                    job_id=job_id,
                    status=job_status,
                    progress=progress
                )
                
                if job_status == "completed":
                    # Extract result URLs from RunPod response
                    vocal_url = None
                    instrument_url = None
                    
                    if status.get("result"):
                        # RunPod result contains the output data
                        output = status["result"]
                        vocal_url = output.get("vocal_audio") or output.get("vocals")
                        instrument_url = output.get("instrument_audio") or output.get("instruments")
                        
                        # Log the response structure for debugging
                        logger.info(f"RunPod separation response for {runpod_request_id}: vocal_url={vocal_url}, instrument_url={instrument_url}")
                    
                    # Update job with completion details (non-blocking)
                    _update_separation_status_non_blocking(
                        job_id=job_id,
                        status="completed",
                        progress=100,
                        vocal_url=vocal_url,
                        instrument_url=instrument_url,
                        details={
                            "completed_at": datetime.now().isoformat(),
                            "processing_time_seconds": attempt * 10,
                            "result_data": status.get("result", {})
                        }
                    )
                    
                    # Auto-deduct credits on successful completion (non-blocking)
                    _deduct_separation_credits_non_blocking(
                        user_id=user_id,
                        job_id=job_id,
                        duration_seconds=duration_seconds
                    )
                    
                    break
                    
                elif job_status == "failed":
                    error_msg = status.get("error", "Audio separation failed")
                    _update_separation_status_non_blocking(
                        job_id=job_id,
                        status="failed",
                        progress=0,
                        error=error_msg
                    )
                    logger.error(f"Separation job {job_id} (RunPod: {runpod_request_id}) failed: {error_msg}")
                    break
                    
                # Wait before next check
                import time
                time.sleep(POLLING_INTERVAL_SECONDS)
                attempt += 1
                
            except Exception as e:
                logger.error(f"Error checking separation job {job_id} (RunPod: {runpod_request_id}) status (attempt {attempt}): {e}")
                import time
                time.sleep(POLLING_INTERVAL_SECONDS)
                attempt += 1
        
        if attempt >= max_attempts:
            logger.warning(f"Separation job {job_id} (RunPod: {runpod_request_id}) monitoring timed out after {max_attempts * POLLING_INTERVAL_SECONDS} seconds")
            _update_separation_status_non_blocking(
                job_id=job_id,
                status="failed",
                progress=0,
                error="Job monitoring timed out"
            )
            
    except Exception as e:
        logger.error(f"Background separation monitoring failed for job {job_id} (RunPod: {runpod_request_id}): {e}")
        try:
            _update_separation_status_non_blocking(
                job_id=job_id,
                status="failed",
                progress=0,
                error=f"Background monitoring error: {str(e)}"
            )
        except:
            pass  # Avoid double error

# Audio Separation Endpoints
@router.post("/audio-separation", response_model=AudioSeparationResponse)
async def start_audio_separation(
    request: AudioSeparationRequest,
    current_user = Depends(get_current_user)
):
    """Start audio separation job with uploaded file and credit pre-check"""
    try:
        user_id = current_user.id
        job_id = request.job_id
        
        # Pre-check: Verify user has sufficient credits
        credit_check = await credit_service.check_sufficient_credits(
            user_id=user_id,
            job_type=JobType.SEPARATION,
            duration_seconds=request.duration
        )
        
        if not credit_check["sufficient"]:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": credit_check["message"],
                    "required_credits": credit_check["required"],
                    "available_credits": credit_check["available"]
                }
            )
        
        # Get uploaded file path from upload status
        from app.utils.shared_memory import get_upload_status as get_upload_status_data, job_exists
        
        if not job_exists(job_id):
            raise HTTPException(status_code=400, detail="Upload job not found")
            
        upload_data = get_upload_status_data(job_id)
        if upload_data.get("status") != "done":
            raise HTTPException(status_code=400, detail=f"Upload not completed. Status: {upload_data.get('status', 'unknown')}")
        
        local_audio_path = upload_data.get("file_url")
        if not local_audio_path or not os.path.exists(local_audio_path):
            raise HTTPException(status_code=400, detail="Uploaded file not found on disk")
            
        # Verify it's an audio file
        if not local_audio_path.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg')):
            raise HTTPException(status_code=400, detail="Uploaded file is not an audio format")
        
        # Upload audio to R2 storage for separation processing
        from app.utils.r2_storage import R2Storage
        r2_storage = R2Storage()
        
        audio_upload_result = r2_storage.upload_file(local_audio_path, f"audio_separation/{job_id}/audio.wav")
        if not audio_upload_result.get("success"):
            raise HTTPException(status_code=500, detail=f"Audio upload failed: {audio_upload_result.get('error')}")
        
        audio_url = audio_upload_result["url"]
        
        # Submit separation request with uploaded audio URL
        from app.utils.runpod_service import runpod_service
        
        runpod_request_id = runpod_service.submit_separation_request(
            audio_url,
            caller_info=request.callerInfo or "audio_separation_api"
        )
        
        # Use upload job_id as separation job_id for consistency
        separation_job_id = job_id
        
        # Create separation job in MongoDB for tracking
        job_data = {
            "job_id": separation_job_id,
            "user_id": user_id,
            "audio_url": audio_url,
            "original_filename": upload_data.get("original_filename", os.path.basename(local_audio_path)),
            "caller_info": request.callerInfo or "audio_separation_api",
            "runpod_request_id": runpod_request_id,
            "details": {
                "duration": request.duration,
                "required_credits": credit_check["required"],
                "upload_job_id": job_id,
                "local_audio_path": local_audio_path
            }
        }
        
        # Save to MongoDB
        job = await separation_job_service.create_job(job_data)
        if not job:
            logger.error(f"Failed to create separation job in MongoDB for {separation_job_id}")
        
        # Set initial pending status
        await separation_job_service.update_job_status(separation_job_id, "pending", 0)
        
        status = runpod_service.get_separation_status(runpod_request_id)
        queue_position = status.get("queue_position") if status else None
        
        # Run separation monitoring in separate thread
        thread = threading.Thread(target=process_audio_separation_background, args=(separation_job_id, runpod_request_id, user_id, request.duration), daemon=True)
        thread.start()
        
        logger.info(f"Started audio separation job {separation_job_id} (RunPod: {runpod_request_id}) for user {user_id} (duration: {request.duration}s)")
        
        return AudioSeparationResponse(
            success=True,
            job_id=separation_job_id,
            message=MSG_PROCESSING_STARTED,
            estimatedTime="5-15 minutes",
            statusCheckUrl=f"/api/jobs/separation/{separation_job_id}",
            queuePosition=queue_position
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start audio separation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start separation: {str(e)}")



# Voice Clone Endpoint
@router.post("/voice-clone-segment", response_model=VoiceCloneResponse)
async def voice_clone_segment(request: VoiceCloneRequest):
    """
    Voice clone a single text segment using reference audio.
    1. Downloads reference audio from URL.
    2. Generates cloned voice with FishSpeech.
    3. Uploads generated audio to R2 bucket.
    4. Returns public URL of cloned audio.
    """
    try:
        from app.utils.r2_storage import R2Storage
        from app.config.settings import settings
        
        r2_storage = R2Storage()
        job_id = r2_storage.generate_job_id()
        job_dir = os.path.join(settings.TEMP_DIR, f"voice_clone_{job_id}")
        os.makedirs(job_dir, exist_ok=True)

        # Download reference audio
        from app.services.dub.audio_utils import AudioUtils
        audio_utils = AudioUtils()
        reference_path = os.path.join(job_dir, "reference.wav")
        download_res = audio_utils.download_audio_file(request.referenceAudioUrl, reference_path)
        if not download_res["success"]:
            raise HTTPException(status_code=400, detail=f"Reference audio download failed: {download_res.get('error')}")

        # Read reference audio bytes
        with open(reference_path, "rb") as f:
            reference_bytes = f.read()

        # Generate cloned audio
        from app.services.dub.fish_speech_service import get_fish_speech_service
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
        from app.services.dub.audio_utils import AudioUtils
        AudioUtils.remove_temp_dir(folder_path=locals().get("job_dir"))

