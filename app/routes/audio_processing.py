from fastapi import APIRouter, HTTPException, Depends, Query
import asyncio
import threading
import gc
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import JSONResponse
import logging
import os
from app.schemas import AudioSeparationRequest, AudioSeparationResponse, SeparationStatusResponse, VoiceCloneRequest, VoiceCloneResponse
from app.dependencies.auth import get_current_user
from app.services.separation_job_service import separation_job_service
from app.services.credit_service import credit_service
from app.config.credit_constants import JobType
from app.config.constants import MAX_ATTEMPTS_DEFAULT, POLLING_INTERVAL_SECONDS, MSG_PROCESSING_STARTED, ERROR_PROCESSING_FAILED
from app.utils.job_utils import job_utils
from datetime import datetime, timezone
import random
# Removed custom logger import - using standard logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Global ThreadPoolExecutor for separation processing
_separation_executor = None
_executor_lock = threading.Lock()

def get_separation_executor():
    """Get or create global ThreadPoolExecutor for separation processing"""
    global _separation_executor
    if _separation_executor is None:
        with _executor_lock:
            if _separation_executor is None:
                _separation_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="separation_worker")
    return _separation_executor

def _update_separation_status_non_blocking(job_id: str, status: str, progress: int = None, **kwargs):
    """Update separation job status using unified status manager"""
    from app.utils.unified_status_manager import get_unified_status_manager, ProcessingStatus
    from app.utils.unified_status_manager import JobType as UnifiedJobType
    
    manager = get_unified_status_manager()
    
    # Convert string status to enum
    try:
        status_enum = ProcessingStatus(status)
    except ValueError:
        logger.warning(f"Unknown status: {status}, defaulting to processing")
        status_enum = ProcessingStatus.PROCESSING
    
    # Use sync version to avoid event loop issues
    try:
        manager.update_status_sync(job_id, UnifiedJobType.SEPARATION, status_enum, progress, kwargs)

    except Exception as e:
        logger.error(f"Failed to update separation status for {job_id}: {e}")

def _deduct_separation_credits_non_blocking(user_id: str, job_id: str, duration_seconds: float):
    """Deduct credits using common utility"""
    from app.utils.db_sync_operations import deduct_credits
    deduct_credits(user_id, job_id, duration_seconds, "separation")

def _cleanup_separation_files_non_blocking(job_id: str):
    """Cleanup separation temp files using common utility"""
    try:
        from app.utils.db_sync_operations import cleanup_separation_files
        cleanup_separation_files(job_id)
        logger.info(f"🧹 Cleaned up separation temp files for job {job_id}")
    except Exception as e:
        logger.warning(f"Failed to cleanup separation files for {job_id}: {e}")

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
            # Check if job was cancelled by user
            from app.utils.shared_memory import is_job_cancelled, _status_manager
            is_cancelled = is_job_cancelled(job_id)

            
            if is_cancelled:
                logger.info(f"🛑 Separation job {job_id} (RunPod: {runpod_request_id}) was cancelled by user")
                _update_separation_status_non_blocking(
                    job_id=job_id,
                    status="cancelled",
                    progress=0,
                    error="Job cancelled by user"
                )
                _cleanup_separation_files_non_blocking(job_id)
                # Unmark as cancelled since monitoring is stopping
                from app.utils.shared_memory import unmark_job_cancelled
                unmark_job_cancelled(job_id)
                break
            
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
                

                
                # Check for cancelled status from RunPod
                if job_status.upper() == "CANCELLED":
                    logger.info(f"🛑 RunPod job {runpod_request_id} was CANCELLED - force stopping monitoring")
                    
                    # Try to update status (but don't fail if it doesn't work)
                    try:
                        _update_separation_status_non_blocking(
                            job_id=job_id,
                            status="cancelled",
                            progress=0,
                            error="Job cancelled by user"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update cancelled status for {job_id}: {e}")
                    
                    # Try cleanup (but don't fail if it doesn't work)
                    try:
                        _cleanup_separation_files_non_blocking(job_id)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup files for {job_id}: {e}")
                    
                    # Always unmark and break regardless of database update success
                    from app.utils.shared_memory import unmark_job_cancelled
                    unmark_job_cancelled(job_id)
                    logger.info(f"🚫 FORCE STOPPED monitoring for {job_id} due to RunPod CANCELLED")
                    break  # Force stop monitoring
                
                # Only update status if job is not cancelled

                
                # 🛡️ CRITICAL: Check if job is already cancelled in database (soft delete protection)
                try:
                    from app.utils.db_sync_operations import SyncDBOperations
                    from pymongo import MongoClient
                    from app.config.settings import settings
                    
                    sync_client = MongoClient(settings.MONGODB_URI)
                    sync_collection = sync_client[settings.DB_NAME]["separation_jobs"]
                    current_job = sync_collection.find_one({"job_id": job_id})
                    sync_client.close()
                    
                    if current_job and current_job.get("status") == "cancelled":
                        logger.info(f"🛡️ Job {job_id} already cancelled in database - stopping background monitoring")
                        from app.utils.shared_memory import unmark_job_cancelled
                        unmark_job_cancelled(job_id)
                        break
                except Exception as e:
                    logger.warning(f"Failed to check job status in database: {e}")
                
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
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                            "processing_time_seconds": attempt * 10,
                            "result_data": status.get("result", {})
                        }
                    )
                    
                    # Memory cleanup after separation completion
                    del status, vocal_url, instrument_url
                    gc.collect()
                    
                    # Complete credit billing using centralized utility (sync context)
                    job_utils.complete_job_billing_sync(job_id, "separation", user_id)
                    
                    # Cleanup temp files after successful completion
                    _cleanup_separation_files_non_blocking(job_id)
                    
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
                    
                    # Refund credits on failure (sync context)
                    job_utils.refund_job_credits_sync(job_id, "separation", "job_failed")
                    
                    # Cleanup temp files even on failure
                    _cleanup_separation_files_non_blocking(job_id)
                    
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
            
            # Refund credits on timeout (sync context)
            job_utils.refund_job_credits_sync(job_id, "separation", "job_timeout")
            
            # Cleanup temp files on timeout
            _cleanup_separation_files_non_blocking(job_id)
            
    except Exception as e:
        logger.error(f"Background separation monitoring failed for job {job_id} (RunPod: {runpod_request_id}): {e}")
        try:
            _update_separation_status_non_blocking(
                job_id=job_id,
                status="failed",
                progress=0,
                error=f"Background monitoring error: {str(e)}"
            )
            
            # Refund credits on monitoring error (sync context)
            job_utils.refund_job_credits_sync(job_id, "separation", "monitoring_error")
            
            # Cleanup temp files on error
            _cleanup_separation_files_non_blocking(job_id)
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
        
        # Get uploaded file path from upload status
        from app.utils.shared_memory import get_upload_status as get_upload_status_data, job_exists
        
        logger.info(f"user_id {user_id} and job_id {job_id}")
        if not job_exists(job_id):
            raise HTTPException(status_code=400, detail="Upload job not found")
            
        upload_data = get_upload_status_data(job_id)
        if upload_data.get("status") not in ["done", "ready"]:
            raise HTTPException(status_code=400, detail=f"Upload not completed. Status: {upload_data.get('status', 'unknown')}")
        
        local_audio_path = upload_data.get("file_url")

        if not local_audio_path:
        # If file_url not found, try to get from video_info
            video_info = upload_data.get("video_info", {})
            local_audio_path = video_info.get("local_path")
        if not local_audio_path or not os.path.exists(local_audio_path):
            raise HTTPException(status_code=400, detail="Uploaded file not found on disk")
            
        # Verify it's an audio file
        if not local_audio_path.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(status_code=400, detail="Uploaded file is not an audio format")
        
        logger.info(f"---> local url {local_audio_path}")
        # Upload audio to R2 storage for separation processing
        from app.services.r2_service import get_r2_service
        r2_service = get_r2_service()
        
        # Get original filename from upload data or fallback to local path basename
        original_filename = upload_data.get("original_filename", os.path.basename(local_audio_path))
        
        # Generate R2 key preserving original filename
        audio_extension = os.path.splitext(original_filename)[1] if original_filename else ".wav"
        r2_audio_key = f"audio_separation/{job_id}/{original_filename}"
        
        audio_upload_result = r2_service.upload_file(local_audio_path, r2_audio_key)
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
        
        # Create separation job data
        job_data = {
            "job_id": separation_job_id,
            "user_id": user_id,
            "audio_url": audio_url,
            "original_filename": upload_data.get("original_filename", os.path.basename(local_audio_path)),
            "caller_info": request.callerInfo or "audio_separation_api",
            "runpod_request_id": runpod_request_id,
            "status": "pending",
            "progress": 0,
            "upload_job_id": job_id,
            "local_audio_path": local_audio_path
        }
        
        # Atomic credit reservation + job creation
        result = await credit_service.reserve_credits_and_create_job(
            user_id=user_id,
            job_data=job_data,
            job_type=JobType.SEPARATION,
            duration_seconds=request.duration
        )
        
        if not result["success"]:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Insufficient credits",
                    "error": result.get("error", "Credit reservation failed")
                }
            )
        
        status = runpod_service.get_separation_status(runpod_request_id)
        queue_position = status.get("queue_position") if status else None
        
        # Run separation monitoring in ThreadPoolExecutor for better resource management
        executor = get_separation_executor()
        future = executor.submit(process_audio_separation_background, separation_job_id, runpod_request_id, user_id, request.duration)
        
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
    job_dir = None  # Initialize to ensure cleanup works
    try:
        from app.services.r2_service import get_r2_service
        from app.config.settings import settings
        
        r2_service = get_r2_service()
        job_id = r2_service.generate_job_id()
        # Voice cloning এর জন্য nested folder structure ব্যবহার করি
        job_dir = os.path.join(settings.TEMP_DIR, "voice_cloning", f"voice_clone_job_{job_id}")
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
            seed=int(random.randint(0, 2**32 - 1))
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Voice generation failed"))

        cloned_path = os.path.join(job_dir, f"{job_id}.wav")
        with open(cloned_path, "wb") as f:
            f.write(result["audio_data"])

        # Upload to R2
        r2_key = r2_service.generate_file_path(job_id, "", f"{job_id}.wav")
        upload_res = r2_service.upload_file(cloned_path, r2_key, content_type="audio/wav")

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
        # Cleanup temp directory properly
        if job_dir and os.path.exists(job_dir):
            try:
                from app.services.dub.audio_utils import AudioUtils
                AudioUtils.remove_temp_dir(folder_path=job_dir)
                logger.info(f"🧹 Cleaned up voice clone temp directory: {job_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup voice clone directory {job_dir}: {cleanup_error}")

