from fastapi import APIRouter, HTTPException, Depends, Query
import asyncio
import threading
import gc
import time
import random
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from pymongo import MongoClient
from pathlib import Path

import logging
from app.schemas import AudioSeparationRequest, AudioSeparationResponse, SeparationStatusResponse, VoiceCloneRequest, VoiceCloneResponse
from app.dependencies.auth import get_current_user
from app.services.separation_job_service import separation_job_service
from app.services.credit_service import credit_service
from app.services.dub.audio_utils import AudioUtils
from app.services.dub.fish_speech_service import get_fish_speech_service
from app.services.r2_service import get_r2_service
from app.utils.unified_status_manager import (
    get_unified_status_manager, ProcessingStatus, JobType as UnifiedJobType
)
from app.utils.runpod_service import runpod_service
from app.utils.shared_memory import is_job_cancelled, unmark_job_cancelled
from app.config.credit_constants import JobType
from app.config.constants import MAX_ATTEMPTS_DEFAULT, POLLING_INTERVAL_SECONDS, MSG_PROCESSING_STARTED
from app.config.settings import settings
from app.utils.cleanup_utils import cleanup_utils
from app.utils.separation_utils import separation_utils
from app.utils.job_utils import job_utils


router = APIRouter()
logger = logging.getLogger(__name__)

class SeparationExecutor:
    def __init__(self):
        self._executor = None
        self._lock = threading.Lock()
        self._last_used = 0
        self._shutdown = False

    def get_executor(self):
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Executor shutdown")

            current_time = time.time()
            if self._executor is None or (current_time - self._last_used) > 300:
                self._create_executor()

            self._last_used = current_time
            return self._executor

    def _create_executor(self):
        if self._executor:
            self._executor.shutdown(wait=True)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sep")

    def shutdown(self):
        with self._lock:
            self._shutdown = True
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None

_separation_manager = SeparationExecutor()

def get_separation_executor():
    return _separation_manager.get_executor()

def _update_separation_status_non_blocking(job_id: str, status: str, progress: int = None, **kwargs):
    """Update separation job status using unified status manager"""

    
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

def _cleanup_separation_files_non_blocking(job_id: str):
    """Cleanup separation temp files using common utility"""
    try:
        cleanup_utils.cleanup_job_comprehensive(job_id, "separation")
        logger.info(f"Cleaned up separation temp files for job {job_id}")
    except Exception as e:
        logger.warning(f"Failed to cleanup separation files for {job_id}: {e}")


def process_audio_separation_background(job_id: str, runpod_request_id: str, user_id: str, duration_seconds: float):
    """Background task to monitor audio separation progress and auto-deduct credits on completion (sync - runs in separate thread)"""
    try:

        import asyncio
        
        logger.info(f"Starting background monitoring for separation job {job_id} (RunPod: {runpod_request_id})")

        # Get job directory for file storage
        job_dir = os.path.join(settings.TEMP_DIR, job_id)

        max_attempts = MAX_ATTEMPTS_DEFAULT
        attempt = 0
        
        while attempt < max_attempts:
            # Check if job was cancelled by user

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
    
                    unmark_job_cancelled(job_id)
                    logger.info(f"🚫 FORCE STOPPED monitoring for {job_id} due to RunPod CANCELLED")
                    break  # Force stop monitoring
                
                # Only update status if job is not cancelled

                
                # 🛡️ CRITICAL: Check if job is already cancelled in database (soft delete protection)
                try:

                    
                    sync_client = MongoClient(settings.MONGODB_URI)
                    sync_collection = sync_client[settings.DB_NAME]["separation_jobs"]
                    current_job = sync_collection.find_one({"job_id": job_id})
                    sync_client.close()
                    
                    if current_job and current_job.get("status") == "cancelled":
                        logger.info(f"🛡️ Job {job_id} already cancelled in database - stopping background monitoring")
        
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
                    # Extract URLs and download files using common utilities
                    vocal_url = None
                    instrument_url = None
                    file_paths = {}

                    if status.get("result"):
                        output = status["result"]
                        runpod_urls = separation_utils.extract_urls_from_clearvocals_response(output)
                        vocal_url = runpod_urls.get('vocal_audio')
                        instrument_url = runpod_urls.get('instrument_audio')

                        download_success, file_paths = separation_utils.download_separation_files(
                            job_id=job_id,
                            job_dir=job_dir,
                            runpod_urls=runpod_urls,
                            on_error_callback=None
                        )

                        if not download_success:
                            logger.warning("Some files failed to download, but continuing with available files")

                    # Update job with completion details
                    _update_separation_status_non_blocking(
                        job_id=job_id,
                        status="completed",
                        progress=100,
                        vocal_url=vocal_url,
                        instrument_url=instrument_url,
                        details={
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                            "processing_time_seconds": attempt * 10,
                            "vocal_file": file_paths.get('vocal'),
                            "instrument_file": file_paths.get('instrument'),
                            "result_data": status.get("result", {})
                        }
                    )
                    
                    # Memory cleanup after separation completion
                    del status, vocal_url, instrument_url, runpod_urls, file_paths
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
        
        logger.info(f"user_id {user_id} and job_id {job_id}")

        # Get uploaded file path directly (no status tracking needed)

        job_dir = os.path.join(settings.TEMP_DIR, job_id)

        if not os.path.exists(job_dir):
            raise HTTPException(status_code=400, detail="Upload directory not found")

        # Find the uploaded file in the job directory
        files = os.listdir(job_dir)
        if not files:
            raise HTTPException(status_code=400, detail="No files found in upload directory")

        # Take the first file (should be the uploaded file)
        local_audio_path = os.path.join(job_dir, files[0])

        if not os.path.exists(local_audio_path):
            raise HTTPException(status_code=400, detail="Uploaded file not found on disk")
            
        # Verify it's an audio file
        if not local_audio_path.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(status_code=400, detail="Uploaded file is not an audio format")
        
        logger.info(f"---> local url {local_audio_path}")
        # Upload audio to R2 storage for separation processing

        r2_service = get_r2_service()
        
        # Get original filename from the uploaded file
        original_filename = os.path.basename(local_audio_path)
        
        # Generate R2 key preserving original filename
        r2_audio_key = f"audio/{job_id}/{original_filename}"
        
        audio_upload_result = r2_service.upload_file(local_audio_path, r2_audio_key)
        if not audio_upload_result.get("success"):
            raise HTTPException(status_code=500, detail=f"Audio upload failed: {audio_upload_result.get('error')}")
        
        audio_url = audio_upload_result["url"]

        # ✅ CREDIT RESERVATION BEFORE RUNPOD REQUEST


        # Create job data first
        job_data = {
            "job_id": job_id,
            "user_id": user_id,
            "audio_url": audio_url,
            "original_filename": original_filename,
            "caller_info": request.callerInfo or "audio_separation_api",
            "status": "pending",
            "progress": 0,
            "local_audio_path": local_audio_path
        }

        # Reserve credits BEFORE expensive RunPod call
        credit_result = await credit_service.reserve_credits_and_create_job(
            user_id=user_id,
            job_data=job_data,
            job_type=JobType.SEPARATION,
            duration_seconds=request.duration
        )

        if not credit_result["success"]:
            raise HTTPException(status_code=400, detail=f"Credit reservation failed: {credit_result.get('error', 'Unknown error')}")

        # ✅ NOW SUBMIT RUNPOD REQUEST (after credits reserved)


        try:
            runpod_request_id = runpod_service.submit_separation_request(
                audio_url,
                caller_info=request.callerInfo or "audio_separation_api"
            )
        except Exception as runpod_error:
            # ❌ RUNPOD FAILED - ROLLBACK CREDITS
            logger.error(f"RunPod request failed, rolling back credits: {runpod_error}")
            await credit_service.refund_job_credits(job_id, JobType.SEPARATION, "runpod_request_failed")
            raise HTTPException(status_code=500, detail=f"Audio separation request failed: {str(runpod_error)}")

        # Job already created with credits reserved, just proceed
        
        status = runpod_service.get_separation_status(runpod_request_id)
        queue_position = status.get("queue_position") if status else None
        
        # Run separation monitoring in ThreadPoolExecutor for better resource management
        executor = get_separation_executor()
        future = executor.submit(process_audio_separation_background, job_id, runpod_request_id, user_id, request.duration)

        logger.info(f"Started audio separation job {job_id} (RunPod: {runpod_request_id}) for user {user_id} (duration: {request.duration}s)")

        return AudioSeparationResponse(
            success=True,
            job_id=job_id,
            message=MSG_PROCESSING_STARTED,
            estimatedTime="5-15 minutes",
            statusCheckUrl=f"/api/jobs/separation/{job_id}",
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


        
        r2_service = get_r2_service()
        job_id = r2_service.generate_job_id()
        # Voice cloning job directory (consistent with other services)
        job_dir = os.path.join(settings.TEMP_DIR, f"voice_clone_job_{job_id}")
        os.makedirs(job_dir, exist_ok=True)

        # Download reference audio

        audio_utils = AudioUtils()
        reference_path = os.path.join(job_dir, "reference.wav")
        download_res = audio_utils.download_audio_file(request.referenceAudioUrl, reference_path)
        if not download_res["success"]:
            raise HTTPException(status_code=400, detail=f"Reference audio download failed: {download_res.get('error')}")

        # Read reference audio bytes
        with open(reference_path, "rb") as f:
            reference_bytes = f.read()

        # Generate cloned audio

        fish_service = get_fish_speech_service()

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
        
                AudioUtils.remove_temp_dir(folder_path=job_dir)
                logger.info(f"🧹 Cleaned up voice clone temp directory: {job_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup voice clone directory {job_dir}: {cleanup_error}")

