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
from app.services.simple_status_service import status_service, JobStatus
from app.utils.runpod_service import runpod_service
from app.config.credit_constants import JobType as CreditJobType
from app.config.constants import MAX_ATTEMPTS_DEFAULT, POLLING_INTERVAL_SECONDS, MSG_PROCESSING_STARTED
from app.config.settings import settings
from app.utils.cleanup_utils import cleanup_utils
from app.utils.separation_utils import separation_utils
from app.utils.job_utils import job_utils
from app.queue.queue_manager import queue_manager

router = APIRouter()
logger = logging.getLogger(__name__)

def _update_separation_status_non_blocking(job_id: str, status: str, progress: int = None, **kwargs):
    """Update separation job status using simple status service"""
    try:
        # Convert string to JobStatus
        status_map = {
            "pending": JobStatus.PENDING,
            "processing": JobStatus.PROCESSING,
            # Note: separation jobs don't use "separating" status (only dub jobs do)
            "completed": JobStatus.COMPLETED,
            "failed": JobStatus.FAILED
        }
        
        status_enum = status_map.get(status, JobStatus.PROCESSING)
        status_service.update_status(job_id, "separation", status_enum, progress, kwargs)
        
    except Exception as e:
        logger.error(f"Failed to update separation status for {job_id}: {e}")

def _cleanup_separation_files_non_blocking(job_id: str):
    """Cleanup separation temp files using common utility"""
    try:
        cleanup_utils.cleanup_job_comprehensive(job_id, "separation")
        logger.info(f"Cleaned up separation temp files for job {job_id}")
    except Exception as e:
        logger.warning(f"Failed to cleanup separation files for {job_id}: {e}")


# Audio Separation Endpoints
@router.post("/audio-separation", response_model=AudioSeparationResponse)
async def start_audio_separation(
    request: AudioSeparationRequest,
    current_user = Depends(get_current_user)
):
    """Start audio separation job using clean service architecture"""
    try:
        user_id = current_user.id
        job_id = request.job_id
        
        logger.info(f"Starting separation job {job_id} for user {user_id}")

        # Validate uploaded file
        job_dir = os.path.join(settings.TEMP_DIR, job_id)
        if not os.path.exists(job_dir):
            raise HTTPException(status_code=400, detail="Upload directory not found")

        files = os.listdir(job_dir)
        if not files:
            raise HTTPException(status_code=400, detail="No files found in upload directory")

        local_audio_path = os.path.join(job_dir, files[0])
        if not os.path.exists(local_audio_path):
            raise HTTPException(status_code=400, detail="Uploaded file not found on disk")
            
        # Verify audio format
        if not local_audio_path.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(status_code=400, detail="Uploaded file is not an audio format")
        
        # Upload to R2 storage
        r2_service = get_r2_service()
        original_filename = os.path.basename(local_audio_path)
        r2_audio_key = f"audio/{job_id}/{original_filename}"
        
        audio_upload_result = r2_service.upload_file(local_audio_path, r2_audio_key)
        if not audio_upload_result.get("success"):
            raise HTTPException(status_code=500, detail=f"Audio upload failed: {audio_upload_result.get('error')}")
        
        audio_url = audio_upload_result["url"]
        
        # Use clean separation service
        from app.services.separation_service import separation_service
        
        result = await separation_service.create_separation_job(
            job_id=job_id,
            user_id=user_id,
            audio_url=audio_url,
            original_filename=original_filename,
            duration=request.duration,
            caller_info=request.callerInfo or "audio_separation_api"
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Enqueue background task
        success = queue_manager.enqueue_separation_task(
            job_id, result["runpod_request_id"], user_id, request.duration
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to enqueue separation task")

        logger.info(f"✅ Separation job created and enqueued: {job_id}")

        return AudioSeparationResponse(
            success=True,
            job_id=job_id,
            message=MSG_PROCESSING_STARTED,
            estimatedTime="5-15 minutes",
            statusCheckUrl=f"/api/jobs/separation/{job_id}",
            queuePosition=None
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

