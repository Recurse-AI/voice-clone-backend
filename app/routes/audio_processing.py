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
from app.schemas import AudioSeparationRequest, AudioSeparationResponse, SeparationStatusResponse
from app.dependencies.auth import get_current_user
from app.services.separation_job_service import separation_job_service
from app.services.credit_service import credit_service
from app.utils.audio import AudioUtils
from app.services.r2_service import R2Service
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


# Audio Separation Endpoints
@router.post("/audio-separation", response_model=AudioSeparationResponse)
async def start_audio_separation(
    request: AudioSeparationRequest,
    current_user = Depends(get_current_user)
):
    try:
        user_id = current_user.id
        job_id = request.job_id
        
        logger.info(f"Starting separation job {job_id} for user {user_id}")

        job_dir = os.path.join(settings.TEMP_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        logger.info(f"📥 Downloading file from: {request.file_url}")
        file_extension = os.path.splitext(request.file_url.split('?')[0])[-1].lower()
        temp_download_path = os.path.join(job_dir, f"original{file_extension}")
        
        try:
            response = requests.get(request.file_url, stream=True, timeout=120)
            response.raise_for_status()
            with open(temp_download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"✅ File downloaded: {temp_download_path}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
        audio_extensions = {'.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma'}
        
        is_video = file_extension in video_extensions
        local_audio_path = temp_download_path
        
        if is_video:
            logger.info(f"🎬 Video detected, extracting audio...")
            audio_utils = AudioUtils()
            audio_path = os.path.join(job_dir, f"{job_id}.wav")
            result = audio_utils.extract_audio_from_video(temp_download_path, audio_path)
            
            if not result.get("success"):
                raise HTTPException(status_code=500, detail=f"Audio extraction failed: {result.get('error')}")
            
            local_audio_path = audio_path
            logger.info(f"✅ Audio extracted: {audio_path}")
        elif file_extension in audio_extensions:
            logger.info(f"🎵 Audio file detected")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        r2_service = R2Service()
        original_filename = os.path.basename(local_audio_path)
        sanitized_filename = r2_service._sanitize_filename(original_filename)
        r2_audio_key = f"audio/{job_id}/{sanitized_filename}"
        
        audio_upload_result = r2_service.upload_file(local_audio_path, r2_audio_key)
        if not audio_upload_result.get("success"):
            raise HTTPException(status_code=500, detail=f"Audio upload failed: {audio_upload_result.get('error')}")
        
        audio_url = audio_upload_result["url"]
        
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



