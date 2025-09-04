from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
import os
import asyncio
import gc
import threading
import uuid
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path
from pymongo import MongoClient

from app.config.settings import settings
from app.schemas import (
    VideoDubRequest,
    VideoDubResponse,
    VideoDubStatusResponse,
    RedubRequest,
    RedubResponse,
)
from app.dependencies.auth import get_current_user
from app.services.dub_job_service import dub_job_service
from app.services.credit_service import credit_service
from app.services.dub.audio_utils import AudioUtils
from app.services.r2_service import get_r2_service
from app.services.dub.simple_dubbed_api import get_simple_dubbed_api
from app.services.job_response_service import job_response_service
from app.utils.unified_status_manager import (
    get_unified_status_manager, ProcessingStatus, JobType,
    JobType as UnifiedJobType
)
from app.config.credit_constants import JobType as CreditJobType
from app.utils.runpod_service import runpod_service
from app.utils.runpod_monitor import monitor_runpod_job

from app.utils.job_utils import job_utils
from app.utils.cleanup_utils import cleanup_utils
from app.utils.video_downloader import video_download_service
from app.services.dub.queue_manager import get_dub_queue_manager
from app.utils.token_helper import generate_url_safe_token
from app.utils.separation_utils import separation_utils
from app.config.database import db


router = APIRouter()

logger = logging.getLogger(__name__)


# Keep workers consistent across executor and scheduler (max concurrent running jobs)
DUB_MAX_WORKERS = 10

# Singleton queue manager instance
_dub_queue_manager = get_dub_queue_manager(max_concurrency=DUB_MAX_WORKERS)

def get_dub_executor():
    """Backward-compat: expose executor (not used directly)."""
    return _dub_queue_manager._get_executor()

def _update_status_non_blocking(job_id: str, status: ProcessingStatus, progress: int, details: dict):
    """Update dub job status using unified status manager"""
    manager = get_unified_status_manager()


    try:
        manager.update_status_sync(job_id, JobType.DUB, status, progress, details)

    except Exception as e:
        logger.error(f"Failed to update status for {job_id}: {e}")


def enqueue_dub_job(request: VideoDubRequest, user_id: str) -> None:
    # Inject runner to avoid circular imports inside manager
    def _run():
        process_video_dub_background(request, user_id)
    _dub_queue_manager.enqueue(request.job_id, _run)

def get_dub_queue_position(job_id: str) -> Optional[int]:
    return _dub_queue_manager.get_position(job_id)


@router.post("/video-dub", response_model=VideoDubResponse)
async def start_video_dub(
    request: VideoDubRequest, 
    current_user = Depends(get_current_user)
):
    try:
        user_id = current_user.id
        # Validate duration server-side to prevent credit calculation errors
        if request.duration is None or request.duration <= 0:
            raise HTTPException(status_code=400, detail="Duration is required and must be greater than 0 seconds")
        
        job_data = {
            "job_id": request.job_id,
            "user_id": user_id,
            "target_language": request.target_language,
            "original_filename": request.project_title,
            "source_video_language": request.source_video_language,
            "duration": request.duration,
            "status": "pending",
            "progress": 0
        }
        

        result = await credit_service.reserve_credits_and_create_job(
            user_id=user_id,
            job_data=job_data,
            job_type=CreditJobType.DUB,
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
        
        # Initialize job status in unified status manager to prevent 404 errors

        manager = get_unified_status_manager()
        
        # Get queue position before enqueuing (to get proper position)
        queue_position = _dub_queue_manager.get_next_position()  # Get position it will have
        
        # Enqueue job 
        enqueue_dub_job(request, user_id)
        
        # Verify queue position after enqueue
        actual_position = get_dub_queue_position(request.job_id)
        if actual_position is not None:
            queue_position = actual_position
        
        # Create status with queue information
        await manager.update_status(
            job_id=request.job_id,
            job_type=JobType.DUB,
            status=ProcessingStatus.PENDING,
            progress=0,
            details={
                "user_id": user_id, 
                "created_at": datetime.now().isoformat(),
                "queue_position": queue_position,
                "phase": "queued"
            },
            user_id=user_id
        )
        
        logger.info(f"Started video dub job {request.job_id} for user {user_id}, queue position: {queue_position}")
        return VideoDubResponse(
            success=True,
            message="Video dub started successfully",
            job_id=request.job_id,
            status_check_url=f"/api/video-dub-status/{request.job_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start video dub: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start video dubbing: {str(e)}")

@router.get("/video-dub-status/{job_id}", response_model=VideoDubStatusResponse)
async def get_video_dub_status(job_id: str):
    try:

        
        # Get status from unified manager (includes queue position and proper caching)  
        manager = get_unified_status_manager()
        status_data = await manager.get_status(job_id, UnifiedJobType.DUB)
        
        if not status_data:
            # Check if job exists in database but not yet in status manager
            job = await dub_job_service.get_job(job_id)
            if job:
                # Job exists in database but not in status manager - return default pending status
                logger.info(f"Job {job_id} found in database but not in status manager, returning default status")
                return VideoDubStatusResponse(
                    job_id=job_id,
                    status="pending",
                    progress=0,
                    message="Job is being initialized...",
                    result_url=None,
                    error=None,
                    details={
                        "files": {},
                        "queue_position": None,
                        "updated_at": datetime.now().isoformat()
                    }
                )
            else:
                raise HTTPException(status_code=404, detail="Job ID not found")
        
        # Get additional job details for files
        job = await dub_job_service.get_job(job_id)
        if job:

            formatted_job = job_response_service.format_dub_job(job)
            files = formatted_job.files
            result_url = formatted_job.result_url
            error = formatted_job.error
        else:
            files = {}
            result_url = None
            error = status_data.details.get("error")

        # Enhanced queue position and status message for PENDING jobs
        enhanced_message = status_data.message
        enhanced_queue_position = status_data.queue_position
        
        if status_data.status == ProcessingStatus.PENDING:
            # Get real-time queue position for PENDING jobs
            current_queue_position = get_dub_queue_position(job_id)
            if current_queue_position is not None:
                enhanced_queue_position = current_queue_position
                if current_queue_position == 1:
                    enhanced_message = "Your job is next in line to start processing"
                else:
                    enhanced_message = f"Your job is #{current_queue_position} in queue, waiting to start"
            else:
                # Job not in queue, probably about to start
                enhanced_message = "Job is being prepared for processing..."
        
        return VideoDubStatusResponse(
            job_id=job_id,
            status=status_data.status.value,
            progress=status_data.progress,
            message=enhanced_message,
            result_url=result_url,
            error=error,
            details={
                "files": files,
                "queue_position": enhanced_queue_position,
                "updated_at": status_data.updated_at.isoformat(),
                "phase": status_data.details.get("phase"),
                "in_queue": status_data.status == ProcessingStatus.PENDING and enhanced_queue_position is not None
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dub job status {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

## Removed per requirement: clients should use details.files only

def process_video_dub_background(request: VideoDubRequest, user_id: str):

    
    r2_service = get_r2_service()
    job_id = request.job_id
    job_dir = os.path.join(settings.TEMP_DIR, job_id)
    
    try:

        # Job picked up from queue - validate first before changing status
        if not os.path.exists(job_dir):
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"error": "Uploaded audio not found"})
            return
            
        allowed_audio_ext = set(settings.ALLOWED_AUDIO_FORMATS)
        audio_candidates = [f for f in os.listdir(job_dir) if f.lower().split('.')[-1] in {e.lstrip('.').lower() for e in allowed_audio_ext}]
        
        if not audio_candidates:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"error": "No audio file found in upload folder"})
            return
            
        audio_path = os.path.join(job_dir, audio_candidates[0])
        
        
            
        # Use R2Service's proper path generation for consistency
        r2_key = r2_service.generate_file_path(job_id, "", f"{job_id}.wav")
        r2_audio_path = r2_service.upload_file(audio_path, r2_key)
        if not r2_audio_path["success"]:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"error": f"Audio upload failed: {r2_audio_path.get('error')}"})
            return
        try:
            # Update to 25% when separation starts
            _update_status_non_blocking(job_id, ProcessingStatus.SEPARATING, 25, {"message": "Starting audio separation...", "phase": "separation"})
            
            request_id = runpod_service.submit_separation_request(r2_audio_path["url"], job_id)
            
            def on_separation_progress(status: str, progress: int):
                # Map RunPod progress (0-100) to separation range (25-45) as per new spec
                separation_progress = 25 + int((progress / 100.0) * 20)  # 25-45% range
                
                # Add debug logging to understand backward transitions
                logger.info(f"Separation progress update: RunPod {progress}% → App {separation_progress}% (status: {status})")
                
                _update_status_non_blocking(job_id, ProcessingStatus.SEPARATING, separation_progress, {
                    "message": f"Audio separation in progress... ({status})",
                    "phase": "separation"
                })
            
            def on_separation_failed():
                _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {
                    "message": "Job failed by RunPod", "error": "Job failed by RunPod"
                })
            
            logger.info(f"Starting separation monitoring for RunPod job {request_id}")
            monitor_result = monitor_runpod_job(
                runpod_request_id=request_id,
                job_id=job_id,
                timeout_seconds=600,
                on_progress=on_separation_progress,
                on_failed=on_separation_failed
            )
            logger.info(f"Separation monitoring completed with result: {monitor_result.get('status')}")
            
            if not monitor_result["success"]:
                if monitor_result["status"] == "FAILED":
                    return
                else:
                    error_msg = monitor_result.get("error", "Audio separation failed")
                    _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {
                        "message": "Audio separation failed.", "error": error_msg
                    })
                    return
            
            status = {"output": monitor_result.get("output", {})}
            
        except Exception as e:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {
                "error": f"Audio separation job submission failed: {str(e)}"
            })
            return
        output = status.get('output', {})
        runpod_urls = separation_utils.extract_urls_from_clearvocals_response(output)
        def on_download_error(message, error):
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {
                "message": message,
                "error": error
            })

        download_success, file_paths = separation_utils.download_separation_files(
            job_id=job_id,
            job_dir=job_dir,
            runpod_urls=runpod_urls,
            on_error_callback=on_download_error
        )

        if not download_success:
            return
            
            
        # This update happens AFTER separation monitor completes
        logger.info(f"Separation completed successfully - now starting transcription phase for job {job_id}")
        _update_status_non_blocking(job_id, ProcessingStatus.TRANSCRIBING, 45, {"message": "Separation completed - starting transcription...", "phase": "transcription", "runpod_urls": runpod_urls})

        # Clean up original audio file after successful separation
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Deleted original audio file after successful separation: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to delete original audio file {audio_path}: {e}")
        

        simple_dubbed_api = get_simple_dubbed_api()
     
        logger.info(f"📝 Starting SimpleDubbedAPI.process_dubbed_audio for job {job_id}")
        pipeline_result = simple_dubbed_api.process_dubbed_audio(
            job_id=job_id,
            target_language=request.target_language,
            source_video_language=request.source_video_language,
            output_dir=job_dir,
            review_mode=getattr(request, "humanReview", False)
        )
        
        if not pipeline_result["success"]:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"message": "Dubbing pipeline failed.", "error": pipeline_result.get("error"), "details": pipeline_result.get("details")})
            if r2_audio_path.get("r2_key"):
                r2_service.delete_file(r2_audio_path["r2_key"])
            return

        if getattr(request, "humanReview", False):
            # Rely on pipeline to set awaiting_review and manifest details
            if r2_audio_path.get("r2_key"):
                r2_service.delete_file(r2_audio_path["r2_key"])
            return

        result_url = pipeline_result.get("result_url") or (pipeline_result.get("result_urls", {}) or {}).get("final_video")
        
        folder_upload = pipeline_result.get("folder_upload", {})

        # Use folder_upload as returned from pipeline (already contains uploaded files)
        
        _update_status_non_blocking(job_id, ProcessingStatus.COMPLETED, 100, {
            "message": "Video dubbing completed.", 
            "result_url": result_url, 
            "details": pipeline_result.get("details"),
            "folder_upload": folder_upload,
            "result_urls": pipeline_result.get("result_urls"),
            "video_upload": pipeline_result.get("video_upload")
        })
        
        # Memory cleanup after processing completion
        del pipeline_result, folder_upload, runpod_urls
        gc.collect()
        
        
        # Complete credit billing using centralized utility (sync context) - charge remaining 25%
        job_utils.complete_job_billing_sync(job_id, "dub", user_id, 0.25)
        
        if r2_audio_path.get("r2_key"):
            r2_service.delete_file(r2_audio_path["r2_key"])
        
        # Immediate cleanup for this specific completed job
        cleanup_utils.cleanup_job_comprehensive(job_id, "dub")

    except Exception as e:
        # If already in review flow, do not override status or cleanup/refund
        try:
            from app.utils.unified_status_manager import get_unified_status_manager, ProcessingStatus as PS
            mgr = get_unified_status_manager()
            current = mgr.get_status_sync(job_id, UnifiedJobType.DUB)
            in_review_flow = current and current.status in {PS.AWAITING_REVIEW, PS.REVIEWING}
        except Exception:
            in_review_flow = False

        if in_review_flow:
            logger.info(f"Job {job_id} is in review flow; skipping failure override and cleanup")
        else:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"message": f"Processing failed: {str(e)}", "error": str(e)})
            # Immediate cleanup for failed job
            cleanup_utils.cleanup_job_comprehensive(job_id, "dub")
            # Refund credits on failure (sync context)
            job_utils.refund_job_credits_sync(job_id, "dub", "job_failed")
        
        try:
            if r2_audio_path.get("r2_key"):
                r2_service.delete_file(r2_audio_path["r2_key"])
        except Exception:
            pass
    finally:
        try:
            logger.info(f"Preserving job directory with vocal/instrument files: {job_dir}")
        except Exception:
            pass

from app.services.dub.manifest_service import load_manifest as _load_manifest_json, ensure_job_dir as _ensure_job_dir

@router.post("/video-dub/{job_id}/approve")
async def approve_and_resume(job_id: str, _: dict = {}, current_user = Depends(get_current_user)):
    job = await dub_job_service.get_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Validate job status - must be awaiting_review to approve
    if job.status != "awaiting_review":
        raise HTTPException(
            status_code=400, 
            detail=f"Job cannot be approved. Current status: {job.status}. Only jobs with 'awaiting_review' status can be approved."
        )
    
    # Validate review_status if available
    review_status = job.review_status or (job.details or {}).get("review_status")
    if review_status and review_status not in ["awaiting", "in_progress"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job cannot be approved. Review status: {review_status}. Job may already be processed."
        )
    
    manifest_url = job.segments_manifest_url or (job.details or {}).get("segments_manifest_url")
    if not manifest_url:
        raise HTTPException(status_code=400, detail="No manifest available for this job")
    manifest = _load_manifest_json(manifest_url)
    
    # Kick off background resume
    executor = get_dub_executor()
    user_id = current_user.id  # Capture user_id for nested function
    def _resume():
        try:
            # Update status to reviewing with proper review_status
            _update_status_non_blocking(job_id, ProcessingStatus.REVIEWING, 80, {
                "message": "Resuming with human edits...",
                "review_status": "approved",
                "phase": "voice_cloning"
            })
    
            api = get_simple_dubbed_api()
            job_dir = _ensure_job_dir(job_id)
            
            
            result = api.process_dubbed_audio(
                job_id=job_id,
                target_language=manifest.get("target_language") or job.target_language,
                source_video_language=job.source_video_language,
                output_dir=job_dir,
                review_mode=False,
                manifest_override=manifest,
            )
            if not result.get("success"):
                _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {
                    "message": "Resume failed", 
                    "error": result.get("error"),
                    "review_status": "rejected"
                })
                return
            result_url = result.get("result_url") or (result.get("result_urls", {}) or {}).get("final_video")
            
            folder_upload = result.get("folder_upload", {})
            logger.info(f"📁 Folder upload contents for job {job_id}: {list(folder_upload.keys())}")
            
            _update_status_non_blocking(job_id, ProcessingStatus.COMPLETED, 100, {
                "message": "Dubbing completed after review.",
                "result_url": result_url,
                "result_urls": result.get("result_urls"),
                "folder_upload": folder_upload,
                "review_status": "completed"
            })
            
            # Memory cleanup after approve completion
            del result, folder_upload
            gc.collect()
            
            logger.info(f"✅ Job {job_id} completed after review")
            
            # Confirm credit usage
            # Complete credit billing using centralized utility (sync context) - charge remaining 25%
            job_utils.complete_job_billing_sync(job_id, "dub", user_id, 0.25)
            
            # ✅ Cleanup ONLY after resume is completely finished

            cleanup_utils.cleanup_job_comprehensive(job_id, "dub")
            
        except Exception as e:
            logger.error(f"Approval resume failed for job {job_id}: {str(e)}")
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {
                "message": f"Approval resume failed: {str(e)}",
                "review_status": "rejected",
                "error": str(e)
            })
            
            # Refund credits on failure
            # Refund credits for failed job (sync context)
            job_utils.refund_job_credits_sync(job_id, "dub", "approval_failed")
            
            # ❌ Cleanup after resume failure too

            cleanup_utils.cleanup_job_comprehensive(job_id, "dub")
    executor.submit(_resume)
    return {"success": True, "message": "Resume started", "job_id": job_id}

@router.post("/video-dub/{job_id}/redub", response_model=RedubResponse)
async def redub_job(job_id: str, request_body: RedubRequest, current_user = Depends(get_current_user)):
    """
    Create a redub job from existing job.
    Returns redub job ID that works with all existing endpoints.
    """
    # Get parent job for redub
    parent_job_id = job_id
    parent_job = await dub_job_service.get_job(parent_job_id)
    if not parent_job or parent_job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Parent job not found")
    
    # Validate job for redub using utility function
    validation_result = await job_utils.validate_job_for_redub(parent_job)
    if not validation_result["valid"]:
        raise HTTPException(status_code=400, detail=validation_result["message"])
    
    manifest_url = validation_result["manifest_url"]
    
    # Load and validate manifest
    try:
        manifest = _load_manifest_json(manifest_url)
        manifest_validation = job_utils.validate_manifest(manifest)
        if not manifest_validation["valid"]:
            raise HTTPException(status_code=400, detail=manifest_validation["message"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load manifest: {str(e)}")
    
    # Generate redub job ID

    r2_service = get_r2_service()
    redub_job_id = r2_service.generate_job_id()

    # Get job details for redub
    user_id = current_user.id
    duration = parent_job.duration or (parent_job.details or {}).get("duration", 0)

    if not duration or duration <= 0:
        raise HTTPException(
            status_code=400,
            detail="Parent job missing duration information. Cannot proceed with redub."
        )

    # Create redub job data
    redub_job_data = {
        "job_id": redub_job_id,
        "user_id": user_id,
        "target_language": request_body.target_language,
        "original_filename": f"Redub - {parent_job.original_filename}",
        "source_video_language": parent_job.source_video_language,
        "duration": duration,
        "status": "pending",
        "progress": 0,
        "parent_job_id": parent_job_id,
        "redub_from": parent_job.target_language
    }
    
    # Atomic credit reservation + job creation
    result = await credit_service.reserve_credits_and_create_job(
        user_id=user_id,
        job_data=redub_job_data,
        job_type=CreditJobType.DUB,
        duration_seconds=duration
    )
    
    if not result["success"]:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Insufficient credits for redub",
                "error": result.get("error", "Credit reservation failed")
            }
        )
    
    # Setup job directory using utility function
    try:
        redub_job_dir = await job_utils.setup_job_directory(parent_job_id, redub_job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Prepare manifest for redub using utility function
    manifest = job_utils.prepare_manifest_for_redub(
        manifest, redub_job_id, request_body.target_language, parent_job_id
    )
    
    # Create redub request similar to original dub request
    try:
        redub_request = VideoDubRequest(
            job_id=redub_job_id,
            target_language=request_body.target_language,
            project_title=f"Redub - {parent_job.original_filename}",
            duration=duration,
            source_video_language=parent_job.source_video_language,
            humanReview=getattr(request_body, "humanReview", False)
        )
        logger.info(f"Created redub request: {redub_request.dict()}")
    except Exception as e:
        logger.error(f"Failed to create redub request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid redub request parameters: {str(e)}")
    
    # Start redub processing using existing pipeline
    from app.services.dub.simple_dubbed_api import get_simple_dubbed_api
    api = get_simple_dubbed_api()

    def _run_redub():  # ← Sync function like existing pattern
        try:
            _update_status_non_blocking(redub_job_id, ProcessingStatus.PROCESSING, 45, {"message": f"Redubbing to {request_body.target_language}", "phase": "initialization"})

            result = api.process_dubbed_audio(
                job_id=redub_job_id,
                target_language=request_body.target_language,
                source_video_language=parent_job.source_video_language,
                output_dir=redub_job_dir,
                review_mode=bool(getattr(request_body, "humanReview", False)),
                manifest_override=manifest,
            )
            
            if not result["success"]:
                _update_status_non_blocking(redub_job_id, ProcessingStatus.FAILED, 0, {
                    "message": "Redub failed",
                    "error": result.get("error")
                })
                return
            
            # Handle review mode or completion
            if getattr(request_body, "humanReview", False):
                if result.get("review"):
                    logger.info(f"✅ Redub job {redub_job_id} reached awaiting_review")
                    return
            
            # Complete redub
            result_url = result.get("result_url") or (result.get("result_urls", {}) or {}).get("final_video")
            folder_upload = result.get("folder_upload", {})
            

            _update_status_non_blocking(redub_job_id, ProcessingStatus.COMPLETED, 100, {
                "message": "Redub completed successfully",
                "result_url": result_url,
                "details": result.get("details"),
                "folder_upload": folder_upload,
                "result_urls": result.get("result_urls"),
                "parent_job_id": parent_job_id
            })
            
            logger.info(f"✅ Redub job {redub_job_id} completed")
            
            # Complete credit billing using centralized utility (sync context for redub) - charge remaining 25%
            job_utils.complete_job_billing_sync(redub_job_id, "dub", user_id, 0.25)

            # Immediate cleanup for this specific completed redub job

            video_download_service.cleanup_specific_job(redub_job_id)
                
        except Exception as e:
            logger.error(f"Redub processing failed: {str(e)}")
            _update_status_non_blocking(redub_job_id, ProcessingStatus.FAILED, 0, {
                "message": f"Redub failed: {str(e)}",
                "error": str(e),
                "parent_job_id": parent_job_id
            })
            
            # Refund credits on failure
            try:
                # Note: For redub failures, we use a simple approach - just log and cleanup
                # Complex refund logic can cause additional event loop issues
                logger.info(f"Redub job {redub_job_id} failed, cleaning up resources")
            except Exception:
                pass
    
    # Enqueue redub job with proper background runner (like existing dub API)
    _dub_queue_manager.enqueue(redub_job_id, _run_redub)

    logger.info(f"Started redub job {redub_job_id} from parent {parent_job_id} for user {user_id}")
    
    return RedubResponse(
        success=True,
        message="Redub job created successfully",
        job_id=redub_job_id,
        status="started",
        details={
            "parent_job_id": parent_job_id,
            "redub_job_id": redub_job_id,
            "target_language": request_body.target_language,
            "status_check_url": f"/api/video-dub-status/{redub_job_id}"
        }
    )

@router.post("/video-dub/{job_id}/share")
async def generate_share_link(
    job_id: str,
    expires_in_hours: int = 24,
    current_user = Depends(get_current_user)
):
    """Generate a shareable link for review access"""
    
    # Verify job exists and user owns it
    job = await dub_job_service.get_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Only allow sharing for review-ready jobs
    if job.status != "awaiting_review":
        raise HTTPException(
            status_code=400, 
            detail=f"Job must be in 'awaiting_review' status to share. Current status: {job.status}"
        )
    
    # Generate share token

    
    share_token = generate_url_safe_token(32)  # 32-byte token
    expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
    
    # Store token in database
    share_tokens_collection = db["share_tokens"]
    await share_tokens_collection.insert_one({
        "token": share_token,
        "job_id": job_id,
        "user_id": current_user.id,
        "expires_at": expires_at,
        "created_at": datetime.now(timezone.utc)
    })
    
    return {
        "success": True,
        "token": share_token,
        "expires_at": expires_at.isoformat(),
        "expires_in_hours": expires_in_hours
    }
