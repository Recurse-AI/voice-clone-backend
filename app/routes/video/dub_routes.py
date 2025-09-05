from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
import os
import gc
from typing import Optional
from datetime import datetime, timezone, timedelta
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
from app.services.simple_status_service import status_service, JobStatus
from app.config.credit_constants import JobType as CreditJobType
from app.utils.runpod_service import runpod_service
from app.utils.runpod_monitor import monitor_runpod_job

from app.utils.job_utils import job_utils
from app.queue.queue_manager import get_dub_queue
from app.utils.cleanup_utils import cleanup_utils
from app.utils.video_downloader import video_download_service
from app.queue.queue_manager import queue_manager
from app.utils.token_helper import generate_url_safe_token
from app.utils.separation_utils import separation_utils
from app.config.database import db


router = APIRouter()

logger = logging.getLogger(__name__)


def _update_status_non_blocking(job_id: str, status: JobStatus, progress: int, details: dict):
    """Update dub job status using simple status service"""
    try:
        status_service.update_status(job_id, "dub", status, progress, details)
    except Exception as e:
        logger.error(f"Failed to update status for {job_id}: {e}")


def enqueue_dub_job(request: VideoDubRequest, user_id: str) -> bool:
    """Enqueue dub job using queue manager"""
    logger.info(f"🚀 ENQUEUING DUB JOB: {request.job_id}")
    success = queue_manager.enqueue_dub_task(request.dict(), user_id)
    if success:
        logger.info(f"✅ DUB JOB ENQUEUED: {request.job_id}")
    else:
        logger.error(f"❌ FAILED TO ENQUEUE: {request.job_id}")
    return success

def get_dub_queue_position(job_id: str) -> Optional[int]:
    return None


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
        
        # Set initial status
        status_service.update_status(
            request.job_id, "dub", JobStatus.PENDING, 0,
            {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "phase": "queued",
                "message": "Job queued for processing"
            }
        )
        logger.info(f"✅ DUB JOB CREATED: {request.job_id} - Status: PENDING")
        
        # Enqueue job for background processing
        success = enqueue_dub_job(request, user_id)
        if not success:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to enqueue job",
                    "error": "Queue system unavailable"
                }
            )
        
        logger.info(f"Started video dub job {request.job_id} for user {user_id}")
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

        
        # Get status from simple service
        status_data = status_service.get_status(job_id, "dub")
        
        if not status_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
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

        return VideoDubStatusResponse(
            job_id=job_id,
            status=status_data["status"],
            progress=status_data["progress"],
            message=status_data.get("details", {}).get("message", f"Status: {status_data['status']}"),
            result_url=result_url,
            error=error,
            details={
                "files": files,
                "updated_at": status_data["updated_at"].isoformat() if status_data.get("updated_at") else None,
                "phase": status_data.get("details", {}).get("phase")
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dub job status {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

## Removed per requirement: clients should use details.files only


from app.services.dub.manifest_service import load_manifest as _load_manifest_json, ensure_job_dir as _ensure_job_dir

def _resume_approved_job(job_id: str, manifest: dict, target_language: str, source_video_language: str, user_id: str):
    """Module-level function to resume approved job for RQ compatibility"""
    try:
        # Update status message to show processing is continuing
        _update_status_non_blocking(job_id, JobStatus.REVIEWING, 80, {
            "message": "Processing approved edits...",
            "review_status": "approved",
            "phase": "voice_cloning"
        })

        api = get_simple_dubbed_api()
        job_dir = _ensure_job_dir(job_id)
        
        # Download missing files before processing
        logger.info(f"Checking for missing files in job directory: {job_dir}")
        try:
            api._download_missing_files(job_id, manifest, job_dir)
            logger.info(f"✅ Missing files check completed for job {job_id}")
        except Exception as e:
            logger.warning(f"⚠️ Missing files download failed for job {job_id}: {e}")
        
        result = api.process_dubbed_audio(
            job_id=job_id,
            target_language=target_language,
            source_video_language=source_video_language,
            output_dir=job_dir,
            review_mode=False,
            manifest_override=manifest,
        )
        if not result.get("success"):
            _update_status_non_blocking(job_id, JobStatus.FAILED, 0, {
                "message": "Resume failed", 
                "error": result.get("error"),
                "review_status": "rejected"
            })
            return

        result_url = result.get("result_url") or (result.get("result_urls", {}) or {}).get("final_video")
        
        folder_upload = result.get("folder_upload", {})
        logger.info(f"📁 Folder upload contents for job {job_id}: {list(folder_upload.keys())}")
        
        _update_status_non_blocking(job_id, JobStatus.COMPLETED, 100, {
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
        _update_status_non_blocking(job_id, JobStatus.FAILED, 0, {
            "message": f"Approval resume failed: {str(e)}",
            "review_status": "rejected",
            "error": str(e)
        })
        
        # Refund credits on failure
        # Refund credits for failed job (sync context)
        job_utils.refund_job_credits_sync(job_id, "dub", "approval_failed")
        
        # ❌ Cleanup after resume failure too
        cleanup_utils.cleanup_job_comprehensive(job_id, "dub")

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
    
    # Immediately update status to reviewing after approve
    _update_status_non_blocking(job_id, JobStatus.REVIEWING, 80, {
        "message": "Approved! Starting review processing...",
        "review_status": "approved",
        "phase": "voice_cloning"
    })
    
    # Enqueue background resume task
    dub_queue = get_dub_queue()
    dub_queue.enqueue(
        _resume_approved_job,
        job_id,
        manifest,
        manifest.get("target_language") or job.target_language,
        job.source_video_language,
        current_user.id
    )
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
            _update_status_non_blocking(redub_job_id, JobStatus.PROCESSING, 45, {"message": f"Redubbing to {request_body.target_language}", "phase": "initialization"})

            result = api.process_dubbed_audio(
                job_id=redub_job_id,
                target_language=request_body.target_language,
                source_video_language=parent_job.source_video_language,
                output_dir=redub_job_dir,
                review_mode=bool(getattr(request_body, "humanReview", False)),
                manifest_override=manifest,
            )
            
            if not result["success"]:
                _update_status_non_blocking(redub_job_id, JobStatus.FAILED, 0, {
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
            

            _update_status_non_blocking(redub_job_id, JobStatus.COMPLETED, 100, {
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
            _update_status_non_blocking(redub_job_id, JobStatus.FAILED, 0, {
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
    
    # Enqueue redub job
    from app.queue.dub_tasks import process_redub_task
    dub_queue = get_dub_queue()
    dub_queue.enqueue(process_redub_task, redub_job_id, request_body.target_language, 
                     parent_job.source_video_language, redub_job_dir, manifest, 
                     bool(getattr(request_body, "humanReview", False)))

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
