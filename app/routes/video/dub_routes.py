from fastapi import APIRouter, HTTPException, Depends, Form, File, UploadFile
from fastapi.responses import JSONResponse
import logging
import gc
import uuid
import os
from typing import Optional
from datetime import datetime, timezone, timedelta

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
from app.services.dub.simple_dubbed_api import get_simple_dubbed_api
from app.services.job_response_service import job_response_service
from app.services.simple_status_service import status_service, JobStatus
from app.services.status_api_service import api_status_service
from app.config.credit_constants import JobType as CreditJobType

from app.utils.job_utils import job_utils
from app.queue.queue_manager import get_dub_queue
from app.queue.queue_manager import queue_manager
from app.utils.cleanup_utils import cleanup_utils
from app.utils.token_helper import generate_url_safe_token
from app.config.database import db
from app.config.pipeline_settings import pipeline_settings


router = APIRouter()

logger = logging.getLogger(__name__)


def safe_isoformat(value):
    """Safely convert datetime or string to ISO format string"""
    if value is None:
        return None
    if isinstance(value, str):
        return value  # Already a string, assume it's in ISO format
    if hasattr(value, 'isoformat'):
        return value.isoformat()  # DateTime object
    return str(value)  # Fallback to string conversion




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
    try:
        q = get_dub_queue()
        if not q:
            return None
        # Try to find exact position by scanning queued jobs' first arg (request_dict)
        try:
            jobs = list(q.jobs)
        except Exception:
            # Fallback to queue length if jobs cannot be loaded
            return len(q)
        for idx, job in enumerate(jobs):
            try:
                req = job.args[0] if getattr(job, "args", None) else None
                if isinstance(req, dict) and req.get("job_id") == job_id:
                    return idx
            except Exception:
                continue
        return None
    except Exception:
        return None


@router.post("/video-dub", response_model=VideoDubResponse)
async def start_video_dub(
    request: str = Form(..., description="JSON string of VideoDubRequest"),
    srt_file: Optional[UploadFile] = File(None, description="SRT subtitle file"),
    current_user = Depends(get_current_user)
):
    try:
        # Parse and validate the JSON request
        import json
        try:
            request_data = json.loads(request)
            logger.info(f"📝 Received request data: {request_data}")
            request_obj = VideoDubRequest(**request_data)
            logger.info(f"✅ Successfully parsed VideoDubRequest: {request_obj}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request field")
        except Exception as e:
            logger.error(f"❌ Validation error: {e} | Request data: {request_data}")
            raise HTTPException(status_code=422, detail=f"Invalid request data: {str(e)}")
        
        user_id = current_user.id
        if request_obj.duration is None or request_obj.duration <= 0:
            raise HTTPException(status_code=400, detail="Duration is required and must be greater than 0 seconds")
        
        # Handle SRT file when video_subtitle is true
        if request_obj.video_subtitle:
            from app.config.settings import settings
            job_dir = os.path.join(settings.TEMP_DIR, request_obj.job_id)
            srt_path = os.path.join(job_dir, f"{request_obj.job_id}.srt")
            
            if srt_file:
                # SRT file provided in this request
                if not srt_file.filename.lower().endswith('.srt'):
                    raise HTTPException(status_code=400, detail="Only SRT files are supported")
                
                os.makedirs(job_dir, exist_ok=True)
                with open(srt_path, "wb") as buffer:
                    content = await srt_file.read()
                    buffer.write(content)
            else:
                # Check if SRT file was uploaded previously via /upload-file
                if not os.path.exists(srt_path):
                    raise HTTPException(
                        status_code=400, 
                        detail="SRT file is required when video_subtitle is true. Upload via /upload-file endpoint first or include in this request."
                    )
        
        job_data = {
            "job_id": request_obj.job_id,
            "user_id": user_id,
            "target_language": request_obj.target_language,
            "original_filename": request_obj.project_title,
            "source_video_language": request_obj.source_video_language,
            "duration": request_obj.duration,
            "status": "pending",
            "progress": 0,
            "video_subtitle": request_obj.video_subtitle
        }
        

        result = await credit_service.reserve_credits_and_create_job(
            user_id=user_id,
            job_data=job_data,
            job_type=CreditJobType.DUB,
            duration_seconds=request_obj.duration
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
            request_obj.job_id, "dub", JobStatus.PENDING, 0,
            {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "phase": "queued",
                "message": "Job queued for processing"
            }
        )
        logger.info(f"✅ DUB JOB CREATED: {request_obj.job_id} - Status: PENDING")
        
        # Enqueue job for background processing
        success = enqueue_dub_job(request_obj, user_id)
        if not success:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to enqueue job",
                    "error": "Queue system unavailable"
                }
            )
        
        logger.info(f"Started video dub job {request_obj.job_id} for user {user_id}")
        return VideoDubResponse(
            success=True,
            message="Video dub started successfully",
            job_id=request_obj.job_id,
            status_check_url=f"/api/video-dub-status/{request_obj.job_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start video dub: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start video dubbing: {str(e)}")

@router.get("/video-dub-status/{job_id}", response_model=VideoDubStatusResponse)
async def get_video_dub_status(job_id: str):
    try:

        
        # Get latest status from API service
        status_data = api_status_service.get_job_status(job_id, "dub")
        
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
            error = status_data.get("details", {}).get("error")

        return VideoDubStatusResponse(
            job_id=job_id,
            status=status_data["status"],
            progress=status_data["progress"],
            message=status_data.get("details", {}).get("message", f"Status: {status_data['status']}"),
            result_url=result_url,
            error=error,
            details={
                "files": files,
                "updated_at": safe_isoformat(status_data.get("updated_at")),
                "phase": status_data.get("details", {}).get("phase")
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dub job status {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

## Removed per requirement: clients should use details.files only


from app.services.dub.manifest_manager import manifest_manager
from app.services.dub.manifest_service import ensure_job_dir as _ensure_job_dir

def _resume_approved_job(job_id: str, manifest: dict, target_language: str, source_video_language: str, user_id: str):
    try:
        from app.utils.pipeline_utils import mark_resume_job, mark_dub_job_active, mark_dub_job_inactive
        
        mark_resume_job(job_id)
        # Do not pre-mark voice_cloning to avoid occupying the slot prematurely
        mark_dub_job_active(job_id, "review_prep")
        
        status_service.update_status(job_id, "dub", JobStatus.REVIEWING, 80, {
            "message": "Processing approved edits...",
            "review_status": "approved",
            "phase": "reviewing"
        })

        # FAST PATH: Reuse existing services, minimal initialization
        import time
        setup_start = time.time()
        
        api = get_simple_dubbed_api()  # Reuses existing singleton
        job_dir = _ensure_job_dir(job_id)
        
        setup_time = time.time() - setup_start
        logger.info(f"⚡ Fast resume setup completed in {setup_time:.2f}s for {job_id}")
        
        # Validate manifest has required URLs before proceeding
        manifest_validation = job_utils.validate_manifest(manifest)
        if not manifest_validation["valid"]:
            logger.error(f"Resume failed - invalid manifest: {manifest_validation['message']}")
            status_service.update_status(job_id, "dub", JobStatus.FAILED, 0, {
                "message": "Resume failed - invalid manifest",
                "error": manifest_validation["message"],
                "review_status": "rejected"
            })
            return

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
            status_service.update_status(job_id, "dub", JobStatus.FAILED, 0, {
                "message": "Resume failed", 
                "error": result.get("error"),
                "review_status": "rejected"
            })
            return

        result_url = result.get("result_url") or (result.get("result_urls", {}) or {}).get("final_video")

        folder_upload = result.get("folder_upload", {})
        logger.info(f"📁 Folder upload contents for job {job_id}: {list(folder_upload.keys())}")

        # Use dub_service.complete_job to ensure email notification
        from app.services.dub_service import dub_service
        completion_details = {
            "folder_upload": folder_upload,
            "result_urls": result.get("result_urls"),
            "review_status": "completed"
        }

        # Extract manifest URL from pipeline result
        manifest_url = result.get("manifest_url")
        manifest_key = result.get("manifest_key")
        if manifest_url:
            completion_details["segments_manifest_url"] = manifest_url
        if manifest_key:
            completion_details["segments_manifest_key"] = manifest_key
        
        success = dub_service.complete_job(job_id, result_url, completion_details, credit_percentage=0.25)
        if not success:
            logger.error(f"Failed to complete dub job {job_id} after review")
            # Fallback billing if complete_job fails
            job_utils.complete_job_billing_sync(job_id, "dub", user_id, 0.25)
        
        # Update status with review-specific message before cleanup
        status_service.update_status(job_id, "dub", JobStatus.COMPLETED, 100, {
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
        
        cleanup_utils.cleanup_job_comprehensive(job_id, "dub")
        
    except Exception as e:
        logger.error(f"Approval resume failed for job {job_id}: {str(e)}")
        status_service.update_status(job_id, "dub", JobStatus.FAILED, 0, {
            "message": f"Approval resume failed: {str(e)}",
            "review_status": "rejected",
            "error": str(e)
        })
        
        job_utils.refund_job_credits_sync(job_id, "dub", "approval_failed")
        cleanup_utils.cleanup_job_comprehensive(job_id, "dub")
    finally:
        from app.utils.pipeline_utils import remove_resume_job
        mark_dub_job_inactive(job_id)
        remove_resume_job(job_id)

@router.post("/video-dub/{job_id}/approve")
async def approve_and_resume(job_id: str, _: dict = {}, current_user = Depends(get_current_user)):
    job = await dub_job_service.get_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    manifest_url = job.segments_manifest_url or (job.details or {}).get("segments_manifest_url")
    if not manifest_url:
        raise HTTPException(status_code=400, detail="No manifest available for this job")

    manifest = manifest_manager.load_manifest(manifest_url)
    manifest = manifest_manager._normalize_manifest(manifest)
    
    # Immediately update status to reviewing after approve
    status_service.update_status(job_id, "dub", JobStatus.REVIEWING, 80, {
        "message": "Approved! Starting review processing...",
        "review_status": "approved",
        "phase": "voice_cloning"
    })
    
    # Enqueue resume task with improved logging
    dub_queue = get_dub_queue()
    
    # Immediate enqueue for faster pickup (dedicated resume worker available)
    job = dub_queue.enqueue(
        _resume_approved_job,
        job_id,
        manifest,
        manifest.get("target_language") or job.target_language,
        job.source_video_language,
        current_user.id,
        job_timeout=pipeline_settings.JOB_TIMEOUT
    )
    
    logger.info(f"🚀 FAST RESUME: Job {job_id} enqueued for instant processing (RQ job: {job.id})")
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
        manifest = manifest_manager.load_manifest(manifest_url)
        manifest = manifest_manager._normalize_manifest(manifest)
        manifest_validation = job_utils.validate_manifest(manifest)
        if not manifest_validation["valid"]:
            raise HTTPException(status_code=400, detail=manifest_validation["message"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load manifest: {str(e)}")
    
    # Generate redub job ID with consistent "dub_" prefix
    redub_job_id = f"dub_{uuid.uuid4()}"

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
