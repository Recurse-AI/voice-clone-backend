from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
import os
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
from app.config.credit_constants import JobType
from app.utils.unified_status_manager import ProcessingStatus
from app.utils.runpod_url_manager import RunPodURLManager
from app.utils.job_utils import job_utils
import asyncio
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import uuid
from datetime import datetime, timezone, timedelta
import shutil
from pathlib import Path
from app.config.settings import settings

router = APIRouter()

logger = logging.getLogger(__name__)

from app.services.dub.queue_manager import get_dub_queue_manager

# Keep workers consistent across executor and scheduler (max concurrent running jobs)
DUB_MAX_WORKERS = 10

# Singleton queue manager instance
_dub_queue_manager = get_dub_queue_manager(max_concurrency=DUB_MAX_WORKERS)

def get_dub_executor():
    """Backward-compat: expose executor (not used directly)."""
    return _dub_queue_manager._get_executor()

def _update_status_non_blocking(job_id: str, status: ProcessingStatus, progress: int, details: dict, job_type: str = "dub"):
    """Update job status using unified status manager"""
    from app.utils.unified_status_manager import get_unified_status_manager
    from app.utils.unified_status_manager import JobType as UnifiedJobType
    
    manager = get_unified_status_manager()
    job_type_enum = UnifiedJobType.DUB if job_type == "dub" else UnifiedJobType.SEPARATION
    
    # Use sync version to avoid event loop issues
    try:
        manager.update_status_sync(job_id, job_type_enum, status, progress, details)

    except Exception as e:
        logger.error(f"Failed to update status for {job_id}: {e}")

def _ensure_resume_files_available(job_id: str, job_dir: str, manifest: dict) -> None:
    """Smart resume: Check and download missing files from manifest if needed"""
    try:
        missing_files = []
        segments = manifest.get("segments", [])
        
        # Check which original audio files are missing
        for seg in segments:
            original_audio_file = seg.get("original_audio_file")
            if original_audio_file:
                file_path = os.path.join(job_dir, original_audio_file)
                if not os.path.exists(file_path):
                    original_url = seg.get("original_audio_url")
                    if original_url:
                        missing_files.append({
                            "filename": original_audio_file,
                            "url": original_url,
                            "path": file_path
                        })
        
        # Download missing files if any are missing
        if missing_files:
            logger.info(f"📥 Resume: {len(missing_files)} files missing for job {job_id}, downloading...")
            
            import requests
            for file_info in missing_files:
                try:
                    resp = requests.get(file_info["url"], timeout=60)
                    resp.raise_for_status()
                    with open(file_info["path"], 'wb') as f:
                        f.write(resp.content)
                    logger.info(f"Downloaded: {file_info['filename']}")
                except Exception as e:
                    logger.warning(f"Failed to download {file_info['filename']}: {e}")
                    
            logger.info(f"Resume files preparation complete for job {job_id}")
        else:
            logger.info(f"Resume: All files already available for job {job_id}")
            
    except Exception as e:
        logger.warning(f"Resume files check failed for job {job_id}: {e}")

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
        
        # Atomic credit reservation + job creation
        result = await credit_service.reserve_credits_and_create_job(
            user_id=user_id,
            job_data=job_data,
            job_type=JobType.DUB,
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
        from app.utils.unified_status_manager import get_unified_status_manager, ProcessingStatus
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
        from app.utils.unified_status_manager import get_unified_status_manager
        from app.utils.unified_status_manager import JobType as UnifiedJobType
        
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
            from app.services.job_response_service import job_response_service
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
    from app.services.dub.audio_utils import AudioUtils
    from app.config.settings import settings
    from app.services.r2_service import get_r2_service
    from app.utils.shared_memory import is_job_cancelled
    
    r2_service = get_r2_service()
    job_id = request.job_id
    job_dir = os.path.join(settings.TEMP_DIR, f"dub_{job_id}")
    
    try:
        # Check cancellation first
        if is_job_cancelled(job_id):
            _update_status_non_blocking(job_id, ProcessingStatus.CANCELLED, 0, {
                "message": "Job cancelled by user", 
                "error": "Job cancelled by user"
            }, "dub")
            return

        # Job picked up from queue - validate first before changing status
        if not os.path.exists(job_dir):
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"error": "Uploaded audio not found"}, "dub")
            return
            
        allowed_audio_ext = set(settings.ALLOWED_AUDIO_FORMATS)
        audio_candidates = [f for f in os.listdir(job_dir) if f.lower().split('.')[-1] in {e.lstrip('.').lower() for e in allowed_audio_ext}]
        
        if not audio_candidates:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"error": "No audio file found in upload folder"}, "dub")
            return
            
        audio_path = os.path.join(job_dir, audio_candidates[0])
        
        # NOW transition from PENDING to PROCESSING - validation successful
        _update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 5, {
            "message": "Starting dubbing process...", 
            "phase": "initialization"
        }, "dub")
        
        if is_job_cancelled(job_id):
            _update_status_non_blocking(job_id, ProcessingStatus.CANCELLED, 0, {
                "message": "Job cancelled by user", 
                "error": "Job cancelled by user"
            }, "dub")
            return
            
        _update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 25, {"message": "Preparing audio for separation...", "phase": "separation"}, "dub")
        # Use R2Service's proper path generation for consistency
        r2_key = r2_service.generate_file_path(job_id, "", f"{job_id}.wav")
        r2_audio_path = r2_service.upload_file(audio_path, r2_key)
        if not r2_audio_path["success"]:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"error": f"Audio upload failed: {r2_audio_path.get('error')}"}, "dub")
            return
        try:
            from app.utils.runpod_service import runpod_service
            from app.utils.runpod_monitor import monitor_runpod_job
            
            request_id = runpod_service.submit_separation_request(r2_audio_path["url"], f"video_dub_{job_id}")
            
            def on_separation_progress(status: str, progress: int):
                # Map RunPod progress (0-100) to separation range (30-45) to fix overlap with transcription
                separation_progress = 30 + int((progress / 100.0) * 15)  # 30-45% range
                
                # Add debug logging to understand backward transitions
                logger.info(f"Separation progress update: RunPod {progress}% → App {separation_progress}% (status: {status})")
                
                _update_status_non_blocking(job_id, ProcessingStatus.SEPARATING, separation_progress, {
                    "message": f"Audio separation in progress... ({status})",
                    "phase": "separation"
                }, "dub")
            
            def on_separation_cancelled():
                _update_status_non_blocking(job_id, ProcessingStatus.CANCELLED, 0, {
                    "message": "Job cancelled by user", "error": "Job cancelled by user"
                }, "dub")
            
            logger.info(f"🔄 Starting separation monitoring for RunPod job {request_id}")
            monitor_result = monitor_runpod_job(
                runpod_request_id=request_id,
                job_id=job_id,
                timeout_seconds=300,
                on_progress=on_separation_progress,
                on_cancelled=on_separation_cancelled
            )
            logger.info(f"✅ Separation monitoring completed with result: {monitor_result.get('status')}")
            
            if not monitor_result["success"]:
                if monitor_result["status"] == "CANCELLED":
                    return
                else:
                    error_msg = monitor_result.get("error", "Audio separation failed")
                    _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {
                        "message": "Audio separation failed.", "error": error_msg
                    }, "dub")
                    return
            
            status = {"output": monitor_result.get("output", {})}
            
        except Exception as e:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {
                "error": f"Audio separation job submission failed: {str(e)}"
            }, "dub")
            return
        runpod_urls = RunPodURLManager.extract_urls_from_runpod_response(status['output'])
        is_valid, error_msg = RunPodURLManager.validate_urls(runpod_urls)
        if not is_valid:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {
                "message": "Invalid RunPod URLs received", 
                "error": error_msg
            }, "dub")
            return
        
        vocal_path = os.path.join(job_dir, f"vocals_{job_id}.wav")
        instrument_path = os.path.join(job_dir, f"instruments_{job_id}.wav")
        
        if runpod_urls.get(RunPodURLManager.VOCALS_KEY):
            download_result = AudioUtils().download_audio_file(
                runpod_urls[RunPodURLManager.VOCALS_KEY], vocal_path
            )
            if not download_result["success"]:
                _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {
                    "message": "Vocal audio download failed.", 
                    "error": download_result.get("error")
                }, "dub")
                return
            
        if runpod_urls.get(RunPodURLManager.INSTRUMENTS_KEY):
            download_result = AudioUtils().download_audio_file(
                runpod_urls[RunPodURLManager.INSTRUMENTS_KEY], instrument_path
            )
            if not download_result["success"]:
                _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {
                    "message": "Instrument audio download failed.", 
                    "error": download_result.get("error")
                }, "dub")
                return
            
        if is_job_cancelled(job_id):
            _update_status_non_blocking(job_id, ProcessingStatus.CANCELLED, 0, {
                "message": "Job cancelled by user", 
                "error": "Job cancelled by user"
            }, "dub")
            return
            
        # Wait for separation to complete before starting transcription
        # This update happens AFTER separation monitor completes
        logger.info(f"🚀 Separation completed successfully - now starting transcription phase for job {job_id}")
        _update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 45, {"message": "Separation completed - starting transcription...", "phase": "transcription"}, "dub")
        
        from app.services.dub.simple_dubbed_api import get_simple_dubbed_api
        simple_dubbed_api = get_simple_dubbed_api()
        audio_url = r2_audio_path["url"]
        
        logger.info(f"📝 Starting SimpleDubbedAPI.process_dubbed_audio for job {job_id}")
        pipeline_result = simple_dubbed_api.process_dubbed_audio(
            job_id=job_id,
            audio_url=audio_url,
            target_language=request.target_language,
            source_video_language=request.source_video_language,
            output_dir=job_dir,
            review_mode=getattr(request, "humanReview", False)
        )
        
        if not pipeline_result["success"]:
            if is_job_cancelled(job_id):
                return
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"message": "Dubbing pipeline failed.", "error": pipeline_result.get("error"), "details": pipeline_result.get("details")}, "dub")
            if r2_audio_path.get("r2_key"):
                r2_service.delete_file(r2_audio_path["r2_key"])
            return

        if getattr(request, "humanReview", False):
            if not pipeline_result.get("review"):
                _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"message": "Review mode failed - no manifest generated", "error": "Failed to generate manifest for review"}, "dub")
                if r2_audio_path.get("r2_key"):
                    r2_service.delete_file(r2_audio_path["r2_key"])
                return
                
            review = pipeline_result["review"]
            if not review.get("segments_manifest_url"):
                _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"message": "Review mode failed - manifest URL missing", "error": "No manifest URL in review data"}, "dub")
                if r2_audio_path.get("r2_key"):
                    r2_service.delete_file(r2_audio_path["r2_key"])
                return
                
            details = {
                "duration": request.duration,
                "required_credits": None,
                "review_required": True,
                "review_status": "awaiting",
                "segments_manifest_url": review.get("segments_manifest_url"),
                "segments_manifest_key": review.get("segments_manifest_key"),
                "segments_count": review.get("segments_count"),
                "transcript_id": review.get("transcript_id"),
                "edited_segments_version": 0,
                "message": "Human review mode: segments ready for review"
            }
            details = RunPodURLManager.store_urls_in_details(details, runpod_urls)
            _update_status_non_blocking(job_id, ProcessingStatus.AWAITING_REVIEW, 80, details, "dub")
            
            from app.utils.shared_memory import unmark_job_cancelled
            unmark_job_cancelled(job_id)
            
            # ✅ Review mode complete - keep files for faster resume (will auto-cleanup later if needed)
            
            if r2_audio_path.get("r2_key"):
                r2_service.delete_file(r2_audio_path["r2_key"])
            return

        result_url = pipeline_result.get("result_url") or (pipeline_result.get("result_urls", {}) or {}).get("final_video")
        
        folder_upload = pipeline_result.get("folder_upload", {})
        folder_upload = RunPodURLManager.add_urls_to_folder_upload(folder_upload, runpod_urls, job_id)
        
        _update_status_non_blocking(job_id, ProcessingStatus.COMPLETED, 100, {
            "message": "Video dubbing completed.", 
            "result_url": result_url, 
            "details": pipeline_result.get("details"),
            "folder_upload": folder_upload,
            "result_urls": pipeline_result.get("result_urls"),
            "video_upload": pipeline_result.get("video_upload")
        }, "dub")
        
        # Memory cleanup after processing completion
        del pipeline_result, folder_upload, runpod_urls
        gc.collect()
        
        from app.utils.shared_memory import unmark_job_cancelled
        unmark_job_cancelled(job_id)
        
        # Complete credit billing using centralized utility (sync context)
        job_utils.complete_job_billing_sync(job_id, "dub", user_id)
        
        if r2_audio_path.get("r2_key"):
            r2_service.delete_file(r2_audio_path["r2_key"])
        
        # Immediate cleanup for this specific completed job
        from app.utils.video_downloader import video_download_service
        video_download_service.cleanup_specific_job(job_id)

    except Exception as e:
        if is_job_cancelled(job_id):
            logger.info(f"Job {job_id} was cancelled - not overriding with failed status on exception")
        else:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"message": f"Processing failed: {str(e)}", "error": str(e)}, "dub")
        
        # Immediate cleanup for failed job
        from app.utils.video_downloader import video_download_service
        video_download_service.cleanup_specific_job(job_id)
        
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
        # Initialize variable to avoid scoping issues
        stored_runpod_urls = {}
        try:
            # Retrieve URLs inside the function to avoid scoping issues
            stored_runpod_urls = RunPodURLManager.retrieve_urls_from_job(job)
            if not stored_runpod_urls:
                logger.warning(f"No stored RunPod URLs found for job {job_id}")
                stored_runpod_urls = {}  # Provide fallback
            else:
                logger.info(f"✅ Retrieved RunPod URLs for job {job_id}: {list(stored_runpod_urls.keys())}")
            # Update status to reviewing with proper review_status
            _update_status_non_blocking(job_id, ProcessingStatus.REVIEWING, 80, {
                "message": "Resuming with human edits...",
                "review_status": "approved",
                "phase": "voice_cloning"
            }, "dub")
            from app.services.dub.simple_dubbed_api import get_simple_dubbed_api
            api = get_simple_dubbed_api()
            job_dir = _ensure_job_dir(job_id)
            
            # ✅ Smart resume: Check if required files exist, download if missing
            _ensure_resume_files_available(job_id, job_dir, manifest)
            
            result = api.process_dubbed_audio(
                audio_url=None,
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
                }, "dub")
                return
            result_url = result.get("result_url") or (result.get("result_urls", {}) or {}).get("final_video")
            
            folder_upload = result.get("folder_upload", {})
            folder_upload = RunPodURLManager.add_urls_to_folder_upload(folder_upload, stored_runpod_urls, job_id)
            logger.info(f"📁 Folder upload contents for job {job_id}: {list(folder_upload.keys())}")
            
            _update_status_non_blocking(job_id, ProcessingStatus.COMPLETED, 100, {
                "message": "Dubbing completed after review.",
                "result_url": result_url,
                "result_urls": result.get("result_urls"),
                "folder_upload": folder_upload,
                "runpod_urls": stored_runpod_urls,
                "review_status": "completed"
            }, "dub")
            
            # Memory cleanup after approve completion
            del result, folder_upload, stored_runpod_urls
            gc.collect()
            
            # ✅ Successfully completed after review - safe to unmark cancellation
            from app.utils.shared_memory import unmark_job_cancelled
            unmark_job_cancelled(job_id)
            logger.info(f"✅ Job {job_id} completed after review - unmarked cancellation")
            
            # Confirm credit usage
            # Complete credit billing using centralized utility (sync context)
            job_utils.complete_job_billing_sync(job_id, "dub", user_id)
            
            # ✅ Cleanup ONLY after resume is completely finished
            from app.utils.video_downloader import video_download_service
            video_download_service.cleanup_specific_job(job_id)
            
        except Exception as e:
            logger.error(f"Approval resume failed for job {job_id}: {str(e)}")
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {
                "message": f"Approval resume failed: {str(e)}",
                "review_status": "rejected",
                "error": str(e)
            }, "dub")
            
            # Refund credits on failure
            # Refund credits for failed job (sync context)
            job_utils.refund_job_credits_sync(job_id, "dub", "approval_failed")
            
            # ❌ Cleanup after resume failure too
            from app.utils.video_downloader import video_download_service
            video_download_service.cleanup_specific_job(job_id)
    executor.submit(_resume)
    return {"success": True, "message": "Resume started", "job_id": job_id}

@router.post("/video-dub/{job_id}/redub", response_model=RedubResponse)
async def redub_job(job_id: str, request_body: RedubRequest, current_user = Depends(get_current_user)):
    """
    Create a new redub job from existing job.
    Returns new job ID that works with all existing endpoints.
    """
    # Get original job
    original_job_id = job_id  # Rename for clarity in the function
    original_job = await dub_job_service.get_job(original_job_id)
    if not original_job or original_job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Original job not found")
    
    # Validate job for redub using utility function
    validation_result = await job_utils.validate_job_for_redub(original_job)
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
    
    # Generate new job ID using same pattern as original (job_{uuid})
    from app.services.r2_service import get_r2_service
    r2_service = get_r2_service()
    new_job_id = r2_service.generate_job_id()
    
    # Get job details for redub
    user_id = current_user.id
    duration = original_job.duration or (original_job.details or {}).get("duration", 0)
    
    if not duration or duration <= 0:
        raise HTTPException(
            status_code=400, 
            detail="Original job missing duration information. Cannot proceed with redub."
        )
    
    # Create new job data
    new_job_data = {
        "job_id": new_job_id,
        "user_id": user_id,
        "target_language": request_body.target_language,
        "original_filename": f"Redub - {original_job.original_filename}",
        "source_video_language": original_job.source_video_language,
        "duration": duration,
        "status": "pending",
        "progress": 0,
        "parent_job_id": original_job_id,
        "redub_from": original_job.target_language
    }
    
    # Atomic credit reservation + job creation
    result = await credit_service.reserve_credits_and_create_job(
        user_id=user_id,
        job_data=new_job_data,
        job_type=JobType.DUB,
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
        new_job_dir = await job_utils.setup_job_directory(original_job_id, new_job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Prepare manifest for redub using utility function
    manifest = job_utils.prepare_manifest_for_redub(
        manifest, new_job_id, request_body.target_language, original_job_id
    )
    
    # Create redub request similar to original dub request
    try:
        redub_request = VideoDubRequest(
            job_id=new_job_id,
            target_language=request_body.target_language,
            project_title=f"Redub - {original_job.original_filename}",
            duration=duration,
            source_video_language=original_job.source_video_language,
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
            _update_status_non_blocking(new_job_id, ProcessingStatus.PROCESSING, 10, {"message": f"Redubbing to {request_body.target_language}", "phase": "initialization"}, "dub")
            
            result = api.process_dubbed_audio(
                audio_url=None,  # Reuse existing files
                job_id=new_job_id,
                target_language=request_body.target_language,
                source_video_language=original_job.source_video_language,
                output_dir=new_job_dir,
                review_mode=bool(getattr(request_body, "humanReview", False)),
                manifest_override=manifest,
            )
            
            if not result["success"]:
                _update_status_non_blocking(new_job_id, ProcessingStatus.FAILED, 0, {
                    "message": "Redub failed", 
                    "error": result.get("error")
                }, "dub")
                return
            
            # Handle review mode or completion
            if getattr(request_body, "humanReview", False):
                if result.get("review"):
                    review = result["review"]
                    details = {
                        "duration": duration,
                        "review_required": True,
                        "review_status": "awaiting",
                        "segments_manifest_url": review.get("segments_manifest_url"),
                        "segments_manifest_key": review.get("segments_manifest_key"),
                        "segments_count": review.get("segments_count"),
                        "transcript_id": review.get("transcript_id"),
                        "edited_segments_version": 0,
                        "parent_job_id": original_job_id
                    }
                    details = RunPodURLManager.copy_urls_from_original_job(original_job, details)
                    _update_status_non_blocking(new_job_id, ProcessingStatus.AWAITING_REVIEW, 80, details, "dub")
                    
                    # ✅ Redub review mode complete - keep files for faster resume (will auto-cleanup later if needed)
                    
                    # ✅ Redub successfully reached review stage - safe to unmark cancellation
                    from app.utils.shared_memory import unmark_job_cancelled
                    unmark_job_cancelled(new_job_id)
                    logger.info(f"✅ Redub job {new_job_id} reached awaiting_review - unmarked cancellation")
                    return
            
            # Complete redub
            result_url = result.get("result_url") or (result.get("result_urls", {}) or {}).get("final_video")
            folder_upload = result.get("folder_upload", {})
            
            # Add stored RunPod URLs to folder upload for redub
            original_urls = RunPodURLManager.retrieve_urls_from_job(original_job)
            if original_urls:
                folder_upload = RunPodURLManager.add_urls_to_folder_upload(folder_upload, original_urls, new_job_id)
            
            _update_status_non_blocking(new_job_id, ProcessingStatus.COMPLETED, 100, {
                "message": "Redub completed successfully",
                "result_url": result_url,
                "details": result.get("details"),
                "folder_upload": folder_upload,
                "result_urls": result.get("result_urls"),
                "parent_job_id": original_job_id
            }, "dub")
            
            # ✅ Successfully completed redub - safe to unmark cancellation
            from app.utils.shared_memory import unmark_job_cancelled
            unmark_job_cancelled(new_job_id)
            logger.info(f"✅ Redub job {new_job_id} completed - unmarked cancellation")
            
            # Complete credit billing using centralized utility (sync context for redub)
            job_utils.complete_job_billing_sync(new_job_id, "dub", user_id)
            
            # Immediate cleanup for this specific completed redub job
            from app.utils.video_downloader import video_download_service
            video_download_service.cleanup_specific_job(new_job_id)
                
        except Exception as e:
            logger.error(f"Redub processing failed: {str(e)}")
            _update_status_non_blocking(new_job_id, ProcessingStatus.FAILED, 0, {
                "message": f"Redub failed: {str(e)}", 
                "error": str(e),
                "original_job_id": original_job_id
            }, "dub")
            
            # Refund credits on failure
            try:
                # Note: For redub failures, we use a simple approach - just log and cleanup
                # Complex refund logic can cause additional event loop issues
                logger.info(f"Redub job {new_job_id} failed, cleaning up resources")
            except Exception:
                pass
    
    # Enqueue redub job with proper background runner (like existing dub API)
    _dub_queue_manager.enqueue(new_job_id, _run_redub)
    
    logger.info(f"Started redub job {new_job_id} from original {original_job_id} for user {user_id}")
    
    return RedubResponse(
        success=True, 
        message="Redub job created successfully", 
        job_id=new_job_id, 
        status="started",
        details={
            "original_job_id": original_job_id,
            "new_job_id": new_job_id,
            "target_language": request_body.target_language,
            "status_check_url": f"/api/video-dub-status/{new_job_id}"
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
    from app.utils.token_helper import generate_url_safe_token
    from datetime import timedelta
    from app.config.database import db
    
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
