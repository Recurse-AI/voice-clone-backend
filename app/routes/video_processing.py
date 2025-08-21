from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import logging
import os
from app.schemas import (
    VideoDubRequest,
    VideoDubResponse,
    VideoDubStatusResponse,
    VideoDownloadRequest,
    VideoDownloadResponse,
    FileDeleteRequest,
    FileDeleteResponse,
    SegmentsResponse,
    SaveEditsRequest,
    RedubRequest,
    RedubResponse,
    RegenerateSegmentRequest,
    RegenerateSegmentResponse,
)
from fastapi import UploadFile, File
from app.dependencies.auth import get_current_user
from app.services.dub_job_service import dub_job_service
from app.config.database import users_collection
from app.services.credit_service import credit_service, JobType
from app.utils.status_manager import status_manager, ProcessingStatus
from app.utils.runpod_url_manager import RunPodURLManager
import asyncio
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import uuid
from datetime import datetime
import shutil
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
    """Update dub job status using common utility"""
    from app.utils.db_sync_operations import update_dub_status
    status_str = status.value if hasattr(status, 'value') else str(status)
    update_dub_status(job_id, status_str, progress, details)

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
            "expected_speaker": request.expected_speaker,
            "duration": request.duration,
            "status": "pending",
            "progress": 0
        }
        
        # Atomic credit reservation + job creation
        result = await credit_service.atomic_reserve_and_create_job(
            user_id=user_id,
            job_data=job_data,
            job_type=JobType.DUB,
            duration_seconds=request.duration,
            collection_name="dub_jobs"
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
        
        enqueue_dub_job(request, user_id)
        
        logger.info(f"Started video dub job {request.job_id} for user {user_id}")
        
        queue_position = get_dub_queue_position(request.job_id)
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
        job = await dub_job_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job ID not found")
        
        from app.services.job_response_service import job_response_service
        formatted_job = job_response_service.format_dub_job(job)
        
        def get_progress_message(status: str, progress: int) -> str:
            if status == "failed":
                return "Processing failed"
            elif status == "completed":
                return "Video dubbing completed successfully"
            elif status == "awaiting_review":
                return "Awaiting human review - Please review dubbed text"
            elif status == "reviewing":
                return "Applying human edits and continuing dubbing"
            elif status == "processing":
                if progress <= 10:
                    return "Starting dubbing process"
                elif progress <= 30:
                    return "Separating audio tracks"
                elif progress <= 45:
                    return "Transcribing audio with AI"
                elif progress <= 55:
                    return "Dubbing text with AI translation"
                elif progress <= 75:
                    return "Reviewing and editing with AI"
                elif progress <= 79:
                    return "Preparing review files"
                elif progress <= 89:
                    return "Voice cloning and reconstructing final audio"
                elif progress <= 93:
                    return "Generating final output"
                elif progress <= 96:
                    return "Uploading results"
                else:
                    return "Finalizing"
            elif status == "pending":
                return "Processing queued"
            else:
                return f"Job is {status}"
        
        descriptive_message = get_progress_message(
            formatted_job.status, 
            formatted_job.progress
        )
        
        queue_position = None
        if formatted_job.status in [ProcessingStatus.PENDING.value, ProcessingStatus.PROCESSING.value]:
            try:
                queue_position = get_dub_queue_position(job_id)
            except Exception:
                queue_position = None

        return VideoDubStatusResponse(
            job_id=job_id,
            status=formatted_job.status,
            progress=formatted_job.progress,
            message=descriptive_message,
            result_url=formatted_job.result_url,
            error=formatted_job.error,
            details={
                "files": formatted_job.files,
                "queue_position": queue_position
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
    from app.utils.job_utils import job_utils
    from app.utils.shared_memory import is_job_cancelled
    
    r2_service = get_r2_service()
    job_id = request.job_id
    job_dir = os.path.join(settings.TEMP_DIR, f"dub_{job_id}")
    
    try:
        if is_job_cancelled(job_id):
            _update_status_non_blocking(job_id, ProcessingStatus.CANCELLED, 0, {
                "message": "Job cancelled by user", 
                "error": "Job cancelled by user"
            }, "dub")
            return
            
        _update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 10, {"message": "Starting dubbing process"}, "dub")
        
        if not os.path.exists(job_dir):
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"error": "Uploaded audio not found"}, "dub")
            return
            
        allowed_audio_ext = set(settings.ALLOWED_AUDIO_FORMATS)
        audio_candidates = [f for f in os.listdir(job_dir) if f.lower().split('.')[-1] in {e.lstrip('.').lower() for e in allowed_audio_ext}]
        
        if not audio_candidates:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"error": "No audio file found in upload folder"}, "dub")
            return
            
        audio_path = os.path.join(job_dir, audio_candidates[0])
        
        if is_job_cancelled(job_id):
            _update_status_non_blocking(job_id, ProcessingStatus.CANCELLED, 0, {
                "message": "Job cancelled by user", "error": "Job cancelled by user"
            }, "dub")
            return
            
        _update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 30, {"message": "Separating audio tracks..."}, "dub")
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
                _update_status_non_blocking(job_id, ProcessingStatus.SEPARATING, 30 + (progress // 4), {
                    "message": f"Audio separation in progress... ({status})"
                }, "dub")
            
            def on_separation_cancelled():
                _update_status_non_blocking(job_id, ProcessingStatus.CANCELLED, 0, {
                    "message": "Job cancelled by user", "error": "Job cancelled by user"
                }, "dub")
            
            monitor_result = monitor_runpod_job(
                runpod_request_id=request_id,
                job_id=job_id,
                timeout_seconds=300,
                on_progress=on_separation_progress,
                on_cancelled=on_separation_cancelled
            )
            
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
                "message": "Job cancelled by user", "error": "Job cancelled by user"
            }, "dub")
            return
            
        _update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 40, {"message": "Starting AI dubbing pipeline..."}, "dub")
        
        from app.services.dub.simple_dubbed_api import get_simple_dubbed_api
        simple_dubbed_api = get_simple_dubbed_api()
        audio_url = r2_audio_path["url"]
        
        pipeline_result = simple_dubbed_api.process_dubbed_audio(
            job_id=job_id,
            audio_url=audio_url,
            target_language=request.target_language,
            speakers_count=int(request.expected_speaker) if request.expected_speaker else 1,
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
        
        # Confirm credit usage
        credit_service.confirm_credit_usage_sync(job_id, JobType.DUB)
        
        if r2_audio_path.get("r2_key"):
            r2_service.delete_file(r2_audio_path["r2_key"])
        
        job_utils.cleanup_job_directories()

    except Exception as e:
        if is_job_cancelled(job_id):
            logger.info(f"Job {job_id} was cancelled - not overriding with failed status on exception")
        else:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"message": f"Processing failed: {str(e)}", "error": str(e)}, "dub")
        
        # Refund credits on failure
        credit_service.refund_reserved_credits_sync(job_id, JobType.DUB, "job_failed")
        
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

from app.services.dub.manifest_service import write_json as _write_temp_json

@router.get("/video-dub/{job_id}/segments", response_model=SegmentsResponse)
async def get_segments(job_id: str, current_user = Depends(get_current_user)):
    job = await dub_job_service.get_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check job status first
    if job.status != "awaiting_review" and job.status != "reviewing":
        raise HTTPException(status_code=400, detail=f"Job is in {job.status} status. Segments are only available for review jobs.")
    
    manifest_url = job.segments_manifest_url or (job.details or {}).get("segments_manifest_url")
    if not manifest_url:
        raise HTTPException(
            status_code=400, 
            detail=f"No manifest available for job {job_id}. Status: {job.status}, Manifest URL: {job.segments_manifest_url}"
        )
    
    try:
        manifest = _load_manifest_json(manifest_url)
        return SegmentsResponse(
            job_id=job_id, 
            segments=manifest.get("segments", []), 
            manifestUrl=manifest_url, 
            version=manifest.get("version"),
            target_language=manifest.get("target_language")
        )
    except Exception as e:
        logger.error(f"Failed to load manifest from {manifest_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load manifest: {str(e)}")

@router.put("/video-dub/{job_id}/segments", response_model=SegmentsResponse)
async def save_segment_edits(job_id: str, request_body: SaveEditsRequest, current_user = Depends(get_current_user)):
    job = await dub_job_service.get_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    manifest_url = job.segments_manifest_url or (job.details or {}).get("segments_manifest_url")
    manifest_key = job.segments_manifest_key or (job.details or {}).get("segments_manifest_key")
    if not manifest_url:
        raise HTTPException(status_code=400, detail="No manifest available for this job")
    manifest = _load_manifest_json(manifest_url)
    id_to_edit = {e.id: e for e in request_body.segments}
    for seg in manifest.get("segments", []):
        if seg["id"] in id_to_edit:
            edit = id_to_edit[seg["id"]]
            if edit.dubbed_text is not None:
                seg["dubbed_text"] = edit.dubbed_text
            if edit.start is not None:
                seg["start"] = edit.start
            if edit.end is not None:
                seg["end"] = edit.end
            seg["duration_ms"] = max(0, seg["end"] - seg["start"])
    manifest["version"] = int(manifest.get("version", 1)) + 1

    # Write and upload manifest back to R2
    job_dir = _ensure_job_dir(job_id)
    manifest_path = os.path.join(job_dir, f"dubbing_manifest_{job_id}.json")
    _write_temp_json(manifest, manifest_path)
    from app.services.r2_service import get_r2_service
    r2 = get_r2_service()
    if manifest_key:
        up_res = r2.upload_file(manifest_path, manifest_key, content_type="application/json")
        manifest_url_out = (up_res or {}).get("url") or manifest_url
    else:
        # If key not known, upload new copy
        r2_key = r2.generate_file_path(job_id, "", os.path.basename(manifest_path))
        res = r2.upload_file(manifest_path, r2_key, content_type="application/json")
        manifest_url_out = res.get("url") if res.get("success") else manifest_url
    # Update DB details and status reviewing
    await dub_job_service.update_job_status(job_id, ProcessingStatus.REVIEWING.value, 80, details={
        "review_required": True,
        "review_status": "in_progress",
        "segments_manifest_url": manifest_url_out,
        "edited_segments_version": (job.edited_segments_version or 0) + 1,
    })
    return SegmentsResponse(
        job_id=job_id, 
        segments=manifest.get("segments", []), 
        manifestUrl=manifest_url_out, 
        version=manifest.get("version"),
        target_language=manifest.get("target_language")
    )

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
    from app.utils.job_utils import job_utils
    executor = get_dub_executor()
    def _resume():
        # Initialize variable to avoid scoping issues
        stored_runpod_urls = {}
        try:
            # Retrieve URLs inside the function to avoid scoping issues
            stored_runpod_urls = RunPodURLManager.retrieve_urls_from_job(job)
            if not stored_runpod_urls:
                logger.warning(f"No stored RunPod URLs found for job {job_id}")
                stored_runpod_urls = {}  # Provide fallback
            # Update status to reviewing with proper review_status
            _update_status_non_blocking(job_id, ProcessingStatus.REVIEWING, 80, {
                "message": "Resuming with human edits...",
                "review_status": "approved"
            }, "dub")
            from app.services.dub.simple_dubbed_api import get_simple_dubbed_api
            api = get_simple_dubbed_api()
            job_dir = _ensure_job_dir(job_id)
            result = api.process_dubbed_audio(
                audio_url=None,
                job_id=job_id,
                target_language=manifest.get("target_language") or job.target_language,
                speakers_count=int(job.expected_speaker) if job.expected_speaker else 1,
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
            credit_service.confirm_credit_usage_sync(job_id, JobType.DUB)
            
            # Cleanup old dub directories after approval completion
            job_utils.cleanup_job_directories()
        except Exception as e:
            logger.error(f"Approval resume failed for job {job_id}: {str(e)}")
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {
                "message": f"Approval resume failed: {str(e)}",
                "review_status": "rejected",
                "error": str(e)
            }, "dub")
            
            # Refund credits on failure
            credit_service.refund_reserved_credits_sync(job_id, JobType.DUB, "approval_failed")
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
    from app.utils.job_utils import job_utils
    
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
        "expected_speaker": original_job.expected_speaker,
        "duration": duration,
        "status": "pending",
        "progress": 0,
        "parent_job_id": original_job_id,
        "redub_from": original_job.target_language
    }
    
    # Atomic credit reservation + job creation
    result = await credit_service.atomic_reserve_and_create_job(
        user_id=user_id,
        job_data=new_job_data,
        job_type=JobType.DUB,
        duration_seconds=duration,
        collection_name="dub_jobs"
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
            expected_speaker=original_job.expected_speaker,
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
        from app.utils.job_utils import job_utils
        try:
            _update_status_non_blocking(new_job_id, ProcessingStatus.PROCESSING, 10, {"message": f"Redubbing to {request_body.target_language}"}, "dub")
            
            result = api.process_dubbed_audio(
                audio_url=None,  # Reuse existing files
                job_id=new_job_id,
                target_language=request_body.target_language,
                speakers_count=int(original_job.expected_speaker) if original_job.expected_speaker else 1,
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
            
            # Confirm credit usage
            credit_service.confirm_credit_usage_sync(new_job_id, JobType.DUB)
            
            # Cleanup old dub directories after redub completion
            job_utils.cleanup_job_directories()
                
        except Exception as e:
            logger.error(f"Redub processing failed: {str(e)}")
            _update_status_non_blocking(new_job_id, ProcessingStatus.FAILED, 0, {
                "message": f"Redub failed: {str(e)}", 
                "error": str(e),
                "original_job_id": original_job_id
            }, "dub")
            
            # Refund credits on failure
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                refund_result = loop.run_until_complete(
                    credit_service.refund_reserved_credits(new_job_id, JobType.DUB, "redub_failed")
                )
                loop.close()
                if refund_result["success"]:
                    logger.info(f"Refunded {refund_result['credits_refunded']} credits for failed redub {new_job_id}")
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

@router.post("/video-dub/{job_id}/segments/{segment_id}/regenerate", response_model=RegenerateSegmentResponse)
async def regenerate_segment(job_id: str, segment_id: str, request_body: RegenerateSegmentRequest, current_user = Depends(get_current_user)):
    job = await dub_job_service.get_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    manifest_url = job.segments_manifest_url or (job.details or {}).get("segments_manifest_url")
    if not manifest_url:
        raise HTTPException(status_code=400, detail="No manifest available for this job")
    manifest = _load_manifest_json(manifest_url)
    seg = next((s for s in manifest.get("segments", []) if s.get("id") == segment_id), None)
    if not seg:
        raise HTTPException(status_code=404, detail="Segment not found")

    # Update text with OpenAI if prompt is provided, otherwise use existing logic
    dubbed_text = request_body.dubbed_text if request_body.dubbed_text is not None else seg.get("dubbed_text")
    
    # If custom prompt is provided, use OpenAI to regenerate text
    if request_body.prompt and dubbed_text:
        try:
            from app.services.dub.assembly_transcription import TranscriptionService
            from app.config.settings import settings
            from openai import OpenAI
            
            openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            target_lang = request_body.target_language or manifest.get("target_language", "Bengali")
            
            system_prompt = (
                f"You are a professional dubbing script writer. "
                f"Rewrite the given text in {target_lang} according to the specific instructions provided. "
                f"Keep the meaning accurate but adapt the style based on the prompt. "
                f"Return only the rewritten text, nothing else."
            )
            
            user_prompt = (
                f"Instructions: {request_body.prompt}\n"
                f"Original text: {dubbed_text}\n"
                f"Language: {target_lang}\n"
                f"Rewrite this text following the instructions:"
            )
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Use OpenAI generated text
            dubbed_text = response.choices[0].message.content.strip()
            
        except Exception as e:
            # If OpenAI fails, fallback to original logic
            logger.warning(f"OpenAI regeneration failed: {e}, using fallback")
            if request_body.prompt:
                dubbed_text = f"[{request_body.prompt}] " + dubbed_text
    
    # Apply tone if provided
    if request_body.tone and dubbed_text:
        dubbed_text = f"({request_body.tone}) " + dubbed_text
    
    seg["dubbed_text"] = dubbed_text
    
    # Store prompt separately for future reference
    if request_body.prompt:
        seg["custom_prompt"] = request_body.prompt
    
    # Store tone separately for future reference  
    if request_body.tone:
        seg["tone"] = request_body.tone

    # If target_language provided, update manifest target_language (optional)
    if request_body.target_language:
        manifest["target_language"] = request_body.target_language

    # Persist manifest (version +1)
    manifest["version"] = int(manifest.get("version", 1)) + 1
    job_dir = _ensure_job_dir(job_id)
    manifest_path = os.path.join(job_dir, f"dubbing_manifest_{job_id}.json")
    _write_temp_json(manifest, manifest_path)
    from app.services.r2_service import get_r2_service
    r2 = get_r2_service()
    # Try to overwrite existing manifest key if present
    manifest_key = job.segments_manifest_key or (job.details or {}).get("segments_manifest_key")
    if manifest_key:
        up = r2.upload_file(manifest_path, manifest_key, content_type="application/json")
        if up.get("success"):
            manifest_url = up.get("url")
    else:
        r2_key = r2.generate_file_path(job_id, "", os.path.basename(manifest_path))
        up = r2.upload_file(manifest_path, r2_key, content_type="application/json")
        if up.get("success"):
            manifest_url = up.get("url")

    # Return updated segment + manifest info
    return RegenerateSegmentResponse(
    success=True,
    message="Segment text updated for re-dub",
    job_id=job_id,
    segment_id=segment_id,
    manifestUrl=manifest_url,
    version=manifest.get("version", 1),
    segment=seg
)

@router.post("/download-media", response_model=VideoDownloadResponse)
async def download_media(request: VideoDownloadRequest):
    """Download media (video/audio) from URL and store locally"""
    try:
        from app.utils.video_downloader import video_download_service
        
        logger.info(f"Media download request: {request.url}")

        result = await video_download_service.download_video(
            url=request.url,
            quality=request.quality,
            resolution=request.resolution,
            max_filesize=request.max_filesize,
            format_preference=request.format_preference,
            audio_quality=request.audio_quality,
            prefer_free_formats=request.prefer_free_formats,
            include_subtitles=request.include_subtitles
        )

        if result["success"]:
            logger.info(f"Media download successful: {result['job_id']}")
            return VideoDownloadResponse(
                success=True,
                message="Download successful",
                job_id=result["job_id"],
                video_info=result["video_info"],
                download_info=result.get("download_info"),
                available_formats=result.get("available_formats")
            )
        else:
            logger.error(f"Media download failed: {result['error']}")
            return VideoDownloadResponse(
                success=False,
                message="Media download failed",
                error=result["error"]
            )
    except Exception as e:
        logger.error(f"Media download endpoint error: {str(e)}")
        return VideoDownloadResponse(
            success=False,
            message="Internal server error",
            error=str(e)
        )

@router.delete("/file-delete", response_model=FileDeleteResponse)
async def delete_downloaded_file(request: FileDeleteRequest):
    """Delete locally stored downloaded file by job_id"""
    try:
        from app.utils.video_downloader import video_download_service
        
        logger.info(f"File delete request for job_id: {request.job_id}")
        
        result = video_download_service.delete_file(request.job_id)
        
        if result["success"]:
            logger.info(f"File delete successful for job_id: {request.job_id}")
            return FileDeleteResponse(
                success=True,
                message="File deleted successfully",
                deleted_files=result["deleted_files"]
            )
        else:
            logger.error(f"File delete failed for job_id {request.job_id}: {result['error']}")
            return FileDeleteResponse(
                success=False,
                message="File delete failed",
                error=result["error"]
            )
    except Exception as e:
        logger.error(f"File delete endpoint error for {request.job_id}: {str(e)}")
        return FileDeleteResponse(
            success=False,
            message="Internal server error",
            error=str(e)
        )

@router.get("/file-serve/{job_id}")
async def serve_downloaded_file(job_id: str):
    """Serve the actual downloaded file content by job_id"""
    try:
        from app.utils.video_downloader import video_download_service
        
        logger.info(f"File serve request for job_id: {job_id}")
        
        result = video_download_service.get_file_path(job_id)
        
        if not result["success"]:
            logger.warning(f"File serve failed for job_id {job_id}: {result['error']}")
            raise HTTPException(status_code=404, detail=result["error"])
        
        file_path = result["file_path"]
        filename = result["filename"]
        
        logger.info(f"Serving file for job_id {job_id}: {filename}")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File serve endpoint error for {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/add-subtitle")
async def add_subtitle_to_video(
    video_file: UploadFile = File(...),
    srt_url: str = Form(..., description="URL to SRT subtitle file")
):
    """
    Add subtitle to video file and upload to R2.
    Simple API: video file + SRT URL → subtitled video URL
    """
    import subprocess
    import requests
    import tempfile
    from pathlib import Path
    
    job_id = str(uuid.uuid4())
    logger.info(f"Starting subtitle addition for job {job_id}")
    
    try:
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Save uploaded video file
            video_filename = video_file.filename or f"video_{job_id}.mp4"
            video_ext = Path(video_filename).suffix or ".mp4"
            video_path = temp_path / f"input_video{video_ext}"
            
            with open(video_path, "wb") as f:
                content = await video_file.read()
                f.write(content)
            
            logger.info(f"Video saved: {video_path} ({len(content)} bytes)")
            
            # 2. Download SRT file
            srt_response = requests.get(srt_url, timeout=60)
            srt_response.raise_for_status()
            
            srt_path = temp_path / f"subtitles_{job_id}.srt"
            with open(srt_path, "wb") as f:
                f.write(srt_response.content)
            
            logger.info(f"SRT downloaded: {srt_path} ({len(srt_response.content)} bytes)")
            
            # 3. Add subtitles using FFmpeg (same design as video renderer)
            output_path = temp_path / f"video_with_subtitles_{job_id}{video_ext}"
            
            # FFmpeg command with professional subtitle styling (copied from video renderer)
            subtitle_style = (
                "Fontname=Arial-Bold,Fontsize=18,Bold=1,PrimaryColour=&H00ffffff,"
                "OutlineColour=&H00000000,Outline=3,Alignment=2,MarginV=30"
            )
            
            # Cross-platform path escaping for FFmpeg (Windows/Linux/macOS)
            import platform
            srt_path_str = str(srt_path)
            
            # Convert to forward slashes (works on all platforms)
            srt_path_escaped = srt_path_str.replace('\\', '/')
            
            # Escape special characters for FFmpeg filter
            srt_path_escaped = srt_path_escaped.replace(':', '\\:').replace("'", "\\'")
            
            logger.info(f"Platform: {platform.system()}, Original path: {srt_path_str}, Escaped: {srt_path_escaped}")
            
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"subtitles='{srt_path_escaped}':force_style='{subtitle_style}'",
                "-c:a", "copy",  # Copy audio without re-encoding
                "-preset", "fast",  # Fast encoding
                str(output_path)
            ]
            
            logger.info(f"Running FFmpeg: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return {
                    "success": False,
                    "error": f"FFmpeg error: {result.stderr}",
                    "job_id": job_id
                }
            
            # 4. Upload to R2
            from app.services.r2_service import get_r2_service
            r2_service = get_r2_service()
            
            # Generate R2 key
            output_filename = f"subtitled_{job_id}{video_ext}"
            r2_key = f"subtitled_videos/{job_id}/{output_filename}"
            
            # Upload file
            upload_result = r2_service.upload_file(
                str(output_path), 
                r2_key, 
                content_type="video/mp4"
            )
            
            if not upload_result.get("success"):
                logger.error(f"R2 upload failed: {upload_result.get('error')}")
                return {
                    "success": False,
                    "error": f"Upload failed: {upload_result.get('error')}",
                    "job_id": job_id
                }
            
            # 5. Success response
            video_url = upload_result["url"]
            file_size = output_path.stat().st_size
            
            logger.info(f"Subtitle addition completed. Output: {video_url}")
            
            return {
                "success": True,
                "message": "Subtitle added successfully",
                "job_id": job_id,
                "video_url": video_url,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "original_filename": video_filename,
                "output_filename": output_filename
            }
            
            # Note: temp_dir cleanup happens automatically when exiting 'with' block
            
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timeout for job {job_id}")
        return {
            "success": False,
            "error": "Video processing timeout (10 minutes limit)",
            "job_id": job_id
        }
    except requests.RequestException as e:
        logger.error(f"SRT download failed for job {job_id}: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to download SRT file: {str(e)}",
            "job_id": job_id
        }
    except Exception as e:
        logger.error(f"Subtitle addition failed for job {job_id}: {str(e)}")
        return {
            "success": False,
            "error": f"Processing failed: {str(e)}",
            "job_id": job_id
        }
    finally:
        # Additional cleanup logging
        logger.info(f"Cleaned up temporary files for subtitle job {job_id}")
        

