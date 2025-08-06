from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
import logging
import os
from app.schemas import VideoDubRequest, VideoDubResponse, VideoDubStatusResponse, VideoDownloadRequest, VideoDownloadResponse
from app.dependencies.auth import get_current_user
from app.services.dub_job_service import dub_job_service
from app.config.database import users_collection
from app.services.credit_service import credit_service, JobType
from app.utils.status_manager import status_manager, ProcessingStatus
import asyncio
import threading

import uuid
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

def _update_status_non_blocking(job_id: str, status: ProcessingStatus, progress: int, details: dict, job_type: str = "dub"):
    """Non-blocking status update to avoid blocking background processing"""
    def run_update():
        try:
            # Check if there's already an event loop in this thread
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
            except RuntimeError:
                # No event loop in this thread, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function
            if loop.is_running():
                # Loop is already running, use create_task instead
                asyncio.create_task(
                    status_manager.update_status(job_id, status, progress, details, job_type)
                )
            else:
                loop.run_until_complete(
                    status_manager.update_status(job_id, status, progress, details, job_type)
                )
                
        except Exception as e:
            logger.error(f"Failed to update status for {job_id}: {e}")
    
    # Run in separate thread to avoid blocking
    thread = threading.Thread(target=run_update, daemon=True)
    thread.start()

@router.post("/video-dub", response_model=VideoDubResponse)
async def start_video_dub(
    request: VideoDubRequest, 
    current_user = Depends(get_current_user)
):
    """
    Start video dubbing job (background).
    ভিডিও upload-file API দিয়ে upload করতে হবে, তারপর এখানে সেই job_id দিতে হবে।
    video_url ফিল্ড নেই, কারণ ভিডিও local-এ সংরক্ষিত থাকবে।
    """
    try:
        user_id = current_user.id
        
        # Pre-check: Verify user has sufficient credits
        credit_check = await credit_service.check_sufficient_credits(
            user_id=user_id,
            job_type=JobType.DUB,
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
        
        # Create dub job in MongoDB for tracking
        job_data = {
            "job_id": request.job_id,
            "user_id": user_id, 
            "target_language": request.target_language,
            "original_filename": request.project_title,
            "source_video_language": request.source_video_language,
            "expected_speaker": request.expected_speaker,
            "subtitle": request.subtitle,
            "instrument": request.instrument,
            "details": {
                "duration": request.duration,
                "required_credits": credit_check["required"]
            }
        }
        
        # Save to MongoDB
        job = await dub_job_service.create_job(job_data)
        if not job:
            logger.error(f"Failed to create dub job in MongoDB for {request.job_id}")
        
                            # Status is managed by dub_job_service, no need for separate status manager

        # Run processing in separate thread to avoid blocking main event loop
        thread = threading.Thread(target=process_video_dub_background, args=(request, user_id), daemon=True)
        thread.start()
        
        logger.info(f"Started video dub job {request.job_id} for user {user_id} (duration: {request.duration}s, credits: {credit_check['required']})")
        
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
    """Get video dubbing job status (hybrid: cache + MongoDB)"""
    try:
        # Use hybrid status manager for fast response
        status_data = await status_manager.get_status(job_id, job_type="dub")
        if not status_data:
            raise HTTPException(status_code=404, detail="Job ID not found")
        
        return VideoDubStatusResponse(
            job_id=job_id,
            status=status_data.get("status"),
            progress=status_data.get("progress", 0),
            message=status_data.get("message", f"Job is {status_data.get('status')}"),
            result_url=status_data.get("result_url"),
            error=status_data.get("error"),
            details=status_data.get("details", {})
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dub job status {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

# Background processing function (sync - runs in separate thread)
def process_video_dub_background(request: VideoDubRequest, user_id: str):
    from app.services.dub.audio_utils import AudioUtils
    from app.config.settings import settings
    from app.utils.r2_storage import R2Storage
    
    r2_storage = R2Storage()
    job_id = request.job_id
    job_dir = os.path.join(settings.TEMP_DIR, f"dub_{job_id}")
    try:
        _update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 10, {"message": "Finding uploaded video..."}, "dub")
        job_dir = os.path.join(settings.TEMP_DIR, f"dub_{job_id}")
        if not os.path.exists(job_dir):
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"error": "Uploaded video not found - No such directory"}, "dub")
            return
        video_files = [f for f in os.listdir(job_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'))]
        if not video_files:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"error": "No video file found in upload folder"}, "dub")
            return
        local_video_path = os.path.join(job_dir, video_files[0])
        _update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 20, {"message": "Extracting audio..."}, "dub")
        from app.services.dub.audio_utils import AudioUtils
        audio_utils = AudioUtils()
        audio_path = os.path.join(job_dir, f"{job_id}.wav")
        extract_result = audio_utils.extract_audio_from_video(local_video_path, audio_path)
        if not extract_result["success"]:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"error": f"Audio extraction failed: {extract_result.get('error')}"}, "dub")
            return
        _update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 30, {"message": "Separating vocals and instruments..."}, "dub")
        r2_audio_path = r2_storage.upload_file(audio_path, f"temp/{job_id}/{job_id}.wav")
        if not r2_audio_path["success"]:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"error": f"Audio upload failed: {r2_audio_path.get('error')}"}, "dub")
            return
        try:
            from app.utils.runpod_service import runpod_service
            request_id = runpod_service.submit_separation_request(r2_audio_path["url"], f"video_dub_{job_id}")
            status = runpod_service.wait_for_completion(request_id)
        except Exception as e:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"error": f"Audio separation job submission failed: {str(e)}"}, "dub")
            return
        if status.get('status') != 'COMPLETED' or not status.get('output'):
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"message": "Audio separation failed.", "error": status.get("error")}, "dub")
            return
        vocal_url = status['output'].get('vocal_audio')
        instrument_url = status['output'].get('instrument_audio')
        vocal_path = os.path.join(job_dir, f"{job_id}_vocal.wav")
        instrument_path = os.path.join(job_dir, f"{job_id}_instrument.wav")
        if vocal_url:
            download_result = audio_utils.download_audio_file(vocal_url, vocal_path)
            if not download_result["success"]:
                _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"message": "Vocal audio download failed.", "error": download_result.get("error")}, "dub")
                return
        if instrument_url:
            download_result = audio_utils.download_audio_file(instrument_url, instrument_path)
            if not download_result["success"]:
                _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"message": "Instrument audio download failed.", "error": download_result.get("error")}, "dub")
                return
        _update_status_non_blocking(job_id, ProcessingStatus.PROCESSING, 40, {"message": "Cloning voice and reconstructing audio..."}, "dub")
        from app.services.dub.simple_dubbed_api import SimpleDubbedAPI
        simple_dubbed_api = SimpleDubbedAPI()
        upload_response = r2_storage.upload_file(audio_path, f"temp/{job_id}/{job_id}.wav")
        if not upload_response["success"]:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"message": "Audio upload failed.", "error": upload_response.get("error")}, "dub")
            return
        audio_url = upload_response["url"]
        pipeline_result = simple_dubbed_api.process_dubbed_audio(
            job_id=job_id,
            audio_url=audio_url,
            video_path=local_video_path,
            instrument_path=instrument_path,
            target_language=request.target_language,
            speakers_count=int(request.expected_speaker) if request.expected_speaker else 1,
            source_video_language=request.source_video_language,
            subtitle=request.subtitle,
            instrument=request.instrument,
            output_dir=job_dir
        )
        if not pipeline_result["success"]:
            _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"message": "Dubbing pipeline failed.", "error": pipeline_result.get("error"), "details": pipeline_result.get("details")}, "dub")
            return
        # Clean up temporary R2 files
        r2_storage.delete_file(upload_response["r2_key"])
        r2_storage.delete_file(r2_audio_path["r2_key"])
        result_url = pipeline_result.get("result_url") or (pipeline_result.get("result_urls", {}) or {}).get("final_video")
        _update_status_non_blocking(job_id, ProcessingStatus.COMPLETED, 100, {
            "message": "Video dubbing completed.", 
            "result_url": result_url, 
            "details": pipeline_result.get("details"),
            "folder_upload": pipeline_result.get("folder_upload"),
            "result_urls": pipeline_result.get("result_urls"),
            "video_upload": pipeline_result.get("video_upload")
        }, "dub")
        
        # Auto-deduct credits on successful completion (non-blocking)
        def deduct_credits_non_blocking():
            try:
                # Check if there's already an event loop in this thread
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError("Event loop is closed")
                except RuntimeError:
                    # No event loop in this thread, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the async function
                if loop.is_running():
                    # Loop is already running, use create_task instead
                    asyncio.create_task(
                        credit_service.deduct_credits_on_completion(
                            user_id=user_id,
                            job_id=job_id,
                            job_type=JobType.DUB,
                            duration_seconds=request.duration
                        )
                    )
                else:
                    credit_result = loop.run_until_complete(
                        credit_service.deduct_credits_on_completion(
                            user_id=user_id,
                            job_id=job_id,
                            job_type=JobType.DUB,
                            duration_seconds=request.duration
                        )
                    )
                    
                    if credit_result["success"]:
                        logger.info(f"Auto-deducted {credit_result['deducted']} credits for completed dub job {job_id}")
                    else:
                        logger.error(f"Failed to auto-deduct credits for dub job {job_id}: {credit_result['message']}")
                        
            except Exception as e:
                logger.error(f"Credit deduction failed for job {job_id}: {e}")
        
        credit_thread = threading.Thread(target=deduct_credits_non_blocking, daemon=True)
        credit_thread.start()
        

    except Exception as e:
        _update_status_non_blocking(job_id, ProcessingStatus.FAILED, 0, {"message": f"Processing failed: {str(e)}", "error": str(e)}, "dub")
    finally:
        # Ensure temp directory is removed in any case
        AudioUtils.remove_temp_dir(folder_path=job_dir)

@router.post("/download-video", response_model=VideoDownloadResponse)
async def download_video(request: VideoDownloadRequest):
    """Download video from URL and store locally"""
    try:
        from app.utils.video_downloader import video_download_service
        
        logger.info(f"Video download request: {request.url}")

        result = await video_download_service.download_video(
            url=request.url,
            quality=request.quality
        )

        if result["success"]:
            logger.info(f"Video download successful: {result['download_id']}")
            return VideoDownloadResponse(
                success=True,
                message=result["message"],
                job_id=result["job_id"],
                video_info=result["video_info"]
            )
        else:
            logger.error(f"Video download failed: {result['error']}")
            return VideoDownloadResponse(
                success=False,
                message="Video download failed",
                error=result["error"]
            )
    except Exception as e:
        logger.error(f"Video download endpoint error: {str(e)}")
        return VideoDownloadResponse(
            success=False,
            message="Internal server error",
            error=str(e)
        )

