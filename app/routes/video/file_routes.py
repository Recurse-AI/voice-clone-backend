from fastapi import APIRouter, HTTPException, Query, File, UploadFile, Form
import logging
from pathlib import Path
import os
import uuid
import json
from typing import Optional
from app.schemas import VideoDownloadRequest, VideoDownloadResponse
from app.utils.video_downloader import video_download_service
from app.config.settings import settings
from app.services.r2_service import R2Service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/available-formats")
async def get_available_formats(url: str = Query(..., description="Video URL to check formats")):
    """Get list of available download formats without downloading."""
    try:
        result = await video_download_service.get_available_formats(url)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to get formats"))
        return result
    except Exception as e:
        logger.error(f"Error getting formats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/download-media", response_model=VideoDownloadResponse)
async def download_media(
    request: str = Form(...),
    cookie_file: Optional[UploadFile] = File(None)
):
    """Download media from YouTube/social media using resilient downloader, then upload to R2 bucket"""
    user_cookie_path = None
    try:
        request_data = json.loads(request)
        request_obj = VideoDownloadRequest(**request_data)
        
        logger.info(f"Media download request: {request_obj.url}")

        if cookie_file:
            cookie_dir = Path(settings.TEMP_DIR) / "cookies"
            cookie_dir.mkdir(parents=True, exist_ok=True)
            user_cookie_path = cookie_dir / f"user_{uuid.uuid4()}.txt"
            
            with open(user_cookie_path, "wb") as f:
                content = await cookie_file.read()
                f.write(content)
            
            logger.info(f"User cookie file saved: {user_cookie_path}")

        # 1) Download locally using resilient service (retries, fallbacks, tuned yt-dlp)
        res = await video_download_service.download_video(
            url=request_obj.url,
            format_id=request_obj.format_id,
            quality=request_obj.quality,
            resolution=request_obj.resolution,
            max_filesize=request_obj.max_filesize,
            format_preference=request_obj.format_preference,
            audio_quality=request_obj.audio_quality,
            prefer_free_formats=bool(request_obj.prefer_free_formats),
            include_subtitles=bool(request_obj.include_subtitles),
            user_cookie_file=str(user_cookie_path) if user_cookie_path else None,
        )

        if not res.get("success"):
            return VideoDownloadResponse(success=False, message="Failed to download video", error=res.get("error"))

        job_id = res.get("job_id")
        filename = res.get("filename")
        local_path = Path(settings.TEMP_DIR) / job_id / filename
        if not local_path.exists():
            return VideoDownloadResponse(success=False, message="Download failed", error="Downloaded file not found")

        # 2) Upload to R2
        r2_service = R2Service()
        sanitized_filename = r2_service._sanitize_filename(filename)
        r2_key = f"downloads/{job_id}/{sanitized_filename}"
        content_type = r2_service._get_content_type(filename)
        upload_result = r2_service.upload_file(str(local_path), r2_key, content_type)
        if not upload_result.get("success"):
            logger.error(f"R2 upload failed: {upload_result.get('error')}")
            return VideoDownloadResponse(success=False, message="Upload to R2 failed", error=upload_result.get("error"))

        # 3) Build response preserving existing shape fields
        video_info = {
            "title": res.get("title", "Unknown"),
            "duration": res.get("duration", 0),
            "filename": filename,
            "file_size": res.get("file_size", 0),
        }
        download_info = {
            "requested_quality": request_obj.quality or "best",
            "resolution": res.get("resolution", "Unknown"),
            "format": res.get("format", "Unknown"),
        }

        return VideoDownloadResponse(
            success=True,
            message="Download and upload successful",
            job_id=job_id,
            r2_url=upload_result["url"],
            r2_key=r2_key,
            video_info=video_info,
            download_info=download_info
        )

    except json.JSONDecodeError:
        return VideoDownloadResponse(success=False, message="Invalid JSON in request field", error="Invalid request format")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Endpoint error: {error_msg}")
        if "401" in error_msg or "Unauthorized" in error_msg:
            message = "Video requires authentication or is not accessible"
        elif "403" in error_msg or "Forbidden" in error_msg:
            message = "Video access forbidden or region-restricted"
        elif "404" in error_msg:
            message = "Video not found"
        else:
            message = "Failed to download video"
        return VideoDownloadResponse(success=False, message=message, error=error_msg)
    finally:
        if user_cookie_path and os.path.exists(user_cookie_path):
            try:
                os.remove(user_cookie_path)
                logger.info(f"Cleaned up user cookie file: {user_cookie_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup cookie file: {e}")
