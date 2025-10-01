from fastapi import APIRouter, HTTPException
import logging
import tempfile
import os
import uuid
import yt_dlp
from app.schemas import VideoDownloadRequest, VideoDownloadResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/download-media", response_model=VideoDownloadResponse)
async def download_media(request: VideoDownloadRequest):
    """Download media from YouTube/social media and upload to R2 bucket"""
    temp_dir = None
    try:
        from app.services.r2_service import R2Service
        
        logger.info(f"Media download request: {request.url}")
        
        temp_dir = tempfile.mkdtemp(prefix="download_")
        output_template = os.path.join(temp_dir, "%(title)s.%(ext)s")
        
        quality_format = "bv*+ba/best"
        if request.resolution:
            res_int = int(request.resolution.replace("p", ""))
            quality_format = f"bv*[height<={res_int}]+ba/best"
        elif request.quality == "worst":
            quality_format = "worst"
        
        ydl_opts = {
            "outtmpl": output_template,
            "format": quality_format,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=True)
        
        downloaded_files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
        if not downloaded_files:
            return VideoDownloadResponse(success=False, message="Download failed", error="No file downloaded")
        
        file_path = os.path.join(temp_dir, downloaded_files[0])
        filename = downloaded_files[0]
        file_size = os.path.getsize(file_path)
        
        r2_service = R2Service()
        file_id = str(uuid.uuid4())
        r2_key = f"downloads/{file_id}/{filename}"
        content_type = r2_service._get_content_type(filename)
        
        upload_result = r2_service.upload_file(file_path, r2_key, content_type)
        
        if not upload_result.get("success"):
            logger.error(f"R2 upload failed: {upload_result.get('error')}")
            return VideoDownloadResponse(
                success=False,
                message="Upload to R2 failed",
                error=upload_result.get("error")
            )
        
        logger.info(f"Upload to R2 successful: {upload_result['url']}")
        
        video_info = {
            "title": info.get("title", "Unknown"),
            "duration": info.get("duration", 0),
            "uploader": info.get("uploader", "Unknown"),
            "filename": filename,
            "file_size": file_size,
            "local_path": "",
            "downloaded_at": "",
        }
        
        download_info = {
            "requested_quality": request.quality or "best",
            "actual_format": info.get("ext", "Unknown"),
            "resolution": f"{info.get('width', 'N/A')}x{info.get('height', 'N/A')}",
            "format": info.get("ext", "Unknown"),
            "filesize_approx": file_size,
        }
        
        return VideoDownloadResponse(
            success=True,
            message="Download and upload successful",
            r2_url=upload_result["url"],
            r2_key=r2_key,
            video_info=video_info,
            download_info=download_info
        )
        
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        return VideoDownloadResponse(
            success=False,
            message="Internal server error",
            error=str(e)
        )
    finally:
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp dir: {cleanup_error}")
