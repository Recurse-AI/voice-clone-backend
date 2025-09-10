from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import logging
from app.schemas import (
    VideoDownloadRequest,
    VideoDownloadResponse,
    FileDeleteRequest,
    FileDeleteResponse,
)

router = APIRouter()

logger = logging.getLogger(__name__)

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
            # Build video_info from the new response structure
            video_info = {
                "title": result.get("title", "Unknown"),
                "duration": result.get("duration", 0),
                "uploader": "Unknown",  # Not in new structure, set default
                "filename": result.get("filename", ""),
                "file_size": result.get("file_size", 0),
                "local_path": "",  # Not in new structure, set empty
                "downloaded_at": "",  # Not in new structure, set empty
            }

            # Build download_info from the new response structure
            download_info = {
                "requested_quality": result.get("quality_note", ""),
                "actual_format": result.get("format", "Unknown"),
                "resolution": result.get("resolution", "Unknown"),
                "format": result.get("format", "Unknown"),
                "filesize_approx": result.get("file_size", "Unknown"),
            }

            return VideoDownloadResponse(
                success=True,
                message="Download successful",
                job_id=result["job_id"],
                video_info=video_info,
                download_info=download_info,
                available_formats=result.get("available_formats", [])
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
