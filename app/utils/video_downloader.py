import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import yt_dlp

from app.services.export_video.constants import (
    DOWNLOAD_TIMEOUT,
    DEFAULT_VIDEO_QUALITY,
    PLATFORM_VIDEO_QUALITY,
)
from app.config.settings import settings
from app.services.r2_service import get_r2_service
from app.utils.shared_memory import set_upload_status, update_upload_status

logger = logging.getLogger(__name__)


class VideoDownloadService:
    """Download media (video/audio) from any supported URL and keep them locally.

    The downloaded file is stored under:
        {settings.TEMP_DIR}/dub_{job_id}/{filename}

    Supports 800+ sites via yt-dlp. No Cloudflare / R2 upload happens here – 
    keeping things simple and local as requested.
    """

    def __init__(self) -> None:
        # R2Service will be initialized lazily when needed
        self._r2_service = None
    
    @property
    def r2_service(self):
        """Get R2Service instance with lazy initialization"""
        if self._r2_service is None:
            self._r2_service = get_r2_service()
        return self._r2_service

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def download_video(self, url: str, quality: str | None = None, platform_optimized: bool = True) -> Dict[str, Any]:
        """Download media (video/audio) from any URL and return metadata.

        Args:
            url:  The media URL from any supported site (800+ sites via yt-dlp).
            quality: Optional yt-dlp format string.
            platform_optimized: If True, use platform-optimized quality for social media uploads.
        Returns:
            A dict ready to be fed into the FastAPI response.
        """
        try:
            # Let yt-dlp handle URL validation naturally - it supports many more sites
            # than our hardcoded list

            # Generate a unique job ID and matching storage directory
            job_id = self.r2_service.generate_job_id()
            job_dir = Path(settings.TEMP_DIR) / f"dub_{job_id}"
            job_dir.mkdir(parents=True, exist_ok=True)

            # Set initial status
            set_upload_status(job_id, {
                "status": "downloading",
                "progress": 0,
                "message": "Starting download...",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "url": url
            })

            # Use platform-optimized quality by default for better social media compatibility
            if quality:
                quality_format = quality
            elif platform_optimized:
                quality_format = PLATFORM_VIDEO_QUALITY
            else:
                quality_format = DEFAULT_VIDEO_QUALITY
            output_template = str(job_dir / "%(title)s.%(ext)s")

            # Update status while getting metadata
            update_upload_status(job_id, {
                "progress": 10,
                "message": "Extracting video information..."
            })

            # Grab basic metadata first
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    update_upload_status(job_id, {
                        "status": "failed",
                        "message": "Could not extract video information"
                    })
                    return {"success": False, "error": "Could not extract video information"}

                video_title = info.get("title", "Unknown")
                video_duration = info.get("duration", 0)
                video_uploader = info.get("uploader", "Unknown")

            # Update status before download
            update_upload_status(job_id, {
                "progress": 30,
                "message": "Downloading video...",
                "video_title": video_title,
                "video_duration": video_duration
            })

            ydl_opts = {
                "outtmpl": output_template,
                "format": quality_format,
                "noplaylist": True,
                "timeout": DOWNLOAD_TIMEOUT,
                "ignoreerrors": False,
                "no_warnings": True,
                "quiet": True,
                "no_color": True,
            }

            # Actual download
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            downloaded_files = list(job_dir.glob("*"))
            if not downloaded_files:
                update_upload_status(job_id, {
                    "status": "failed",
                    "message": "Download completed but file not found"
                })
                return {"success": False, "error": "Download completed but file not found"}

            downloaded_file = downloaded_files[0]
            file_size = downloaded_file.stat().st_size

            # Set final success status
            update_upload_status(job_id, {
                "status": "done",
                "progress": 100,
                "message": "Download completed successfully",
                "file_url": str(downloaded_file),
                "original_filename": downloaded_file.name,
                "file_size": file_size,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "video_info": {
                    "title": video_title,
                    "duration": video_duration,
                    "uploader": video_uploader,
                    "filename": downloaded_file.name,
                    "file_size": file_size,
                    "local_path": str(downloaded_file)
                }
            })

            return {
                "success": True,
                "message": "Media downloaded successfully",
                "job_id": job_id,
                "video_info": {
                    "title": video_title,
                    "duration": video_duration,
                    "uploader": video_uploader,
                    "filename": downloaded_file.name,
                    "file_size": file_size,
                    "local_path": str(downloaded_file),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        except Exception as e:
            logger.error(f"Video download error: {e}")
            if 'job_id' in locals():
                update_upload_status(job_id, {
                    "status": "failed",
                    "progress": 0,
                    "message": f"Download failed: {str(e)}"
                })
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------
    def cleanup_old_files(self, max_age_hours: int = settings.LOCAL_STORAGE_RETENTION_HOURS) -> None:
        """Remove dub_* folders older than *max_age_hours* inside TEMP_DIR."""
        try:
            cutoff_ts = time.time() - max_age_hours * 3600
            root_dir = Path(settings.TEMP_DIR)
            for path in root_dir.glob("dub_*"):
                try:
                    if path.is_dir() and path.stat().st_mtime < cutoff_ts:
                        shutil.rmtree(path, ignore_errors=True)
                        logger.info(f"🧹 Removed old dub folder: {path}")
                except Exception:
                    continue
        except Exception:
            pass


# Shared singleton instance
video_download_service = VideoDownloadService()