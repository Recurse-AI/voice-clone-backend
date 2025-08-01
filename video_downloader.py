import logging
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import yt_dlp

from export_video.constants import (
    DOWNLOAD_TIMEOUT,
    DEFAULT_VIDEO_QUALITY,
    SUPPORTED_DOWNLOAD_SITES,
)
from config import settings
from r2_storage import R2Storage

logger = logging.getLogger(__name__)


class VideoDownloadService:
    """Download videos from supported platforms and keep them locally.

    The downloaded file is stored under:
        {settings.TEMP_DIR}/dub_{job_id}/{video_filename}

    No Cloudflare / R2 upload happens here – keeping things simple and local as
    requested.
    """

    def __init__(self) -> None:
        # Re-use the same ID generator used elsewhere for a consistent format
        self.r2_storage = R2Storage()

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def _is_supported_url(self, url: str) -> bool:
        try:
            url_l = url.lower()
            return any(site in url_l for site in SUPPORTED_DOWNLOAD_SITES)
        except Exception:
            return False

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def download_video(self, url: str, quality: str | None = None) -> Dict[str, Any]:
        """Download a video and return metadata.

        Args:
            url:  The video URL (YouTube, Vimeo, …).
            quality: Optional yt-dl format string.
        Returns:
            A dict ready to be fed into the FastAPI response.
        """
        try:
            if not self._is_supported_url(url):
                return {
                    "success": False,
                    "error": "Unsupported video platform or invalid URL",
                }

            # Generate a unique job ID and matching storage directory
            job_id = self.r2_storage.generate_job_id()
            job_dir = Path(settings.TEMP_DIR) / f"dub_{job_id}"
            job_dir.mkdir(parents=True, exist_ok=True)

            quality_format = quality or DEFAULT_VIDEO_QUALITY
            output_template = str(job_dir / "%(title)s.%(ext)s")

            # Grab basic metadata first – helpful for the final response
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    return {"success": False, "error": "Could not extract video information"}

                video_title = info.get("title", "Unknown")
                video_duration = info.get("duration", 0)
                video_uploader = info.get("uploader", "Unknown")

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
                return {"success": False, "error": "Download completed but file not found"}

            downloaded_file = downloaded_files[0]
            file_size = downloaded_file.stat().st_size

            return {
                "success": True,
                "message": "Video downloaded successfully",
                "job_id": job_id,
                "video_info": {
                    "title": video_title,
                    "duration": video_duration,
                    "uploader": video_uploader,
                    "filename": downloaded_file.name,
                    "file_size": file_size,
                    "local_path": str(downloaded_file),
                    "downloaded_at": datetime.now().isoformat(),
                },
            }
        except Exception as e:
            logger.error(f"Video download error: {e}")
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
