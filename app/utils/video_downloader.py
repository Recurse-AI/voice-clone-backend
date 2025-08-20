import logging
import shutil
import time
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any

import yt_dlp

from app.config.settings import settings

logger = logging.getLogger(__name__)


class VideoDownloadService:
    """Simple video downloader service.
    
    Downloads videos from URLs using yt-dlp and stores locally.
    Files auto-delete after 30 minutes.
    """

    def __init__(self) -> None:
        # Track downloaded files for auto cleanup
        self._downloaded_files = {}  # {job_id: {"path": str, "created_at": datetime}}
    
    def _generate_job_id(self) -> str:
        """Generate unique job ID"""
        import uuid
        return str(uuid.uuid4())

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def download_video(self, url: str, quality: str | None = None) -> Dict[str, Any]:
        """Download video from URL and store locally.
        
        Args:
            url: Video URL to download
            quality: yt-dlp quality format (default: "best")
        
        Returns:
            Dict with success status and file info
        """
        try:
            # Generate unique job ID and directory
            job_id = self._generate_job_id()
            job_dir = Path(settings.TEMP_DIR) / f"dub_{job_id}"
            job_dir.mkdir(parents=True, exist_ok=True)
            
            # Set quality format
            quality_format = quality or "best"
            output_template = str(job_dir / "%(title)s.%(ext)s")
            
            # Get video info first
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    return {"success": False, "error": "Could not extract video information"}
                
                video_title = info.get("title", "Unknown")
                video_duration = info.get("duration", 0)
                video_uploader = info.get("uploader", "Unknown")
            
            # Download options
            ydl_opts = {
                "outtmpl": output_template,
                "format": quality_format,
                "noplaylist": True,
                "timeout": 300,  # 5 minutes timeout
                "ignoreerrors": False,
                "no_warnings": True,
                "quiet": True,
                "no_color": True,
            }
            
            # Download the file
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Find downloaded file
            downloaded_files = list(job_dir.glob("*"))
            if not downloaded_files:
                return {"success": False, "error": "Download completed but file not found"}
            
            downloaded_file = downloaded_files[0]
            file_size = downloaded_file.stat().st_size
            
            # Track file for auto cleanup
            self._downloaded_files[job_id] = {
                "path": str(downloaded_file),
                "job_dir": str(job_dir),
                "created_at": datetime.now(timezone.utc),
                "filename": downloaded_file.name,
                "file_size": file_size
            }
            
            # Schedule auto cleanup after 30 minutes
            self._schedule_auto_cleanup(job_id)
            
            logger.info(f"Successfully downloaded: {downloaded_file.name} ({file_size} bytes)")
            
            return {
                "success": True,
                "message": "Download successful",
                "job_id": job_id,
                "video_info": {
                    "title": video_title,
                    "duration": video_duration,
                    "uploader": video_uploader,
                    "filename": downloaded_file.name,
                    "file_size": file_size,
                    "local_path": str(downloaded_file),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }
            }
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # File Management Methods
    # ------------------------------------------------------------------
    def get_file_path(self, job_id: str) -> Dict[str, Any]:
        """Get file path for serving, with expiry check"""
        try:
            if job_id not in self._downloaded_files:
                return {"success": False, "error": "File not found or expired"}
            
            file_info = self._downloaded_files[job_id]
            file_path = Path(file_info["path"])
            
            # Check if file still exists
            if not file_path.exists():
                # Clean up tracking if file doesn't exist
                del self._downloaded_files[job_id]
                return {"success": False, "error": "File not found on disk"}
            
            # Check if file is expired (30+ minutes old)
            created_at = file_info["created_at"]
            if datetime.now(timezone.utc) - created_at > timedelta(minutes=30):
                self._cleanup_file(job_id)
                return {"success": False, "error": "File has expired"}
            
            return {
                "success": True,
                "file_path": str(file_path),
                "filename": file_info["filename"]
            }
        except Exception as e:
            logger.error(f"Error getting file path for {job_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_file(self, job_id: str) -> Dict[str, Any]:
        """Manually delete downloaded file by job_id"""
        try:
            if job_id not in self._downloaded_files:
                return {"success": False, "error": "File not found"}
            
            file_info = self._downloaded_files[job_id]
            deleted_files = self._cleanup_file(job_id)
            
            return {
                "success": True,
                "message": "File deleted successfully",
                "deleted_files": deleted_files
            }
        except Exception as e:
            logger.error(f"Error deleting file {job_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _schedule_auto_cleanup(self, job_id: str) -> None:
        """Schedule automatic file cleanup after 30 minutes"""
        def delayed_cleanup():
            time.sleep(30 * 60)  # Wait 30 minutes
            self._cleanup_file(job_id)
        
        # Run cleanup in background thread
        cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
        cleanup_thread.start()
        logger.info(f"Scheduled auto cleanup for {job_id} in 30 minutes")
    
    def _cleanup_file(self, job_id: str) -> list:
        """Remove file and job directory for given job_id"""
        deleted_files = []
        try:
            if job_id in self._downloaded_files:
                file_info = self._downloaded_files[job_id]
                job_dir = Path(file_info["job_dir"])
                
                # Remove entire job directory
                if job_dir.exists() and job_dir.is_dir():
                    # List files before deletion
                    for file_path in job_dir.rglob("*"):
                        if file_path.is_file():
                            deleted_files.append(str(file_path))
                    
                    shutil.rmtree(job_dir, ignore_errors=True)
                    logger.info(f"🧹 Cleaned up job directory: {job_dir}")
                
                # Remove from tracking
                del self._downloaded_files[job_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up {job_id}: {e}")
        
        return deleted_files

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------
    def cleanup_old_files(self) -> None:
        """Clean up any orphaned dub_* folders that might exist."""
        try:
            root_dir = Path(settings.TEMP_DIR)
            for path in root_dir.glob("dub_*"):
                try:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                        logger.info(f"🧹 Removed orphaned folder: {path}")
                except Exception:
                    continue
        except Exception:
            pass


# Shared singleton instance
video_download_service = VideoDownloadService()