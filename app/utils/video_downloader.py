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
    
    def _build_quality_format(
        self, 
        quality: str | None, 
        resolution: str | None, 
        max_filesize: str | None,
        format_preference: str | None, 
        audio_quality: str | None, 
        prefer_free_formats: bool
    ) -> str:
        """Build simplified and robust yt-dlp format selector."""
        
        # Handle common quality presets with simple, reliable selectors
        if quality == "worst":
            return "worst"
        
        # If no specific requirements, use simple "best" 
        if not resolution and not max_filesize and not format_preference:
            return "best"
        
        # Build a simple format chain with only essential options
        format_options = []
        
        # Try user preferences first, but keep it simple
        if resolution:
            res_num = resolution.replace("p", "")
            if max_filesize:
                format_options.append(f"best[height<={res_num}][filesize<{max_filesize}]")
            format_options.append(f"best[height<={res_num}]")
        
        if format_preference:
            if max_filesize:
                format_options.append(f"best[ext={format_preference}][filesize<{max_filesize}]")
            format_options.append(f"best[ext={format_preference}]")
        
        if max_filesize:
            format_options.append(f"best[filesize<{max_filesize}]")
        
        # Always add reliable fallbacks
        format_options.append("best")
        format_options.append("worst")
        
        # Join options with "/" for fallback chain (max 5 options for simplicity)
        return "/".join(format_options[:5])
    
    def _analyze_available_formats(self, formats: list, requested_format: str) -> Dict[str, Any]:
        """Analyze available formats and provide detailed information."""
        
        if not formats:
            return {
                "available_formats": [],
                "selected_format": {},
                "resolution": "Unknown",
                "ext": "Unknown",
                "vcodec": "Unknown",
                "acodec": "Unknown",
                "filesize": "Unknown"
            }
        
        # Extract useful format information
        available_formats = []
        for fmt in formats:
            if fmt.get("vcodec") != "none":  # Skip audio-only formats for main list
                format_info = {
                    "format_id": fmt.get("format_id", ""),
                    "ext": fmt.get("ext", ""),
                    "resolution": f"{fmt.get('width', 'N/A')}x{fmt.get('height', 'N/A')}",
                    "filesize": fmt.get("filesize") or fmt.get("filesize_approx", "Unknown"),
                    "vcodec": fmt.get("vcodec", ""),
                    "acodec": fmt.get("acodec", ""),
                    "fps": fmt.get("fps", ""),
                    "quality": fmt.get("quality", ""),
                }
                available_formats.append(format_info)
        
        # Sort by quality/resolution (best first)
        available_formats.sort(
            key=lambda x: (x.get("quality", 0) or 0, 
                          int(x["resolution"].split("x")[1]) if "x" in str(x["resolution"]) and x["resolution"].split("x")[1].isdigit() else 0), 
            reverse=True
        )
        
        # Find the best match for requested format (simplified)
        selected_format = available_formats[0] if available_formats else {}
        
        return {
            "available_formats": available_formats[:10],  # Limit to top 10 to avoid huge responses
            "selected_format": selected_format,
            "resolution": selected_format.get("resolution", "Unknown"),
            "ext": selected_format.get("ext", "Unknown"),
            "vcodec": selected_format.get("vcodec", "Unknown"),
            "acodec": selected_format.get("acodec", "Unknown"),
            "filesize": selected_format.get("filesize", "Unknown")
        }

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def download_video(
        self, 
        url: str, 
        quality: str | None = None,
        resolution: str | None = None,
        max_filesize: str | None = None,
        format_preference: str | None = None,
        audio_quality: str | None = None,
        prefer_free_formats: bool = False,
        include_subtitles: bool = False
    ) -> Dict[str, Any]:
        """Download video from URL with advanced quality controls.
        
        Args:
            url: Video URL to download
            quality: yt-dlp quality format (default: "best")
            resolution: Preferred resolution height (e.g., "720", "1080")
            max_filesize: Maximum file size (e.g., "100M", "1G")
            format_preference: Preferred video format (e.g., "mp4", "webm")
            audio_quality: Audio quality preference
            prefer_free_formats: Whether to prefer open formats over proprietary
            include_subtitles: Whether to download subtitles if available
        
        Returns:
            Dict with success status and detailed file/format info
        """
        try:
            # Generate unique job ID and directory
            job_id = self._generate_job_id()
            job_dir = Path(settings.TEMP_DIR) / f"dub_{job_id}"
            job_dir.mkdir(parents=True, exist_ok=True)
            
            # Build smart quality format based on parameters
            quality_format = self._build_quality_format(
                quality, resolution, max_filesize, format_preference, 
                audio_quality, prefer_free_formats
            )
            
            output_template = str(job_dir / "%(title)s.%(ext)s")
            
            # Get video info and available formats first
            with yt_dlp.YoutubeDL({"quiet": True, "listformats": False}) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    return {"success": False, "error": "Could not extract video information from URL"}
                
                video_title = info.get("title", "Unknown")
                video_duration = info.get("duration", 0)
                video_uploader = info.get("uploader", "Unknown")
                available_formats = info.get("formats", [])
                
                # Validate that formats are available
                if not available_formats:
                    return {
                        "success": False, 
                        "error": "No video formats available for this URL",
                        "video_info": {
                            "title": video_title,
                            "duration": video_duration,
                            "uploader": video_uploader
                        }
                    }
                
                # Get format details for the selected quality
                format_info = self._analyze_available_formats(available_formats, quality_format)
                
                # Log the quality format being used for debugging
                logger.info(f"Using quality format: {quality_format}")
                logger.info(f"Available formats count: {len(available_formats)}")
            
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
                "extractaudio": False,
                "embed_subs": include_subtitles,
                "writesubtitles": include_subtitles,
                "writeautomaticsub": include_subtitles,
            }
            
            # Download the file with enhanced error handling
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except Exception as download_error:
                error_msg = str(download_error)
                
                # Handle specific yt-dlp errors with helpful messages
                if "Requested format is not available" in error_msg:
                    # Try one more time with just "best" as a final fallback
                    logger.info("Format not available, trying simple 'best' fallback")
                    try:
                        ydl_opts["format"] = "best"
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([url])
                    except Exception:
                        return {
                            "success": False,
                            "error": "No compatible video formats found for download. This video may be restricted or in an unsupported format.",
                            "video_info": {
                                "title": video_title,
                                "duration": video_duration,
                                "uploader": video_uploader
                            },
                            "available_formats": format_info.get("available_formats", [])[:5]
                        }
                elif "Video unavailable" in error_msg:
                    return {"success": False, "error": "Video is unavailable or private"}
                elif "Sign in to confirm your age" in error_msg:
                    return {"success": False, "error": "Video requires age verification and cannot be downloaded"}
                elif "timeout" in error_msg.lower():
                    return {"success": False, "error": "Download timeout - the video server may be slow or unavailable"}
                else:
                    return {"success": False, "error": f"Download failed: {error_msg}"}
            
            # Find downloaded file
            downloaded_files = list(job_dir.glob("*"))
            if not downloaded_files:
                return {"success": False, "error": "Download completed but file not found in expected directory"}
            
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
                },
                "download_info": {
                    "requested_quality": quality_format,
                    "actual_format": format_info.get("selected_format", {}),
                    "resolution": format_info.get("resolution", "Unknown"),
                    "format": format_info.get("ext", "Unknown"),
                    "video_codec": format_info.get("vcodec", "Unknown"),
                    "audio_codec": format_info.get("acodec", "Unknown"),
                    "filesize_approx": format_info.get("filesize", "Unknown"),
                },
                "available_formats": format_info.get("available_formats", [])
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
        """Clean up only truly orphaned dub_* folders that are older than 1 hour."""
        try:
            from datetime import datetime, timezone, timedelta
            
            root_dir = Path(settings.TEMP_DIR)
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
            
            for path in root_dir.glob("dub_*"):
                try:
                    if path.is_dir():
                        # Get directory modification time
                        dir_mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                        
                        # Only delete if directory is older than 1 hour
                        if dir_mtime < cutoff_time:
                            # Additional safety check: ensure no recent file activity
                            has_recent_activity = False
                            for file_path in path.rglob("*"):
                                if file_path.is_file():
                                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)
                                    if file_mtime >= cutoff_time:
                                        has_recent_activity = True
                                        break
                            
                            if not has_recent_activity:
                                shutil.rmtree(path, ignore_errors=True)
                                logger.info(f"🧹 Removed orphaned folder (older than 1h): {path}")
                            else:
                                logger.debug(f"🧹 Skipping folder with recent activity: {path}")
                        else:
                            logger.debug(f"🧹 Skipping recent folder: {path}")
                except Exception as e:
                    logger.warning(f"🧹 Error checking folder {path}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"🧹 Error during cleanup: {e}")
            pass
    
    def cleanup_specific_job(self, job_id: str) -> None:
        """Immediately clean up folders for specific completed/failed/cancelled job"""
        try:
            from app.services.dub.audio_utils import AudioUtils
            import os
            
            # Clean up specific job directories
            temp_patterns = [
                f"dub_{job_id}",                    # Main job folder  
                f"voice_cloning/dub_job_{job_id}"   # Voice cloning folder
            ]
            
            for pattern in temp_patterns:
                temp_dir = os.path.join(settings.TEMP_DIR, pattern)
                if os.path.exists(temp_dir):
                    AudioUtils.remove_temp_dir(folder_path=temp_dir)
                    logger.info(f"🧹 Immediately removed {job_id} directory: {temp_dir}")
            
            # Remove from tracking if present
            if job_id in self._downloaded_files:
                del self._downloaded_files[job_id]
                logger.info(f"🧹 Removed {job_id} from tracking")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup job {job_id}: {e}")


# Shared singleton instance
video_download_service = VideoDownloadService()