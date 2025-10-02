import logging
import shutil
import time
import threading
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any

import yt_dlp

from app.config.settings import settings
from app.utils.cleanup_utils import cleanup_utils

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

    def _get_format_selector(self, quality: str | None, resolution: str | None, max_filesize: str | None) -> str:
        if quality == "worst":
            return "worst"

        size_filter = f"[filesize<{max_filesize}]" if max_filesize else ""

        if resolution:
            try:
                res_int = int(resolution.replace("p", ""))
                return f"bv*[height<={res_int}]+ba[height<={res_int}]{size_filter}/best[height<={res_int}]{size_filter}/bv*+ba{size_filter}/best{size_filter}"
            except ValueError:
                pass

        return f"bv*+ba{size_filter}/best{size_filter}"

    async def _download_with_retry(self, ydl_opts: Dict[str, Any], url: str) -> yt_dlp.YoutubeDL:
        for attempt in range(3):
            try:
                ydl = yt_dlp.YoutubeDL(ydl_opts)
                ydl.download([url])
                return ydl
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise e

                # Simple exponential backoff
                delay = 2 ** attempt
                await asyncio.sleep(delay)

    def _get_fallback_formats(self, error: Exception) -> list[str] | None:
        error_msg = str(error).lower()

        if "requested format is not available" in error_msg:
            return ["best", "bestvideo+bestaudio/best", "bestvideo+bestaudio", "worst"]

        elif "video unavailable" in error_msg:
            raise Exception("Video is private or deleted")

        elif "sign in to confirm your age" in error_msg:
            raise Exception("Age-restricted content")

        return None


    def _analyze_available_formats(self, formats: list, requested_format: str, requested_resolution: str | None = None) -> Dict[str, Any]:
        if not formats:
            return {"available_formats": [], "selected_format": {}, "resolution": "Unknown",
                   "ext": "Unknown", "filesize": "Unknown", "resolution_match": False}

        video_formats = [f for f in formats if f.get("vcodec") != "none"]
        if not video_formats:
            return {"available_formats": [], "selected_format": {}, "resolution": "Unknown",
                   "ext": "Unknown", "filesize": "Unknown", "resolution_match": False}

        # Sort by quality (height first)
        video_formats.sort(key=lambda f: f.get("height", 0) or 0, reverse=True)
        selected = video_formats[0]

        # Check if requested resolution matches
        resolution_match = False
        if requested_resolution:
            req_height = int(requested_resolution.replace("p", "")) if requested_resolution.replace("p", "").isdigit() else 0
            available_heights = [f.get("height", 0) for f in video_formats if f.get("height")]
            resolution_match = any(h >= req_height for h in available_heights)

        return {
            "available_formats": video_formats[:10],
            "selected_format": selected,
            "resolution": f"{selected.get('width', 'N/A')}x{selected.get('height', 'N/A')}",
            "ext": selected.get("ext", "Unknown"),
            "filesize": selected.get("filesize") or selected.get("filesize_approx", "Unknown"),
            "resolution_match": resolution_match,
            "best_available_height": selected.get("height", 0)
        }


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
            job_dir = Path(settings.TEMP_DIR) / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            
            quality_format = self._get_format_selector(quality, resolution, max_filesize)
            logger.info(f"Initial format selector: {quality_format}")

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
                
                if not available_formats:
                    return {"success": False, "error": "No video formats available"}

                format_info = self._analyze_available_formats(available_formats, quality_format, resolution)
            
            # Download options with optimizations
            ydl_opts = {
                "outtmpl": output_template,
                "format": quality_format,
                "noplaylist": True,
                "timeout": 600,
                "ignoreerrors": False,
                "no_warnings": True,
                "quiet": True,
                "no_color": True,
                "extractaudio": False,
                "embed_subs": include_subtitles,
                "writesubtitles": include_subtitles,
                "writeautomaticsub": include_subtitles,
                "merge_output_format": "mp4",
                "hls_prefer_native": True,
                "geo_bypass": True,
                "geo_bypass_country": "US",
                "concurrent_fragments": 3,
                "fragment_retries": 10,
                "retry_sleep": 5,
                "socket_timeout": 30,
                "http_chunk_size": 10485760,
                "extractor_args": {"youtube": {"player_client": ["android", "web"], "player_skip": ["js", "configs"]}},
            }
            
            # Download with retry and fallback mechanism
            try:
                await self._download_with_retry(ydl_opts, url)
            except Exception as download_error:
                fallback_formats = self._get_fallback_formats(download_error)
                if fallback_formats:
                    last_error = download_error
                    for fallback_format in fallback_formats:
                        ydl_opts["format"] = fallback_format
                        try:
                            ydl = yt_dlp.YoutubeDL(ydl_opts)
                            ydl.download([url])
                            logger.info(f"Successfully downloaded with fallback format: {fallback_format}")
                            break
                        except Exception as e:
                            last_error = e
                            continue
                    else:
                        return {"success": False, "error": f"Download failed: {str(last_error)}"}
                else:
                    return {"success": False, "error": f"Download failed: {str(download_error)}"}
            
            # Find downloaded file!
            downloaded_files = list(job_dir.glob("*"))
            if not downloaded_files:
                return {"success": False, "error": "Download completed but file not found in expected directory"}
            
            downloaded_file = downloaded_files[0]
            file_size = downloaded_file.stat().st_size
            
            actual_duration = video_duration
            if not video_duration or video_duration <= 0:
                try:
                    import subprocess
                    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(downloaded_file)]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0 and result.stdout.strip():
                        actual_duration = float(result.stdout.strip())
                except Exception:
                    actual_duration = 0
            
            self._downloaded_files[job_id] = {
                "path": str(downloaded_file),
                "job_dir": str(job_dir),
                "created_at": datetime.now(timezone.utc),
                "filename": downloaded_file.name,
                "file_size": file_size
            }

            self._schedule_auto_cleanup(job_id)

            response = {
                "success": True,
                "job_id": job_id,
                "title": video_title,
                "duration": actual_duration,
                "filename": downloaded_file.name,
                "file_size": file_size,
                "resolution": format_info.get("resolution", "Unknown"),
                "format": format_info.get("ext", "Unknown"),
                "best_available_height": format_info.get("best_available_height", 0)
            }

            # Add resolution info if user requested specific resolution
            if resolution:
                requested_height = int(resolution.replace("p", "")) if resolution.replace("p", "").isdigit() else 0
                actual_height = format_info.get("best_available_height", 0)
                response["requested_resolution"] = f"{requested_height}p"
                response["resolution_matched"] = format_info.get("resolution_match", False)

                if requested_height > 0 and actual_height > 0:
                    if actual_height >= requested_height:
                        response["quality_note"] = f"Downloaded {actual_height}p (matches or exceeds {requested_height}p request)"
                    else:
                        response["quality_note"] = f"Downloaded {actual_height}p (best available, requested {requested_height}p not available)"

            return response
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return {"success": False, "error": str(e)}


    def get_file_path(self, job_id: str) -> Dict[str, Any]:
        """Get file path for serving, with expiry check"""
        try:
            # First try memory tracking
            if job_id in self._downloaded_files:
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

            # Fallback: Check file system directly (for RunPod/serverless environments)
            temp_dir = Path(settings.TEMP_DIR)
            job_dir = temp_dir / job_id

            if job_dir.exists() and job_dir.is_dir():
                # Look for video files in the job directory
                video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv']
                for file_path in job_dir.glob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                        # Check if file is too old (more than 1 hour to be safe)
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)
                        if datetime.now(timezone.utc) - file_mtime > timedelta(hours=1):
                            # Clean up old file
                            try:
                                import shutil
                                shutil.rmtree(job_dir)
                            except:
                                pass
                            return {"success": False, "error": "File has expired"}

                        return {
                            "success": True,
                            "file_path": str(file_path),
                            "filename": file_path.name
                        }

            return {"success": False, "error": "File not found or expired"}

        except Exception as e:
            logger.error(f"Error getting file path for {job_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_file(self, job_id: str) -> Dict[str, Any]:
        try:
            deleted_files = self._cleanup_file(job_id)
            
            if not deleted_files:
                job_dirs = [
                    Path(settings.TEMP_DIR) / job_id,
                    Path("tmp") / job_id,
                    Path("tmp") / "downloads" / job_id
                ]
                
                files_found = False
                for job_dir in job_dirs:
                    if job_dir.exists():
                        files_found = True
                        break
                        
                if not files_found:
                    return {"success": False, "error": f"No files found for job_id: {job_id}"}
            
            return {
                "success": True,
                "message": "File deleted successfully",
                "deleted_files": deleted_files
            }
        except Exception as e:
            logger.error(f"Error deleting file {job_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _schedule_auto_cleanup(self, job_id: str) -> None:
        cleanup_utils.schedule_auto_cleanup(job_id, 30)

    def _cleanup_file(self, job_id: str) -> list:
        try:
            deleted_files = []
            
            if job_id in self._downloaded_files:
                file_info = self._downloaded_files[job_id]
                job_dir_path = str(file_info["job_dir"])
                deleted_files = cleanup_utils.delete_file_path(job_dir_path)
                del self._downloaded_files[job_id]
            else:
                cleanup_utils.cleanup_job(job_id)
                
            return deleted_files
        except Exception as e:
            logger.error(f"Error cleaning up {job_id}: {e}")
            return []



    
    def cleanup_old_files(self) -> None:
        cleanup_utils.cleanup_old_files()

    def cleanup_specific_job(self, job_id: str) -> None:
        cleanup_utils.cleanup_job(job_id)
        if job_id in self._downloaded_files:
            del self._downloaded_files[job_id]

video_download_service = VideoDownloadService()