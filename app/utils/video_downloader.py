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

    def _build_quality_format(
        self,
        quality: str | None,
        resolution: str | None,
        max_filesize: str | None
    ) -> str:
        if quality == "worst":
            return "worst"

        # Always prefer merged formats, with comprehensive fallbacks
        base_formats = [
            "bv*+ba/best",  # Best merged video+audio
            "bv+ba/bestvideo+bestaudio",  # Alternative merged
            "best[height<=1080]",  # Best up to 1080p
            "best[height<=720]",  # Best up to 720p
            "best[ext=mp4]",  # Best MP4
            "best"  # Any best format
        ]

        size_filter = f"[filesize<{max_filesize}]" if max_filesize else ""

        if resolution:
            res_num = resolution.replace("p", "")
            try:
                res_int = int(res_num)
                # Smart resolution logic with size filtering
                resolution_formats = [
                    f"bv*[height>={res_int}]+ba[height>={res_int}]{size_filter}",
                    f"best[height>={res_int}]{size_filter}",
                    f"bv*[height<={res_int}]+ba[height<={res_int}]{size_filter}",
                    f"best[height<={res_int}]{size_filter}"
                ]
                return "/".join(resolution_formats + base_formats)
            except ValueError:
                logger.warning(f"Invalid resolution: {resolution}")

        # Apply size filter to all base formats
        if size_filter:
            base_formats = [f"{fmt}{size_filter}" for fmt in base_formats]

        return "/".join(base_formats)

    async def _download_with_retry(self, ydl_opts: Dict[str, Any], url: str, max_retries: int = 5) -> yt_dlp.YoutubeDL:
        last_error = None
        for attempt in range(max_retries):
            try:
                # For 403 errors, try different extractor args on retry
                if attempt > 0:
                    ydl_opts_copy = ydl_opts.copy()
                    # Try different client configurations for retries
                    clients = [
                        {"youtube": {"player_client": ["android", "web"], "player_skip": ["js", "configs"]}},
                        {"youtube": {"player_client": ["web"], "player_skip": ["js"]}},
                        {"youtube": {"player_client": ["ios"], "player_skip": ["js", "configs"]}},
                        {"youtube": {"player_client": ["tvhtml5"], "player_skip": ["js"]}}
                    ]
                    ydl_opts_copy["extractor_args"] = clients[attempt % len(clients)]
                else:
                    ydl_opts_copy = ydl_opts

                ydl = yt_dlp.YoutubeDL(ydl_opts_copy)
                ydl.download([url])
                return ydl
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()

                if attempt < max_retries - 1:
                    # Longer delays for YouTube-specific errors
                    if "403" in error_msg or "forbidden" in error_msg:
                        delay = 5 + (attempt * 3)  # 5s, 8s, 11s, 14s
                        logger.warning(f"YouTube 403 error, retrying with different client in {delay}s...")
                    elif "unavailable" in error_msg:
                        delay = 3 + (attempt * 2)  # 3s, 5s, 7s, 9s
                    elif "timeout" in error_msg or "connection" in error_msg:
                        delay = 2 ** attempt  # 1s, 2s, 4s, 8s
                    else:
                        delay = 2 ** attempt

                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_retries} download attempts failed")

        raise last_error

    def _handle_download_error(self, error: Exception, ydl_opts: Dict[str, Any], url: str, attempt: int = 0) -> yt_dlp.YoutubeDL:
        error_msg = str(error).lower()

        # Try multiple fallback strategies for format issues
        if "requested format is not available" in error_msg:
            fallback_formats = [
                "bv*+ba/best",  # Best merged
                "best[height<=1080]",  # Best up to 1080p
                "best[height<=720]",  # Best up to 720p
                "best[ext=mp4]",  # Best MP4
                "best"  # Any best format
            ]
            if attempt < len(fallback_formats):
                ydl_opts["format"] = fallback_formats[attempt]
                logger.info(f"Trying fallback format: {fallback_formats[attempt]}")
                return yt_dlp.YoutubeDL(ydl_opts)

        elif "video unavailable" in error_msg:
            raise Exception("Video is private or deleted")

        elif "sign in to confirm your age" in error_msg:
            raise Exception("Age-restricted content")

        elif "geo" in error_msg or "blocked" in error_msg or "403" in error_msg:
            # Enhanced geo-bypass with different strategies
            geo_strategies = [
                {"geo_bypass": True, "geo_bypass_country": "US"},
                {"geo_bypass": True, "geo_bypass_country": "US", "source_address": "0.0.0.0"},
                {"geo_bypass": True, "geo_bypass_country": "US", "extractor_args": {"youtube": {"player_client": ["android", "web"]}}},
                {"geo_bypass": True, "geo_bypass_country": "US", "extractor_args": {"youtube": {"player_client": ["ios"], "player_skip": ["js"]}}}
            ]
            if attempt < len(geo_strategies):
                ydl_opts.update(geo_strategies[attempt])
                logger.info(f"Trying geo-bypass strategy {attempt + 1}")
                return yt_dlp.YoutubeDL(ydl_opts)

        raise error

    def _analyze_available_formats(self, formats: list, requested_format: str, requested_resolution: str | None = None) -> Dict[str, Any]:
        if not formats:
            return {"available_formats": [], "selected_format": {}, "resolution": "Unknown",
                   "ext": "Unknown", "filesize": "Unknown", "resolution_match": False}

        video_formats = [f for f in formats if f.get("vcodec") != "none"]
        if not video_formats:
            return {"available_formats": [], "selected_format": {}, "resolution": "Unknown",
                   "ext": "Unknown", "filesize": "Unknown", "resolution_match": False}

        def sort_key(fmt):
            height = fmt.get("height", 0) or 0
            return (height, fmt.get("fps", 0) or 0, fmt.get("tbr", 0) or 0)

        video_formats.sort(key=sort_key, reverse=True)
        selected = video_formats[0]

        # Check if requested resolution is available
        resolution_match = False
        if requested_resolution:
            req_height = int(requested_resolution.replace("p", "")) if requested_resolution.replace("p", "").isdigit() else 0
            if req_height > 0:
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
            
            quality_format = self._build_quality_format(quality, resolution, max_filesize)
            
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
            
            # Download with enhanced retry and fallback mechanism
            format_attempt = 0
            max_format_attempts = 5

            while format_attempt < max_format_attempts:
                try:
                    await self._download_with_retry(ydl_opts, url)
                    break  # Success, exit the format retry loop
                except Exception as download_error:
                    error_msg = str(download_error).lower()

                    # Check if this is a format availability error that we can retry
                    if ("requested format is not available" in error_msg or
                        "403" in error_msg or "forbidden" in error_msg) and format_attempt < max_format_attempts - 1:

                        format_attempt += 1
                        logger.warning(f"Format attempt {format_attempt} failed: {error_msg[:100]}... Trying fallback...")

                        try:
                            # Try different format/geobypass strategy
                            ydl = self._handle_download_error(download_error, ydl_opts, url, format_attempt - 1)
                            ydl.download([url])
                            break  # Success with fallback
                        except Exception as fallback_error:
                            if format_attempt >= max_format_attempts - 1:
                                # All format attempts exhausted
                                error_msg = str(fallback_error).lower()
                                if "video unavailable" in error_msg:
                                    return {"success": False, "error": "Video is private or deleted"}
                                elif "age" in error_msg:
                                    return {"success": False, "error": "Age-restricted content"}
                                elif "timeout" in error_msg:
                                    return {"success": False, "error": "Download timeout"}
                                else:
                                    return {"success": False, "error": f"Download failed after {format_attempt} format attempts: {str(fallback_error)}"}
                            # Continue to next format attempt
                    else:
                        # Not a format/geobypass error, or we've exhausted attempts
                        if "video unavailable" in error_msg:
                            return {"success": False, "error": "Video is private or deleted"}
                        elif "age" in error_msg:
                            return {"success": False, "error": "Age-restricted content"}
                        elif "timeout" in error_msg:
                            return {"success": False, "error": "Download timeout"}
                        else:
                            return {"success": False, "error": f"Download failed: {str(download_error)}"}
            
            # Find downloaded file
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
        cleanup_utils.schedule_auto_cleanup(job_id, 30)

    def _cleanup_file(self, job_id: str) -> list:
        deleted_files = []

        try:
            if job_id in self._downloaded_files:
                file_info = self._downloaded_files[job_id]
                job_dir = Path(file_info["job_dir"])

                # List files before deletion for return value
                if job_dir.exists() and job_dir.is_dir():
                    for file_path in job_dir.rglob("*"):
                        if file_path.is_file():
                            deleted_files.append(str(file_path))

                # Use centralized cleanup
                cleanup_utils.cleanup_job(job_id)

                # Remove from tracking
                del self._downloaded_files[job_id]

        except Exception as e:
            logger.error(f"Error cleaning up {job_id}: {e}")

        return deleted_files



    
    def cleanup_old_files(self) -> None:
        cleanup_utils.cleanup_old_files()

    def cleanup_specific_job(self, job_id: str) -> None:
        cleanup_utils.cleanup_job(job_id)
        if job_id in self._downloaded_files:
            del self._downloaded_files[job_id]

video_download_service = VideoDownloadService()