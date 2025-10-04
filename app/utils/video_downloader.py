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
    
    def _preprocess_facebook_url(self, url: str) -> str:
        """Preprocess Facebook URLs to handle share links better"""
        if "facebook.com/share/v/" in url.lower():
            # Extract video ID from share URL
            import re
            match = re.search(r'/share/v/([^/?]+)', url)
            if match:
                video_id = match.group(1)
                # Try to construct a direct video URL
                direct_url = f"https://www.facebook.com/video.php?v={video_id}"
                logger.info(f"Facebook share URL detected, trying direct URL: {direct_url}")
                return direct_url
            else:
                logger.warning("Facebook share URL detected but couldn't extract video ID")
        
        return url

    def _generate_job_id(self) -> str:
        """Generate unique job ID"""
        import uuid
        return str(uuid.uuid4())

    def _get_format_selector(self, quality: str | None, resolution: str | None, max_filesize: str | None, is_audio: bool = False) -> str:
        if is_audio:
            if quality == "worst":
                return "ba/worst"
            return "ba/b"
        
        if quality == "worst":
            return "worst[ext=mp4]/worst"
        return "best[ext=mp4]/best/18"

    def _progress_hook(self, d):
        if d['status'] == 'downloading':
            if 'total_bytes' in d:
                percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
                print(f"\rDownloading: {d['downloaded_bytes']/1024/1024:.1f}MB / {d['total_bytes']/1024/1024:.1f}MB ({percent:.1f}%)", end='', flush=True)
            elif 'total_bytes_estimate' in d:
                percent = (d['downloaded_bytes'] / d['total_bytes_estimate']) * 100
                print(f"\rDownloading: {d['downloaded_bytes']/1024/1024:.1f}MB / ~{d['total_bytes_estimate']/1024/1024:.1f}MB ({percent:.1f}%)", end='', flush=True)
            else:
                print(f"\rDownloading: {d['downloaded_bytes']/1024/1024:.1f}MB", end='', flush=True)
        elif d['status'] == 'finished':
            print(f"\n✓ Download completed: {d['filename']}")

    async def _download_with_retry(self, ydl_opts: Dict[str, Any], url: str) -> yt_dlp.YoutubeDL:
        for attempt in range(2):
            try:
                ydl = yt_dlp.YoutubeDL(ydl_opts)
                ydl.download([url])
                return ydl
            except Exception as e:
                if attempt == 1:
                    raise e
                await asyncio.sleep(2)

    def _get_fallback_configs(self, error: Exception, is_audio: bool = False, is_direct_audio: bool = False) -> list[dict] | None:
        error_msg = str(error).lower()

        if "requested format is not available" in error_msg or "http error 403" in error_msg or "forbidden" in error_msg:
            if is_direct_audio:
                # For direct audio files, just try different formats
                return [
                    {"format": "best"},
                    {"format": "worst"},
                ]
            elif is_audio:
                return [
                    {"format": "ba/b", "extractor_args": {"youtube": {"player_client": ["web"]}}},
                    {"format": "ba/worst", "extractor_args": {"youtube": {"player_client": ["mweb"]}}},
                    {"format": "ba/b", "extractor_args": {"youtube": {"player_client": ["android", "web"]}}},
                    {"format": "ba/b", "extractor_args": {"youtube": {"player_client": ["android"]}}},
                ]
            else:
                # Video fallbacks: try formats with audio first (more reliable)
                return [
                    {"format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"},  # Try mp4 with audio
                    {"format": "bestvideo+bestaudio/best"},  # Any format with audio
                    {"format": "best[ext=mp4]/best"},  # Best mp4 or any
                    {"format": "18"},  # Format 18 (360p with audio) - most reliable
                ]

        elif "video unavailable" in error_msg:
            raise Exception("Media is private or deleted")

        elif "sign in to confirm your age" in error_msg:
            raise Exception("Age-restricted content")

        elif "unsupported url" in error_msg and "facebook.com" in error_msg:
            raise Exception("Facebook URL is not accessible. Try using the direct video URL instead of share link.")

        return None


    async def get_available_formats(self, url: str) -> Dict[str, Any]:
        """Get list of available formats without downloading."""
        try:
            url = self._preprocess_facebook_url(url)
            
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    return {"success": False, "error": "Could not extract video information"}
                
                formats = info.get("formats", [])
                # Include ALL video formats (don't filter by URL - some formats have it missing)
                video_formats = [f for f in formats if f.get("vcodec") != "none"]
                audio_formats = [f for f in formats if f.get("vcodec") == "none" and f.get("acodec") != "none"]
                
                best_audio = max(audio_formats, key=lambda x: x.get("abr", 0)) if audio_formats else None
                best_audio_id = best_audio.get("format_id") if best_audio else None
                best_audio_size = best_audio.get("filesize") or best_audio.get("filesize_approx", 0) if best_audio else 0
                
                format_list = []
                seen_formats = {}  # Track best format for each resolution+ext combo
                
                # Show ALL video formats (including video-only for high quality)
                # Prefer formats with audio (more reliable to download)
                for f in sorted(video_formats, key=lambda x: (x.get("acodec", "none") != "none"), reverse=True):
                    height = f.get("height", 0)
                    if height <= 0:
                        continue
                    
                    ext = f.get("ext", "mp4")
                    filesize = f.get("filesize") or f.get("filesize_approx", 0)
                    acodec = f.get("acodec", "none")
                    has_audio = acodec != "none"
                    format_id = f.get("format_id")
                    
                    # Create unique key for resolution+extension
                    format_key = f"{height}_{ext}"
                    
                    # Prefer formats with audio (more downloadable)
                    if format_key in seen_formats:
                        existing = seen_formats[format_key]
                        # If existing has audio, keep it (unless new one is also with audio and bigger)
                        if existing.get("has_audio"):
                            if not has_audio:
                                continue  # Keep existing with audio
                            elif filesize <= existing.get("_filesize", 0):
                                continue  # Keep existing larger one
                        # If new has audio but existing doesn't, replace
                        elif has_audio:
                            pass  # Will replace below
                        # Both no audio, keep larger one
                        elif filesize <= existing.get("_filesize", 0):
                            continue
                    
                    # Estimate final size (video + audio if needed)
                    estimated_size = filesize
                    if not has_audio and best_audio_size:
                        estimated_size = filesize + best_audio_size
                    
                    # Check if format is likely downloadable (has audio or is streaming format)
                    is_downloadable = has_audio or filesize == 0 or "-" in str(format_id)
                    
                    format_obj = {
                        "format_id": format_id,
                        "resolution": f"{height}p",
                        "ext": ext,
                        "filesize_mb": round(estimated_size / (1024*1024), 2) if estimated_size else 0,
                        "fps": f.get("fps", 30),
                        "vcodec": f.get("vcodec", "").split(".")[0] if f.get("vcodec") else "unknown",
                        "has_audio": has_audio,
                        "needs_audio_merge": not has_audio,
                        "audio_format_id": best_audio_id if not has_audio else None,
                        "note": f"Includes audio (recommended)" if has_audio else "High quality (will merge with audio)",
                        "quality": f.get("quality", 0),
                        "is_downloadable": is_downloadable,
                        "_filesize": filesize  # Internal tracking
                    }
                    
                    seen_formats[format_key] = format_obj
                
                # Convert to list and sort (formats with audio first)
                format_list = list(seen_formats.values())
                for fmt in format_list:
                    fmt.pop("_filesize", None)  # Remove internal field
                
                # Sort: resolution desc, but formats with audio come first at same resolution
                format_list.sort(key=lambda x: (
                    int(x["resolution"].replace("p", "")),
                    x["has_audio"]  # True comes after False, so formats with audio last
                ), reverse=True)
                
                return {
                    "success": True,
                    "title": info.get("title"),
                    "duration": info.get("duration", 0),
                    "uploader": info.get("uploader"),
                    "thumbnail": info.get("thumbnail"),
                    "formats": format_list,
                    "has_audio_only": len(audio_formats) > 0,
                    "default_format": format_list[0] if format_list else None
                }
        except Exception as e:
            logger.error(f"Error getting formats: {e}")
            return {"success": False, "error": str(e)}

    def _analyze_available_formats(self, formats: list, requested_format: str, requested_resolution: str | None = None) -> Dict[str, Any]:
        if not formats:
            return {"available_formats": [], "selected_format": {}, "resolution": "Unknown",
                   "ext": "Unknown", "filesize": "Unknown", "resolution_match": False}

        video_formats = [f for f in formats if f.get("vcodec") != "none"]
        if not video_formats:
            return {"available_formats": [], "selected_format": {}, "resolution": "Unknown",
                   "ext": "Unknown", "filesize": "Unknown", "resolution_match": False}

        video_formats.sort(key=lambda f: f.get("height", 0) or 0, reverse=True)
        selected = video_formats[0]

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


    async def download_video_with_audio(
        self,
        url: str,
        video_format_id: str,
        audio_format_id: str = "bestaudio",
        output_format: str = "mp4"
    ) -> Dict[str, Any]:
        """Download video and audio separately, then merge with FFmpeg."""
        try:
            url = self._preprocess_facebook_url(url)
            job_id = self._generate_job_id()
            job_dir = Path(settings.TEMP_DIR) / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading video={video_format_id} + audio={audio_format_id}")
            
            # Combined format selector: video + audio
            format_selector = f"{video_format_id}+{audio_format_id}"
            output_template = str(job_dir / "%(title)s.%(ext)s")
            
            ydl_opts = {
                "outtmpl": output_template,
                "format": format_selector,
                "merge_output_format": output_format,
                "postprocessor_args": ["-ar", "44100"],  # Audio sample rate
                "noplaylist": True,
                "quiet": False,
            }
            
            await self._download_with_retry(ydl_opts, url)
            
            downloaded_files = list(job_dir.glob("*"))
            if not downloaded_files:
                return {"success": False, "error": "Download failed"}
            
            downloaded_file = downloaded_files[0]
            file_size = downloaded_file.stat().st_size
            
            self._downloaded_files[job_id] = {
                "path": str(downloaded_file),
                "job_dir": str(job_dir),
                "created_at": datetime.now(timezone.utc),
                "filename": downloaded_file.name,
                "file_size": file_size
            }
            
            self._schedule_auto_cleanup(job_id)
            
            return {
                "success": True,
                "job_id": job_id,
                "filename": downloaded_file.name,
                "file_size": file_size
            }
        except Exception as e:
            logger.error(f"Download with audio merge error: {e}")
            return {"success": False, "error": str(e)}

    async def download_video(
        self, 
        url: str, 
        format_id: str | None = None,
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
            format_id: Specific format ID (overrides quality/resolution)
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
            # Preprocess URL to handle Facebook share links
            url = self._preprocess_facebook_url(url)
            
            # Generate unique job ID and directory
            job_id = self._generate_job_id()
            job_dir = Path(settings.TEMP_DIR) / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            
            # Detect if URL is audio-only (platforms only, not direct files)
            is_audio = (
                'soundcloud.com' in url.lower() or
                'spotify.com' in url.lower() or
                'bandcamp.com' in url.lower()
            )
            
            # Check if it's a direct audio file (different handling)
            is_direct_audio = any(url.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'])
            
            if format_id:
                # Smart format selection with fallbacks
                # Try: specific format + audio, then format alone, then similar quality with audio
                logger.info(f"Using format_id {format_id} with automatic audio merge if needed")
                
                # If format has "-" it's likely a streaming format (more reliable)
                if "-" in str(format_id):
                    quality_format = format_id  # Direct download
                else:
                    # Try to merge with audio, with fallbacks
                    quality_format = f"{format_id}+bestaudio/{format_id}/bestvideo+bestaudio/best"
            else:
                quality_format = self._get_format_selector(quality, resolution, max_filesize, is_audio)
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
                    return {"success": False, "error": "No media formats available"}

                # Enhanced audio detection - check if only audio formats are available
                audio_only_formats = [f for f in available_formats if f.get("vcodec") == "none" and f.get("acodec") != "none"]
                if len(audio_only_formats) > 0 and len([f for f in available_formats if f.get("vcodec") != "none"]) == 0:
                    is_audio = True
                    quality_format = self._get_format_selector(quality, resolution, max_filesize, is_audio)
                    logger.info(f"Detected audio-only content, updated format selector: {quality_format}")

                format_info = self._analyze_available_formats(available_formats, quality_format, resolution)
            
            # Download options with optimizations
            ydl_opts = {
                "outtmpl": output_template,
                "format": quality_format,
                "noplaylist": True,
                "timeout": 600,
                "ignoreerrors": False,
                "no_warnings": False,
                "quiet": False,
                "no_color": True,
                "embed_subs": include_subtitles,
                "writesubtitles": include_subtitles,
                "writeautomaticsub": include_subtitles,
                "hls_prefer_native": True,
                "concurrent_fragments": 3,
                "fragment_retries": 5,
                "extractor_args": {
                    "youtube": {"player_client": ["android", "web"]},
                    "facebook": {"player_client": ["mobile", "web"]}
                },
                "http_headers": {
                    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5.1 Mobile/15E148 Safari/604.1"
                },
                "progress_hooks": [self._progress_hook],
            }
            
            # Configure options based on content type
            if is_audio:
                # For audio-only platform URLs, use extract-audio flag (like yt-dlp preset)
                ydl_opts["extractaudio"] = True
                ydl_opts["audioformat"] = format_preference or "mp3"
                ydl_opts["audioquality"] = audio_quality or "192"
            elif is_direct_audio:
                # For direct audio files, just download as-is (no extraction needed)
                logger.info("Direct audio file detected, downloading as-is")
            else:
                # For video URLs, merge video and audio
                ydl_opts["merge_output_format"] = format_preference or "mp4"
            
            # Download with retry and fallback mechanism
            download_success = False
            try:
                await self._download_with_retry(ydl_opts, url)
                download_success = True
            except Exception as download_error:
                logger.error(f"Initial download failed, trying fallbacks: {str(download_error)[:100]}")
                fallback_configs = self._get_fallback_configs(download_error, is_audio, is_direct_audio)
                if fallback_configs:
                    last_error = download_error
                    for idx, config in enumerate(fallback_configs):
                        client_info = config.get("extractor_args", {}).get("youtube", {}).get("player_client", ["direct"])[0]
                        logger.info(f"Trying fallback {idx+1}/{len(fallback_configs)}: format={config['format']}, client={client_info}")
                        ydl_opts["format"] = config["format"]
                        if "extractor_args" in config:
                            ydl_opts["extractor_args"] = config["extractor_args"]
                        ydl_opts["quiet"] = False
                        
                        # Ensure audio extraction is maintained for audio platforms
                        if is_audio:
                            ydl_opts["extractaudio"] = True
                            ydl_opts["audioformat"] = format_preference or "mp3"
                            ydl_opts["audioquality"] = audio_quality or "192"
                        try:
                            ydl = yt_dlp.YoutubeDL(ydl_opts)
                            ydl.download([url])
                            logger.info(f"✓ Fallback successful: format={config['format']}, client={client_info}")
                            download_success = True
                            break
                        except Exception as e:
                            logger.error(f"✗ Fallback {idx+1} failed: {str(e)[:100]}")
                            last_error = e
                            continue
                    
                    if not download_success:
                        return {"success": False, "error": f"All download attempts failed. Last error: {str(last_error)}"}
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
                # Look for media files in the job directory (video and audio)
                media_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv', '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac']
                for file_path in job_dir.glob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in media_extensions:
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