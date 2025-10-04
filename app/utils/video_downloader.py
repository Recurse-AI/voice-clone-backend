import logging
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any

import yt_dlp

from app.config.settings import settings
from app.utils.cleanup_utils import cleanup_utils

logger = logging.getLogger(__name__)


class VideoDownloadService:

    def __init__(self) -> None:
        self._downloaded_files = {}
        self.cookie_file = "youtube_cookies.txt"
        self.user_cookie_file = None
    
    def _preprocess_facebook_url(self, url: str) -> str:
        if "facebook.com/share/v/" in url.lower():
            import re
            match = re.search(r'/share/v/([^/?]+)', url)
            if match:
                return f"https://www.facebook.com/video.php?v={match.group(1)}"
        return url

    def _generate_job_id(self) -> str:
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
        if d['status'] == 'finished':
            logger.info(f"Download complete: {d.get('filename', 'file')}")

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
                return [{"format": "best"}, {"format": "worst"}]
            elif is_audio:
                return [
                    {"format": "ba/b", "extractor_args": {"youtube": {"player_client": ["web"]}}},
                    {"format": "ba/worst", "extractor_args": {"youtube": {"player_client": ["mweb"]}}},
                ]
            else:
                return [
                    {"format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"},
                    {"format": "bestvideo+bestaudio/best"},
                    {"format": "18"},
                ]

        elif "video unavailable" in error_msg:
            raise Exception("Media is private or deleted")
        elif "sign in to confirm your age" in error_msg:
            raise Exception("Age-restricted content")
        elif "unsupported url" in error_msg and "facebook.com" in error_msg:
            raise Exception("Facebook URL not accessible")

        return None


    async def get_available_formats(self, url: str) -> Dict[str, Any]:
        try:
            url = self._preprocess_facebook_url(url)
            
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    return {"success": False, "error": "Could not extract video information"}
                
                formats = info.get("formats", [])
                video_formats = [f for f in formats if f.get("vcodec") != "none"]
                audio_formats = [f for f in formats if f.get("vcodec") == "none" and f.get("acodec") != "none"]
                
                best_audio = max(audio_formats, key=lambda x: x.get("abr", 0)) if audio_formats else None
                best_audio_id = best_audio.get("format_id") if best_audio else None
                best_audio_size = best_audio.get("filesize") or best_audio.get("filesize_approx", 0) if best_audio else 0
                
                seen_formats = {}
                
                for f in sorted(video_formats, key=lambda x: (x.get("acodec", "none") != "none"), reverse=True):
                    height = f.get("height", 0)
                    if height <= 0:
                        continue
                    
                    ext = f.get("ext", "mp4")
                    filesize = f.get("filesize") or f.get("filesize_approx", 0)
                    has_audio = f.get("acodec", "none") != "none"
                    format_id = f.get("format_id")
                    format_key = f"{height}_{ext}"
                    
                    if format_key in seen_formats:
                        existing = seen_formats[format_key]
                        if existing.get("has_audio") and not has_audio:
                            continue
                        if filesize <= existing.get("_filesize", 0):
                            continue
                    
                    estimated_size = filesize + (best_audio_size if not has_audio else 0)
                    
                    seen_formats[format_key] = {
                        "format_id": format_id,
                        "resolution": f"{height}p",
                        "ext": ext,
                        "filesize_mb": round(estimated_size / (1024*1024), 2) if estimated_size else 0,
                        "fps": f.get("fps", 30),
                        "vcodec": f.get("vcodec", "").split(".")[0] if f.get("vcodec") else "unknown",
                        "has_audio": has_audio,
                        "needs_audio_merge": not has_audio,
                        "audio_format_id": best_audio_id if not has_audio else None,
                        "note": "With audio" if has_audio else "HD (auto-merge audio)",
                        "quality": f.get("quality", 0),
                        "_filesize": filesize
                    }
                
                format_list = list(seen_formats.values())
                for fmt in format_list:
                    fmt.pop("_filesize", None)
                
                format_list.sort(key=lambda x: (int(x["resolution"].replace("p", "")), x["has_audio"]), reverse=True)
                
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
            logger.error(f"Format extraction error: {e}")
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


    async def download_video_with_audio(self, url: str, video_format_id: str, audio_format_id: str = "bestaudio", output_format: str = "mp4") -> Dict[str, Any]:
        try:
            url = self._preprocess_facebook_url(url)
            job_id = self._generate_job_id()
            job_dir = Path(settings.TEMP_DIR) / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            
            output_template = str(job_dir / "%(title)s.%(ext)s")
            ydl_opts = {
                "outtmpl": output_template,
                "format": f"{video_format_id}+{audio_format_id}",
                "merge_output_format": output_format,
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
            logger.error(f"Merge download error: {e}")
            return {"success": False, "error": str(e)}

    async def download_video(self, url: str, format_id: str | None = None, quality: str | None = None, 
                           resolution: str | None = None, max_filesize: str | None = None,
                           format_preference: str | None = None, audio_quality: str | None = None,
                           prefer_free_formats: bool = False, include_subtitles: bool = False,
                           user_cookie_file: str | None = None) -> Dict[str, Any]:
        self.user_cookie_file = user_cookie_file
        try:
            url = self._preprocess_facebook_url(url)
            job_id = self._generate_job_id()
            job_dir = Path(settings.TEMP_DIR) / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            
            is_audio = any(x in url.lower() for x in ['soundcloud.com', 'spotify.com', 'bandcamp.com'])
            is_direct_audio = any(url.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'])
            
            if format_id:
                quality_format = format_id if "-" in str(format_id) else f"{format_id}+bestaudio/{format_id}/bestvideo+bestaudio/best"
            else:
                quality_format = self._get_format_selector(quality, resolution, max_filesize, is_audio)

            output_template = str(job_dir / "%(title)s.%(ext)s")
            
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    return {"success": False, "error": "Could not extract video information"}
                
                video_title = info.get("title", "Unknown")
                video_duration = info.get("duration", 0)
                available_formats = info.get("formats", [])
                
                if not available_formats:
                    return {"success": False, "error": "No formats available"}

                audio_only_formats = [f for f in available_formats if f.get("vcodec") == "none" and f.get("acodec") != "none"]
                if audio_only_formats and not [f for f in available_formats if f.get("vcodec") != "none"]:
                    is_audio = True
                    quality_format = self._get_format_selector(quality, resolution, max_filesize, is_audio)

                format_info = self._analyze_available_formats(available_formats, quality_format, resolution)
            
            ydl_opts = {
                "outtmpl": output_template,
                "format": quality_format,
                "noplaylist": True,
                "timeout": 600,
                "quiet": False,
                "embed_subs": include_subtitles,
                "writesubtitles": include_subtitles,
                "progress_hooks": [self._progress_hook],
            }
            
            # Add cookie file for HD YouTube downloads
            import os
            cookie_to_use = None
            
            if self.user_cookie_file and os.path.exists(self.user_cookie_file):
                cookie_to_use = self.user_cookie_file
                logger.info(f"Using user cookies: {self.user_cookie_file}")
            elif os.path.exists(self.cookie_file):
                cookie_to_use = self.cookie_file
                logger.info(f"Using server cookies: {self.cookie_file}")
            
            if cookie_to_use:
                ydl_opts["cookiefile"] = cookie_to_use
            else:
                ydl_opts["extractor_args"] = {
                    "youtube": {"player_client": ["android", "web"]},
                    "facebook": {"player_client": ["mobile", "web"]}
                }
                ydl_opts["http_headers"] = {
                    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5_1 like Mac OS X) AppleWebKit/605.1.15"
                }
            
            if is_audio:
                ydl_opts["extractaudio"] = True
                ydl_opts["audioformat"] = format_preference or "mp3"
                ydl_opts["audioquality"] = audio_quality or "192"
            elif not is_direct_audio:
                ydl_opts["merge_output_format"] = format_preference or "mp4"
            
            download_success = False
            try:
                await self._download_with_retry(ydl_opts, url)
                download_success = True
            except Exception as download_error:
                logger.error(f"Download failed: {str(download_error)[:100]}")
                fallback_configs = self._get_fallback_configs(download_error, is_audio, is_direct_audio)
                
                if fallback_configs:
                    last_error = download_error
                    for idx, config in enumerate(fallback_configs):
                        ydl_opts["format"] = config["format"]
                        if "extractor_args" in config:
                            ydl_opts["extractor_args"] = config["extractor_args"]
                        if is_audio:
                            ydl_opts["extractaudio"] = True
                            ydl_opts["audioformat"] = format_preference or "mp3"
                        try:
                            ydl = yt_dlp.YoutubeDL(ydl_opts)
                            ydl.download([url])
                            download_success = True
                            break
                        except Exception as e:
                            last_error = e
                            continue
                    
                    if not download_success:
                        return {"success": False, "error": f"All attempts failed: {str(last_error)[:100]}"}
                
                if not download_success:
                    return {"success": False, "error": str(download_error)[:100]}
            
            downloaded_files = list(job_dir.glob("*"))
            if not downloaded_files:
                return {"success": False, "error": "File not found"}
            
            downloaded_file = downloaded_files[0]
            file_size = downloaded_file.stat().st_size
            
            actual_duration = video_duration
            if not video_duration:
                try:
                    import subprocess
                    result = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(downloaded_file)],
                        capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        actual_duration = float(result.stdout.strip())
                except Exception:
                    pass
            
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

            if resolution:
                requested_height = int(resolution.replace("p", "")) if resolution.replace("p", "").isdigit() else 0
                actual_height = format_info.get("best_available_height", 0)
                response["requested_resolution"] = f"{requested_height}p"
                response["resolution_matched"] = format_info.get("resolution_match", False)
                if requested_height > 0 and actual_height > 0:
                    response["quality_note"] = f"{actual_height}p ({'match' if actual_height >= requested_height else 'best available'})"

            return response
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return {"success": False, "error": str(e)}


    def get_file_path(self, job_id: str) -> Dict[str, Any]:
        try:
            if job_id in self._downloaded_files:
                file_info = self._downloaded_files[job_id]
                file_path = Path(file_info["path"])

                if not file_path.exists():
                    del self._downloaded_files[job_id]
                    return {"success": False, "error": "File not found"}

                if datetime.now(timezone.utc) - file_info["created_at"] > timedelta(minutes=30):
                    self._cleanup_file(job_id)
                    return {"success": False, "error": "File expired"}

                return {"success": True, "file_path": str(file_path), "filename": file_info["filename"]}

            job_dir = Path(settings.TEMP_DIR) / job_id
            if job_dir.exists():
                media_ext = ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.mp3', '.wav', '.m4a']
                for file_path in job_dir.glob("*"):
                    if file_path.suffix.lower() in media_ext:
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)
                        if datetime.now(timezone.utc) - file_mtime > timedelta(hours=1):
                            try:
                                import shutil
                                shutil.rmtree(job_dir)
                            except:
                                pass
                            return {"success": False, "error": "File expired"}
                        return {"success": True, "file_path": str(file_path), "filename": file_path.name}

            return {"success": False, "error": "File not found"}

        except Exception as e:
            logger.error(f"Get file error: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_file(self, job_id: str) -> Dict[str, Any]:
        try:
            deleted_files = self._cleanup_file(job_id)
            if not deleted_files:
                return {"success": False, "error": "File not found"}
            return {"success": True, "message": "Deleted", "deleted_files": deleted_files}
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return {"success": False, "error": str(e)}
    
    def _schedule_auto_cleanup(self, job_id: str) -> None:
        cleanup_utils.schedule_auto_cleanup(job_id, 30)

    def _cleanup_file(self, job_id: str) -> list:
        try:
            if job_id in self._downloaded_files:
                file_info = self._downloaded_files[job_id]
                deleted_files = cleanup_utils.delete_file_path(str(file_info["job_dir"]))
                del self._downloaded_files[job_id]
                return deleted_files
            cleanup_utils.cleanup_job(job_id)
            return []
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return []

    def cleanup_old_files(self) -> None:
        cleanup_utils.cleanup_old_files()

    def cleanup_specific_job(self, job_id: str) -> None:
        cleanup_utils.cleanup_job(job_id)
        if job_id in self._downloaded_files:
            del self._downloaded_files[job_id]

video_download_service = VideoDownloadService()