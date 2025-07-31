import os
import uuid
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

import yt_dlp
from export_video.constants import (
    DOWNLOAD_TIMEOUT, DOWNLOAD_TEMP_DIR, MAX_VIDEO_SIZE, 
    DEFAULT_VIDEO_QUALITY, SUPPORTED_DOWNLOAD_SITES
)
from r2_storage import R2Storage
from config import settings

logger = logging.getLogger(__name__)

class VideoDownloadService:
    def __init__(self):
        self.r2_storage = R2Storage()
        self.temp_dir = Path(DOWNLOAD_TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_download_id(self) -> str:
        """Generate unique download ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"video_{timestamp}_{unique_id}"
    
    def is_supported_url(self, url: str) -> bool:
        """Check if URL is from supported video platforms"""
        try:
            for site in SUPPORTED_DOWNLOAD_SITES:
                if site in url.lower():
                    return True
            return False
        except Exception:
            return False
    
    async def download_video(self, url: str, quality: str = None) -> Dict[str, Any]:
        """Download video from URL and upload to Cloudflare R2"""
        try:
            if not self.is_supported_url(url):
                return {
                    "success": False,
                    "error": "Unsupported video platform or invalid URL"
                }
            
            download_id = self.generate_download_id()
            logger.info(f"Starting video download {download_id} from URL: {url}")
            
            # Configure yt-dlp options
            quality_format = quality or DEFAULT_VIDEO_QUALITY
            output_template = str(self.temp_dir / f"{download_id}_%(title)s.%(ext)s")
            
            ydl_opts = {
                'outtmpl': output_template,
                'format': quality_format,
                'noplaylist': True,
                'extractaudio': False,
                'audioformat': 'mp3',
                'timeout': DOWNLOAD_TIMEOUT,
                'max_filesize': MAX_VIDEO_SIZE,
                'ignoreerrors': False,
                'no_warnings': False,
                'quiet': True,
                'no_color': True
            }
            
            # Download video metadata first
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                try:
                    info = ydl.extract_info(url, download=False)
                    if not info:
                        return {
                            "success": False,
                            "error": "Could not extract video information"
                        }
                    
                    video_title = info.get('title', 'Unknown')
                    video_duration = info.get('duration', 0)
                    video_uploader = info.get('uploader', 'Unknown')
                    
                    logger.info(f"Video info - Title: {video_title}, Duration: {video_duration}s")
                    
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to extract video info: {str(e)}"
                    }
            
            # Download the actual video
            downloaded_file = None
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                # Find the downloaded file
                downloaded_files = list(self.temp_dir.glob(f"{download_id}_*"))
                if not downloaded_files:
                    return {
                        "success": False,
                        "error": "Download completed but file not found"
                    }
                
                downloaded_file = downloaded_files[0]
                file_size = downloaded_file.stat().st_size
                
                logger.info(f"Video downloaded successfully: {downloaded_file.name}, Size: {file_size} bytes")
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Download failed: {str(e)}"
                }
            
            # Store in local storage and upload to Cloudflare R2 simultaneously
            try:
                # Read file content for local storage
                with open(downloaded_file, 'rb') as f:
                    file_content = f.read()
                

                
                # Upload to Cloudflare R2
                upload_result = await self.upload_to_cloudflare(download_id, downloaded_file, info)
                
                    
            except Exception as e:
                self.cleanup_local_file(downloaded_file)
                return {
                    "success": False,
                    "error": f"Upload to Cloudflare failed: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Video download service error: {str(e)}")
            return {
                "success": False,
                "error": f"Service error: {str(e)}"
            }
    
    async def upload_to_cloudflare(self, download_id: str, file_path: Path, video_info: Dict) -> Dict[str, Any]:
        """Upload video file to Cloudflare R2"""
        try:
            # Generate R2 key with organized structure (using same pattern as uploads for consistency)
            file_extension = file_path.suffix
            clean_title = "".join(c for c in video_info.get('title', 'video') if c.isalnum() or c in (' ', '-', '_')).strip()
            clean_title = clean_title.replace(' ', '_')[:50]  # Limit length
            
            # Use same pattern as uploads so file_id extraction works
            filename = f"{clean_title}{file_extension}"
            r2_key = f"uploads/{download_id}/{filename}"
            
            # Determine content type
            content_type = "video/mp4"
            if file_extension.lower() in ['.webm']:
                content_type = "video/webm"
            elif file_extension.lower() in ['.mkv']:
                content_type = "video/x-matroska"
            elif file_extension.lower() in ['.avi']:
                content_type = "video/x-msvideo"
            
            # Upload to R2
            upload_result = self.r2_storage.upload_file(
                str(file_path),
                r2_key,
                content_type
            )
            
            if upload_result["success"]:
                return {
                    "success": True,
                    "cloudflare": {
                        "url": upload_result["url"],
                        "r2_key": upload_result["r2_key"],
                        "bucket": upload_result["bucket"],
                        "size": upload_result["size"],
                        "content_type": upload_result["content_type"],
                        "uploaded_at": datetime.now().isoformat()
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"R2 upload failed: {upload_result.get('error', 'Unknown error')}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Upload process failed: {str(e)}"
            }
# Global instance
video_download_service = VideoDownloadService() 