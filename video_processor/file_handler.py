"""
File Handler

Handles video file downloads and uploads with progress tracking.
Responsible for the first 10% of processing progress.
"""

import os
import requests
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FileHandler:
    """Handles video file operations for queue processing"""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    def handle_video_source(self, video_source: str, audio_id: str, is_file_upload: bool,
                          status_manager=None) -> Tuple[bool, str, Optional[str]]:
        """
        Handle video source (URL download or file upload processing)
        
        Args:
            video_source: Video URL or file path
            audio_id: Unique audio ID for this processing job
            is_file_upload: Whether this is a file upload or URL
            status_manager: Status manager for progress updates
            
        Returns:
            Tuple of (success: bool, local_file_path: str, error: Optional[str])
        """
        try:
            if status_manager:
                from status_manager import ProcessingStatus
                status_manager.update_status(audio_id, ProcessingStatus.DOWNLOADING, 5)
            
            if is_file_upload:
                return self._handle_file_upload(video_source, audio_id, status_manager)
            else:
                return self._handle_url_download(video_source, audio_id, status_manager)
                
        except Exception as e:
            error_msg = f"File handling failed: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg
    
    def _handle_file_upload(self, file_path: str, audio_id: str, status_manager=None) -> Tuple[bool, str, Optional[str]]:
        """Handle file upload - file is already saved locally"""
        try:
            if not os.path.exists(file_path):
                return False, "", "Uploaded file not found"
            
            # Validate file
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "", "Uploaded file is empty"
            
            # Check file extension
            allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in allowed_extensions:
                return False, "", f"Unsupported video format: {file_ext}"
            
            # File is already in temp directory, just verify it's readable
            try:
                with open(file_path, 'rb') as f:
                    # Read first few bytes to verify file is not corrupted
                    header = f.read(1024)
                    if len(header) == 0:
                        return False, "", "Uploaded file appears to be corrupted"
            except Exception as e:
                return False, "", f"Cannot read uploaded file: {str(e)}"
            
            # Update progress to 10%
            if status_manager:
                status_manager.set_progress(audio_id, 10)
            
            logger.info(f"File upload processed successfully for audio_id {audio_id}: {file_path}")
            return True, file_path, None
            
        except Exception as e:
            error_msg = f"File upload processing failed: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg
    
    def _handle_url_download(self, video_url: str, audio_id: str, status_manager=None) -> Tuple[bool, str, Optional[str]]:
        """Handle URL download with proper progress tracking"""
        try:
            # Parse URL and create filename
            parsed_url = urllib.parse.urlparse(video_url)
            filename = os.path.basename(parsed_url.path)
            if not filename or '.' not in filename:
                filename = "video.mp4"
            
            # Create local file path
            file_extension = Path(filename).suffix.lower()
            local_filename = f"{audio_id}_video{file_extension}"
            local_file_path = os.path.join(self.temp_dir, local_filename)
            
            # Start download with initial progress
            if status_manager:
                status_manager.set_progress(audio_id, 5)
            
            response = requests.get(video_url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Get content length for progress tracking
            content_length = response.headers.get('content-length')
            if content_length:
                content_length = int(content_length)
            
            downloaded = 0
            chunk_size = 8192
            progress_updated = False
            
            with open(local_file_path, "wb") as buffer:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        buffer.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progressive download progress (0% to 20%)
                        if content_length and status_manager and audio_id:
                            download_percent = (downloaded / content_length) * 100
                            # Map download progress to 0-20% range
                            progress = min(19, int(download_percent * 0.2))  # 0-20% range
                            status_manager.set_progress(audio_id, progress)
            
            # Verify downloaded file
            if not os.path.exists(local_file_path):
                return False, "", "Download failed - file not created"
            
            file_size = os.path.getsize(local_file_path)
            if file_size == 0:
                os.unlink(local_file_path)
                return False, "", "Download failed - empty file"
            
            # Validate file format
            allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
            if file_extension not in allowed_extensions:
                os.unlink(local_file_path)
                return False, "", f"Unsupported video format: {file_extension}"
            
            # Final progress update to 20%
            if status_manager:
                status_manager.set_progress(audio_id, 20)
                logger.info("Download completed - 20%")
            
            logger.info(f"URL download completed successfully for audio_id {audio_id}: {video_url} -> {local_file_path}")
            return True, local_file_path, None
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Download failed: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg
        except Exception as e:
            error_msg = f"URL download processing failed: {str(e)}"
            logger.error(error_msg)
            
            # Clean up partial download
            if 'local_file_path' in locals() and os.path.exists(local_file_path):
                try:
                    os.unlink(local_file_path)
                except:
                    pass
            
            return False, "", error_msg
    
    def validate_video_source(self, video_source: str, is_file_upload: bool) -> Tuple[bool, Optional[str]]:
        """
        Validate video source before processing
        
        Args:
            video_source: Video URL or file path
            is_file_upload: Whether this is a file upload or URL
            
        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        try:
            if is_file_upload:
                return self._validate_file_upload(video_source)
            else:
                return self._validate_url(video_source)
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _validate_file_upload(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate uploaded file"""
        if not file_path or not file_path.strip():
            return False, "File path is empty"
        
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check file size (basic validation)
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "File is empty"
            
            # Check if file is too large (e.g., 1GB limit)
            max_size = 1024 * 1024 * 1024  # 1GB
            if file_size > max_size:
                return False, f"File is too large ({file_size / (1024*1024):.1f}MB). Maximum size is 1GB"
        except Exception:
            return False, "Cannot access file size"
        
        # Check file extension
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in allowed_extensions:
            return False, f"Unsupported video format: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        
        return True, None
    
    def _validate_url(self, video_url: str) -> Tuple[bool, Optional[str]]:
        """Validate video URL"""
        if not video_url or not video_url.strip():
            return False, "Video URL is empty"
        
        # Check URL format
        if not video_url.startswith(('http://', 'https://')):
            return False, "Invalid URL format. Must start with http:// or https://"
        
        try:
            parsed_url = urllib.parse.urlparse(video_url)
            if not parsed_url.netloc:
                return False, "Invalid URL format"
        except Exception:
            return False, "Invalid URL format"
        
        # Optional: Check if URL is accessible (HEAD request)
        try:
            response = requests.head(video_url, timeout=10, allow_redirects=True)
            if response.status_code not in [200, 201, 202, 206]:
                return False, f"URL not accessible (status: {response.status_code})"
            
            # Check content type if available
            content_type = response.headers.get('content-type', '').lower()
            if content_type and not any(vid_type in content_type for vid_type in ['video', 'application/octet-stream']):
                # Allow octet-stream as some servers don't set proper content-type
                if 'text/html' in content_type:
                    return False, "URL points to a webpage, not a video file"
        except requests.exceptions.RequestException:
            # URL validation failed, but we'll let the download attempt proceed
            # This allows for URLs that don't respond to HEAD requests but work with GET
            pass
        
        return True, None
    
    def cleanup_file(self, file_path: str):
        """Clean up a temporary file"""
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {e}")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a file"""
        try:
            if not os.path.exists(file_path):
                return {}
            
            stat = os.stat(file_path)
            return {
                "filename": os.path.basename(file_path),
                "file_path": file_path,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "extension": Path(file_path).suffix.lower(),
                "created_at": stat.st_ctime,
                "modified_at": stat.st_mtime
            }
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return {} 