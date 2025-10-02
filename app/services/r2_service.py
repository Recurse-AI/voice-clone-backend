import boto3
import os
import logging
import uuid
import time
import concurrent.futures
from typing import Dict, Any, List
from app.config.settings import settings

logger = logging.getLogger(__name__)

class R2Service:
    """Clean and simple R2 Storage service"""
    
    def __init__(self):
        """Initialize R2 client with settings - lazy initialization for faster startup"""
        # Cloudflare R2 uses 'auto' but boto3 needs 'us-east-1' for compatibility
        self._region = settings.R2_REGION if settings.R2_REGION != "auto" else "us-east-1"
        self._client = None
        self.bucket_name = settings.R2_BUCKET_NAME
        logger.info(f"R2Service configured - bucket: {self.bucket_name}")
    
    @property
    def client(self):
        """Lazy initialization of boto3 client to speed up startup"""
        if self._client is None:
            self._client = boto3.client(
                's3',
                endpoint_url=settings.R2_ENDPOINT_URL,
                aws_access_key_id=settings.R2_ACCESS_KEY_ID,
                aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
                region_name=self._region,
                config=boto3.session.Config(
                    retries={'max_attempts': 2},
                    read_timeout=10,
                    connect_timeout=5
                )
            )
            logger.info(f"R2 client initialized with optimized timeouts")
        return self._client
    
    def upload_file(self, local_path: str, r2_key: str, content_type: str = "application/octet-stream") -> Dict[str, Any]:
        """Upload file to R2 storage with automatic optimization for large files"""
        try:
            if not os.path.exists(local_path):
                return {"success": False, "error": f"File not found: {local_path}"}

            file_size = os.path.getsize(local_path)

            # Use chunked upload for files larger than 100MB
            from app.config.constants import LARGE_FILE_THRESHOLD_MB
            threshold_bytes = LARGE_FILE_THRESHOLD_MB * 1024 * 1024

            def _do_upload():
                if file_size > threshold_bytes:
                    return self._chunked_upload(local_path, r2_key, content_type)
                else:
                    return self._standard_upload(local_path, r2_key, content_type)

            # Simple exponential backoff retry
            max_attempts = 4
            base_delay = 0.5
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return _do_upload()
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(f"Upload attempt {attempt} failed for {r2_key}: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        break

            logger.error(f"❌ Upload failed after retries: {r2_key} → {last_error}")
            return {"success": False, "error": str(last_error) if last_error else "Upload failed"}
        
        except Exception as e:
            logger.error(f"❌ Upload failed: {r2_key} → {e}")
            return {"success": False, "error": str(e)}
    
    def _standard_upload(self, local_path: str, r2_key: str, content_type: str) -> Dict[str, Any]:
        """Standard upload for smaller files"""
        with open(local_path, 'rb') as file:
            self.client.upload_fileobj(
                file,
                self.bucket_name,
                r2_key,
                ExtraArgs={'ContentType': content_type}
            )
        
        # Generate public URL
        public_url = f"{settings.R2_PUBLIC_URL}/{r2_key}" if settings.R2_PUBLIC_URL else f"https://{self.bucket_name}.r2.cloudflarestorage.com/{r2_key}"
        
        file_size = os.path.getsize(local_path)
        logger.info(f"Standard upload: {r2_key} ({file_size} bytes)")
        
        return {
            "success": True,
            "r2_key": r2_key,
            "url": public_url,
            "size": file_size
        }
    
    def _chunked_upload(self, local_path: str, r2_key: str, content_type: str) -> Dict[str, Any]:
        """Memory-efficient upload for large files using streaming"""
        file_size = os.path.getsize(local_path)
        
        # Custom file reader that streams in chunks
        class ChunkedReader:
            def __init__(self, file_path, chunk_size=50*1024*1024):  # 50MB chunks
                self.file_path = file_path
                self.chunk_size = chunk_size
                self._file = None
                self._total_read = 0
                
            def __enter__(self):
                self._file = open(self.file_path, 'rb')
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self._file:
                    self._file.close()
                    
            def read(self, size=-1):
                if size == -1:
                    return self._file.read()
                
                # Read in optimized chunks
                chunk = self._file.read(min(size, self.chunk_size))
                self._total_read += len(chunk)
                
                # Memory cleanup every 100MB read
                if self._total_read % (100 * 1024 * 1024) == 0:
                    import gc
                    gc.collect()
                    
                return chunk
        
        # Upload with streaming
        with ChunkedReader(local_path) as reader:
            self.client.upload_fileobj(
                reader,
                self.bucket_name,
                r2_key,
                ExtraArgs={'ContentType': content_type}
            )
        
        # Generate public URL
        public_url = f"{settings.R2_PUBLIC_URL}/{r2_key}" if settings.R2_PUBLIC_URL else f"https://{self.bucket_name}.r2.cloudflarestorage.com/{r2_key}"
        
        logger.info(f"Chunked upload: {r2_key} ({file_size} bytes)")
        
        return {
            "success": True,
            "r2_key": r2_key,
            "url": public_url,
            "size": file_size
        }
    
    def delete_file(self, r2_key: str) -> Dict[str, Any]:
        """Delete file from R2 storage"""
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=r2_key)
            logger.info(f"🗑️ Deleted: {r2_key}")
            return {"success": True}
        except Exception as e:
            logger.error(f"❌ Delete failed: {r2_key} → {e}")
            return {"success": False, "error": str(e)}

    def _calculate_optimal_workers(self, file_count: int, avg_file_size_mb: float = None) -> int:
        """Smart worker calculation based on file count and size"""
        # Base calculation on file count
        if file_count <= 3:
            base_workers = 2
        elif file_count <= 10:
            base_workers = 3
        elif file_count <= 25:
            base_workers = 4
        elif file_count <= 50:
            base_workers = 6
        elif file_count <= 100:
            base_workers = 8
        elif file_count <= 200:
            base_workers = 10
        else:
            base_workers = 12

        # Adjust for file size (if provided)
        if avg_file_size_mb:
            if avg_file_size_mb > 100:  # Large files (>100MB)
                base_workers = max(2, base_workers - 2)  # Reduce workers for large files
            elif avg_file_size_mb < 1:  # Very small files (<1MB)
                base_workers = min(8, base_workers + 1)  # Can use more workers

        return min(base_workers, 16)  # Cap at 16

    def upload_directory(self, job_id: str, local_dir: str, exclude_files: List[str] = None, max_workers: int = None) -> Dict[str, Any]:
        """
        Upload all files from a directory to R2 using parallel processing
        max_workers: Auto-calculated if None, or manual override
        Best for: Large folders, good network bandwidth, time-sensitive uploads
        """
        if not os.path.exists(local_dir):
            return {"success": False, "error": f"Directory not found: {local_dir}"}

        exclude_files = exclude_files or []
        results = {}
        upload_tasks = []

        # Collect valid files and calculate average size
        total_size = 0
        valid_files = []

        for filename in os.listdir(local_dir):
            file_path = os.path.join(local_dir, filename)

            if not os.path.isfile(file_path) or filename in exclude_files:
                continue

            # Only upload supported audio/video files
            if not filename.lower().endswith(('.wav', '.mp3', '.mp4', '.json', '.srt', '.flac', '.m4a', '.aac', '.ogg')):
                continue

            file_size = os.path.getsize(file_path)
            total_size += file_size

            content_type = self._get_content_type(filename)
            base_path = settings.R2_BASE_PATH.rstrip('/')
            sanitized_filename = self._sanitize_filename(filename)
            r2_key = f"{base_path}/temp/{job_id}/{sanitized_filename}"

            upload_tasks.append({
                'filename': filename,
                'file_path': file_path,
                'r2_key': r2_key,
                'content_type': content_type
            })

        if not upload_tasks:
            return {"message": "No valid files found for upload"}

        # Calculate optimal workers dynamically
        total_files = len(upload_tasks)
        avg_file_size_mb = (total_size / total_files) / (1024 * 1024) if total_files > 0 else 0

        if max_workers is None:
            max_workers = self._calculate_optimal_workers(total_files, avg_file_size_mb)
        else:
            # Respect manual override but cap at reasonable limit
            max_workers = min(max_workers, 16)

        logger.info(f"📊 Auto-selected {max_workers} workers for {total_files} files ({avg_file_size_mb:.1f}MB avg)")

        # Execute parallel uploads
        completed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._upload_single_file, task): task['filename']
                for task in upload_tasks
            }

            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                completed += 1

                try:
                    results[filename] = future.result()

                    # Progress logging every 10% or at completion
                    if completed % max(1, total_files // 10) == 0 or completed == total_files:
                        progress = (completed / total_files) * 100
                        logger.info(f"📊 Upload progress: {completed}/{total_files} ({progress:.1f}%)")

                except Exception as exc:
                    logger.error(f"❌ Upload failed: {filename} → {exc}")
                    results[filename] = {"success": False, "error": str(exc)}

        success_count = sum(1 for r in results.values() if r.get("success"))
        logger.info(f"📁 Parallel upload completed: {success_count}/{total_files} files")
        return results

    def _upload_single_file(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Helper for parallel file upload"""
        return self.upload_file(task['file_path'], task['r2_key'], task['content_type'])

    def generate_file_path(self, job_id: str, file_type: str = "", filename: str = "") -> str:
        """Generate R2 file path with base path"""
        base_path = settings.R2_BASE_PATH.rstrip('/')
        if filename:
            sanitized_filename = self._sanitize_filename(filename)
            return f"{base_path}/temp/{job_id}/{sanitized_filename}"
        return f"{base_path}/temp/{job_id}/"
    
    def generate_job_id(self) -> str:
        """Generate unique job ID with prefix"""
        return f"job_{uuid.uuid4()}"
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for URL-safe storage by replacing special characters"""
        import re
        
        # Extract extension
        name_parts = filename.rsplit('.', 1)
        name = name_parts[0]
        ext = f".{name_parts[1]}" if len(name_parts) > 1 else ""
        
        # Replace spaces with hyphens
        name = name.replace(' ', '-')
        
        # Keep only ASCII alphanumeric, hyphen, underscore, dot
        # Remove all other characters (including Unicode)
        name = re.sub(r'[^a-zA-Z0-9\-_.]', '-', name)
        
        # Remove multiple consecutive hyphens
        name = re.sub(r'-+', '-', name)
        
        # Remove leading/trailing hyphens
        name = name.strip('-')
        
        # If name becomes empty after sanitization, use timestamp
        if not name:
            from datetime import datetime
            name = f"file_{int(datetime.now().timestamp())}"
        
        return f"{name}{ext}"
    
    def _get_content_type(self, filename: str) -> str:
        """Get content type based on file extension"""
        ext = filename.lower().split('.')[-1]
        content_types = {
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'flac': 'audio/flac',
            'm4a': 'audio/mp4',
            'aac': 'audio/aac',
            'ogg': 'audio/ogg',
            'mp4': 'video/mp4',
            'json': 'application/json',
            'srt': 'application/x-subrip'
        }
        return content_types.get(ext, 'application/octet-stream')

# No global state - each caller creates its own instance
# This is much simpler and faster than singleton pattern
