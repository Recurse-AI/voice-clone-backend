import boto3
import os
import logging
import uuid
import time
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
    
    def upload_directory(self, job_id: str, local_dir: str, exclude_files: List[str] = None) -> Dict[str, Any]:
        """Upload all files from a directory to R2"""
        results = {}

        try:
            exclude_files = exclude_files or []
            for filename in os.listdir(local_dir):
                file_path = os.path.join(local_dir, filename)

                if not os.path.isfile(file_path):
                    continue

                # Skip excluded files
                if filename in exclude_files:
                    logger.info(f"⏭️ Skipping excluded file: {filename}")
                    continue
                
                # Only upload supported files
                if not filename.lower().endswith(('.wav', '.mp3', '.mp4', '.json', '.srt', '.flac', '.m4a', '.aac', '.ogg')):
                    continue
                
                # Determine content type
                content_type = self._get_content_type(filename)
                
                # Generate R2 key with base path
                base_path = settings.R2_BASE_PATH.rstrip('/')
                r2_key = f"{base_path}/temp/{job_id}/{filename}"
                
                # Upload file
                result = self.upload_file(file_path, r2_key, content_type)
                results[filename] = result
            
            success_count = sum(1 for r in results.values() if r.get("success"))
            logger.info(f"📁 Uploaded {success_count}/{len(results)} files for job: {job_id}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Directory upload failed: {job_id} → {e}")
            return {"error": str(e)}
    
    def generate_file_path(self, job_id: str, file_type: str = "", filename: str = "") -> str:
        """Generate R2 file path with base path"""
        base_path = settings.R2_BASE_PATH.rstrip('/')
        return f"{base_path}/temp/{job_id}/{filename}" if filename else f"{base_path}/temp/{job_id}/"
    
    def generate_job_id(self) -> str:
        """Generate unique job ID with prefix"""
        return f"job_{uuid.uuid4()}"
    
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
