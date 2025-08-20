import boto3
import os
import logging
import uuid
from typing import Dict, Any
from app.config.settings import settings

logger = logging.getLogger(__name__)

class R2Service:
    """Clean and simple R2 Storage service"""
    
    def __init__(self):
        """Initialize R2 client with settings"""
        # Cloudflare R2 uses 'auto' but boto3 needs 'us-east-1' for compatibility
        region = settings.R2_REGION if settings.R2_REGION != "auto" else "us-east-1"
        
        self.client = boto3.client(
            's3',
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            region_name=region
        )
        self.bucket_name = settings.R2_BUCKET_NAME
        logger.info(f"✅ R2Service initialized - bucket: {self.bucket_name}")
    
    def upload_file(self, local_path: str, r2_key: str, content_type: str = "application/octet-stream") -> Dict[str, Any]:
        """Upload file to R2 storage"""
        try:
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
            logger.info(f"✅ Uploaded: {r2_key} ({file_size} bytes)")
            
            return {
                "success": True,
                "r2_key": r2_key,
                "url": public_url,
                "size": file_size
            }
            
        except Exception as e:
            logger.error(f"❌ Upload failed: {r2_key} → {e}")
            return {
                "success": False,
                "error": str(e)
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
    
    def upload_directory(self, job_id: str, local_dir: str) -> Dict[str, Any]:
        """Upload all files from a directory to R2"""
        results = {}
        
        try:
            for filename in os.listdir(local_dir):
                file_path = os.path.join(local_dir, filename)
                
                if not os.path.isfile(file_path):
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

# Global R2 service instance
_r2_service = None

def get_r2_service() -> R2Service:
    """Get R2Service singleton instance"""
    global _r2_service
    if _r2_service is None:
        _r2_service = R2Service()
    return _r2_service

def reset_r2_service():
    """Reset R2Service singleton"""
    global _r2_service
    _r2_service = None
