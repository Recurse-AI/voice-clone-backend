import boto3
import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid
from app.config.settings import settings

class R2Storage:
    def __init__(self):
        try:
            self.client = boto3.client(
                's3',
                endpoint_url=settings.R2_ENDPOINT_URL,
                aws_access_key_id=settings.R2_ACCESS_KEY_ID,
                aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
                region_name=settings.R2_REGION
            )
            self.bucket_name = settings.R2_BUCKET_NAME
            self.base_path = settings.R2_BASE_PATH

            print(f"R2Storage initialized with bucket: {self.bucket_name}, base_path: {self.base_path}")
        except Exception as e:
            self.client = None
            self.bucket_name = None
            self.base_path = settings.R2_BASE_PATH
            print(f"R2Storage initialized with bucket: {self.bucket_name}, base_path: {self.base_path}")
    
    def generate_job_id(self) -> str:
        """Generate unique audio processing ID"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"job_{timestamp}_{unique_id}"
    
    def generate_file_path(self, audio_id: str, file_type: str, filename: str) -> str:
        """Generate R2 file path. If file_type is empty/None it is omitted to store files directly inside the job_id folder."""
        date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        # Omit file_type folder if it is None or an empty string
        file_type_part = f"/{file_type}" if file_type else ""
        return f"{self.base_path}/{date_prefix}/{audio_id}{file_type_part}/{filename}"
    
    def upload_file(self, local_path: str, r2_key: str, content_type: str = "audio/wav") -> Dict[str, Any]:
        """Upload file to R2 bucket"""
        if not self.client:
            return {"success": False, "error": "R2 client not initialized"}
            
        try:
            with open(local_path, 'rb') as file:
                self.client.upload_fileobj(
                    file,
                    self.bucket_name,
                    r2_key,
                    ExtraArgs={'ContentType': content_type}
                )
            
            file_size = os.path.getsize(local_path)
            
            # Use public URL if configured, otherwise use default format
            if settings.R2_PUBLIC_URL:
                public_url = f"{settings.R2_PUBLIC_URL}/{r2_key}"
            else:
                public_url = f"https://{self.bucket_name}.r2.cloudflarestorage.com/{r2_key}"
            
            return {
                "success": True,
                "r2_key": r2_key,
                "bucket": self.bucket_name,
                "size": file_size,
                "content_type": content_type,
                "url": public_url
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
  
    def _enable_public_access(self):
        """Enable public access for bucket"""
        try:
            # Set bucket policy for public read access
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "PublicReadGetObject",
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{self.bucket_name}/*"
                    }
                ]
            }
            
            self.client.put_bucket_policy(
                Bucket=self.bucket_name,
                Policy=json.dumps(bucket_policy)
            )
            
            # Set CORS for web access
            cors_config = {
                'CORSRules': [
                    {
                        'AllowedHeaders': ['*'],
                        'AllowedMethods': ['GET', 'PUT', 'POST', 'DELETE'],
                        'AllowedOrigins': ['*'],
                        'ExposeHeaders': ['ETag']
                    }
                ]
            }
            
            self.client.put_bucket_cors(
                Bucket=self.bucket_name,
                CORSConfiguration=cors_config
            )
            
        except Exception:
            pass
    
    def get_storage_info(self, audio_id: str) -> Dict[str, Any]:
        """Get storage information for an audio processing job"""
        date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        return {
            "bucket": self.bucket_name,
            "base_path": f"{self.base_path}/{date_prefix}/{audio_id}",
            "audio_id": audio_id,
            "date": date_prefix,
            "access_url": f"https://{self.bucket_name}.r2.cloudflarestorage.com/{self.base_path}/{date_prefix}/{audio_id}"
        }
    
    def upload_audio_segments(self, job_id: str, process_temp_dir: str):
        """Upload all audio segment and info JSON files for a job. Returns per-file upload results."""
        if not self.client:
            return {"success": False, "error": "R2 client not initialized"}

        results = {}
        overall_success = True
        try:
            for fname in os.listdir(process_temp_dir):
                fpath = os.path.join(process_temp_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                # Allowed extensions for upload
                allowed_ext = (".wav", ".json", ".mp4", ".srt")
                if not fname.lower().endswith(allowed_ext):
                    continue

                # Store segments directly under the job_id folder in R2
                r2_key = self.generate_file_path(job_id, "", fname)
                if fname.endswith(".wav"):
                    content_type = "audio/wav"
                elif fname.endswith(".mp4"):
                    content_type = "video/mp4"
                elif fname.endswith(".srt"):
                    content_type = "application/x-subrip"
                else:
                    content_type = "application/json"
                upload_res = self.upload_file(fpath, r2_key, content_type=content_type)
                results[fname] = upload_res
                if not upload_res.get("success"):
                    overall_success = False
        except Exception as e:
            return {"success": False, "error": str(e)}

        return results

    def get_segment_urls(self, job_id: str, segment_uploads: dict):
        """Extract public URLs for successfully uploaded segment files."""
        if not segment_uploads:
            return {}
        urls = {}
        for fname, res in segment_uploads.items():
            if res.get("success") and res.get("url"):
                urls[fname] = res["url"]
        return urls

    def delete_file(self, r2_key: str):
        """Delete file from R2 bucket"""
        if not self.client:
            return {"success": False, "error": "R2 client not initialized"}
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=r2_key)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}