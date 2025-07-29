import boto3
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid
from config import settings

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
            
            # Skip connection test and public access setup during initialization
            # These will be done lazily when needed
            
        except Exception:
            self.client = None
            self.bucket_name = None
            self.base_path = settings.R2_BASE_PATH
    
    def generate_audio_id(self) -> str:
        """Generate unique audio processing ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"audio_{timestamp}_{unique_id}"
    
    def generate_file_path(self, audio_id: str, file_type: str, filename: str) -> str:
        """Generate R2 file path with good identification"""
        date_prefix = datetime.now().strftime("%Y/%m/%d")
        return f"{self.base_path}/{date_prefix}/{audio_id}/{file_type}/{filename}"
    
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
    
    def upload_audio_segments(self, audio_id: str, segments_dir: str) -> Dict[str, Any]:
        """Upload all audio segments and metadata to R2 with direct storage"""
        upload_results = {
            "audio_id": audio_id,
            "segments": {},
            "metadata": {},
            "cloned": {},
            "success": False,
            "files_uploaded": 0
        }
        
        segments_path = Path(segments_dir)
        
        try:
            # Upload segments directly
            segments_folder = segments_path / "segments"
            if segments_folder.exists():
                for file_path in segments_folder.iterdir():
                    if file_path.is_file():
                        file_type = "audio" if file_path.suffix in [".wav", ".mp3"] else "metadata"
                        r2_key = self.generate_file_path(audio_id, "segments", file_path.name)
                        
                        result = self.upload_file(
                            str(file_path),
                            r2_key,
                            "audio/wav" if file_path.suffix == ".wav" else "application/json"
                        )
                        
                        if result["success"]:
                            upload_results["segments"][file_path.name] = result
                            upload_results["files_uploaded"] += 1
            
            # Upload cloned segments directly
            cloned_folder = segments_path / "cloned_segments"  # Fixed: match voice_cloning service directory name
            if cloned_folder.exists():
                for cloned_file in cloned_folder.iterdir():
                    if cloned_file.is_file() and cloned_file.suffix == ".wav":
                        r2_key = self.generate_file_path(
                            audio_id, 
                            "segments/cloned_segments",  # Fixed: match voice_cloning service directory name
                            cloned_file.name
                        )
                        
                        result = self.upload_file(
                            str(cloned_file),
                            r2_key,
                            "audio/wav"
                        )
                        
                        if result["success"]:
                            upload_results["cloned"][cloned_file.name] = result
                            upload_results["files_uploaded"] += 1
            
            # Upload metadata
            metadata_dir = segments_path / "metadata"
            if metadata_dir.exists():
                for file_path in metadata_dir.iterdir():
                    if file_path.is_file():
                        r2_key = self.generate_file_path(audio_id, "metadata", file_path.name)
                        
                        result = self.upload_file(
                            str(file_path),
                            r2_key,
                            "audio/wav" if file_path.suffix == ".wav" else "application/json"
                        )
                        
                        if result["success"]:
                            upload_results["metadata"][file_path.name] = result
                            upload_results["files_uploaded"] += 1
            
            upload_results["success"] = True
            return upload_results
            
        except Exception as e:
            upload_results["error"] = str(e)
            return upload_results
    
    def upload_final_audio(self, audio_id: str, final_audio_path: str) -> Dict[str, Any]:
        """Upload final processed audio"""
        filename = f"final_output_{audio_id}.wav"
        r2_key = self.generate_file_path(audio_id, "final", filename)
        
        return self.upload_file(final_audio_path, r2_key, "audio/wav")
    
    def create_processing_summary(self, audio_id: str, processing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create and upload processing summary"""
        summary = {
            "audio_id": audio_id,
            "processed_at": datetime.now().isoformat(),
            "processing_data": processing_data,
            "r2_bucket": self.bucket_name,
            "r2_base_path": f"{self.base_path}/{datetime.now().strftime('%Y/%m/%d')}/{audio_id}"
        }
        
        # Save summary locally first
        summary_path = f"/tmp/processing_summary_{audio_id}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Upload summary
        r2_key = self.generate_file_path(audio_id, "summary", f"processing_summary_{audio_id}.json")
        result = self.upload_file(summary_path, r2_key, "application/json")
        
        # Clean up local file
        os.remove(summary_path)
        
        return result
    

    
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
        date_prefix = datetime.now().strftime("%Y/%m/%d")
        return {
            "bucket": self.bucket_name,
            "base_path": f"{self.base_path}/{date_prefix}/{audio_id}",
            "audio_id": audio_id,
            "date": date_prefix,
            "access_url": f"https://{self.bucket_name}.r2.cloudflarestorage.com/{self.base_path}/{date_prefix}/{audio_id}"
        }
    
    def cleanup_temp_files(self, audio_id: str):
        """Clean up temporary files uploaded to R2 for processing"""
        try:
            if not self.client:
                return
                
            # List and delete temp files in R2
            temp_prefix = f"temp/{audio_id}"
            
            # List objects with the temp prefix
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=temp_prefix
            )
            
            # Delete temp files
            if 'Contents' in response:
                delete_objects = []
                for obj in response['Contents']:
                    delete_objects.append({'Key': obj['Key']})
                
                if delete_objects:
                    self.client.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': delete_objects}
                    )
                    
        except Exception:
            pass 

    def get_segment_urls(self, audio_id: str, upload_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get all URLs for uploaded segments with direct storage"""
        urls = {
            "metadata": {},
            "segments": {},
            "cloned": {}
        }
        
        if "metadata" in upload_results:
            for filename, result in upload_results["metadata"].items():
                if result.get("success"):
                    urls["metadata"][filename] = result.get("url")
        
        # Handle direct segments storage
        if "segments" in upload_results:
            for filename, result in upload_results["segments"].items():
                if result.get("success"):
                    urls["segments"][filename] = result.get("url")
        
        # Handle direct cloned storage
        if "cloned" in upload_results:
            for filename, result in upload_results["cloned"].items():
                if result.get("success"):
                    urls["cloned"][filename] = result.get("url")
        
        return urls 
    
    def generate_cloned_segment_url(self, audio_id: str, speaker: str, segment_index: int) -> str:
        """Generate R2 URL for a cloned segment using unified structure"""
        cloned_filename = f"cloned_segment_{segment_index:03d}.wav"
        # Use unified cloned folder structure
        r2_key = self.generate_file_path(audio_id, "segments/cloned_segments", cloned_filename)
        return f"{settings.R2_PUBLIC_URL}/{r2_key}" 