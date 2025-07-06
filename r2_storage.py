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
        self.client = boto3.client(
            's3',
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            region_name=settings.R2_REGION
        )
        self.bucket_name = settings.R2_BUCKET_NAME
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
        try:
            with open(local_path, 'rb') as file:
                self.client.upload_fileobj(
                    file,
                    self.bucket_name,
                    r2_key,
                    ExtraArgs={'ContentType': content_type}
                )
            
            file_size = os.path.getsize(local_path)
            return {
                "success": True,
                "r2_key": r2_key,
                "bucket": self.bucket_name,
                "size": file_size,
                "content_type": content_type,
                "url": f"https://{self.bucket_name}.r2.cloudflarestorage.com/{r2_key}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def upload_audio_segments(self, audio_id: str, segments_dir: str) -> Dict[str, Any]:
        """Upload all audio segments and metadata to R2"""
        upload_results = {
            "audio_id": audio_id,
            "segments": {},
            "references": {},
            "metadata": {},
            "silent_parts": {}
        }
        
        segments_path = Path(segments_dir)
        
        # Upload segments by speaker
        for speaker_dir in segments_path.iterdir():
            if speaker_dir.is_dir() and speaker_dir.name.startswith("speaker_"):
                speaker_id = speaker_dir.name
                upload_results["segments"][speaker_id] = {}
                upload_results["references"][speaker_id] = {}
                
                # Upload speaker segments
                segments_subdir = speaker_dir / "segments"
                if segments_subdir.exists():
                    for file_path in segments_subdir.iterdir():
                        if file_path.is_file():
                            file_type = "audio" if file_path.suffix in [".wav", ".mp3"] else "metadata"
                            r2_key = self.generate_file_path(audio_id, f"segments/{speaker_id}", file_path.name)
                            
                            result = self.upload_file(
                                str(file_path),
                                r2_key,
                                "audio/wav" if file_path.suffix == ".wav" else "application/json"
                            )
                            
                            if result["success"]:
                                upload_results["segments"][speaker_id][file_path.name] = result
                
                # Upload speaker references
                reference_subdir = speaker_dir / "reference"
                if reference_subdir.exists():
                    for file_path in reference_subdir.iterdir():
                        if file_path.is_file():
                            r2_key = self.generate_file_path(audio_id, f"references/{speaker_id}", file_path.name)
                            
                            result = self.upload_file(
                                str(file_path),
                                r2_key,
                                "audio/wav" if file_path.suffix == ".wav" else "application/json"
                            )
                            
                            if result["success"]:
                                upload_results["references"][speaker_id][file_path.name] = result
        
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
        
        # Upload silent parts
        silent_dir = segments_path / "silent_parts"
        if silent_dir.exists():
            for file_path in silent_dir.iterdir():
                if file_path.is_file():
                    r2_key = self.generate_file_path(audio_id, "silent_parts", file_path.name)
                    
                    result = self.upload_file(
                        str(file_path),
                        r2_key,
                        "audio/wav"
                    )
                    
                    if result["success"]:
                        upload_results["silent_parts"][file_path.name] = result
        
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