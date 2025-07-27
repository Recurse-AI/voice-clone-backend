"""
Utility functions for the application
"""

import os
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import threading

from config import settings

logger = logging.getLogger(__name__)

class LocalStorageManager:
    """Manages local video storage with automatic cleanup"""
    
    def __init__(self):
        self.storage_dir = Path(settings.LOCAL_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.retention_hours = settings.LOCAL_STORAGE_RETENTION_HOURS
        self.metadata_file = self.storage_dir / "storage_metadata.json"
        self._cleanup_lock = threading.Lock()
        
    def store_video(self, file_id: str, video_content: bytes, filename: str) -> Dict[str, Any]:
        """Store video locally and return path info"""
        try:
            video_dir = self.storage_dir / file_id
            video_dir.mkdir(exist_ok=True)
            
            video_path = video_dir / filename
            with open(video_path, 'wb') as f:
                f.write(video_content)
            
            # Store metadata
            metadata = {
                "file_id": file_id,
                "filename": filename,
                "path": str(video_path),
                "stored_at": datetime.now().isoformat(),
                "size": len(video_content),
                "expires_at": (datetime.now() + timedelta(hours=self.retention_hours)).isoformat()
            }
            
            self._update_metadata(file_id, metadata)
            
            return {
                "success": True,
                "local_path": str(video_path),
                "file_id": file_id,
                "expires_at": metadata["expires_at"]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_video_path(self, file_id: str) -> Optional[str]:
        """Get local video path if exists and not expired"""
        try:
            metadata = self._get_metadata(file_id)
            if not metadata:
                return None
            
            # Check if expired
            expires_at = datetime.fromisoformat(metadata["expires_at"])
            if datetime.now() > expires_at:
                self._remove_video(file_id)
                return None
            
            video_path = Path(metadata["path"])
            if video_path.exists():
                return str(video_path)
            else:
                # File doesn't exist, cleanup metadata
                self._remove_metadata(file_id)
                return None
                
        except Exception as e:
            logger.warning(f"Error getting video path for {file_id}: {e}")
            return None
    
    def move_to_processing(self, file_id: str, target_dir: str) -> Optional[str]:
        """Move video from storage to processing directory"""
        try:
            local_path = self.get_video_path(file_id)
            if not local_path:
                return None
            
            metadata = self._get_metadata(file_id)
            if not metadata:
                return None
            
            # Create target path
            target_path = Path(target_dir) / metadata["filename"]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file to target location
            shutil.copy2(local_path, target_path)
            
            return str(target_path)
            
        except Exception as e:
            logger.error(f"Error moving video to processing: {e}")
            return None
    
    def cleanup_expired(self):
        """Clean up expired videos"""
        with self._cleanup_lock:
            try:
                current_time = datetime.now()
                metadata_dict = self._load_all_metadata()
                
                expired_files = []
                for file_id, metadata in metadata_dict.items():
                    try:
                        expires_at = datetime.fromisoformat(metadata["expires_at"])
                        if current_time > expires_at:
                            expired_files.append(file_id)
                    except:
                        expired_files.append(file_id)  # Invalid metadata
                
                for file_id in expired_files:
                    self._remove_video(file_id)
                
                if expired_files:
                    logger.info(f"Cleaned up {len(expired_files)} expired videos")
                    
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    def _update_metadata(self, file_id: str, metadata: Dict[str, Any]):
        """Update metadata for a file"""
        try:
            metadata_dict = self._load_all_metadata()
            metadata_dict[file_id] = metadata
            
            import json
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
    
    def _get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file"""
        try:
            metadata_dict = self._load_all_metadata()
            return metadata_dict.get(file_id)
        except:
            return None
    
    def _load_all_metadata(self) -> Dict[str, Any]:
        """Load all metadata"""
        try:
            if self.metadata_file.exists():
                import json
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except:
            return {}
    
    def _remove_metadata(self, file_id: str):
        """Remove metadata for a file"""
        try:
            metadata_dict = self._load_all_metadata()
            if file_id in metadata_dict:
                del metadata_dict[file_id]
                import json
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Error removing metadata: {e}")
    
    def _remove_video(self, file_id: str):
        """Remove video files and metadata"""
        try:
            metadata = self._get_metadata(file_id)
            if metadata and "path" in metadata:
                video_path = Path(metadata["path"])
                video_dir = video_path.parent
                
                # Remove the video directory
                if video_dir.exists():
                    shutil.rmtree(video_dir, ignore_errors=True)
            
            # Remove metadata
            self._remove_metadata(file_id)
            
        except Exception as e:
            logger.error(f"Error removing video {file_id}: {e}")

# Global instance
local_storage = LocalStorageManager()

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def cleanup_temp_files(audio_id: str, audio_temp_path: Optional[str] = None, 
                      instruments_temp_path: Optional[str] = None, 
                      video_temp_path: Optional[str] = None):
    """Clean up temporary files and log cache statistics"""
    try:
        from config import settings
        
        # Get audio processor for cache cleanup
        from video_processor import get_audio_processor
        audio_processor = get_audio_processor()
        
        # Log cache statistics before cleanup
        cache_stats = audio_processor.get_cache_stats()
        logger.info(f"Cache stats before cleanup for {audio_id}: {cache_stats}")
        
        # Clean up processor files and caches
        audio_processor.cleanup_temp_files(audio_id)
        
        # Clean up specific temp files
        temp_files_to_clean = []
        if audio_temp_path:
            temp_files_to_clean.append(audio_temp_path)
        if instruments_temp_path:
            temp_files_to_clean.append(instruments_temp_path)
        if video_temp_path:
            temp_files_to_clean.append(video_temp_path)
        
        for temp_file in temp_files_to_clean:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
        
        # Clean up any remaining temp files for this audio_id
        temp_dir = Path(settings.TEMP_DIR)
        for temp_file in temp_dir.glob(f"*{audio_id}*"):
            if temp_file.is_file():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
                    
        logger.info(f"Cleanup completed for {audio_id}")
        
    except Exception as e:
        logger.error(f"Error during cleanup for {audio_id}: {str(e)}") 