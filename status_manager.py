"""
Status Manager - Simplified
"""
import time
import logging
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Processing status enum"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    SEPARATING = "separating"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


class StatusManager:
    """Simplified status manager"""
    
    def __init__(self):
        self._statuses: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        
    def _get_status_message(self, status: ProcessingStatus) -> str:
        """Get status message"""
        messages = {
            ProcessingStatus.PENDING: "Processing queued",
            ProcessingStatus.DOWNLOADING: "Downloading video...",
            ProcessingStatus.SEPARATING: "Separating audio tracks...",
            ProcessingStatus.TRANSCRIBING: "Transcribing audio...",
            ProcessingStatus.PROCESSING: "Processing audio and video...",
            ProcessingStatus.UPLOADING: "Uploading results...",
            ProcessingStatus.COMPLETED: "Processing completed successfully",
            ProcessingStatus.FAILED: "Processing failed"
        }
        return messages.get(status, "Unknown status")
    
    def initialize_status(self, audio_id: str) -> None:
        """Initialize status for new processing job"""
        with self._lock:
            self._statuses[audio_id] = {
                "status": ProcessingStatus.PENDING.value,
                "message": self._get_status_message(ProcessingStatus.PENDING),
                "progress": 0,
                "audio_id": audio_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "details": {}
            }
    
    def update_status(self, audio_id: str, status: ProcessingStatus, progress: int = None, 
                     details: Optional[Dict[str, Any]] = None):
        """Update processing status"""
        with self._lock:
            if audio_id not in self._statuses:
                self._statuses[audio_id] = {}
            
            self._statuses[audio_id].update({
                "status": status.value,
                "message": self._get_status_message(status),
                "audio_id": audio_id,
                "updated_at": datetime.now().isoformat()
            })
            
            if progress is not None:
                self._statuses[audio_id]["progress"] = progress
            
            if details:
                if "details" not in self._statuses[audio_id]:
                    self._statuses[audio_id]["details"] = {}
                self._statuses[audio_id]["details"].update(details)
    
    def set_progress(self, audio_id: str, progress: int) -> None:
        """Update progress percentage"""
        with self._lock:
            if audio_id in self._statuses:
                self._statuses[audio_id]["progress"] = min(100, max(0, progress))
                self._statuses[audio_id]["updated_at"] = datetime.now().isoformat()
    
    def get_status(self, audio_id: str) -> Dict[str, Any]:
        """Get current status"""
        with self._lock:
            if audio_id in self._statuses:
                return self._statuses[audio_id].copy()
            else:
                #check mongo db for status
                status = self.get_status_from_db(audio_id)
                if status:
                    return status
                else:
                    return None
    
    def complete_processing(self, audio_id: str, details: Optional[Dict[str, Any]] = None):
        """Mark processing as completed"""
        self.update_status(audio_id, ProcessingStatus.COMPLETED, 100, details)
    
    def fail_processing(self, audio_id: str, error: str):
        """Mark processing as failed"""
        self.update_status(audio_id, ProcessingStatus.FAILED, details={"error": error})
    
    def cleanup_old_statuses(self):
        """Clean up old statuses (older than 24 hours)"""
        try:
            cutoff_time = datetime.now().timestamp() - (24 * 3600)  # 24 hours ago
            
            with self._lock:
                keys_to_remove = []
                for audio_id, status_info in self._statuses.items():
                    try:
                        created_at = datetime.fromisoformat(status_info.get("created_at", ""))
                        if created_at.timestamp() < cutoff_time:
                            keys_to_remove.append(audio_id)
                    except:
                        continue
                
                for key in keys_to_remove:
                    del self._statuses[key]
                    
        except Exception:
            pass
    
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get all statuses"""
        with self._lock:
            return {k: v.copy() for k, v in self._statuses.items()}


# Global status manager instance
status_manager = StatusManager() 