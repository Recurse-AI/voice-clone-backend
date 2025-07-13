"""
Smart Status Manager - Local Memory + MongoDB Final States
Memory: Active jobs only | MongoDB: Final states only
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
    """Manages processing status for audio jobs"""
    
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
                return {
                    "status": "not_found",
                    "message": f"No processing job found with ID: {audio_id}",
            "audio_id": audio_id
        }
    
    def complete_processing(self, audio_id: str, details: Optional[Dict[str, Any]] = None):
        """Mark processing as completed"""
        self.update_status(audio_id, ProcessingStatus.COMPLETED, 100, details)
    
    def fail_processing(self, audio_id: str, error: str):
        """Mark processing as failed"""
        self.update_status(audio_id, ProcessingStatus.FAILED, details={"error": error})


# Global status manager instance
status_manager = StatusManager() 