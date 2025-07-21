"""
Status Manager - With MongoDB Backup
"""
import time
import logging
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime
from threading import Lock
from config import settings

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
    """Status manager with MongoDB backup"""
    
    def __init__(self):
        self._statuses: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._mongo_client = None
        self._mongo_db = None
        self._mongo_collection = None
        self._init_mongodb()
    
    def _init_mongodb(self):
        """Initialize MongoDB connection"""
        try:
            if settings.MONGODB_URI:
                import pymongo
                self._mongo_client = pymongo.MongoClient(settings.MONGODB_URI)
                self._mongo_db = self._mongo_client.voice_cloning
                self._mongo_collection = self._mongo_db.status
                logger.info("MongoDB connected for status backup")
            else:
                logger.warning("MongoDB URI not configured, status backup disabled")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            self._mongo_client = None
    
    def _save_to_mongodb(self, audio_id: str, status_data: Dict[str, Any]):
        """Save status to MongoDB"""
        if self._mongo_collection is None:
            return
        
        try:
            self._mongo_collection.replace_one(
                {"audio_id": audio_id},
                status_data,
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to save status to MongoDB: {e}")
    
    def _get_from_mongodb(self, audio_id: str) -> Optional[Dict[str, Any]]:
        """Get status from MongoDB"""
        if self._mongo_collection is None:
            return None
        
        try:
            result = self._mongo_collection.find_one({"audio_id": audio_id})
            if result:
                result.pop('_id', None)  # Remove MongoDB ObjectId
                return result
        except Exception as e:
            logger.error(f"Failed to get status from MongoDB: {e}")
        
        return None
        
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
        status_data = {
            "status": ProcessingStatus.PENDING.value,
            "message": self._get_status_message(ProcessingStatus.PENDING),
            "progress": 0,
            "audio_id": audio_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "details": {}
        }
        
        with self._lock:
            self._statuses[audio_id] = status_data
        
        # Save to MongoDB
        self._save_to_mongodb(audio_id, status_data)
    
    def update_status(self, audio_id: str, status: ProcessingStatus, progress: int = None, 
                     details: Optional[Dict[str, Any]] = None):
        """Update processing status"""
        with self._lock:
            if audio_id not in self._statuses:
                self._statuses[audio_id] = {}
            
            # Use custom message if provided in details, otherwise use default
            message = self._get_status_message(status)
            if details and details.get("message"):
                message = details["message"]
            
            self._statuses[audio_id].update({
                "status": status.value,
                "message": message,
                "audio_id": audio_id,
                "updated_at": datetime.now().isoformat()
            })
            
            if progress is not None:
                self._statuses[audio_id]["progress"] = progress
            
            if details:
                if "details" not in self._statuses[audio_id]:
                    self._statuses[audio_id]["details"] = {}
                self._statuses[audio_id]["details"].update(details)
            
            # Save to MongoDB
            self._save_to_mongodb(audio_id, self._statuses[audio_id])
    
    def set_progress(self, audio_id: str, progress: int) -> None:
        """Update progress percentage"""
        with self._lock:
            if audio_id in self._statuses:
                self._statuses[audio_id]["progress"] = min(100, max(0, progress))
                self._statuses[audio_id]["updated_at"] = datetime.now().isoformat()
                
                # Save to MongoDB
                self._save_to_mongodb(audio_id, self._statuses[audio_id])
    
    def get_status(self, audio_id: str) -> Optional[Dict[str, Any]]:
        """Get current status with MongoDB fallback and queue information"""
        with self._lock:
            status_data = None
            if audio_id in self._statuses:
                status_data = self._statuses[audio_id].copy()
            else:
                # Fallback to MongoDB
                status_data = self._get_from_mongodb(audio_id)
            
            if status_data:
                # Add queue information if available
                try:
                    from video_processor.video_queue_manager import video_queue_manager
                    
                    # Try to find queue request by audio_id
                    queue_request = None
                    for request_id, request in video_queue_manager.requests.items():
                        if request.audio_id == audio_id:
                            queue_request = video_queue_manager.get_request_status(request_id)
                            break
                    
                    if queue_request:
                        queue_status = queue_request["status"]
                        
                        # Sync main status with queue status if inconsistent
                        if queue_status in ["failed", "timeout", "cancelled"] and status_data.get("status") not in ["failed", "completed"]:
                            status_data["status"] = ProcessingStatus.FAILED.value
                            error_msg = queue_request.get("error", "Processing failed")
                            status_data["message"] = f"Processing failed: {error_msg}"
                            status_data["updated_at"] = datetime.now().isoformat()
                            if "details" not in status_data:
                                status_data["details"] = {}
                            status_data["details"]["error"] = error_msg
                            
                            # Save updated status
                            self._statuses[audio_id] = status_data
                            self._save_to_mongodb(audio_id, status_data)
                        
                        # Enhance message with queue information
                        queue_position = queue_request.get("queue_position")
                        if queue_position and queue_position > 0:
                            status_data["message"] = f"{status_data.get('message', 'Processing queued')} (Queue position: {queue_position})"
                        
                        status_data["queue_info"] = {
                            "request_id": queue_request["request_id"],
                            "queue_status": queue_status,
                            "queue_position": queue_position,
                            "estimated_time": queue_request.get("estimated_time"),
                            "timeout_in": queue_request.get("timeout_in"),
                            "started_at": queue_request.get("started_at"),
                            "created_at": queue_request.get("created_at")
                        }
                    else:
                        # No queue info available - might be legacy processing
                        status_data["queue_info"] = None
                        
                except Exception as e:
                    logger.warning(f"Failed to get queue info for {audio_id}: {e}")
                    status_data["queue_info"] = None
            
            return status_data
    
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