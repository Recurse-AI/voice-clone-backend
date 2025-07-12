"""
Smart Status Manager - Local Memory + MongoDB Final States
Memory: Active jobs only | MongoDB: Final states only
"""
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime, timedelta
import logging
from config import settings

# Configure logging
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Processing status enum"""
    NOT_FOUND = "not_found"
    STARTING = "starting"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StatusInfo:
    """Status information dataclass"""
    status: ProcessingStatus
    message: str
    audio_id: str
    progress: int = 0
    details: Dict[str, Any] = None
    started_at: float = None
    updated_at: float = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.started_at is None:
            self.started_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()


class StatusManager:
    """Smart status manager: Local memory + MongoDB"""
    
    def __init__(self, mongodb_uri: str = None):
        # Local memory for active jobs
        self._statuses: Dict[str, StatusInfo] = {}
        self._lock = threading.Lock()
        
        # Status messages
        self._status_messages = {
            ProcessingStatus.NOT_FOUND: "No processing found",
            ProcessingStatus.STARTING: "Processing starting...",
            ProcessingStatus.DOWNLOADING: "Downloading video...",
            ProcessingStatus.PROCESSING: "Processing audio and video...",
            ProcessingStatus.UPLOADING: "Uploading results...",
            ProcessingStatus.COMPLETED: "Processing completed successfully",
            ProcessingStatus.FAILED: "Processing failed"
        }
        
        # MongoDB setup
        self._mongodb_uri = mongodb_uri or settings.MONGODB_URI
        self._mongo_client = None
        self._mongo_collection = None
        self._init_mongodb()
    
    def _init_mongodb(self):
        """Initialize MongoDB connection"""
        # Check if MongoDB URI is provided
        if not self._mongodb_uri or self._mongodb_uri.strip() == "":
            logger.info("MongoDB URI not provided, status persistence disabled (statuses will only be kept in memory)")
            self._mongo_client = None
            return
        
        try:
            logger.info("Initializing MongoDB connection...")
            self._mongo_client = MongoClient(self._mongodb_uri, serverSelectionTimeoutMS=5000)
            
            # Test the connection
            self._mongo_client.admin.command('ismaster')
            self._mongo_collection = self._mongo_client.get_default_database().job_status
            
            logger.info("MongoDB connection established successfully")
            
        except Exception as e:
            logger.warning(f"Failed to connect to MongoDB: {str(e)}")
            logger.info("Status persistence disabled - statuses will only be kept in memory")
            self._mongo_client = None
    
    def start_processing(self, audio_id: str) -> None:
        """Start processing for an audio ID"""
        logger.info(f"Starting processing status tracking for audio_id: {audio_id}")
        with self._lock:
            self._statuses[audio_id] = StatusInfo(
                status=ProcessingStatus.STARTING,
                message=self._status_messages[ProcessingStatus.STARTING],
                audio_id=audio_id,
                progress=0
            )
        logger.info(f"Processing status initialized for audio_id: {audio_id}")
    
    def update_status(self, audio_id: str, status: ProcessingStatus, 
                     progress: int = None, details: Dict[str, Any] = None) -> None:
        """Update processing status"""
        logger.info(f"Updating status for audio_id: {audio_id} to {status.value}")
        with self._lock:
            if audio_id not in self._statuses:
                logger.warning(f"Attempted to update status for non-existent audio_id: {audio_id}")
                return
                
            status_info = self._statuses[audio_id]
            status_info.status = status
            status_info.message = self._status_messages[status]
            status_info.updated_at = time.time()
            
            if progress is not None:
                status_info.progress = min(100, max(0, progress))
            
            if details:
                status_info.details.update(details)
            
            # Save final states to MongoDB and cleanup
            if status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                logger.info(f"Final status reached for audio_id: {audio_id} - {status.value}")
                if self._save_final_state(status_info):
                    del self._statuses[audio_id]
                    logger.info(f"Status moved to MongoDB and cleared from memory for audio_id: {audio_id}")
                else:
                    logger.info(f"Status kept in memory for audio_id: {audio_id} (MongoDB not available)")
    
    def set_progress(self, audio_id: str, progress: int) -> None:
        """Set progress percentage"""
        with self._lock:
            if audio_id in self._statuses:
                self._statuses[audio_id].progress = min(100, max(0, progress))
                self._statuses[audio_id].updated_at = time.time()
                logger.debug(f"Progress updated for audio_id: {audio_id} to {progress}%")
    
    def complete_processing(self, audio_id: str, details: Dict[str, Any] = None) -> None:
        """Mark processing as completed"""
        logger.info(f"Completing processing for audio_id: {audio_id}")
        self.update_status(audio_id, ProcessingStatus.COMPLETED, 100, details)
    
    def fail_processing(self, audio_id: str, error: str, details: Dict[str, Any] = None) -> None:
        """Mark processing as failed"""
        logger.error(f"Failing processing for audio_id: {audio_id} with error: {error}")
        fail_details = {"error": error}
        if details:
            fail_details.update(details)
        self.update_status(audio_id, ProcessingStatus.FAILED, None, fail_details)
    
    def get_status(self, audio_id: str) -> Dict[str, Any]:
        """Get processing status - local first, MongoDB fallback"""
        logger.debug(f"Getting status for audio_id: {audio_id}")
        
        # Check local memory first
        with self._lock:
            if audio_id in self._statuses:
                logger.debug(f"Found status in memory for audio_id: {audio_id}")
                status_info = self._statuses[audio_id]
                result = asdict(status_info)
                result["status"] = status_info.status.value
                result["elapsed_time"] = time.time() - status_info.started_at
                return result
        
        logger.debug(f"Status not found in memory for audio_id: {audio_id}, checking MongoDB")
        
        # Fallback to MongoDB
        mongodb_result = self._get_from_mongodb(audio_id)
        
        # If not found in MongoDB, check if there's a log file for this audio_id
        # This indicates processing was started but status was lost
        if mongodb_result["status"] == ProcessingStatus.NOT_FOUND.value:
            logger.warning(f"Status not found in MongoDB for audio_id: {audio_id}")
            log_file_exists = self._check_log_file_exists(audio_id)
            if log_file_exists:
                logger.warning(f"Log file exists for audio_id: {audio_id} but status not found - may indicate lost processing")
                return {
                    "status": ProcessingStatus.FAILED.value,
                    "message": "Processing was started but status was lost. Check logs for details.",
                    "audio_id": audio_id,
                    "details": {"log_file_available": True}
                }
        
        return mongodb_result
    
    def _save_final_state(self, status_info: StatusInfo) -> bool:
        """Save final state to MongoDB"""
        if not self._mongo_client:
            logger.info(f"MongoDB not configured - final status for audio_id: {status_info.audio_id} kept in memory")
            return False
        
        try:
            doc = {
                "audio_id": status_info.audio_id,
                "status": status_info.status.value,
                "message": status_info.message,
                "progress": status_info.progress,
                "details": status_info.details,
                "started_at": status_info.started_at,
                "completed_at": status_info.updated_at
            }
            
            self._mongo_collection.find_and_modify(
                query={"audio_id": status_info.audio_id},
                update={"$set": doc},
                upsert=True
            )
            logger.info(f"Final state saved to MongoDB for audio_id: {status_info.audio_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save final state to MongoDB for audio_id: {status_info.audio_id}, error: {str(e)}")
            return False
    
    def _get_from_mongodb(self, audio_id: str) -> Dict[str, Any]:
        """Get job status from MongoDB"""
        if not self._mongo_client:
            logger.debug(f"MongoDB not configured, cannot retrieve status from persistent storage for audio_id: {audio_id}")
            return {
                "status": ProcessingStatus.NOT_FOUND.value,
                "message": self._status_messages[ProcessingStatus.NOT_FOUND],
                "audio_id": audio_id
            }
        
        try:
            doc = self._mongo_collection.find_one({"audio_id": audio_id})
            if doc:
                logger.debug(f"Found status in MongoDB for audio_id: {audio_id}")
                return {
                    "status": doc["status"],
                    "message": doc.get("message", ""),
                    "audio_id": doc["audio_id"],
                    "progress": doc.get("progress", 100),
                    "details": doc.get("details", {}),
                    "started_at": doc.get("started_at", 0),
                    "elapsed_time": doc.get("completed_at", 0) - doc.get("started_at", 0)
                }
            else:
                logger.debug(f"No status found in MongoDB for audio_id: {audio_id}")
        except Exception as e:
            logger.error(f"Failed to retrieve status from MongoDB for audio_id: {audio_id}, error: {str(e)}")
            pass
        
        return {
            "status": ProcessingStatus.NOT_FOUND.value,
            "message": self._status_messages[ProcessingStatus.NOT_FOUND],
            "audio_id": audio_id
        }
    
    def _check_log_file_exists(self, audio_id: str) -> bool:
        """Check if a log file exists for the given audio_id"""
        try:
            from pathlib import Path
            log_dir = Path("logs")
            if log_dir.exists():
                # Look for any log file containing this audio_id
                for log_file in log_dir.glob(f"*{audio_id}*"):
                    if log_file.is_file():
                        return True
        except:
            pass
        return False
    
    def cleanup_old_statuses(self, max_age_hours: int = 24) -> None:
        """Remove old status entries from memory and handle stuck processing"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        stuck_timeout_seconds = 30 * 60  # 30 minutes for stuck processing
        
        with self._lock:
            to_remove = []
            to_fail = []
            
            for audio_id, status_info in self._statuses.items():
                age_seconds = current_time - status_info.updated_at
                
                # Mark stuck processing as failed (30 minutes without update)
                if (status_info.status in [ProcessingStatus.PROCESSING, ProcessingStatus.DOWNLOADING, ProcessingStatus.UPLOADING] 
                    and age_seconds > stuck_timeout_seconds):
                    to_fail.append(audio_id)
                
                # Remove very old entries
                elif age_seconds > max_age_seconds:
                    to_remove.append(audio_id)
            
            # Fail stuck processing
            for audio_id in to_fail:
                self.fail_processing(audio_id, "Processing timed out (stuck for more than 30 minutes)")
            
            # Remove old entries
            for audio_id in to_remove:
                if audio_id in self._statuses:  # Check again as fail_processing might have removed it
                    del self._statuses[audio_id]
    
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get all active processing statuses"""
        with self._lock:
            return {
                audio_id: self.get_status(audio_id) 
                for audio_id in self._statuses.keys()
            }


# Global status manager instance
status_manager = StatusManager() 