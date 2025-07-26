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
                queue_request = None
                try:
                    from video_processor.video_queue_manager import video_queue_manager
                    
                    # Try to find queue request by audio_id with better error handling
                    found_request_id = None
                    found_request = None
                    
                    with video_queue_manager._lock:
                        for request_id, request in video_queue_manager.requests.items():
                            if request.audio_id == audio_id:
                                found_request_id = request_id
                                found_request = request
                                break
                    
                    if found_request_id and found_request:
                        # Get detailed queue status
                        queue_request = video_queue_manager.get_request_status(found_request_id)
                        
                        if queue_request:
                            queue_status = queue_request["status"]
                            
                            # Check for critical failures that need immediate termination
                            if queue_status in ["failed", "timeout", "cancelled"]:
                                error_msg = queue_request.get("error", "Processing failed")
                                logger.error(f"Critical failure detected for {audio_id}: {error_msg}")
                                
                                # Immediate termination logic
                                self._terminate_failed_process(audio_id, found_request_id, error_msg)
                                
                                # Update status to reflect failure
                                status_data["status"] = ProcessingStatus.FAILED.value
                                status_data["message"] = f"Processing terminated: {error_msg}"
                                status_data["updated_at"] = datetime.now().isoformat()
                                if "details" not in status_data:
                                    status_data["details"] = {}
                                status_data["details"]["error"] = error_msg
                                status_data["details"]["terminated"] = True
                                
                                # Save updated status
                                self._statuses[audio_id] = status_data
                                self._save_to_mongodb(audio_id, status_data)
                            
                            # Enhance message with queue information for active processes
                            elif queue_status in ["pending", "processing"]:
                                queue_position = queue_request.get("queue_position")
                                if queue_position and queue_position > 0:
                                    status_data["message"] = f"{status_data.get('message', 'Processing queued')} (Queue position: {queue_position})"
                            
                            # Always include queue_info when request is found
                            status_data["queue_info"] = {
                                "request_id": queue_request["request_id"],
                                "queue_status": queue_status,
                                "queue_position": queue_request.get("queue_position"),
                                "estimated_time": queue_request.get("estimated_time"),
                                "timeout_in": queue_request.get("timeout_in"),
                                "started_at": queue_request.get("started_at"),
                                "created_at": queue_request.get("created_at")
                            }
                        else:
                            # Request found but status retrieval failed
                            logger.warning(f"Found queue request for {audio_id} but failed to get status")
                            status_data["queue_info"] = {
                                "request_id": found_request_id,
                                "queue_status": found_request.status.value,
                                "queue_position": found_request.queue_position,
                                "error": "Failed to retrieve detailed status"
                            }
                    else:
                        # No queue request found - could be legacy processing or completed
                        logger.info(f"No active queue request found for {audio_id}")
                        
                        # Check if this is a stale process that needs cleanup
                        if status_data.get("status") in ["downloading", "processing"] and status_data.get("progress", 0) < 100:
                            # Check how long it's been since last update
                            try:
                                last_update = datetime.fromisoformat(status_data.get("updated_at", datetime.now().isoformat()))
                                time_diff = (datetime.now() - last_update).total_seconds()
                                
                                # If no update for more than 15 minutes, consider it stale
                                if time_diff > 900:  # 15 minutes
                                    logger.error(f"Stale process detected for {audio_id}, terminating")
                                    self._terminate_stale_process(audio_id)
                                    
                                    status_data["status"] = ProcessingStatus.FAILED.value
                                    status_data["message"] = "Processing terminated due to inactivity"
                                    status_data["updated_at"] = datetime.now().isoformat()
                                    if "details" not in status_data:
                                        status_data["details"] = {}
                                    status_data["details"]["error"] = "Process became unresponsive"
                                    status_data["details"]["terminated"] = True
                                    
                                    self._statuses[audio_id] = status_data
                                    self._save_to_mongodb(audio_id, status_data)
                            except Exception as e:
                                logger.error(f"Error checking stale process for {audio_id}: {e}")
                        
                        # Set queue_info to indicate no active queue request
                        status_data["queue_info"] = None
                        
                except Exception as e:
                    logger.error(f"Failed to get queue info for {audio_id}: {e}")
                    # Don't fail the entire status check, just log the error
                    status_data["queue_info"] = None
                    if "details" not in status_data:
                        status_data["details"] = {}
                    status_data["details"]["queue_error"] = str(e)
            
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

    def _terminate_failed_process(self, audio_id: str, request_id: str, error_msg: str):
        """Terminate a failed process and clean up resources"""
        try:
            logger.info(f"Terminating failed process for {audio_id} (request_id: {request_id})")
            
            # Cancel the queue request
            from video_processor.video_queue_manager import video_queue_manager
            video_queue_manager.cancel_request(request_id)
            
            # Clean up temp files
            self._cleanup_process_files(audio_id)
            
            # Update internal tracking
            if audio_id in self._statuses:
                self._statuses[audio_id]["terminated"] = True
                self._statuses[audio_id]["termination_reason"] = error_msg
                
        except Exception as e:
            logger.error(f"Error terminating failed process {audio_id}: {e}")
    
    def _terminate_stale_process(self, audio_id: str):
        """Terminate a stale process that has no active queue request"""
        try:
            logger.info(f"Terminating stale process for {audio_id}")
            
            # Try to find and cancel any related queue requests
            from video_processor.video_queue_manager import video_queue_manager
            with video_queue_manager._lock:
                for request_id, request in video_queue_manager.requests.items():
                    if request.audio_id == audio_id:
                        video_queue_manager.cancel_request(request_id)
                        break
            
            # Clean up temp files
            self._cleanup_process_files(audio_id)
            
            # Update internal tracking
            if audio_id in self._statuses:
                self._statuses[audio_id]["terminated"] = True
                self._statuses[audio_id]["termination_reason"] = "Stale process cleanup"
                
        except Exception as e:
            logger.error(f"Error terminating stale process {audio_id}: {e}")
    
    def _cleanup_process_files(self, audio_id: str):
        """Clean up temporary files for a terminated process"""
        try:
            # Import cleanup utilities
            from utils import cleanup_temp_files
            from config import settings
            import os
            import shutil
            from pathlib import Path
            
            # Clean up temp files
            cleanup_temp_files(audio_id, None, None, None)
            
            # Clean up any remaining audio-specific temp files
            temp_dir = Path(settings.TEMP_DIR)
            if temp_dir.exists():
                for file_path in temp_dir.glob(f"*{audio_id}*"):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {file_path}: {e}")
            
            # Clean up segments directory
            segments_dir = temp_dir / f"segments_{audio_id}"
            if segments_dir.exists():
                try:
                    shutil.rmtree(segments_dir)
                except Exception as e:
                    logger.warning(f"Could not remove segments directory {segments_dir}: {e}")
                    
            logger.info(f"Cleaned up temporary files for {audio_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up files for {audio_id}: {e}")
    
    def force_terminate_process(self, audio_id: str, reason: str = "Manual termination"):
        """Force terminate a process immediately"""
        logger.warning(f"Force terminating process {audio_id}: {reason}")
        
        try:
            # Update status to failed
            self.fail_processing(audio_id, reason)
            
            # Find and cancel queue request
            from video_processor.video_queue_manager import video_queue_manager
            with video_queue_manager._lock:
                for request_id, request in video_queue_manager.requests.items():
                    if request.audio_id == audio_id:
                        video_queue_manager.cancel_request(request_id)
                        self._terminate_failed_process(audio_id, request_id, reason)
                        break
                else:
                    # No active queue request, just clean up
                    self._terminate_stale_process(audio_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error force terminating {audio_id}: {e}")
            return False


# Global status manager instance
status_manager = StatusManager() 