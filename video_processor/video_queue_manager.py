"""
Video Queue Manager

Manages video processing queue with timeout enforcement and status tracking.
"""

import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock, Thread
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class VideoQueueStatus(Enum):
    """Video processing queue status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class VideoQueueRequest:
    """Individual video processing request"""
    
    def __init__(self, request_id: str, video_source: str, audio_id: str, 
                 is_file_upload: bool, parameters: Dict[str, Any]):
        self.request_id = request_id
        self.video_source = video_source
        self.audio_id = audio_id
        self.is_file_upload = is_file_upload
        self.parameters = parameters
        
        self.status = VideoQueueStatus.PENDING
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.progress = 0
        self.error = None
        self.result = None
        self.queue_position = 0
        
        # 30 minutes timeout
        self.timeout_seconds = 30 * 60


class VideoQueueManager:
    """Manages video processing queue with configurable concurrency"""
    
    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self.requests: Dict[str, VideoQueueRequest] = {}
        self.queue: List[str] = []  # Request IDs waiting to be processed
        self.processing: List[str] = []  # Request IDs currently being processed
        self._lock = Lock()
        self._processing_threads: Dict[str, Thread] = {}
        
        # Keep completed requests for 30 minutes
        self.completed_retention_seconds = 30 * 60
        
        # Start cleanup thread
        self._cleanup_thread = Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"VideoQueueManager initialized with max_concurrent={max_concurrent}")
    
    def _periodic_cleanup(self):
        """Periodically clean up old completed requests"""        
        while True:
            try:
                time.sleep(300)  # Check every 5 minutes
                self._cleanup_old_requests()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    def _cleanup_old_requests(self):
        """Remove old completed requests to prevent memory buildup"""
        current_time = datetime.now()
        to_remove = []
        
        with self._lock:
            for request_id, request in self.requests.items():
                if request.status in [VideoQueueStatus.COMPLETED, VideoQueueStatus.FAILED, 
                                    VideoQueueStatus.CANCELLED, VideoQueueStatus.TIMEOUT]:
                    if request.completed_at:
                        time_since_completion = (current_time - request.completed_at).total_seconds()
                        if time_since_completion > self.completed_retention_seconds:
                            to_remove.append(request_id)
            
            # Remove old requests
            for request_id in to_remove:
                if request_id in self.requests:
                    logger.info(f"Cleaning up old completed request: {request_id}")
                    del self.requests[request_id]
                    
                if request_id in self._processing_threads:
                    del self._processing_threads[request_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old completed requests")
    
    def submit_request(self, video_source: str, audio_id: str, is_file_upload: bool,
                      parameters: Dict[str, Any]) -> str:
        """Submit a new video processing request"""
        request_id = str(uuid.uuid4())
        
        request = VideoQueueRequest(
            request_id=request_id,
            video_source=video_source,
            audio_id=audio_id,
            is_file_upload=is_file_upload,
            parameters=parameters
        )
        
        with self._lock:
            self.requests[request_id] = request
            self.queue.append(request_id)
            self._update_queue_positions()
            
            # Try to start processing immediately if slots available
            self._start_next_request()
        
        logger.info(f"Submitted video processing request {request_id} for audio_id {audio_id}")
        return request_id
    
    def _update_queue_positions(self):
        """Update queue positions for all pending requests"""
        for position, request_id in enumerate(self.queue):
            request = self.requests.get(request_id)
            if request:
                request.queue_position = position + 1
    
    def _start_next_request(self):
        """Start the next queued request if slots available"""
        if len(self.processing) >= self.max_concurrent or not self.queue:
            return
        
        # Get next request from queue
        request_id = self.queue.pop(0)
        request = self.requests.get(request_id)
        
        if not request or request.status != VideoQueueStatus.PENDING:
            return
        
        # Move to processing
        self.processing.append(request_id)
        request.status = VideoQueueStatus.PROCESSING
        request.started_at = datetime.now()
        request.queue_position = 0
        
        # Update queue positions for remaining requests
        self._update_queue_positions()
        
        # Start processing thread
        thread = Thread(target=self._process_request, args=(request_id,), daemon=True)
        self._processing_threads[request_id] = thread
        thread.start()
        
        logger.info(f"Started processing request {request_id}")
    
    def _process_request(self, request_id: str):
        """Process a video request in a separate thread"""
        try:
            request = self.requests.get(request_id)
            if not request:
                return
            
            # Import here to avoid circular imports  
            from video_processing import process_video_with_queue
            
            # Check if still should process (not cancelled/timed out)
            if request.status != VideoQueueStatus.PROCESSING:
                return
            
            # Check for timeout
            if request.started_at:
                elapsed = (datetime.now() - request.started_at).total_seconds()
                if elapsed > request.timeout_seconds:
                    self._handle_timeout(request_id)
                    return
            
            # Process the video
            result = process_video_with_queue(request)
            
            # Update request status
            with self._lock:
                if request.status == VideoQueueStatus.PROCESSING:
                    if result.get("success"):
                        request.status = VideoQueueStatus.COMPLETED
                        request.result = result
                    else:
                        request.status = VideoQueueStatus.FAILED
                        request.error = result.get("error", "Unknown error")
                        
                        # Update status manager
                        from status_manager import status_manager
                        status_manager.fail_processing(request.audio_id, request.error)
                    
                    request.completed_at = datetime.now()
                    request.progress = 100
                    
                    # Remove from processing
                    if request_id in self.processing:
                        self.processing.remove(request_id)
                    
                    # Remove thread reference
                    if request_id in self._processing_threads:
                        del self._processing_threads[request_id]
                    
                    # Start next queued request
                    self._start_next_request()
        
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            
            with self._lock:
                request = self.requests.get(request_id)
                if request:
                    request.status = VideoQueueStatus.FAILED
                    request.error = f"Processing error: {str(e)}"
                    request.completed_at = datetime.now()
                    
                    # Update status manager
                    from status_manager import status_manager
                    status_manager.fail_processing(request.audio_id, request.error)
                    
                    if request_id in self.processing:
                        self.processing.remove(request_id)
                    
                    if request_id in self._processing_threads:
                        del self._processing_threads[request_id]
                    
                    self._start_next_request()

    def _handle_timeout(self, request_id: str):
        """Handle a timed out request"""
        with self._lock:
            request = self.requests.get(request_id)
            if request and request.status == VideoQueueStatus.PROCESSING:
                request.status = VideoQueueStatus.TIMEOUT
                request.error = "Processing timed out after 30 minutes"
                request.completed_at = datetime.now()
                
                # Remove from processing
                if request_id in self.processing:
                    self.processing.remove(request_id)
                
                if request_id in self._processing_threads:
                    del self._processing_threads[request_id]
                
                logger.warning(f"Request {request_id} timed out after 30 minutes")
                
                # Update status manager
                from status_manager import status_manager
                status_manager.fail_processing(
                    request.audio_id, 
                    "Processing timed out after 30 minutes - automatically cancelled"
                )
                
                # Start next queued request
                self._start_next_request()
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request"""
        with self._lock:
            request = self.requests.get(request_id)
            if not request:
                return None
            
            return {
                "request_id": request_id,
                "audio_id": request.audio_id,
                "status": request.status.value,
                "progress": request.progress,
                "queue_position": request.queue_position if request.status == VideoQueueStatus.PENDING else None,
                "created_at": request.created_at.isoformat(),
                "started_at": request.started_at.isoformat() if request.started_at else None,
                "completed_at": request.completed_at.isoformat() if request.completed_at else None,
                "error": request.error,
                "result": request.result
            }
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending or processing request"""
        with self._lock:
            request = self.requests.get(request_id)
            if not request:
                return False
            
            if request.status in [VideoQueueStatus.COMPLETED, VideoQueueStatus.FAILED, 
                                VideoQueueStatus.CANCELLED, VideoQueueStatus.TIMEOUT]:
                return False
            
            # Cancel the request
            request.status = VideoQueueStatus.CANCELLED
            request.error = "Request cancelled by user"
            request.completed_at = datetime.now()
            
            # Remove from queue or processing
            if request_id in self.queue:
                self.queue.remove(request_id)
                self._update_queue_positions()
            
            if request_id in self.processing:
                self.processing.remove(request_id)
                
            if request_id in self._processing_threads:
                del self._processing_threads[request_id]
            
            # Update status manager
            from status_manager import status_manager
            status_manager.fail_processing(request.audio_id, "Processing cancelled by user")
            
            # Start next queued request if we freed up a processing slot
            self._start_next_request()
            
            return True
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics"""
        with self._lock:
            pending = sum(1 for r in self.requests.values() if r.status == VideoQueueStatus.PENDING)
            processing = sum(1 for r in self.requests.values() if r.status == VideoQueueStatus.PROCESSING)
            completed = sum(1 for r in self.requests.values() if r.status == VideoQueueStatus.COMPLETED)
            failed = sum(1 for r in self.requests.values() 
                        if r.status in [VideoQueueStatus.FAILED, VideoQueueStatus.TIMEOUT])
            
            return {
                "total_requests": len(self.requests),
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "failed": failed,
                "max_concurrent": self.max_concurrent,
                "queue_length": len(self.queue)
            }


# Global instance
video_queue_manager = VideoQueueManager() 