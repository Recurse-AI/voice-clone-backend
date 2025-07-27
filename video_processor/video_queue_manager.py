"""
Video Queue Manager

Manages video processing queue with timeout enforcement and status tracking.
Similar to RunPod queue but for video processing operations.
"""

import time
import threading
import uuid
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from threading import Lock, Thread
import logging

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
    """Individual video processing request
    
    request_id: Unique identifier for tracking this specific processing request
                Used for queue management, cancellation, and debugging
    audio_id:   User-facing identifier for the audio/video being processed
                Used for status checking and file organization
    """
    
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
        
        # 30 minutes timeout from when processing starts
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
        
        # Cleanup policy - keep completed requests for 30 minutes
        self.completed_retention_seconds = 30 * 60
        
        # Start cleanup thread
        self._cleanup_thread = Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"VideoQueueManager initialized with max_concurrent={max_concurrent}")
    
    def _start_timeout_monitor(self):
        """Start the timeout monitoring thread"""
        def monitor_timeouts():
            while not self._shutdown:
                try:
                    self._check_timeouts()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Timeout monitor error: {e}")
        
        self._timeout_monitor_thread = Thread(target=monitor_timeouts, daemon=True)
        self._timeout_monitor_thread.start()
    
    def _check_timeouts(self):
        """Check for timed out requests and mark them as failed"""
        current_time = datetime.now()
        timeout_requests = []
        
        with self._lock:
            for request_id in self.processing.copy():
                request = self.requests.get(request_id)
                if request and request.started_at:
                    elapsed = (current_time - request.started_at).total_seconds()
                    if elapsed > request.timeout_seconds:
                        timeout_requests.append(request_id)
        
        # Handle timeouts outside of lock to avoid deadlock
        for request_id in timeout_requests:
            self._handle_timeout(request_id)
    
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
                
                # Stop the processing thread if it exists
                if request_id in self._processing_threads:
                    # Thread will check status and exit gracefully
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
    
    def _periodic_cleanup(self):
        """Periodically clean up old completed requests"""
        import time
        
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
                    
                # Also clean up any lingering thread references
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
                "result": request.result,
                "estimated_time": self._estimate_remaining_time(request),
                "timeout_in": self._get_timeout_remaining(request)
            }
    
    def _estimate_remaining_time(self, request: VideoQueueRequest) -> Optional[str]:
        """Estimate remaining time for a request"""
        if request.status == VideoQueueStatus.COMPLETED:
            return None
        elif request.status == VideoQueueStatus.PROCESSING:
            return "Processing - up to 30 minutes"
        elif request.status == VideoQueueStatus.PENDING:
            # Estimate based on queue position
            avg_processing_time = 15  # minutes
            estimated_minutes = request.queue_position * avg_processing_time
            return f"~{estimated_minutes} minutes (queue position: {request.queue_position})"
        else:
            return None
    
    def _get_timeout_remaining(self, request: VideoQueueRequest) -> Optional[int]:
        """Get remaining seconds before timeout"""
        if request.status != VideoQueueStatus.PROCESSING or not request.started_at:
            return None
        
        elapsed = (datetime.now() - request.started_at).total_seconds()
        remaining = request.timeout_seconds - elapsed
        return max(0, int(remaining))
    
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
                # Processing thread will check status and exit
                
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
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the queue system"""
        with self._lock:
            current_time = datetime.now()
            
            # Count requests by status
            status_counts = {status.value: 0 for status in VideoQueueStatus}
            stuck_requests = []
            
            for request_id, request in self.requests.items():
                status_counts[request.status.value] += 1
                
                # Check for stuck requests (processing for too long)
                if request.status == VideoQueueStatus.PROCESSING and request.started_at:
                    processing_time = (current_time - request.started_at).total_seconds()
                    if processing_time > request.timeout_seconds:
                        stuck_requests.append({
                            "request_id": request_id,
                            "audio_id": request.audio_id,
                            "processing_time_seconds": processing_time,
                            "timeout_seconds": request.timeout_seconds
                        })
            
            # Calculate queue health metrics
            queue_health = {
                "total_requests": len(self.requests),
                "queue_size": len(self.queue),
                "processing_count": len(self.processing),
                "max_concurrent": self.max_concurrent,
                "status_counts": status_counts,
                "stuck_requests": stuck_requests,
                "threads_active": len(self._processing_threads),
                "retention_policy_seconds": self.completed_retention_seconds,
                "is_healthy": len(stuck_requests) == 0 and len(self.processing) <= self.max_concurrent
            }
            
            return queue_health
    
    def force_cleanup_stuck_requests(self) -> int:
        """Force cleanup of stuck requests and return count of cleaned up requests"""
        cleaned_count = 0
        current_time = datetime.now()
        
        with self._lock:
            stuck_request_ids = []
            
            for request_id, request in self.requests.items():
                if request.status == VideoQueueStatus.PROCESSING and request.started_at:
                    processing_time = (current_time - request.started_at).total_seconds()
                    if processing_time > request.timeout_seconds:
                        stuck_request_ids.append(request_id)
            
            # Clean up stuck requests
            for request_id in stuck_request_ids:
                logger.warning(f"Force cleaning up stuck request: {request_id}")
                request = self.requests[request_id]
                request.status = VideoQueueStatus.TIMEOUT
                request.completed_at = current_time
                request.error = "Force terminated due to timeout"
                
                # Remove from processing list
                if request_id in self.processing:
                    self.processing.remove(request_id)
                
                # Clean up thread reference
                if request_id in self._processing_threads:
                    del self._processing_threads[request_id]
                
                cleaned_count += 1
                
                # Update status manager
                try:
                    from status_manager import status_manager
                    status_manager.force_terminate_process(request.audio_id, "Force terminated due to timeout")
                except Exception as e:
                    logger.error(f"Error updating status manager for stuck request {request_id}: {e}")
            
            # Start next requests if slots are available
            if cleaned_count > 0:
                for _ in range(min(cleaned_count, len(self.queue))):
                    self._start_next_request()
        
        if cleaned_count > 0:
            logger.info(f"Force cleaned up {cleaned_count} stuck requests")
        
        return cleaned_count

    def shutdown(self):
        """Shutdown the queue manager"""
        self._shutdown = True
        logger.info("VideoQueueManager shutdown initiated")


# Global instance
video_queue_manager = VideoQueueManager() 