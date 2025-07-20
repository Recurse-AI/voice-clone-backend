"""
RunPod Queue Management System

Manages a queue for RunPod audio separation requests to prevent overloading the 3rd party service.
Limits concurrent requests to maximum 2 and provides status tracking.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

logger = logging.getLogger(__name__)


class QueueStatus(Enum):
    """Queue request status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueueRequest:
    """Represents a queued RunPod request"""
    request_id: str
    audio_url: str
    status: QueueStatus = QueueStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: int = 0
    caller_info: Optional[str] = None  # To track which API called this


class RunPodQueueManager:
    """
    Queue manager for RunPod audio separation requests.
    
    Features:
    - Maximum 2 concurrent requests
    - Queue management with status tracking
    - Thread-safe operations
    - Progress tracking
    - Error handling and retry logic
    """
    
    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self.requests: Dict[str, QueueRequest] = {}
        self.pending_queue = queue.Queue()  # Thread-safe queue
        self.processing_count = 0
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._runpod_service = None
        self._stop_event = threading.Event()
        
        # Start queue processor threads
        for i in range(max_concurrent):
            worker_thread = threading.Thread(
                target=self._queue_processor_worker,
                name=f"QueueWorker-{i+1}",
                daemon=True
            )
            worker_thread.start()
        
        logger.info(f"RunPod Queue Manager initialized with max_concurrent={max_concurrent}")
    
    def set_runpod_service(self, service):
        """Set the RunPod service instance"""
        self._runpod_service = service
    
    def submit_request(self, audio_url: str, caller_info: str = None) -> str:
        """
        Submit a new audio separation request to the queue.
        
        Args:
            audio_url: URL of the audio file to process
            caller_info: Information about which API/process called this
            
        Returns:
            request_id: Unique identifier for tracking the request
        """
        request_id = str(uuid.uuid4())
        
        with self.lock:
            request = QueueRequest(
                request_id=request_id,
                audio_url=audio_url,
                caller_info=caller_info
            )
            self.requests[request_id] = request
            
            # Add to queue for processing
            self.pending_queue.put(request_id)
        
        logger.info(f"Request {request_id} submitted to queue (caller: {caller_info})")
        return request_id
    
    def _queue_processor_worker(self):
        """Worker thread that processes queued requests"""
        while not self._stop_event.is_set():
            try:
                # Wait for a request (blocking with timeout)
                try:
                    request_id = self.pending_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the request
                self._process_single_request(request_id)
                
            except Exception as e:
                logger.error(f"Error in queue processor worker: {e}")
    
    def _process_single_request(self, request_id: str):
        """Process a single RunPod request"""
        try:
            with self.lock:
                if request_id not in self.requests:
                    return
                request = self.requests[request_id]
                
                # Check if request was cancelled
                if request.status == QueueStatus.CANCELLED:
                    return
                
                # Mark as processing
                request.status = QueueStatus.PROCESSING
                request.started_at = datetime.now()
                request.progress = 10
                self.processing_count += 1
            
            logger.info(f"Started processing request {request_id}")
            
            if not self._runpod_service:
                raise Exception("RunPod service not initialized")
            
            # Update progress
            request.progress = 20
            
            # Submit to RunPod
            separation_result = self._runpod_service.process_audio_separation(request.audio_url)
            
            if not separation_result or not separation_result.get("id"):
                raise Exception("RunPod service returned invalid response")
            
            # Update progress
            request.progress = 40
            
            # Wait for completion
            completion_result = self._runpod_service.wait_for_completion(separation_result["id"])
            
            # Update progress
            request.progress = 90
            
            if completion_result.get("status") != "COMPLETED":
                raise Exception(f"RunPod job failed: {completion_result.get('error', 'Unknown error')}")
            
            # Mark as completed
            with self.lock:
                request.status = QueueStatus.COMPLETED
                request.completed_at = datetime.now()
                request.result = completion_result
                request.progress = 100
                self.processing_count -= 1
            
            logger.info(f"Request {request_id} completed successfully")
            
        except Exception as e:
            # Mark as failed
            with self.lock:
                request.status = QueueStatus.FAILED
                request.completed_at = datetime.now()
                request.error = str(e)
                request.progress = 0
                self.processing_count -= 1
            
            logger.error(f"Request {request_id} failed: {str(e)}")
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request"""
        with self.lock:
            if request_id not in self.requests:
                return None
            
            request = self.requests[request_id]
            
            # Calculate queue position for pending requests
            queue_position = None
            if request.status == QueueStatus.PENDING:
                # Approximate queue position (rough estimate)
                queue_position = self.pending_queue.qsize()
            
            return {
                "request_id": request_id,
                "status": request.status.value,
                "progress": request.progress,
                "created_at": request.created_at.isoformat(),
                "started_at": request.started_at.isoformat() if request.started_at else None,
                "completed_at": request.completed_at.isoformat() if request.completed_at else None,
                "queue_position": queue_position,
                "result": request.result,
                "error": request.error,
                "caller_info": request.caller_info
            }
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending or processing request"""
        with self.lock:
            if request_id not in self.requests:
                return False
            
            request = self.requests[request_id]
            
            if request.status in [QueueStatus.COMPLETED, QueueStatus.FAILED, QueueStatus.CANCELLED]:
                return False
            
            # Cancel the request
            request.status = QueueStatus.CANCELLED
            request.completed_at = datetime.now()
            request.error = "Cancelled by user"
        
        logger.info(f"Request {request_id} cancelled")
        return True
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get overall queue statistics"""
        with self.lock:
            total_requests = len(self.requests)
            pending_count = sum(1 for r in self.requests.values() if r.status == QueueStatus.PENDING)
            processing_count = self.processing_count
            completed_count = sum(1 for r in self.requests.values() if r.status == QueueStatus.COMPLETED)
            failed_count = sum(1 for r in self.requests.values() if r.status == QueueStatus.FAILED)
            
            return {
                "total_requests": total_requests,
                "pending": pending_count,
                "processing": processing_count,
                "completed": completed_count,
                "failed": failed_count,
                "max_concurrent": self.max_concurrent,
                "queue_length": self.pending_queue.qsize()
            }
    
    def cleanup_old_requests(self, max_age_hours: int = 24):
        """Clean up old completed/failed requests"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        with self.lock:
            to_remove = []
            for request_id, request in self.requests.items():
                if (request.status in [QueueStatus.COMPLETED, QueueStatus.FAILED, QueueStatus.CANCELLED] and
                    request.completed_at and request.completed_at.timestamp() < cutoff_time):
                    to_remove.append(request_id)
            
            for request_id in to_remove:
                del self.requests[request_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old requests")
    
    def shutdown(self):
        """Shutdown the queue manager"""
        self._stop_event.set()
        self.executor.shutdown(wait=True)


# Global queue manager instance
runpod_queue_manager = RunPodQueueManager(max_concurrent=2) 