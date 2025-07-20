"""
RunPod Queue Management System

Manages a queue for RunPod audio separation requests to prevent overloading the 3rd party service.
Limits concurrent requests to maximum 2 and provides status tracking.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

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
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.requests: Dict[str, QueueRequest] = {}
        self.queue: List[str] = []  # Request IDs in queue order
        self.processing: Dict[str, asyncio.Task] = {}  # Currently processing requests
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._runpod_service = None
        
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
            self.queue.append(request_id)
        
        logger.info(f"Request {request_id} submitted to queue (caller: {caller_info})")
        
        # Start processing if slots available
        asyncio.create_task(self._process_queue())
        
        return request_id
    
    async def _process_queue(self):
        """Process queued requests with concurrency control"""
        while True:
            with self.lock:
                # Check if we have pending requests and available slots
                pending_requests = [rid for rid in self.queue 
                                  if self.requests[rid].status == QueueStatus.PENDING]
                
                if not pending_requests:
                    break
                
                # Check available processing slots
                active_processing = len([task for task in self.processing.values() 
                                       if not task.done()])
                
                if active_processing >= self.max_concurrent:
                    break
                
                # Get next request to process
                request_id = pending_requests[0]
                request = self.requests[request_id]
                
                # Mark as processing
                request.status = QueueStatus.PROCESSING
                request.started_at = datetime.now()
                request.progress = 10
            
            # Create processing task
            task = asyncio.create_task(self._process_single_request(request_id))
            self.processing[request_id] = task
            
            logger.info(f"Started processing request {request_id}")
    
    async def _process_single_request(self, request_id: str):
        """Process a single RunPod request"""
        try:
            request = self.requests[request_id]
            
            if not self._runpod_service:
                raise Exception("RunPod service not initialized")
            
            # Update progress
            request.progress = 20
            
            # Submit to RunPod (sync call in executor)
            loop = asyncio.get_event_loop()
            separation_result = await loop.run_in_executor(
                self.executor,
                self._runpod_service.process_audio_separation,
                request.audio_url
            )
            
            if not separation_result or not separation_result.get("id"):
                raise Exception("RunPod service returned invalid response")
            
            # Update progress
            request.progress = 40
            
            # Wait for completion (sync call in executor)
            completion_result = await loop.run_in_executor(
                self.executor,
                self._runpod_service.wait_for_completion,
                separation_result["id"]
            )
            
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
            
            logger.info(f"Request {request_id} completed successfully")
            
        except Exception as e:
            # Mark as failed
            with self.lock:
                request.status = QueueStatus.FAILED
                request.completed_at = datetime.now()
                request.error = str(e)
                request.progress = 0
            
            logger.error(f"Request {request_id} failed: {str(e)}")
        
        finally:
            # Clean up
            if request_id in self.processing:
                del self.processing[request_id]
            
            # Try to process next request
            asyncio.create_task(self._process_queue())
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request"""
        with self.lock:
            if request_id not in self.requests:
                return None
            
            request = self.requests[request_id]
            
            # Calculate queue position for pending requests
            queue_position = None
            if request.status == QueueStatus.PENDING:
                try:
                    queue_position = self.queue.index(request_id) + 1
                except ValueError:
                    queue_position = None
            
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
            
            # Remove from queue if pending
            if request_id in self.queue:
                self.queue.remove(request_id)
            
            # Cancel processing task if running
            if request_id in self.processing:
                task = self.processing[request_id]
                task.cancel()
                del self.processing[request_id]
        
        logger.info(f"Request {request_id} cancelled")
        return True
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get overall queue statistics"""
        with self.lock:
            total_requests = len(self.requests)
            pending_count = sum(1 for r in self.requests.values() if r.status == QueueStatus.PENDING)
            processing_count = sum(1 for r in self.requests.values() if r.status == QueueStatus.PROCESSING)
            completed_count = sum(1 for r in self.requests.values() if r.status == QueueStatus.COMPLETED)
            failed_count = sum(1 for r in self.requests.values() if r.status == QueueStatus.FAILED)
            
            return {
                "total_requests": total_requests,
                "pending": pending_count,
                "processing": processing_count,
                "completed": completed_count,
                "failed": failed_count,
                "max_concurrent": self.max_concurrent,
                "queue_length": len(self.queue)
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
                if request_id in self.queue:
                    self.queue.remove(request_id)
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old requests")


# Global queue manager instance
runpod_queue_manager = RunPodQueueManager(max_concurrent=2) 