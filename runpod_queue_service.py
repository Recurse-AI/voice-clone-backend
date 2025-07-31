"""
RunPod Queue Service

Wrapper service that integrates the queue manager with RunPod service.
Provides a clean interface for sync audio separation requests.
"""

import time
from typing import Dict, Any, Optional
import logging
from dub.runpod_service import RunPodService
from runpod_queue_manager import runpod_queue_manager, QueueStatus

logger = logging.getLogger(__name__)


class RunPodQueueService:
    """
    Service wrapper that uses queue management for RunPod requests.
    
    This service ensures that no more than 2 concurrent requests are sent to
    the 3rd party RunPod service, preventing overloading.
    """
    
    def __init__(self):
        self.runpod_service = RunPodService()
        # Initialize the queue manager with the RunPod service
        runpod_queue_manager.set_runpod_service(self.runpod_service)
        logger.info("RunPod Queue Service initialized")
    
    def submit_separation_request(self, audio_url: str, caller_info: str = None) -> str:
        """
        Submit an audio separation request to the queue.
        
        Args:
            audio_url: URL of the audio file to separate
            caller_info: Information about which API/process is calling
            
        Returns:
            request_id: Unique identifier for tracking the request
        """
        request_id = runpod_queue_manager.submit_request(audio_url, caller_info)
        logger.info(f"Submitted separation request {request_id} from {caller_info}")
        return request_id
    
    def get_separation_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a separation request.
        
        Args:
            request_id: The request ID returned from submit_separation_request
            
        Returns:
            Status dictionary with progress, result, and error information
        """
        return runpod_queue_manager.get_request_status(request_id)
    
    def cancel_separation_request(self, request_id: str) -> bool:
        """
        Cancel a pending or processing separation request.
        
        Args:
            request_id: The request ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        return runpod_queue_manager.cancel_request(request_id)
    
    def wait_for_completion(self, request_id: str, timeout: int = 900, 
                          check_interval: int = 5) -> Dict[str, Any]:
        """
        Wait for a separation request to complete (blocking).
        
        This method is useful for synchronous workflows where you need to wait
        for the result before proceeding.
        
        Args:
            request_id: The request ID to wait for
            timeout: Maximum time to wait in seconds
            check_interval: How often to check status in seconds
            
        Returns:
            Final result dictionary with vocal and instrument URLs
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_separation_status(request_id)
            
            if not status:
                return {
                    "success": False,
                    "error": "Request not found",
                    "request_id": request_id
                }
            
            if status["status"] == QueueStatus.COMPLETED.value:
                return {
                    "success": True,
                    "result": status["result"],
                    "request_id": request_id,
                    "processing_time": (time.time() - start_time)
                }
            
            elif status["status"] == QueueStatus.FAILED.value:
                return {
                    "success": False,
                    "error": status["error"],
                    "request_id": request_id
                }
            
            elif status["status"] == QueueStatus.CANCELLED.value:
                return {
                    "success": False,
                    "error": "Request was cancelled",
                    "request_id": request_id
                }
            
            # Still pending or processing, wait and check again
            time.sleep(check_interval)
        
        # Timeout reached
        return {
            "success": False,
            "error": "Request timeout - separation took too long",
            "request_id": request_id
        }
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics"""
        return runpod_queue_manager.get_queue_stats()
    
    def process_audio_separation_sync(self, audio_url: str, caller_info: str = None) -> Dict[str, Any]:
        """
        Process audio separation synchronously using the queue.
        
        This is a convenience method that submits the request and waits for completion.
        Use this for backwards compatibility with existing sync code.
        
        Args:
            audio_url: URL of the audio file to separate
            caller_info: Information about which API/process is calling
            
        Returns:
            Separation result with vocal and instrument URLs
        """
        # Submit request
        request_id = self.submit_separation_request(audio_url, caller_info)
        
        # Wait for completion
        result = self.wait_for_completion(request_id)
        
        if result["success"]:
            # Return in the same format as the original RunPod service
            return result["result"]
        else:
            # Return error in expected format
            return {
                "status": "FAILED",
                "error": result["error"]
            }
    
    def cleanup_old_requests(self, max_age_hours: int = 24):
        """Clean up old completed/failed requests"""
        runpod_queue_manager.cleanup_old_requests(max_age_hours)


# Global instance for use throughout the application
runpod_queue_service = RunPodQueueService() 