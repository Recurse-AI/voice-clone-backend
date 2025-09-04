import time
import logging
from typing import Dict, Any, Optional, Callable
from app.utils.runpod_service import runpod_service

logger = logging.getLogger(__name__)

class RunPodMonitor:
    def __init__(self, 
                 runpod_request_id: str, 
                 job_id: str, 
                 timeout_seconds: int = 600,
                 poll_interval: int = 10):
        self.runpod_request_id = runpod_request_id
        self.job_id = job_id
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.is_monitoring = False
        
    def monitor_until_completion(self, 
                               on_progress: Optional[Callable[[str, int], None]] = None,
                               on_failed: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
        self.is_monitoring = True
        max_attempts = self.timeout_seconds // self.poll_interval
        attempt = 0
        
        try:
            while attempt < max_attempts and self.is_monitoring:
                
                try:
                    status = runpod_service.get_separation_status(self.runpod_request_id)
                    
                    if not status:
                        attempt += 1
                        time.sleep(self.poll_interval)
                        continue
                        
                    job_status = status.get("status", "unknown").upper()
                    progress = status.get("progress", 0)
                    
                    if on_progress:
                        on_progress(job_status.lower(), progress)
                    
                    if job_status == "COMPLETED":
                        return {
                            "success": True,
                            "status": "COMPLETED",
                            "result": status.get("result", {}),
                            "output": status.get("result", {})
                        }
                        
                    elif job_status == "FAILED":
                        error_msg = status.get("error", "RunPod job failed")
                        return {
                            "success": False,
                            "status": "FAILED", 
                            "error": error_msg
                        }
                        
                    elif job_status == "CANCELLED":
                        if on_failed:
                            on_failed()
                        return {
                            "success": False,
                            "status": "FAILED",
                            "error": "Job failed by RunPod"
                        }
                    
                    attempt += 1
                    time.sleep(self.poll_interval)
                    
                except Exception as e:
                    logger.warning(f"Error checking RunPod status: {e}")
                    attempt += 1
                    time.sleep(self.poll_interval)
                    continue
            
            return {
                "success": False,
                "status": "TIMEOUT",
                "error": f"Job monitoring timed out after {self.timeout_seconds} seconds"
            }
            
        finally:
            self.is_monitoring = False
            
    def stop_monitoring(self):
        self.is_monitoring = False

def monitor_runpod_job(runpod_request_id: str, 
                      job_id: str, 
                      timeout_seconds: int = 600,
                      poll_interval: int = 10,
                      on_progress: Optional[Callable[[str, int], None]] = None,
                      on_failed: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
    monitor = RunPodMonitor(runpod_request_id, job_id, timeout_seconds, poll_interval)
    return monitor.monitor_until_completion(on_progress, on_failed)
