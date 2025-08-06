"""
Simplified RunPod Service for Audio Separation
"""
import time
import logging
from typing import Dict, Any, Optional
import requests
from datetime import datetime
from app.config.settings import settings

logger = logging.getLogger(__name__)


class RunPodService:
    """Simplified service for vocal/instrument separation using RunPod API"""
    
    def __init__(self):
        self.base_url = f"https://api.runpod.ai/v2/{settings.RUNPOD_ENDPOINT_ID}"
        self.api_key = settings.API_ACCESS_TOKEN
        
        if not self.api_key:
            raise ValueError("RunPod API key not configured")
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
    
    def submit_separation_request(self, audio_url: str, caller_info: str = None) -> str:
        """Submit audio separation request and return request ID"""
        if not audio_url:
            raise ValueError('Audio URL is required')
        
        payload = {
            "input": {
                "input_audio": audio_url
            },
            "policy": {
                "executionTimeout": 600000  # 10 minutes
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/run",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                request_id = data.get('id')
                if request_id:
                    logger.info(f"Submitted separation request {request_id} from {caller_info}")
                    return request_id
                else:
                    raise Exception("No request ID returned from RunPod")
            else:
                raise Exception(f"RunPod API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to submit separation request: {e}")
            raise
    
    def get_separation_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of separation request"""
        try:
            response = requests.get(
                f"{self.base_url}/status/{request_id}",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Map RunPod status to our simplified status
                runpod_status = data.get('status')
                progress = 0
                status = "pending"
                
                if runpod_status == 'IN_QUEUE':
                    status = "pending"
                    progress = 10
                elif runpod_status == 'IN_PROGRESS':
                    status = "processing" 
                    progress = 50
                elif runpod_status == 'COMPLETED':
                    status = "completed"
                    progress = 100
                elif runpod_status == 'FAILED':
                    status = "failed"
                    progress = 0
                
                return {
                    "status": status,
                    "progress": progress,
                    "result": data.get('output'),
                    "error": data.get('error'),
                    "created_at": data.get('created_at') or datetime.now().isoformat(),
                    "started_at": data.get('started_at'),
                    "completed_at": data.get('completed_at')
                }
            elif response.status_code == 404:
                logger.error(f"Job not found: {request_id}")
                return None
            else:
                logger.error(f"Failed to get status: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get separation status: {e}")
            return None
    
    def wait_for_completion(self, request_id: str, timeout: int = 900) -> Dict[str, Any]:
        """Wait for separation to complete (blocking)"""
        start_time = time.time()
        check_interval = 10  # Check every 10 seconds
        
        while time.time() - start_time < timeout:
            status = self.get_separation_status(request_id)
            
            if not status:
                return {
                    "success": False,
                    "error": "Request not found",
                    "request_id": request_id
                }
            
            if status["status"] == "completed":
                result = status.get("result", {})
                return {
                    "success": True,
                    "status": "COMPLETED",
                    "output": result,
                    "request_id": request_id
                }
            elif status["status"] == "failed":
                return {
                    "success": False,
                    "error": status.get("error", "Unknown error"),
                    "request_id": request_id
                }
            
            # Still processing, wait and check again
            time.sleep(check_interval)
        
        # Timeout reached
        return {
            "success": False,
            "error": f"Request timed out after {timeout} seconds",
            "request_id": request_id
        }


# Global service instance
runpod_service = RunPodService()