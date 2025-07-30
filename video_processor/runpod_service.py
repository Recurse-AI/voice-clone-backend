"""
RunPod Service Integration

Handles vocal and instrument separation using RunPod API.
"""

import os
import time
from typing import Dict, Any, Optional

import requests

from config import settings


class RunPodService:
    """Service for vocal/instrument separation using RunPod API"""
    
    def __init__(self):
        self.base_url = f"https://api.runpod.ai/v2/{settings.RUNPOD_ENDPOINT_ID}"
        self.api_key = settings.API_ACCESS_TOKEN
        
        if not self.api_key:
            raise ValueError("RunPod API key not configured")
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
    
    def process_audio_separation(self, audio_url: str) -> Dict[str, Any]:
        """Process audio separation asynchronously"""
        if not audio_url:
            raise ValueError('Audio URL is required')
        
        payload = {
            "input": {
                "input_audio": audio_url
            },
            "policy": {
                "executionTimeout": 600000  # Reduced from 900000 to 600000 (10 minutes)
            }
        }
        
        response = requests.post(
            f"{self.base_url}/run",
            json=payload,
            headers=self.headers,
            timeout=30  # Reduced from 60 to 30 seconds
        )
        
        if response.status_code != 200:
            raise Exception(f"RunPod API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        if not result.get('id'):
            raise Exception('Invalid response from RunPod API: Missing job ID')
        
        return result
    
    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check the status of separation job"""
        if not job_id:
            raise ValueError('Job ID is required')
        
        response = requests.get(
            f"{self.base_url}/status/{job_id}",
            headers=self.headers,
            timeout=20  # Reduced from 30 to 20 seconds
        )
        
        if response.status_code != 200:
            raise Exception(f"RunPod API error: {response.status_code} - {response.text}")
        
        data = response.json()
        
        if data.get('status') == 'COMPLETED' and data.get('output'):
            if data['output'].get('processing_status') == 'COMPLETED':
                return {
                    'id': data['id'],
                    'status': data['status'],
                    'output': {
                        'processing_status': 'COMPLETED',
                        'vocal_audio': data['output'].get('vocal_audio'),
                        'instrument_audio': data['output'].get('instrument_audio')
                    }
                }
            elif data['output'].get('processing_status') == 'FAILED':
                return {
                    'id': data['id'],
                    'status': 'FAILED',
                    'error': data['output'].get('failed_reason', 'Processing failed')
                }
        elif data.get('status') == 'FAILED':
            return {
                'id': data['id'],
                'status': 'FAILED',
                'error': data.get('error', 'Job failed')
            }
        elif data.get('status') in ['IN_QUEUE', 'IN_PROGRESS']:
            return {
                'id': data['id'],
                'status': data['status']
            }
        
        return data

    def wait_for_completion(self, job_id: str, timeout: int = 900) -> Dict[str, Any]:
        """Wait for job completion with optimized timing"""
        start_time = time.time()
        check_interval = 5  # Start with 5 seconds
        max_interval = 20   # Maximum 20 seconds
        
        while time.time() - start_time < timeout:
            try:
                status = self.check_job_status(job_id)
                
                if status.get('status') == 'COMPLETED':
                    return status
                elif status.get('status') == 'FAILED':
                    return status
                elif status.get('status') in ['IN_QUEUE', 'IN_PROGRESS']:
                    # Progressive wait - start with 5s, increase gradually
                    elapsed = time.time() - start_time
                    if elapsed < 60:  # First minute: 5s intervals
                        check_interval = 10
                    elif elapsed < 300:  # Next 4 minutes: 10s intervals
                        check_interval = 15
                    else:  # After 5 minutes: 20s intervals
                        check_interval = max_interval
                    
                    time.sleep(check_interval)
                    continue
                else:
                    time.sleep(15)  # Reduced from 8 to 5 seconds
                    
            except Exception as e:
                return {
                    'id': job_id,
                    'status': 'FAILED',
                    'error': f'Error checking job status: {str(e)}'
                }
        
        return {
            'id': job_id,
            'status': 'FAILED',
            'error': 'Job timeout - processing took too long'
        } 