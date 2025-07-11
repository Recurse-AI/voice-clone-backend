"""
RunPod Service Integration

Handles vocal and instrument separation using RunPod API.
"""

import os
import time
import requests
from typing import Dict, Any, Optional
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
                "executionTimeout": 900000
            }
        }
        
        response = requests.post(
            f"{self.base_url}/run",
            json=payload,
            headers=self.headers,
            timeout=60
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
            timeout=30
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
    
    def wait_for_completion(self, job_id: str, max_wait: int = 900) -> Dict[str, Any]:
        """Wait for job completion with polling"""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            result = self.check_job_status(job_id)
            
            if result.get('status') == 'COMPLETED':
                return result
            elif result.get('status') == 'FAILED':
                return result
            
            time.sleep(15) #sleep for 15 seconds
        
        return {
            'id': job_id,
            'status': 'FAILED',
            'error': 'Job timed out'
        } 