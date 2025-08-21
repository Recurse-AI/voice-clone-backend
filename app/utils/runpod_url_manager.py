"""RunPod URL Management Utility"""
import logging
from typing import Dict, Any, Tuple
from app.models.dub_job import DubJob

logger = logging.getLogger(__name__)

class RunPodURLManager:
    VOCALS_KEY = "vocals"
    INSTRUMENTS_KEY = "instruments" 
    RUNPOD_URLS_KEY = "runpod_urls"
    
    @classmethod
    def extract_urls_from_runpod_response(cls, runpod_output: Dict[str, Any]) -> Dict[str, str]:
        """Extract vocal and instrument URLs from RunPod response"""
        urls = {}
        vocal_url = runpod_output.get('vocal_audio') or runpod_output.get('vocals')
        instrument_url = runpod_output.get('instrument_audio') or runpod_output.get('instruments')
        
        if vocal_url:
            urls[cls.VOCALS_KEY] = vocal_url
        if instrument_url:
            urls[cls.INSTRUMENTS_KEY] = instrument_url
            
        return urls
    
    @classmethod
    def store_urls_in_details(cls, details: Dict[str, Any], runpod_urls: Dict[str, str]) -> Dict[str, Any]:
        """Store RunPod URLs in job details"""
        if runpod_urls:
            details[cls.RUNPOD_URLS_KEY] = runpod_urls
        return details
    
    @classmethod
    def retrieve_urls_from_job(cls, job: DubJob) -> Dict[str, str]:
        """Retrieve stored RunPod URLs from job details"""
        if not job.details:
            return {}
        return job.details.get(cls.RUNPOD_URLS_KEY, {})
    
    @classmethod
    def add_urls_to_folder_upload(cls, folder_upload: Dict[str, Any], runpod_urls: Dict[str, str], job_id: str) -> Dict[str, Any]:
        """Add RunPod URLs to folder_upload structure"""
        if not folder_upload:
            folder_upload = {}
            
        if not runpod_urls:
            logger.warning(f"No RunPod URLs provided for job {job_id}")
            return folder_upload
            
        if runpod_urls.get(cls.VOCALS_KEY):
            vocal_filename = f"vocals_{job_id}.wav"
            folder_upload[vocal_filename] = {
                "url": runpod_urls[cls.VOCALS_KEY],
                "type": "audio",
                "success": True
            }
            logger.info(f"Added vocal URL to folder upload for job {job_id}: {vocal_filename}")
            
        if runpod_urls.get(cls.INSTRUMENTS_KEY):
            instrument_filename = f"instruments_{job_id}.wav"
            folder_upload[instrument_filename] = {
                "url": runpod_urls[cls.INSTRUMENTS_KEY], 
                "type": "audio",
                "success": True
            }
            logger.info(f"Added instrument URL to folder upload for job {job_id}: {instrument_filename}")
            
        return folder_upload
    
    @classmethod
    def validate_urls(cls, runpod_urls: Dict[str, str]) -> Tuple[bool, str]:
        """Validate RunPod URLs structure and content"""
        if not isinstance(runpod_urls, dict):
            return False, "RunPod URLs must be a dictionary"
            
        valid_keys = {cls.VOCALS_KEY, cls.INSTRUMENTS_KEY}
        invalid_keys = set(runpod_urls.keys()) - valid_keys
        
        if invalid_keys:
            return False, f"Invalid keys in RunPod URLs: {invalid_keys}"
            
        for key, url in runpod_urls.items():
            if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
                return False, f"Invalid URL format for {key}: {url}"
                
        return True, ""
    
    @classmethod
    def copy_urls_from_original_job(cls, original_job: DubJob, target_details: Dict[str, Any]) -> Dict[str, Any]:
        """Copy RunPod URLs from original job to target details (for redub scenarios)"""
        original_urls = cls.retrieve_urls_from_job(original_job)
        if original_urls:
            target_details = cls.store_urls_in_details(target_details, original_urls)
        return target_details
