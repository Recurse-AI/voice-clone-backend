"""
WhisperX Service Worker - Dedicated VRAM worker for transcription
Handles serial processing of WhisperX transcription requests
"""
import logging
import os
import uuid
import time

logger = logging.getLogger(__name__)


def process_whisperx_request(request_data: dict) -> dict:
    """Background worker function for WhisperX transcription processing"""
    from app.utils.pipeline_utils import mark_service_worker_active, mark_service_worker_inactive, store_service_result
    from app.services.dub.whisperx_transcription import get_whisperx_transcription_service
    
    request_id = request_data.get("request_id")
    audio_path = request_data.get("audio_path")
    language_code = request_data.get("language_code")
    job_id = request_data.get("job_id")
    worker_id = f"whisperx_worker_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"🎯 WhisperX Worker {worker_id} starting request {request_id}")
    
    def handle_error(error_msg: str) -> dict:
        logger.error(f"❌ {error_msg} for {request_id}")
        result = {"error": error_msg, "success": False}
        store_service_result("whisperx", request_id, result)
        return result
    
    if not audio_path or not os.path.exists(audio_path):
        return handle_error(f"Audio file not found: {audio_path}")
    
    if not mark_service_worker_active("whisperx", worker_id):
        return handle_error("Failed to mark worker as active")
    
    try:
        transcription_service = get_whisperx_transcription_service()
        
        if not transcription_service.is_initialized:
            logger.info(f"🚀 Loading WhisperX model for worker {worker_id}")
            if not transcription_service.load_model():
                return handle_error("Failed to load WhisperX model")
        
        logger.info(f"🎤 Processing transcription for {job_id}: {audio_path} (language: {language_code})")
        start_time = time.time()
        
        transcription_result = transcription_service._transcribe_direct(audio_path, language_code, job_id)
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "segments": transcription_result.get("segments", []),
            "language": transcription_result.get("language"),
            "processing_time": processing_time,
            "worker_id": worker_id
        }
        
        store_service_result("whisperx", request_id, result)
        logger.info(f"✅ WhisperX transcription completed in {processing_time:.2f}s for {job_id}")
        return result
        
    except Exception as e:
        return handle_error(f"WhisperX processing failed: {str(e)}")
        
    finally:
        mark_service_worker_inactive("whisperx", worker_id)
        logger.info(f"🎯 WhisperX Worker {worker_id} released")


