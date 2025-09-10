"""
CPU WhisperX Service Worker - CPU-based transcription worker
Handles WhisperX transcription requests using CPU processing
"""
import logging
import os
import uuid
import time

logger = logging.getLogger(__name__)


def process_cpu_whisperx_request(request_data: dict) -> dict:
    """Background CPU worker function for WhisperX transcription processing"""
    from app.utils.pipeline_utils import mark_service_worker_active, mark_service_worker_inactive, store_service_result
    from app.services.dub.whisperx_transcription import get_whisperx_transcription_service

    request_id = request_data.get("request_id")
    audio_path = request_data.get("audio_path")
    language_code = request_data.get("language_code")
    job_id = request_data.get("job_id")
    worker_id = f"cpu_whisperx_worker_{uuid.uuid4().hex[:8]}"

    logger.info(f"🎯 CPU WhisperX Worker {worker_id} starting request {request_id}")

    def handle_error(error_msg: str) -> dict:
        logger.error(f"❌ {error_msg} for {request_id}")
        result = {"error": error_msg, "success": False}
        store_service_result("cpu_whisperx", request_id, result)
        return result

    if not audio_path or not os.path.exists(audio_path):
        return handle_error(f"Audio file not found: {audio_path}")

    if not mark_service_worker_active("cpu_whisperx", worker_id):
        return handle_error("Failed to mark worker as active")

    try:
        # Force CPU device for this worker
        os.environ['WHISPER_DEVICE'] = 'cpu'
        os.environ['WHISPER_COMPUTE_TYPE'] = 'float32'

        transcription_service = get_whisperx_transcription_service()

        if not transcription_service.is_initialized:
            logger.info(f"🚀 Loading CPU WhisperX model for worker {worker_id}")
            if not transcription_service.load_model():
                return handle_error("Failed to load CPU WhisperX model")

        logger.info(f"🎤 CPU Processing transcription for {job_id}: {audio_path}")
        start_time = time.time()

        transcription_result = transcription_service._transcribe_direct(audio_path, language_code, job_id)
        processing_time = time.time() - start_time

        result = {
            "success": True,
            "segments": transcription_result.get("segments", []),
            "language": transcription_result.get("language"),
            "processing_time": processing_time,
            "worker_id": worker_id,
            "device": "cpu"
        }

        store_service_result("cpu_whisperx", request_id, result)
        logger.info(f"✅ CPU WhisperX transcription completed in {processing_time:.2f}s for {job_id}")
        return result

    except Exception as e:
        return handle_error(f"CPU WhisperX processing failed: {str(e)}")

    finally:
        # Cleanup resources
        try:
            # Force garbage collection for CPU memory cleanup
            import gc
            gc.collect()

            # Clear any cached data
            if 'transcription_service' in locals():
                # Clear service instance if it exists
                transcription_service = None

        except Exception as cleanup_error:
            logger.warning(f"⚠️ CPU WhisperX cleanup warning: {cleanup_error}")

        mark_service_worker_inactive("cpu_whisperx", worker_id)
        logger.info(f"🎯 CPU WhisperX Worker {worker_id} released")
