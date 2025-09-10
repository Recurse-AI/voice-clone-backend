"""
CPU Fish Speech Service Worker - CPU-based voice cloning worker
Handles Fish Speech voice cloning requests using CPU processing
"""
import logging
import os
import uuid
import json
import time
import base64
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _handle_error(request_id: str, error_msg: str) -> dict:
    """Helper function to handle errors consistently"""
    logger.error(f"❌ {error_msg} for {request_id}")
    from app.utils.pipeline_utils import store_service_result
    result = {"error": error_msg, "success": False}
    store_service_result("cpu_fish_speech", request_id, result)
    return result


def _validate_request_data(request_data: dict) -> tuple[str, str, str, str]:
    """Validate and extract request parameters"""
    request_id = request_data.get("request_id")
    text = request_data.get("text")
    reference_audio_bytes_b64 = request_data.get("reference_audio_bytes")
    output_path = request_data.get("output_path")

    if not text:
        raise ValueError("No text provided for voice cloning")
    if not reference_audio_bytes_b64:
        raise ValueError("No reference audio provided")

    return request_id, text, reference_audio_bytes_b64, output_path


def _process_cloning_result(cloning_result: dict, output_path: str, processing_time: float, worker_id: str) -> dict:
    """Process voice cloning result and format response"""
    if cloning_result.get("success"):
        if output_path and os.path.exists(output_path):
            return {
                "success": True,
                "output_path": output_path,
                "processing_time": processing_time,
                "worker_id": worker_id,
                "audio_duration": cloning_result.get("audio_duration"),
                "device": "cpu"
            }
        else:
            return {"error": f"Output file not created: {output_path}", "success": False}
    else:
        error_msg = cloning_result.get("error", "Voice cloning failed")
        return {"error": error_msg, "success": False}


def process_cpu_fish_speech_request(request_data: dict) -> dict:
    """
    Background CPU worker function for Fish Speech voice cloning processing
    This runs in a dedicated CPU worker process
    """
    request_id = request_data.get("request_id")
    worker_id = f"cpu_fish_speech_worker_{uuid.uuid4().hex[:8]}"

    logger.info(f"🐟 CPU Fish Speech Worker {worker_id} starting request {request_id}")

    # Mark worker as active for serial processing
    from app.utils.pipeline_utils import mark_service_worker_active, mark_service_worker_inactive

    if not mark_service_worker_active("cpu_fish_speech", worker_id):
        return _handle_error(request_id, "Failed to mark worker as active")

    try:
        # Validate request data
        try:
            request_id, text, reference_audio_bytes_b64, output_path = _validate_request_data(request_data)
        except ValueError as e:
            return _handle_error(request_id, str(e))

        # Force CPU device for this worker
        os.environ['FISH_SPEECH_DEVICE'] = 'cpu'
        os.environ['FISH_SPEECH_PRECISION'] = 'float32'

        # Load Fish Speech service (CPU mode)
        from app.services.dub.fish_speech_service import get_fish_speech_service
        fish_speech_service = get_fish_speech_service()

        # Ensure model is loaded
        if not fish_speech_service.is_initialized:
            logger.info(f"🚀 Loading CPU Fish Speech model for worker {worker_id}")
            if not fish_speech_service.load_model():
                return _handle_error(request_id, "Failed to load CPU Fish Speech model")

        # Decode reference audio bytes
        try:
            reference_audio_bytes = base64.b64decode(reference_audio_bytes_b64)
        except Exception as e:
            return _handle_error(request_id, f"Failed to decode reference audio: {str(e)}")

        logger.info(f"🎵 CPU Processing voice cloning: '{text[:50]}...' -> {output_path}")
        start_time = time.time()

        # Perform voice cloning using CPU
        cloning_result = fish_speech_service._generate_direct(
            text, reference_audio_bytes, "", request_id
        )

        # Save audio data to file if cloning was successful
        if cloning_result.get("success") and cloning_result.get("audio_data"):
            try:
                with open(output_path, "wb") as f:
                    f.write(cloning_result["audio_data"])
                logger.info(f"✅ Saved CPU voice cloning output to: {output_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save output file {output_path}: {e}")
                cloning_result = {"success": False, "error": f"Failed to save output file: {str(e)}"}

        processing_time = time.time() - start_time
        logger.info(f"✅ CPU Fish Speech voice cloning completed in {processing_time:.2f}s")

        # Process result using helper function
        result = _process_cloning_result(cloning_result, output_path, processing_time, worker_id)

        # Store result in Redis
        from app.utils.pipeline_utils import store_service_result
        store_service_result("cpu_fish_speech", request_id, result)

        logger.info(f"🐟 CPU Fish Speech Worker {worker_id} completed request {request_id}")
        return result

    except Exception as e:
        return _handle_error(request_id, f"CPU Fish Speech processing failed: {str(e)}")

    finally:
        # Cleanup resources
        try:
            # Force garbage collection for CPU memory cleanup
            import gc
            gc.collect()

            # Clear any cached data
            if 'fish_speech_service' in locals():
                # Clear service instance if it exists
                fish_speech_service = None

        except Exception as cleanup_error:
            logger.warning(f"⚠️ CPU Fish Speech cleanup warning: {cleanup_error}")

        # Always mark worker as inactive
        mark_service_worker_inactive("cpu_fish_speech", worker_id)
        logger.info(f"🐟 CPU Fish Speech Worker {worker_id} released")
