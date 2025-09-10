"""
Fish Speech Service Worker - Dedicated VRAM worker for voice cloning
Handles serial processing of Fish Speech voice cloning requests
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
    store_service_result("fish_speech", request_id, result)
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
                "details": cloning_result.get("details", {})
            }
        else:
            return {"error": f"Output file not created: {output_path}", "success": False}
    else:
        error_msg = cloning_result.get("error", "Voice cloning failed")
        return {"error": error_msg, "success": False}


def process_fish_speech_request(request_data: dict) -> dict:
    """
    Background worker function for Fish Speech voice cloning processing
    This runs in a dedicated worker process with full VRAM access
    """
    request_id = request_data.get("request_id")
    worker_id = f"fish_speech_worker_{uuid.uuid4().hex[:8]}"
    
    logger.info(f"🐟 Fish Speech Worker {worker_id} starting request {request_id}")
    
    # Mark worker as active for serial processing
    from app.utils.pipeline_utils import mark_service_worker_active, mark_service_worker_inactive
    
    if not mark_service_worker_active("fish_speech", worker_id):
        return _handle_error(request_id, "Failed to mark worker as active")
    
    try:
        # Validate request data
        try:
            request_id, text, reference_audio_bytes_b64, output_path = _validate_request_data(request_data)
        except ValueError as e:
            return _handle_error(request_id, str(e))
        
        params = request_data.get("params") or {}
        reference_text = request_data.get("reference_text", "")
        
        # Load Fish Speech service (this loads the model into VRAM)
        from app.services.dub.fish_speech_service import get_fish_speech_service
        fish_speech_service = get_fish_speech_service()
        
        # Ensure model is loaded
        if not fish_speech_service.is_initialized:
            logger.info(f"🚀 Loading Fish Speech model for worker {worker_id}")
            if not fish_speech_service.load_model():
                return _handle_error(request_id, "Failed to load Fish Speech model")
        
        # Decode reference audio bytes
        try:
            reference_audio_bytes = base64.b64decode(reference_audio_bytes_b64)
        except Exception as e:
            return _handle_error(request_id, f"Failed to decode reference audio: {str(e)}")
        
        logger.info(f"🎵 Processing voice cloning: '{text[:50]}...' -> {output_path}")
        start_time = time.time()
        
        # Perform actual voice cloning using existing logic
        cloning_result = fish_speech_service._generate_direct(
            text,
            reference_audio_bytes,
            reference_text,
            request_id,
            **{k: v for k, v in params.items() if v is not None}
        )
        
        # Save audio data to file if cloning was successful
        if cloning_result.get("success") and cloning_result.get("audio_data"):
            try:
                with open(output_path, "wb") as f:
                    f.write(cloning_result["audio_data"])
                logger.info(f"✅ Saved voice cloning output to: {output_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save output file {output_path}: {e}")
                cloning_result = {"success": False, "error": f"Failed to save output file: {str(e)}"}
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Fish Speech voice cloning completed in {processing_time:.2f}s")
        
        # Process result using helper function
        result = _process_cloning_result(cloning_result, output_path, processing_time, worker_id)
        
        # Store result in Redis
        from app.utils.pipeline_utils import store_service_result
        store_service_result("fish_speech", request_id, result)
        
        logger.info(f"🐟 Fish Speech Worker {worker_id} completed request {request_id}")
        return result
        
    except Exception as e:
        return _handle_error(request_id, f"Fish Speech processing failed: {str(e)}")
        
    finally:
        # Always mark worker as inactive
        mark_service_worker_inactive("fish_speech", worker_id)
        logger.info(f"🐟 Fish Speech Worker {worker_id} released")


def cleanup_fish_speech_worker():
    """Cleanup function for Fish Speech worker"""
    try:
        # Clear VRAM
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 Fish Speech VRAM cache cleared")
    except Exception as e:
        logger.warning(f"⚠️ Fish Speech cleanup warning: {e}")


# Worker health check
def fish_speech_worker_health_check() -> dict:
    """Health check for Fish Speech worker"""
    try:
        import torch
        
        health = {
            "worker_type": "fish_speech_service",
            "cuda_available": torch.cuda.is_available(),
            "cuda_memory_allocated": 0,
            "cuda_memory_reserved": 0
        }
        
        if torch.cuda.is_available():
            health["cuda_memory_allocated"] = torch.cuda.memory_allocated() // 1024 // 1024  # MB
            health["cuda_memory_reserved"] = torch.cuda.memory_reserved() // 1024 // 1024  # MB
        
        return health
        
    except Exception as e:
        return {"error": str(e), "worker_type": "fish_speech_service"}


