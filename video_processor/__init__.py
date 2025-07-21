"""
Video Processor Module

Provides shared instances for video processing components.
Ensures Dia model is loaded only once globally.
"""

import threading
import logging
from config import settings
from .base_processor import AudioProcessor

logger = logging.getLogger(__name__)

# Global shared audio processor instance
_audio_processor = None
_model_loaded = False
_loading_lock = threading.Lock()

def get_audio_processor(load_model: bool = True):
    """
    Get the shared audio processor instance with optional model loading
    
    Args:
        load_model: Whether to load the Dia model (default: True)
    
    Returns:
        AudioProcessor: Shared instance with model loaded
    """
    global _audio_processor, _model_loaded
    
    with _loading_lock:
        if _audio_processor is None:
            logger.info("Creating shared AudioProcessor instance...")
            _audio_processor = AudioProcessor(settings.TEMP_DIR)
            logger.info("AudioProcessor instance created")
        
        if load_model and not _model_loaded:
            logger.info("Loading Dia model globally...")
            try:
                success = _audio_processor.load_dia_model(
                    repo_id=settings.DIA_MODEL_REPO
                )
                if success:
                    _model_loaded = True
                    logger.info("Dia model loaded successfully (global instance)")
                    print("Dia model loaded successfully (global instance)")
                else:
                    logger.error("Failed to load Dia model on global instance")
                    print("ERROR: Failed to load Dia model on global instance")
            except Exception as e:
                logger.error(f"Exception during global model loading: {e}")
                print(f"EXCEPTION during global model loading: {e}")
    
    return _audio_processor

def is_model_loaded():
    """Check if the Dia model is loaded globally"""
    return _model_loaded 