"""
Video Processor Module

Provides shared instances for video processing components.
"""

import logging
import threading

from config import settings
from .base_processor import AudioProcessor

logger = logging.getLogger(__name__)

# Global shared audio processor instance
_audio_processor = None
_loading_lock = threading.Lock()

def get_audio_processor(load_model: bool = True):
    """
    Get the shared audio processor instance
    
    Args:
        load_model: Ignored parameter (for backward compatibility)
    
    Returns:
        AudioProcessor: Shared instance
    """
    global _audio_processor
    
    with _loading_lock:
        if _audio_processor is None:
            logger.info("Creating shared AudioProcessor instance...")
            _audio_processor = AudioProcessor(settings.TEMP_DIR)
            logger.info("AudioProcessor instance created")
    
    return _audio_processor

def is_model_loaded():
    """Check if model is loaded - always returns False (DIA model disabled)"""
    return False 