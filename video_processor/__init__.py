"""
Video Processor Module

Provides shared instances for video processing components.
AudioProcessor now handles only core functionality - voice cloning handled separately.
"""

import threading
import logging
from config import settings
from .base_processor import AudioProcessor

logger = logging.getLogger(__name__)

# Global shared audio processor instance (no voice cloning)
_audio_processor = None
_loading_lock = threading.Lock()

def get_audio_processor():
    """
    Get the shared audio processor instance for core functionality
    (Voice cloning handled separately by CleanAudioProcessor)
    
    Returns:
        AudioProcessor: Shared instance for video/audio processing
    """
    global _audio_processor
    
    with _loading_lock:
        if _audio_processor is None:
            logger.info("Creating shared AudioProcessor instance (no voice cloning)...")
            _audio_processor = AudioProcessor(settings.TEMP_DIR)
            logger.info("✅ AudioProcessor instance created for core functionality")
    
    return _audio_processor

def cleanup_audio_processor():
    """Clean up the global audio processor"""
    global _audio_processor
    with _loading_lock:
        if _audio_processor:
            _audio_processor = None
            logger.info("🧹 Global AudioProcessor cleaned up") 