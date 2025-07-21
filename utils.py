"""
Utility functions for the application
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def cleanup_temp_files(audio_id: str, audio_temp_path: Optional[str] = None, 
                      instruments_temp_path: Optional[str] = None, 
                      video_temp_path: Optional[str] = None):
    """Clean up temporary files and log cache statistics"""
    try:
        from config import settings
        
        # Get audio processor for cache cleanup
        from video_processor import get_audio_processor
        audio_processor = get_audio_processor()
        
        # Log cache statistics before cleanup
        cache_stats = audio_processor.get_cache_stats()
        logger.info(f"Cache stats before cleanup for {audio_id}: {cache_stats}")
        
        # Clean up processor files and caches
        audio_processor.cleanup_temp_files(audio_id)
        
        # Clean up specific temp files
        temp_files_to_clean = []
        if audio_temp_path:
            temp_files_to_clean.append(audio_temp_path)
        if instruments_temp_path:
            temp_files_to_clean.append(instruments_temp_path)
        if video_temp_path:
            temp_files_to_clean.append(video_temp_path)
        
        for temp_file in temp_files_to_clean:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
        
        # Clean up any remaining temp files for this audio_id
        temp_dir = Path(settings.TEMP_DIR)
        for temp_file in temp_dir.glob(f"*{audio_id}*"):
            if temp_file.is_file():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
                    
        logger.info(f"Cleanup completed for {audio_id}")
        
    except Exception as e:
        logger.error(f"Error during cleanup for {audio_id}: {str(e)}") 