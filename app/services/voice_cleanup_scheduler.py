import asyncio
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

class VoiceCleanupScheduler:
    def __init__(self):
        self.running = False
        self.cleanup_interval = 3600
    
    async def start(self):
        if self.running:
            return
        
        self.running = True
        logger.info("Voice cleanup scheduler started (runs every 1 hour)")
        
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_old_voices()
            except Exception as e:
                logger.error(f"Voice cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_voices(self):
        try:
            from app.services.dub.elevenlabs_service import get_elevenlabs_service
            from app.services.dub.fish_audio_api_service import get_fish_audio_api_service
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
            logger.info(f"Cleaning voices older than {cutoff_time}")
            
            try:
                elevenlabs = get_elevenlabs_service()
                result = elevenlabs.cleanup_old_voices(keep_count=0)
                logger.info(f"ElevenLabs cleanup: {result.get('deleted', 0)} deleted")
            except Exception as e:
                logger.error(f"ElevenLabs cleanup failed: {e}")
            
            try:
                fish_api = get_fish_audio_api_service()
                result = fish_api.cleanup_old_voices(keep_count=0)
                logger.info(f"Fish Audio cleanup: {result.get('deleted', 0)} deleted")
            except Exception as e:
                logger.error(f"Fish Audio cleanup failed: {e}")
            
        except Exception as e:
            logger.error(f"Voice cleanup failed: {e}")
    
    def stop(self):
        self.running = False

voice_cleanup_scheduler = VoiceCleanupScheduler()

