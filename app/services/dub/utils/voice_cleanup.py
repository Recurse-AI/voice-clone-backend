import logging

logger = logging.getLogger(__name__)

class VoiceCleanup:
    @staticmethod
    def cleanup_voices(voice_ids: list, model_type: str):
        if not voice_ids:
            return
        
        try:
            if model_type == "best":
                from app.services.dub.elevenlabs_service import get_elevenlabs_service
                service = get_elevenlabs_service()
            elif model_type == "medium":
                from app.services.dub.fish_audio_api_service import get_fish_audio_api_service
                service = get_fish_audio_api_service()
            else:
                return
            
            for voice_id in voice_ids:
                try:
                    service.delete_voice(voice_id)
                except Exception:
                    pass
        except Exception:
            pass

