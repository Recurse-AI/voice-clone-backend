import logging
import os
from typing import Dict, Any, List, Optional
from io import BytesIO
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class ElevenLabsService:
    def __init__(self, api_key: str = None):
        from app.config.settings import settings
        self.api_key = api_key or settings.ELEVENLABS_API_KEY
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not configured")
        
        try:
            from elevenlabs.client import ElevenLabs
            self.client = ElevenLabs(api_key=self.api_key)
        except ImportError:
            raise ImportError("elevenlabs package not installed. Run: pip install elevenlabs")
    
    def create_voice_reference(self, audio_bytes: bytes, name: str) -> Dict[str, Any]:
        try:
            logger.info(f"Creating ElevenLabs voice clone: {name}")
            
            voice = self.client.voices.ivc.create(
                name=name,
                files=[BytesIO(audio_bytes)]
            )
            
            voice_id = voice.voice_id
            logger.info(f"Voice clone created successfully: {voice_id}")
            
            return {
                "success": True,
                "voice_id": voice_id,
                "reference_id": voice_id,
                "name": name
            }
        except Exception as e:
            logger.error(f"Failed to create voice clone: {e}")
            return {"success": False, "error": str(e)}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, Exception)),
        reraise=True
    )
    def _generate_with_retry(self, text: str, voice_id: str, speed: float = 1.0):
        voice_settings = {"speed": speed} if speed != 1.0 else None
        
        return self.client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_v3",
            voice_settings=voice_settings
        )
    
    def generate_speech(self, text: str, voice_id: str, target_language: str = None, job_id: str = None, segment_index: int = 0, target_duration_ms: int = None) -> Dict[str, Any]:
        try:
            from app.config.settings import settings
            import soundfile as sf
            
            if not voice_id:
                return {"success": False, "error": "voice_id is required"}
            
            temp_dir = os.path.join(settings.TEMP_DIR, job_id or "temp")
            os.makedirs(temp_dir, exist_ok=True)
            output_path = os.path.join(temp_dir, f"cloned_elevenlabs_{job_id}_{segment_index:03d}.mp3")
            
            if target_duration_ms:
                words = len(text.split())
                chars = len(text)
                
                estimated_duration_ms = max(words * 400, chars * 80)
                
                if estimated_duration_ms > target_duration_ms:
                    speed = min(1.2, max(0.8, estimated_duration_ms / target_duration_ms))
                else:
                    speed = 1.0
            else:
                speed = 1.0
            
            audio = self._generate_with_retry(text, voice_id, speed)
            audio_bytes = b"".join(chunk for chunk in audio)
            
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            
            if target_duration_ms:
                try:
                    audio_data, sample_rate = sf.read(output_path)
                    actual_duration_ms = int(len(audio_data) / sample_rate * 1000)
                    
                    if actual_duration_ms > target_duration_ms * 1.1:
                        speed_ratio = actual_duration_ms / target_duration_ms
                        new_speed = min(1.2, max(0.8, speed * speed_ratio))
                        
                        if abs(new_speed - speed) >= 0.1:
                            audio = self._generate_with_retry(text, voice_id, new_speed)
                            audio_bytes = b"".join(chunk for chunk in audio)
                            
                            with open(output_path, "wb") as f:
                                f.write(audio_bytes)
                except Exception:
                    pass
            
            return {"success": True, "output_path": output_path}
        except Exception as e:
            logger.error(f"ElevenLabs failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_all_voices(self) -> Dict[str, Any]:
        try:
            voices_response = self.client.voices.get_all()
            
            normalized_voices = []
            for voice in voices_response.voices:
                normalized_voices.append(self._normalize_voice_to_fish_format(voice))
            
            return {
                "success": True,
                "items": normalized_voices,
                "total": len(normalized_voices)
            }
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return {"success": False, "error": str(e)}
    
    def get_voice_details(self, voice_id: str) -> Dict[str, Any]:
        try:
            all_voices = self.client.voices.get_all()
            voice = next((v for v in all_voices.voices if v.voice_id == voice_id), None)
            
            if not voice:
                return {"success": False, "error": "Voice not found"}
            
            return {
                "success": True,
                "data": self._normalize_voice_to_fish_format(voice)
            }
        except Exception as e:
            logger.error(f"Failed to get voice details: {e}")
            return {"success": False, "error": str(e)}
    
    def cleanup_old_voices(self, keep_count: int = 0) -> Dict[str, Any]:
        try:
            voices = self.client.voices.get_all().voices
            custom_voices = [v for v in voices if hasattr(v, 'category') and v.category == 'cloned']
            
            if not custom_voices:
                return {"success": True, "deleted": 0}
            
            to_delete = custom_voices[:-keep_count] if keep_count > 0 else custom_voices
            deleted = sum(1 for v in to_delete if self._delete_voice(v.voice_id))
            
            logger.info(f"🧹 Cleaned {deleted}/{len(to_delete)} ElevenLabs voices")
            return {"success": True, "deleted": deleted}
        except Exception as e:
            logger.error(f"ElevenLabs cleanup failed: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_voice(self, voice_id: str) -> bool:
        """Delete a specific voice by ID"""
        try:
            self.client.voices.delete(voice_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete voice {voice_id}: {e}")
            return False
    
    def _delete_voice(self, voice_id: str) -> bool:
        """Internal delete method for cleanup_old_voices"""
        return self.delete_voice(voice_id)
    
    def _normalize_voice_to_fish_format(self, voice) -> Dict[str, Any]:
        tags = []
        if hasattr(voice, 'labels') and voice.labels:
            if isinstance(voice.labels, dict):
                tags = [f"{k}:{v}" for k, v in voice.labels.items() if v]
            else:
                tags = list(voice.labels)
        
        if hasattr(voice, 'category') and voice.category:
            tags.append(voice.category)
        
        samples = []
       
        if hasattr(voice, 'preview_url') and voice.preview_url:
            samples.append({
                "audio": voice.preview_url,
                "title": "Voice Preview",
                "text": f"Preview of {voice.name}"
            })
        
        supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 
                              'ar', 'hi', 'tr', 'pl', 'nl', 'sv', 'uk', 'vi']
        
        return {
            "_id": voice.voice_id,
            "title": voice.name,
            "cover_image": None,
            "languages": supported_languages,
            "tags": tags,
            "samples": samples,
            "description": voice.description if hasattr(voice, 'description') and voice.description else "",
            "like_count": 0,
            "task_count": 0,
            "author": {
                "nickname": "ElevenLabs"
            },
            "visibility": "private" if hasattr(voice, 'category') and voice.category == "cloned" else "public",
            "train_mode": "instant",
            "model_type": "elevenlabs"
        }


_service_instance = None

def get_elevenlabs_service() -> ElevenLabsService:
    global _service_instance
    if _service_instance is None:
        _service_instance = ElevenLabsService()
    return _service_instance

