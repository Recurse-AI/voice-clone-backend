import logging
import os
from typing import Dict, Any, List, Optional
from io import BytesIO

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
    
    def generate_speech(self, text: str, voice_id: str, target_language: str = None, job_id: str = None, segment_index: int = 0) -> Dict[str, Any]:
        try:
            from app.config.settings import settings
            import tempfile
            
            if not voice_id:
                return {"success": False, "error": "voice_id is required"}
            
            logger.info(f"Generating speech with ElevenLabs (voice: {voice_id[:8]}...)")
            
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_v3"
            )
            
            audio_bytes = b"".join(chunk for chunk in audio)
            
            temp_dir = os.path.join(settings.TEMP_DIR, job_id or "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Use same naming pattern as Fish API and local model: cloned_{job_id}_{segment_index:03d}.wav
            output_path = os.path.join(temp_dir, f"cloned_elevenlabs_{job_id}_{segment_index:03d}.mp3")
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            
            logger.info(f"Speech generated successfully: {output_path}")
            
            return {
                "success": True,
                "output_path": output_path
            }
        except Exception as e:
            logger.error(f"ElevenLabs generation failed: {e}")
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
        if hasattr(voice, 'samples') and voice.samples is not None and len(voice.samples) > 0:
            for sample in voice.samples:
                sample_url = f"https://api.elevenlabs.io/v1/voices/{voice.voice_id}/samples/{sample.sample_id}/audio"
                samples.append({
                    "audio": sample_url,
                    "title": sample.file_name if hasattr(sample, 'file_name') else "Voice Sample",
                    "text": f"Sample audio ({sample.duration_secs:.1f}s)" if hasattr(sample, 'duration_secs') else "Sample audio"
                })
        elif hasattr(voice, 'preview_url') and voice.preview_url:
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

