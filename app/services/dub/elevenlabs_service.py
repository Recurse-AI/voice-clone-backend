import logging
import os
from typing import Dict, Any, List, Optional
from io import BytesIO
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class BlockedVoiceError(Exception):
    """Raised when ElevenLabs detects a blocked voice"""
    pass

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
            error_str = str(e)
            if "detected_blocked_voice" in error_str or "violate our Terms of Service" in error_str:
                logger.error(f"Blocked voice detected during creation: {error_str}")
                raise BlockedVoiceError("Voice blocked by ElevenLabs for ToS violation")
            logger.error(f"Failed to create voice clone: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_with_retry(self, text: str, voice_id: str, speed: float = 1.0):
        try:
            from elevenlabs import VoiceSettings
            voice_settings = VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True,
                speed=max(0.7, min(1.2, speed))
            )
            return self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_v3",
                voice_settings=voice_settings
            )
        except Exception as e:
            error_str = str(e)
            if "detected_blocked_voice" in error_str or "violate our Terms of Service" in error_str:
                logger.error(f"Blocked voice detected: {error_str}")
                raise BlockedVoiceError("Voice blocked by ElevenLabs for ToS violation")
            raise
    
    def generate_speech(self, text: str, voice_id: str, target_language: str = None, job_id: str = None, segment_index: int = 0, target_duration_ms: int = None, speed: float = 1.0) -> Dict[str, Any]:
        success = False
        try:
            from app.config.settings import settings
            
            if not voice_id:
                return {"success": False, "error": "voice_id is required"}
            
            temp_dir = os.path.join(settings.TEMP_DIR, job_id or "temp")
            os.makedirs(temp_dir, exist_ok=True)
            output_path = os.path.join(temp_dir, f"cloned_elevenlabs_{job_id}_{segment_index:03d}.mp3")
            
            audio = self._generate_with_retry(text, voice_id, speed)
            audio_bytes = b"".join(chunk for chunk in audio)
            
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            
            success = True
            return {"success": True, "output_path": output_path}
        except BlockedVoiceError:
            raise
        except Exception as e:
            logger.error(f"ElevenLabs failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            if job_id:
                from app.services.analytics_service import AnalyticsService
                AnalyticsService.track_api_call_sync(job_id, "elevenlabs", chars=len(text), success=success)
    
    def get_all_voices(self, include_library: bool = True) -> Dict[str, Any]:
        try:
            normalized_voices = []
            
            user_voices_response = self.client.voices.get_all()
            for voice in user_voices_response.voices:
                normalized_voices.append(self._normalize_voice_to_fish_format(voice))
            
            if include_library:
                try:
                    import requests
                    headers = {"xi-api-key": self.api_key}
                    page_size = 100
                    page = 0
                    
                    while True:
                        response = requests.get(
                            "https://api.elevenlabs.io/v1/shared-voices",
                            headers=headers,
                            params={
                                "page_size": page_size,
                                "page": page
                            },
                            timeout=30
                        )
                        
                        if response.status_code != 200:
                            break
                        
                        data = response.json()
                        voices = data.get("voices", [])
                        
                        if not voices:
                            break
                        
                        for voice_data in voices:
                            normalized_voices.append(self._normalize_shared_voice(voice_data))
                        
                        if len(voices) < page_size:
                            break
                        
                        page += 1
                    
                    logger.info(f"Fetched {len(normalized_voices)} total voices from ElevenLabs")
                except Exception as lib_error:
                    logger.warning(f"Failed to fetch Voice Library: {lib_error}")
            
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
        
        from app.services.language_service import LanguageService
        supported_languages = sorted(list(LanguageService.ELEVENLABS_V3_LANGUAGES))
        
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
    
    def _normalize_shared_voice(self, voice_data: Dict) -> Dict[str, Any]:
        from app.services.language_service import LanguageService
        
        tags = []
        if voice_data.get("category"):
            tags.append(voice_data["category"])
        if voice_data.get("labels"):
            labels = voice_data["labels"]
            if isinstance(labels, dict):
                tags.extend([f"{k}:{v}" for k, v in labels.items() if v])
            elif isinstance(labels, list):
                tags.extend(labels)
        
        samples = []
        if voice_data.get("preview_url"):
            samples.append({
                "audio": voice_data["preview_url"],
                "title": "Voice Preview",
                "text": f"Preview of {voice_data.get('name', 'Voice')}"
            })
        
        return {
            "_id": voice_data.get("public_owner_id"),
            "title": voice_data.get("name", "Unknown Voice"),
            "cover_image": None,
            "languages": sorted(list(LanguageService.ELEVENLABS_V3_LANGUAGES)),
            "tags": tags,
            "samples": samples,
            "description": voice_data.get("description", ""),
            "like_count": voice_data.get("like_count", 0),
            "task_count": voice_data.get("usage_character_count_1y", 0),
            "author": {
                "nickname": voice_data.get("creator_name", "ElevenLabs Community")
            },
            "visibility": "public",
            "train_mode": "instant",
            "model_type": "elevenlabs"
        }


_service_instance = None

def get_elevenlabs_service() -> ElevenLabsService:
    global _service_instance
    if _service_instance is None:
        _service_instance = ElevenLabsService()
    return _service_instance

