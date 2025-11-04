import logging
import os
import uuid
import time
import random
import soundfile as sf
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from fish_audio_sdk import Session, TTSRequest, ReferenceAudio
    import httpx
    import ormsgpack
    FISH_SDK_AVAILABLE = True
except ImportError:
    FISH_SDK_AVAILABLE = False

from app.config.settings import settings

logger = logging.getLogger(__name__)


class FishAudioAPIService:
    def __init__(self):
        self.api_key = settings.FISH_AUDIO_API_KEY
        if FISH_SDK_AVAILABLE and self.api_key:
            self.session = Session(self.api_key)
        else:
            self.session = None
    
    def create_voice_reference(self, audio_bytes: bytes, name: str) -> Dict[str, Any]:
        if not self.api_key:
            return {"success": False, "error": "Fish Audio API key not configured"}
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            multipart_data = [
                ("visibility", (None, "private")),
                ("type", (None, "tts")),
                ("title", (None, name)),
                ("train_mode", (None, "fast")),
                ("enhance_audio_quality", (None, "false")),
                ("voices", ("audio.wav", audio_bytes, "audio/wav"))
            ]
            
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    "https://api.fish.audio/model",
                    headers=headers,
                    files=multipart_data
                )
                response.raise_for_status()
                result = response.json()
                
                return {
                    "success": True,
                    "reference_id": result.get("_id"),
                    "voice_id": result.get("_id"),
                    "name": name
                }
        except Exception as e:
            logger.error(f"Failed to create Fish Audio voice: {e}")
            return {"success": False, "error": str(e)}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, Exception)),
        reraise=True
    )
    def _fish_api_request_with_retry(self, text: str, reference_id: str) -> bytes:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(
                "https://api.fish.audio/v1/tts",
                content=ormsgpack.packb({
                    "text": text,
                    "reference_id": reference_id,
                    "format": "wav",
                    "normalize": True,
                    "latency": "normal",
                    "chunk_length": settings.FISH_SPEECH_CHUNK_SIZE
                }),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/msgpack",
                    "model": "s1"
                }
            )
            
            if response.status_code == 402:
                raise Exception("Payment Required: Check Fish Audio account credits")
            elif response.status_code != 200:
                raise Exception(f"API error: {response.status_code}")
            
            return response.content
    
    def generate_voice_clone(self, text: str, reference_id: str, job_id: str = None, target_language_code: str = None, **kwargs) -> Dict[str, Any]:
        if not FISH_SDK_AVAILABLE:
            return {"success": False, "error": "Fish Audio SDK not installed"}
        
        if not self.session:
            return {"success": False, "error": "Fish Audio API key not configured"}
        
        if not reference_id:
            return {"success": False, "error": "reference_id is required"}
        
        try:
            logger.info(f"Fish API: {len(text)} chars, ref_id: {reference_id[:8]}...")
            
            output_dir = os.path.join(settings.TEMP_DIR, job_id or "temp")
            os.makedirs(output_dir, exist_ok=True)
            
            request_id = f"fish_api_{job_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            output_path = os.path.join(output_dir, f"output_{request_id}.wav")
            
            time.sleep(random.uniform(0.3, 0.7))
            
            audio_content = self._fish_api_request_with_retry(text, reference_id)
            
            with open(output_path, "wb") as f:
                f.write(audio_content)
            
            logger.info(f"Fish API: {len(audio_content)} bytes")
            
            # Validate and repair audio if needed
            try:
                data, sample_rate = sf.read(output_path)
                
                if len(data) == 0:
                    # Repair corrupted WAV header in-place
                    with open(output_path, "rb") as f:
                        raw_bytes = f.read()
                    
                    data_pos = raw_bytes.find(b'data')
                    if data_pos > 0:
                        actual_data_size = len(raw_bytes) - data_pos - 8
                        fixed_bytes = bytearray(raw_bytes)
                        fixed_bytes[4:8] = (len(raw_bytes) - 8).to_bytes(4, 'little')
                        fixed_bytes[data_pos+4:data_pos+8] = actual_data_size.to_bytes(4, 'little')
                        
                        with open(output_path, 'wb') as f:
                            f.write(fixed_bytes)
                        
                        data, sample_rate = sf.read(output_path)
                
                if len(data) == 0:
                    return {"success": False, "error": "Fish API returned empty audio"}

                sf.write(output_path, data, sample_rate, subtype='PCM_16')
                
                logger.info(f"✅ Fish API success: {len(data)} samples at {sample_rate}Hz")
                return {"success": True, "output_path": output_path}
                
            except Exception as e:
                logger.error(f"Audio validation failed: {e}")
                return {"success": False, "error": f"Invalid audio: {e}"}
                
        except Exception as e:
            logger.error(f"❌ Fish API failed after retries: {e}")
            return {"success": False, "error": str(e)}
    
    def cleanup_old_voices(self, keep_count: int = 0) -> Dict[str, Any]:
        """
        Clean up old Fish Audio voice models
        
        Previous Issue: Single httpx.Client with 30s timeout was shared for GET + all DELETE operations.
        When deleting multiple models (e.g., 20-30), the accumulated time exceeded 30s causing 
        "read operation timed out" errors.
        
        Solution: Separate the GET operation from DELETE operations. Each DELETE gets its own client 
        with fresh 30s timeout, preventing timeout accumulation. Individual error handling ensures 
        one failure doesn't stop the entire cleanup process.
        """
        
        try:
            if not self.api_key:
                return {"success": False, "error": "API key not configured"}
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Fetch models list with isolated timeout
            with httpx.Client(timeout=30.0) as client:
                response = client.get("https://api.fish.audio/model", headers=headers)
                response.raise_for_status()
                
                custom_models = [m for m in response.json().get("items", []) if m.get("visibility") == "private"]
                
                if not custom_models:
                    return {"success": True, "deleted": 0}
                
                to_delete = custom_models[:-keep_count] if keep_count > 0 else custom_models
            
            # Delete each model with fresh timeout per operation (prevents timeout accumulation)
            deleted = 0
            for model in to_delete:
                try:
                    with httpx.Client(timeout=30.0) as delete_client:
                        delete_client.delete(f"https://api.fish.audio/model/{model.get('_id')}", headers=headers).raise_for_status()
                        deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete model {model.get('_id')}: {e}")
                    continue
            
            logger.info(f"🧹 Cleaned {deleted}/{len(to_delete)} Fish Audio models")
            return {"success": True, "deleted": deleted}
        except Exception as e:
            logger.error(f"Fish Audio cleanup failed: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_voice(self, voice_id: str) -> bool:
        """Delete a specific voice model by ID"""
        try:
            if not self.api_key:
                return False
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            with httpx.Client(timeout=30.0) as client:
                response = client.delete(f"https://api.fish.audio/model/{voice_id}", headers=headers)
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Failed to delete voice {voice_id}: {e}")
            return False
    


_fish_api_service_instance = None

def get_fish_audio_api_service() -> FishAudioAPIService:
    global _fish_api_service_instance
    if _fish_api_service_instance is None:
        _fish_api_service_instance = FishAudioAPIService()
    return _fish_api_service_instance
