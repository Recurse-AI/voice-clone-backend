import logging
import os
import uuid
import time
import soundfile as sf
from typing import Dict, Any

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
    
    def generate_voice_clone(self, text: str, reference_audio_bytes: bytes, reference_text: str, job_id: str = None) -> Dict[str, Any]:
        if not FISH_SDK_AVAILABLE:
            return {"success": False, "error": "Fish Audio SDK not installed. Run: pip install fish-audio-sdk"}
        
        if not self.session:
            return {"success": False, "error": "Fish Audio API key not configured"}
        
        try:
            logger.info(f"Fish API request - text: {len(text)} chars, audio: {len(reference_audio_bytes)} bytes")
            
            # Create output directory
            output_dir = os.path.join(settings.TEMP_DIR, job_id or "temp")
            os.makedirs(output_dir, exist_ok=True)
            
            request_id = f"fish_api_{job_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            output_path = os.path.join(output_dir, f"output_{request_id}.wav")
            
            # Try using raw HTTP request with model header (as per Fish Audio docs)
            request_data = {
                "text": text,
                "references": [{
                    "audio": reference_audio_bytes,
                    "text": reference_text
                }],
                "format": "wav",
                "normalize": True,
                "latency": "normal",
                "chunk_length": settings.FISH_SPEECH_CHUNK_SIZE
            }
            
            # Use direct HTTP request with proper model header
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    "https://api.fish.audio/v1/tts",
                    content=ormsgpack.packb(request_data),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/msgpack",
                        "model": "s1"
                    }
                )
                
                if response.status_code == 402:
                    logger.error("Fish Audio API: 402 Payment Required")
                    return {"success": False, "error": "Payment Required: Check Fish Audio account credits"}
                elif response.status_code != 200:
                    logger.error(f"Fish API error: {response.status_code}")
                    return {"success": False, "error": f"API error: {response.status_code}"}
                
                # Write response content to file
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"Fish API response: {len(response.content)} bytes")
            
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

                # Apply audio processing once
                from app.services.dub.audio_utils import AudioUtils
                data = AudioUtils.trim_silence(data, top_db=40.0).astype('float32')
                data = AudioUtils.fade_in_out(data, fade_duration=0.005, sample_rate=sample_rate)
                sf.write(output_path, data, sample_rate)
                
                logger.info(f"✅ Fish API success: {len(data)} samples at {sample_rate}Hz")
                return {"success": True, "output_path": output_path}
                
            except Exception as e:
                logger.error(f"Audio validation failed: {e}")
                return {"success": False, "error": f"Invalid audio: {e}"}
                
        except Exception as e:
            logger.error(f"Fish Audio API failed: {e}")
            return {"success": False, "error": str(e)}
