import logging
import os
import uuid
import time
from typing import Dict, Any
import httpx
import ormsgpack
import soundfile as sf
from app.config.settings import settings

logger = logging.getLogger(__name__)

class FishAudioAPIService:
    def __init__(self):
        self.api_key = settings.FISH_AUDIO_API_KEY
        self.base_url = "https://api.fish.audio/v1/tts"
    
    def generate_voice_clone(self, text: str, reference_audio_bytes: bytes, reference_text: str, job_id: str = None) -> Dict[str, Any]:
        try:
            request_data = {
                "text": text,
                "references": [{
                    "audio": reference_audio_bytes,
                    "text": reference_text
                }],
                "format": "wav",
                "normalize": True,
                "latency": "normal"
            }
            
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    self.base_url,
                    content=ormsgpack.packb(request_data),
                    headers={
                        "authorization": f"Bearer {self.api_key}",
                        "content-type": "application/msgpack",
                        "model": "s1"
                    }
                )
                
                if response.status_code == 200:
                    if len(response.content) < 1000:  # Basic check for minimal audio content
                        return {"success": False, "error": f"Invalid audio response: {len(response.content)} bytes"}
                    
                    output_dir = os.path.join(settings.TEMP_DIR, job_id or "temp")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    request_id = f"fish_api_{job_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                    output_path = os.path.join(output_dir, f"output_{request_id}.wav")
                    
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    
                    try:
                        data, sample_rate = sf.read(output_path)
                        if len(data) == 0:
                            os.remove(output_path)
                            return {"success": False, "error": "Empty audio data"}
                        return {"success": True, "output_path": output_path}
                    except Exception as e:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        return {"success": False, "error": f"Invalid audio: {e}"}
                else:
                    return {"success": False, "error": f"API error: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Fish Audio API failed: {e}")
            return {"success": False, "error": str(e)}
