import logging
from typing import Dict, Any
import httpx
import ormsgpack
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
                    import os
                    import uuid
                    import time
                    from app.config.settings import settings
                    
                    output_dir = os.path.join(settings.TEMP_DIR, job_id or "temp")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    request_id = f"fish_api_{job_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                    output_path = os.path.join(output_dir, f"output_{request_id}.wav")
                    
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    
                    return {"success": True, "output_path": output_path}
                else:
                    error_msg = f"API returned status {response.status_code}"
                    try:
                        error_details = response.text
                        if error_details:
                            error_msg += f": {error_details}"
                    except:
                        pass
                    logger.error(f"Fish Audio API error: {error_msg}")
                    return {"success": False, "error": error_msg}
                    
        except Exception as e:
            logger.error(f"Fish Audio API request failed: {e}")
            return {"success": False, "error": str(e)}
