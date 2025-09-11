import logging
import os
import uuid
import time
import io
import numpy as np
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
            processed_audio_bytes = self._preprocess_reference_audio(reference_audio_bytes)
            
            request_data = {
                "text": text,
                "references": [{"audio": processed_audio_bytes, "text": reference_text}],
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
                    output_dir = os.path.join(settings.TEMP_DIR, job_id or "temp")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    request_id = f"fish_api_{job_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                    output_path = os.path.join(output_dir, f"output_{request_id}.wav")
                    
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    
                    try:
                        data, sample_rate = sf.read(output_path)
                        if len(data) == 0:
                            logger.warning(f"Empty audio file preserved at: {output_path}")
                            return {"success": False, "error": "Fish API service returned empty audio - fallback recommended", "debug_path": output_path}
                        
                        if np.all(data == 0):
                            logger.warning(f"Silent audio file preserved at: {output_path}")
                            return {"success": False, "error": "Audio file is completely silent", "debug_path": output_path}
                        
                        return {"success": True, "output_path": output_path}
                    except Exception as e:
                        logger.error(f"Invalid audio file preserved at: {output_path}")
                        return {"success": False, "error": f"Invalid audio: {e}", "debug_path": output_path}
                else:
                    return {"success": False, "error": f"API error: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Fish Audio API failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _preprocess_reference_audio(self, reference_audio_bytes: bytes) -> bytes:
        try:
            buffer = io.BytesIO(reference_audio_bytes)
            data, sample_rate = sf.read(buffer)
            
            if len(data.shape) > 1:
                data = data[:, 0]
            
            if sample_rate != 44100:
                try:
                    from scipy import signal
                    num_samples = int(len(data) * 44100 / sample_rate)
                    data = signal.resample(data, num_samples)
                    sample_rate = 44100
                except ImportError:
                    pass
            
            max_samples = 44100 * 10
            if len(data) > max_samples:
                data = data[:max_samples]
            
            output_buffer = io.BytesIO()
            sf.write(output_buffer, data, sample_rate, format='WAV')
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            return reference_audio_bytes
