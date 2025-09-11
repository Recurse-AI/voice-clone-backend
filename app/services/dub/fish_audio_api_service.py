import logging
import os
import uuid
import time
import io
import numpy as np
from typing import Dict, Any, Literal
import httpx
import ormsgpack
import soundfile as sf
from pydantic import BaseModel, conint
from app.config.settings import settings

logger = logging.getLogger(__name__)

class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: conint(ge=100, le=300, strict=True) = 200
    format: Literal["wav", "pcm", "mp3"] = "wav"
    mp3_bitrate: Literal[64, 128, 192] = 128
    references: list[ServeReferenceAudio] = []
    reference_id: str | None = None
    normalize: bool = True
    latency: Literal["normal", "balanced"] = "normal"

class FishAudioAPIService:
    def __init__(self):
        self.api_key = settings.FISH_AUDIO_API_KEY
        self.base_url = "https://api.fish.audio/v1/tts"
    
    def generate_voice_clone(self, text: str, reference_audio_bytes: bytes, reference_text: str, job_id: str = None) -> Dict[str, Any]:
        try:
            # Use original audio bytes directly (like Fish Audio docs example)
            processed_audio_bytes = reference_audio_bytes
            
            # Use official Fish Audio API format
            request = ServeTTSRequest(
                text=text,
                references=[ServeReferenceAudio(audio=processed_audio_bytes, text=reference_text)],
                format="wav",
                normalize=True,
                latency="normal",
                chunk_length=200
            )
            
            logger.info(f"Fish API request - text: {len(text)} chars, audio: {len(processed_audio_bytes)} bytes")
            
            output_dir = os.path.join(settings.TEMP_DIR, job_id or "temp")
            os.makedirs(output_dir, exist_ok=True)
            
            request_id = f"fish_api_{job_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            output_path = os.path.join(output_dir, f"output_{request_id}.wav")
            
            # Use streaming approach as per official documentation
            with httpx.Client(timeout=300.0) as client:
                with client.stream(
                    "POST",
                    self.base_url,
                    content=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
                    headers={
                        "authorization": f"Bearer {self.api_key}",
                        "content-type": "application/msgpack",
                        "model": "s1"  # Use s1 model as requested
                    }
                ) as response:
                    if response.status_code != 200:
                        logger.error(f"Fish API error: {response.status_code} - {response.text}")
                        return {"success": False, "error": f"API error: {response.status_code}"}
                    
                    # Stream chunks to file (official approach)
                    with open(output_path, "wb") as f:
                        total_bytes = 0
                        for chunk in response.iter_bytes():
                            f.write(chunk)
                            total_bytes += len(chunk)
                    
                    logger.info(f"Fish API streamed {total_bytes} bytes to {output_path}")
            
            # Validate the generated audio
            try:
                data, sample_rate = sf.read(output_path)
                if len(data) == 0:
                    logger.warning(f"Empty audio file preserved at: {output_path}")
                    return {"success": False, "error": "Fish API returned empty audio", "debug_path": output_path}
                
                if np.all(data == 0):
                    logger.warning(f"Silent audio file preserved at: {output_path}")
                    return {"success": False, "error": "Audio file is silent", "debug_path": output_path}
                
                logger.info(f"✅ Fish API success: {len(data)} samples at {sample_rate}Hz")
                return {"success": True, "output_path": output_path}
                
            except Exception as e:
                logger.error(f"Invalid audio file: {e}")
                return {"success": False, "error": f"Invalid audio: {e}", "debug_path": output_path}
                
        except Exception as e:
            logger.error(f"Fish Audio API failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _preprocess_reference_audio(self, reference_audio_bytes: bytes) -> bytes:
        """Simple preprocessing to ensure compatibility"""
        try:
            # Keep it simple - just ensure mono and reasonable length
            buffer = io.BytesIO(reference_audio_bytes)
            data, sample_rate = sf.read(buffer)
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data[:, 0]
            
            # Limit to 10 seconds max
            max_samples = sample_rate * 10
            if len(data) > max_samples:
                data = data[:max_samples]
            
            # Return as-is (let Fish API handle the rest)
            output_buffer = io.BytesIO()
            sf.write(output_buffer, data, sample_rate, format='WAV')
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed, using original: {e}")
            return reference_audio_bytes
