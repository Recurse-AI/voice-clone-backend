"""
Fish Speech Voice Cloning Service

Core functionality for voice cloning using Fish Speech OpenAudio models.
Handles model loading, inference, and voice generation.
"""

import os
import sys
import torch
import logging
import numpy as np
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Generator, Union

# Add fish-speech to Python path
# Go up 3 levels: services/dub -> services -> app -> root, then to fish-speech
fish_speech_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'fish-speech')
if fish_speech_path not in sys.path:
    sys.path.insert(0, fish_speech_path)

try:
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
except ImportError as e:
    logging.error(f"Failed to import fish_speech modules: {e}")
    raise ImportError("Fish Speech modules not found. Please ensure fish-speech is properly installed.")

from app.config.settings import settings

logger = logging.getLogger(__name__)


class FishSpeechService:
    """Service for voice cloning using Fish Speech models"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = torch.half if torch.cuda.is_available() else torch.float32
        self.checkpoint_path = "checkpoints/openaudio-s1-mini"
        self.decoder_checkpoint_path = "checkpoints/openaudio-s1-mini/codec.pth"
        
        # Memory optimization flags
        self.use_memory_efficient_attention = True
        self.use_flash_attention = torch.cuda.is_available()
        self.is_initialized = False
        
        # TTSInferenceEngine components
        self.llama_queue = None
        self.decoder_model = None
        self.inference_engine = None
        
        logger.info(f"Fish Speech Service initializing on device: {self.device}")
    
    def load_model(self) -> bool:
        """Load Fish Speech TTSInferenceEngine for proper reference audio support"""
        try:
            if self.is_initialized:
                return True
            
            # Quick validation
            if not Path(self.checkpoint_path).exists() or not Path(self.decoder_checkpoint_path).exists():
                logger.warning("Fish Speech models not found - voice cloning disabled")
                return False
            
            import warnings
            warnings.filterwarnings("ignore")
            
            # Load LLAMA model queue
            compile_model = getattr(settings, 'FISH_SPEECH_COMPILE', False)
            checkpoint_path = Path(self.checkpoint_path)
            decoder_path = Path(self.decoder_checkpoint_path)
            
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device=self.device,
                precision=self.precision,
                compile=compile_model
            )
            
            # Load VQ-GAN decoder model
            self.decoder_model = load_decoder_model(
                config_name="modded_dac_vq",
                checkpoint_path=decoder_path,
                device=self.device,
            )
            
            # Create TTSInferenceEngine
            self.inference_engine = TTSInferenceEngine(
                llama_queue=self.llama_queue,
                decoder_model=self.decoder_model,
                compile=compile_model,
                precision=self.precision,
            )
            
            # Warm up with dry run
            list(self.inference_engine.inference(
                ServeTTSRequest(
                    text="Hello world.",
                    references=[],
                    max_new_tokens=1024,
                    format="wav"
                )
            ))
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Fish Speech TTSInferenceEngine: {e}")
            return False
    

    
    def generate_with_reference_audio(self, text: str, reference_audio_bytes: bytes, 
                                     reference_text: str, job_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Generate voice cloning with reference audio (proper fish-speech way)
        
        Args:
            text: Text to convert to speech
            reference_audio_bytes: Reference audio as bytes
            reference_text: Text that was spoken in reference audio
            **kwargs: Additional TTS parameters
            
        Returns:
            Dict with generation results including audio bytes
        """
        if not self.is_initialized:
            if not self.load_model():
                return {"success": False, "error": "Failed to load TTSInferenceEngine"}
        
        try:
            # Create ServeReferenceAudio object
            reference = ServeReferenceAudio(
                audio=reference_audio_bytes,
                text=reference_text
            )
            
            # Create TTS request with reference
            tts_request = ServeTTSRequest(
                text=text,
                references=[reference],
                max_new_tokens=kwargs.get("max_new_tokens", 2048),
                top_p=kwargs.get("top_p", 0.9),
                repetition_penalty=kwargs.get("repetition_penalty", 1.2),
                temperature=kwargs.get("temperature", 0.8),
                format="wav",
                chunk_length=kwargs.get("chunk_length", 200)
            )
            
            logger.info(f"Generating voice with reference audio for text: {text[:50]}...")
            
            # Generate audio using TTSInferenceEngine
            audio_data = b""
            for result in self.inference_engine.inference(tts_request):
                # 🛡️ Check job cancellation during inference generation
                if job_id:
                    from app.utils.shared_memory import is_job_cancelled
                    if is_job_cancelled(job_id):
                        logger.info(f"🛑 Fish speech inference cancelled for job {job_id}")
                        return {"success": False, "error": "Job cancelled by user"}
                
                if result.code in ("chunk", "final"):
                    # Accumulate audio chunks
                    sample_rate, audio_chunk = result.audio
                    if isinstance(audio_chunk, np.ndarray):
                        # Convert numpy array to bytes
                        import soundfile as sf
                        import io
                        audio_buffer = io.BytesIO()
                        sf.write(audio_buffer, audio_chunk, sample_rate, format='WAV')
                        audio_data += audio_buffer.getvalue()
                elif result.code == "error":
                    logger.error(f"Voice generation error: {result.error}")
                    return {"success": False, "error": str(result.error)}
            
            if audio_data:
                logger.info("Voice generation with reference completed")
                return {
                    "success": True,
                    "audio_data": audio_data,
                    "sample_rate": sample_rate if 'sample_rate' in locals() else 44100
                }
            else:
                return {"success": False, "error": "No audio data generated"}
                
        except Exception as e:
            logger.error(f"Error in voice generation with reference: {e}")
            return {"success": False, "error": str(e)}

    def cleanup(self):
        """Clean up TTSInferenceEngine and free memory"""
        try:
            if self.inference_engine is not None:
                del self.inference_engine
                self.inference_engine = None
            
            if self.decoder_model is not None:
                del self.decoder_model
                self.decoder_model = None
                
            if self.llama_queue is not None:
                del self.llama_queue
                self.llama_queue = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Collect inter-process GPU memory to avoid fragmentation
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            
            self.is_initialized = False
            logger.info("Fish Speech TTSInferenceEngine cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global service instance
fish_speech_service = None
_service_lock = threading.Lock()


def get_fish_speech_service() -> FishSpeechService:
    """Get or create global Fish Speech service instance (thread-safe)"""
    global fish_speech_service
    
    if fish_speech_service is None:
        with _service_lock:
            if fish_speech_service is None:
                fish_speech_service = FishSpeechService()
    
    return fish_speech_service


def initialize_fish_speech() -> bool:
    """Initialize Fish Speech service (called from main.py)"""
    try:
        service = get_fish_speech_service()
        return service.load_model()
    except Exception as e:
        logger.error(f"Failed to initialize Fish Speech: {e}")
        return False


def cleanup_fish_speech():
    """Cleanup Fish Speech service (called from main.py)"""
    global fish_speech_service
    
    if fish_speech_service is not None:
        fish_speech_service.cleanup()
        fish_speech_service = None