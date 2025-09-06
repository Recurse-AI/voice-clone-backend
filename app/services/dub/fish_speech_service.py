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
        from app.config.settings import settings
        
        # Device configuration from settings
        if settings.FISH_SPEECH_DEVICE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = settings.FISH_SPEECH_DEVICE
        
        # Precision configuration from settings
        if settings.FISH_SPEECH_PRECISION == "auto":
            self.precision = torch.half if self.device == "cuda" else torch.float32
        elif settings.FISH_SPEECH_PRECISION == "float16":
            self.precision = torch.half
        else:
            self.precision = torch.float32
        
        # Model paths from settings
        self.checkpoint_path = settings.FISH_SPEECH_CHECKPOINT
        self.decoder_checkpoint_path = settings.FISH_SPEECH_DECODER
        
        # Model configuration from settings
        self.use_memory_efficient_attention = not settings.FISH_SPEECH_LOW_MEMORY
        self.use_flash_attention = self.device == "cuda" and not settings.FISH_SPEECH_LOW_MEMORY
        self.max_batch_size = settings.FISH_SPEECH_MAX_BATCH_SIZE
        self.is_initialized = False
        self._compile_enabled = settings.FISH_SPEECH_COMPILE
        
        # GPU optimization for faster compilation
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # TTSInferenceEngine components
        self.llama_queue = None
        self.decoder_model = None
        self.inference_engine = None
        
        logger.info(f"Fish Speech Service initializing on device: {self.device} (compile: {self._compile_enabled})")
    
    def _check_model_health(self) -> bool:
        """Check if models are still loaded and healthy"""
        try:
            if not self.is_initialized:
                return False
            
            # Check if model components exist
            if self.inference_engine is None or self.llama_queue is None or self.decoder_model is None:
                logger.warning("Model components missing - will reinitialize...")
                self.is_initialized = False
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Model health check failed: {e}")
            self.is_initialized = False
            return False
    
    def load_model(self) -> bool:
        """Load Fish Speech TTSInferenceEngine for proper reference audio support"""
        try:
            # Check model health first
            if self._check_model_health():
                return True
            
            # Quick validation
            if not Path(self.checkpoint_path).exists() or not Path(self.decoder_checkpoint_path).exists():
                logger.warning("Fish Speech models not found - voice cloning disabled")
                return False
            
            import warnings
            warnings.filterwarnings("ignore")
            
            # Use compile setting from constructor (based on settings)
            compile_model = self._compile_enabled
            checkpoint_path = Path(self.checkpoint_path)
            decoder_path = Path(self.decoder_checkpoint_path)
            
            if compile_model:
                logger.info("🚀 Loading model with compilation for optimized performance")
            else:
                logger.info("🚀 Loading model without compilation for faster startup")
            
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
            
            # Create TTSInferenceEngine without compilation
            self.inference_engine = TTSInferenceEngine(
                llama_queue=self.llama_queue,
                decoder_model=self.decoder_model,
                compile=compile_model,
                precision=self.precision,
            )
            
            # No warmup needed - ready to use immediately
            logger.info("✅ Model loaded without compilation - ready for immediate use")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Fish Speech TTSInferenceEngine: {e}")
            return False
    

    
    def generate_with_reference_audio(self, text: str, reference_audio_bytes: bytes, 
                                     reference_text: str, job_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Generate voice cloning with reference audio (optimized for speed)
        """
        import time
        start_time = time.time()
        
        # Health check and auto-reload if needed
        if not self._check_model_health():
            logger.info("Model unhealthy, reloading...")
            if not self.load_model():
                return {"success": False, "error": "Failed to reload model"}
        
        # Smart GPU cleanup - only if memory usage is high
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            if current_memory > 2.0:  # Only cleanup if using > 2GB
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        if not self.is_initialized:
            logger.info("🔄 Fish Speech model not loaded, loading now...")
            load_start = time.time()
            if not self.load_model():
                return {"success": False, "error": "Failed to load TTSInferenceEngine"}
            load_time = time.time() - load_start
            logger.info(f"⚡ Fish Speech model loaded in {load_time:.2f}s")
        
        try:
            # Create optimized TTS request with timeout handling
            import concurrent.futures
            import threading
            
            def _generate_audio():
                reference = ServeReferenceAudio(
                    audio=reference_audio_bytes,
                    text=reference_text
                )
                
                # User preferred parameters
                tts_request = ServeTTSRequest(
                    text=text,
                    references=[reference],
                    max_new_tokens=kwargs.get("max_new_tokens", 1024),  # User preference
                    top_p=kwargs.get("top_p", 0.6),                     # Optimized for speed
                    repetition_penalty=kwargs.get("repetition_penalty", 1.05),  # Minimal penalty
                    temperature=kwargs.get("temperature", 0.6),         # Lower for speed
                    format="wav",
                    chunk_length=kwargs.get("chunk_length", 200)        # User preference
                )
                
                # Generate with timeout protection
                audio_data = b""
                sample_rate = 44100
                
                for result in self.inference_engine.inference(tts_request):
                    if result.code in ("chunk", "final"):
                        sample_rate, audio_chunk = result.audio
                        if isinstance(audio_chunk, np.ndarray):
                            import soundfile as sf
                            import io
                            audio_buffer = io.BytesIO()
                            sf.write(audio_buffer, audio_chunk, sample_rate, format='WAV')
                            audio_data += audio_buffer.getvalue()
                    elif result.code == "error":
                        return {"success": False, "error": str(result.error)}
                
                return {"success": True, "audio_data": audio_data, "sample_rate": sample_rate}
            
            # Execute with timeout (120 seconds max per segment for better reliability)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_generate_audio)
                try:
                    result = future.result(timeout=120)
                    elapsed = time.time() - start_time
                    logger.info(f"Voice generation completed in {elapsed:.2f}s for text: {text[:30]}...")
                    return result
                except concurrent.futures.TimeoutError:
                    logger.error(f"Voice generation timeout (120s) for text: {text[:30]}...")
                    return {"success": False, "error": "Generation timeout after 120 seconds"}
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Voice generation error after {elapsed:.2f}s: {e}")
            return {"success": False, "error": str(e)}
        finally:
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        logger.info("🚀 Preloading FishSpeech model during startup...")
        service = get_fish_speech_service()
        
        # Force model loading during startup for better performance
        success = service.load_model()
        if success:
            logger.info("✅ FishSpeech model preloaded successfully!")
        else:
            logger.warning("⚠️ FishSpeech model preloading failed - will load on first use")
        
        return success
    except Exception as e:
        logger.error(f"Failed to initialize Fish Speech: {e}")
        return False


def cleanup_fish_speech():
    """Cleanup Fish Speech service (called from main.py)"""
    global fish_speech_service
    
    if fish_speech_service is not None:
        fish_speech_service.cleanup()
        fish_speech_service = None