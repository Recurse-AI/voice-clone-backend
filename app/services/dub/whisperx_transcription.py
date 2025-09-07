"""
WhisperX Transcription Service - Optimized for fastest transcription with 2 workers
"""

import logging
import os
import threading
from typing import Dict, Any, List, Optional
import torch

# Optimize PyTorch memory allocation
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from app.config.settings import settings
from app.services.language_service import language_service
from app.utils.cleanup_utils import cleanup_utils

logger = logging.getLogger(__name__)

class WhisperXTranscriptionService:
    """Optimized WhisperX service for fastest transcription with 2 workers"""
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.preloaded_align_models = {}
        
        self.model_size = settings.WHISPER_MODEL_SIZE
        self.alignment_device = settings.WHISPER_ALIGNMENT_DEVICE
        self._setup_device_config()
        
        logger.info(f"WhisperX service configured - Model: {self.model_size}, Device: {self.device}")
    
    def _setup_device_config(self):
        """Setup device and compute type configuration"""
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16" if settings.WHISPER_COMPUTE_TYPE == "auto" else settings.WHISPER_COMPUTE_TYPE
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = "cpu"
            self.compute_type = "int8" if settings.WHISPER_COMPUTE_TYPE == "auto" else settings.WHISPER_COMPUTE_TYPE
    
    def load_model(self) -> bool:
        """Load WhisperX model optimized for fastest transcription with 2 workers"""
        if self.is_initialized:
            return True
            
        try:
            logger.info(f"Loading WhisperX model ({self.model_size}) for fastest transcription")
            
            import whisperx
            self._setup_optimal_memory()
            
            # Skip aggressive cleanup for fastest loading
            # Only basic cleanup if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load model with optimal settings for speed
            self.model = whisperx.load_model(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type,
                download_root=None,  # Use default cache - no cleanup needed
                in_memory=True       # Keep in VRAM for fastest access
            )
            
            self._optimize_cuda_performance()
            self.is_initialized = True
            logger.info(f"✅ WhisperX model loaded for maximum speed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load WhisperX model: {e}")
            return False
    
    def _setup_optimal_memory(self):
        """Setup optimal memory allocation for 2-worker setup"""
        if self.device == "cuda":
            # Optimal memory fraction for 2 workers
            torch.cuda.set_per_process_memory_fraction(0.4)
            # Optimized memory allocation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    def _optimize_cuda_performance(self):
        """Apply CUDA optimizations for fastest inference"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _cleanup_gpu_memory(self):
        """Minimal GPU cleanup for fastest performance"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def health_check(self) -> dict:
        """Quick health check for WhisperX service"""
        try:
            status = {
                "model_initialized": self.is_initialized,
                "device": self.device,
                "model_size": self.model_size,
                "ready": self.is_initialized
            }
            
            if torch.cuda.is_available() and self.device == "cuda":
                status["gpu_memory_mb"] = int(torch.cuda.memory_allocated() / 1024**2)
                
            return status
            
        except Exception as e:
            return {"error": str(e), "ready": False}

    def transcribe_audio_file(self, audio_path: str, language: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Fast transcribe audio file with WhisperX"""
        try:
            if not self.is_initialized:
                if not self.load_model():
                    raise Exception("Failed to load WhisperX model")
            
            # Use service worker if available
            from app.config import pipeline_settings
            if pipeline_settings.USE_WHISPERX_SERVICE_WORKER:
                return self._transcribe_via_service_worker(audio_path, language, job_id)
            else:
                return self._transcribe_direct(audio_path, language, job_id)
                
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise Exception(f"Transcription failed: {str(e)}")

    def _transcribe_via_service_worker(self, audio_path: str, language_code: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Submit transcription to service worker for processing"""
        try:
            from app.utils.pipeline_utils import submit_service_request, get_service_result
            import uuid
            
            request_id = f"whisperx_{job_id}_{uuid.uuid4().hex[:8]}"
            request_data = {
                "request_id": request_id,
                "audio_path": audio_path,
                "language_code": language_code,
                "job_id": job_id
            }
            
            # Submit to service worker
            submit_success = submit_service_request("whisperx", request_id, request_data)
            if not submit_success:
                raise Exception("Failed to submit to WhisperX service worker")
            
            # Get result from service worker
            result = get_service_result("whisperx", request_id, timeout=300)
            if not result.get("success"):
                raise Exception(f"WhisperX service worker error: {result['error']}")
            
            return {
                "segments": result["segments"],
                "language": result["language"]
            }
            
        except Exception as e:
            logger.error(f"Service worker transcription failed: {e}")
            raise

    def _transcribe_direct(self, audio_path: str, language_code: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Direct transcription with loaded model"""
        try:
            import whisperx
            
            # Load and transcribe audio
            audio = whisperx.load_audio(audio_path)
            result = self.model.transcribe(audio, batch_size=16, language=language_code)
            
            # Get alignment model and align
            model_a, metadata = self._get_alignment_model(language_code)
            if model_a:
                result = whisperx.align(result["segments"], model_a, metadata, audio, self.alignment_device, return_char_alignments=False)
            
            # Convert segments to expected format
            segments = []
            for i, seg in enumerate(result.get("segments", [])):
                segments.append({
                    "id": f"seg_{i:03d}",
                    "segment_index": i,
                    "start": int(seg["start"] * 1000),
                    "end": int(seg["end"] * 1000),
                    "duration_ms": int((seg["end"] - seg["start"]) * 1000),
                    "text": seg["text"].strip(),
                    "confidence": seg.get("confidence", 0.9)
                })
            
            return {
                "segments": segments,
                "language": result.get("language", language_code)
            }
            
        except Exception as e:
            logger.error(f"Direct transcription failed: {e}")
            raise Exception(f"Transcription failed: {str(e)}")
    
    def _get_alignment_model(self, language_code: str):
        """Get alignment model for language"""
        try:
            import whisperx
            
            # Check cache first
            if language_code in self.preloaded_align_models:
                cached = self.preloaded_align_models[language_code]
                return cached['model'], cached['metadata']
            
            # Load alignment model
            model_a, metadata = whisperx.load_align_model(language_code=language_code, device=self.alignment_device)
            
            # Cache for reuse
            self.preloaded_align_models[language_code] = {
                'model': model_a,
                'metadata': metadata
            }
            
            logger.info(f"Loaded alignment model for {language_code}")
            return model_a, metadata
            
        except Exception as e:
            logger.warning(f"Failed to load alignment model for {language_code}: {e}")
            return None, None


# Global service instance with thread safety
_whisperx_service = None
_service_lock = threading.Lock()

def get_whisperx_transcription_service() -> WhisperXTranscriptionService:
    """Get or create global WhisperX transcription service instance (thread-safe)"""
    global _whisperx_service
    
    if _whisperx_service is None:
        with _service_lock:
            if _whisperx_service is None:
                _whisperx_service = WhisperXTranscriptionService()
    
    return _whisperx_service

def initialize_whisperx_transcription() -> bool:
    """Initialize WhisperX transcription service for fastest performance"""
    try:
        logger.info("Initializing WhisperX for maximum speed...")
        service = get_whisperx_transcription_service()
        
        if not service.load_model():
            logger.error("Failed to load WhisperX model")
            return False
        
        logger.info("✅ WhisperX ready for fastest transcription")
        return True
        
    except Exception as e:
        logger.error(f"❌ WhisperX initialization failed: {e}")
        return False

def cleanup_whisperx_transcription():
    """Cleanup WhisperX transcription service"""
    global _whisperx_service
    try:
        if _whisperx_service:
            # Clean up alignment models
            if hasattr(_whisperx_service, 'preloaded_align_models'):
                _whisperx_service.preloaded_align_models.clear()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            _whisperx_service = None
            logger.info("✅ WhisperX cleanup complete")
    except Exception as e:
        logger.error(f"❌ WhisperX cleanup error: {e}")