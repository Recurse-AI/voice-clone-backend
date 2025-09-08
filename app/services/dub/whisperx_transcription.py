"""
WhisperX Transcription Service - Optimized for fastest transcription with 2 workers
"""

import logging
import os
import threading
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
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
        # Traditional initialization - simple and reliable
        self.model = None
        self.is_initialized = False
        self.preloaded_align_models = {}
        
        self.model_size = settings.WHISPER_MODEL_SIZE
        self.alignment_device = settings.WHISPER_ALIGNMENT_DEVICE
        self.cache_dir = settings.WHISPER_CACHE_DIR
        self._setup_device_config()
        self._setup_cache_directory()
        
        logger.info(f"WhisperX service configured - Model: {self.model_size}, Device: {self.device}, Cache: {self.cache_dir}")
    
    def _setup_device_config(self):
        """Setup device and compute type configuration"""
        # Force CUDA setup first
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Verify CUDA is available
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            self.device = "cuda"
            self.compute_type = "float16" if settings.WHISPER_COMPUTE_TYPE == "auto" else settings.WHISPER_COMPUTE_TYPE
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info(f"✅ WhisperX configured for GPU: {self.device}")
        else:
            logger.warning("⚠️ CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.compute_type = "int8" if settings.WHISPER_COMPUTE_TYPE == "auto" else settings.WHISPER_COMPUTE_TYPE
    
    def _setup_cache_directory(self):
        """Setup persistent cache directory for WhisperX models"""
        # Create cache directory if it doesn't exist
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Set HuggingFace cache environment variables to prevent repeated downloads
        os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_path / "huggingface")
        os.environ['HF_HOME'] = str(cache_path / "huggingface")
        os.environ['TRANSFORMERS_CACHE'] = str(cache_path / "transformers")
        
        logger.info(f"Cache directory configured: {self.cache_dir}")
    
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
            
            # Load model with persistent cache for reuse
            self.model = whisperx.load_model(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type,
                download_root=self.cache_dir  # Use persistent cache directory
            )
            
            self._optimize_cuda_performance()
            self.is_initialized = True
            logger.info(f"✅ WhisperX model loaded for maximum speed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load WhisperX model: {e}")
            return False
    
    def _setup_optimal_memory(self):
        """Setup optimal memory allocation for 16GB VRAM"""
        if self.device == "cuda":
            # Conservative memory fraction for 16GB VRAM
            torch.cuda.set_per_process_memory_fraction(0.8)  # Increased for better performance
            # Memory allocation optimized for 16GB with garbage collection
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8'
            # Clear any existing cache
            torch.cuda.empty_cache()
    
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
            # Check service worker FIRST (before any model loading)
            from app.config.pipeline_settings import pipeline_settings
            logger.info(f"🔧 Service worker config: {pipeline_settings.USE_WHISPERX_SERVICE_WORKER}")
            
            if pipeline_settings.USE_WHISPERX_SERVICE_WORKER:
                logger.info("🎯 Routing to WhisperX service worker (fast path)")
                return self._transcribe_via_service_worker(audio_path, language, job_id)
            
            # Fallback to direct transcription only if running in service worker
            logger.info("📝 Using direct transcription (fallback)")
            if not self.is_initialized:
                # Only load model if this is running in a service worker
                import os
                worker_name = os.getenv('RQ_WORKER_NAME', '')
                if 'whisperx_service_worker' in worker_name:
                    logger.info("🔄 Loading WhisperX model in service worker...")
                    if not self.load_model():
                        raise Exception("Failed to load transcription model")
                else:
                    raise Exception("Model not loaded and not running in transcription service worker. Service worker should handle this.")
            
            return self._transcribe_direct(audio_path, language, job_id)
                
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise Exception(f"Transcription failed: {str(e)}")

    def _transcribe_via_service_worker(self, audio_path: str, language_code: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Submit transcription to service worker for processing"""
        try:
            # Normalize language input using language service
            normalized_language = language_service.get_language_code_for_transcription(language_code)
            logger.info(f"Service worker transcription: {audio_path} (language: {language_code} -> {normalized_language})")
            
            request_id = f"whisperx_{job_id}_{uuid.uuid4().hex[:8]}"
            request_data = {
                "request_id": request_id,
                "audio_path": audio_path,
                "language_code": normalized_language,  # Send normalized language to worker
                "job_id": job_id
            }
            
            # Enqueue request to service worker via queue manager
            from app.queue.queue_manager import queue_manager
            success = queue_manager.enqueue_whisperx_service_task(request_data)
            
            if not success:
                logger.error(f"Failed to enqueue WhisperX request for {job_id}")
                raise Exception("Failed to submit to transcription service worker")
            
            # Wait for result from service worker
            from app.utils.pipeline_utils import wait_for_service_result, cleanup_service_result
            result = wait_for_service_result("whisperx", request_id, timeout=600)  # Increased to 10 minutes
            if "error" in result:
                raise Exception(f"Transcription service worker error: {result['error']}")
            
            # Cleanup result after use
            cleanup_service_result("whisperx", request_id)
            
            return {
                "success": True,
                "segments": result["segments"],
                "sentences": result["segments"],  # Alias for compatibility
                "language": result["language"]
            }
            
        except Exception as e:
            logger.error(f"Service worker transcription failed: {e}")
            raise

    def _transcribe_direct(self, audio_path: str, language_code: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Direct transcription with loaded model"""
        try:
            import whisperx
            
            # Normalize language input using language service
            normalized_language = language_service.get_language_code_for_transcription(language_code)
            logger.info(f"Transcribing: {audio_path} (language: {language_code} -> {normalized_language})")
            
            # Load and transcribe audio
            audio = whisperx.load_audio(audio_path)
            result = self.model.transcribe(
                audio, 
                batch_size=16, 
                language=normalized_language if normalized_language != "auto_detect" else None
            )
            
            # Get alignment model and align (use detected language if auto-detect was used)
            detected_language = result.get("language", normalized_language)
            model_a, metadata = self._get_alignment_model(detected_language)
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
                "success": True,
                "segments": segments,
                "sentences": segments,  # Alias for compatibility
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
        logger.info("🚀 Initializing WhisperX for auto-load and maximum speed...")
        service = get_whisperx_transcription_service()
        
        if not service.load_model():
            logger.error("❌ Failed to load WhisperX model")
            return False
        
        logger.info("✅ WhisperX preloaded and ready for fastest transcription (language weights load on-demand)")
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