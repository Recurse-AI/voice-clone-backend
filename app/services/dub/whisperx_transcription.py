"""
WhisperX Transcription Service - Clean transcription only
High-performance transcription using WhisperX with automatic model loading
"""

import logging
import os
import time
import threading
from typing import Dict, Any, List, Optional
import torch

# Set PyTorch CUDA memory allocation config for better memory management
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from app.config.settings import settings
from app.services.language_service import language_service
from app.utils.cleanup_utils import cleanup_utils

logger = logging.getLogger(__name__)

class WhisperXTranscriptionService:
    """Clean WhisperX transcription service - transcription only, no audio manipulation"""
    
    def __init__(self):
        """Initialize WhisperX service with model pool for concurrent processing"""
        self.model_pool = []
        self.pool_size = getattr(settings, 'WHISPER_POOL_SIZE', 2)
        self.pool_semaphore = threading.Semaphore(self.pool_size)
        self.is_initialized = False
        self.preloaded_align_models = {}
        self._pool_lock = threading.Lock()
        

        self.model_size = settings.WHISPER_MODEL_SIZE
        self.alignment_device = settings.WHISPER_ALIGNMENT_DEVICE
        self._setup_device_config()
        
        logger.info(f"WhisperX pool configured - Size: {self.pool_size}, Model: {self.model_size}")
    
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
        """Load WhisperX model pool for concurrent processing"""
        if self.is_initialized:
            return True
            
        try:
            logger.info(f"Loading {self.pool_size} WhisperX models ({self.model_size})")
            
            for i in range(self.pool_size):
                model = self._load_single_model_with_timeout()
                if not model:
                    return False
                self.model_pool.append(model)
                logger.info(f"Model {i+1}/{self.pool_size} loaded")
            
            self.is_initialized = True
            logger.info(f"✅ WhisperX pool ready: {len(self.model_pool)} models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model pool: {e}")
            return False
    
    def _load_single_model_with_timeout(self):
        """Load single WhisperX model with timeout"""
        import concurrent.futures
        import whisperx
        
        def _load():
            self._aggressive_gpu_cleanup()
            return whisperx.load_model(self.model_size, device=self.device, compute_type=self.compute_type)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_load)
            try:
                return future.result(timeout=settings.WHISPER_MODEL_TIMEOUT)
            except concurrent.futures.TimeoutError:
                logger.error(f"Model loading timeout ({settings.WHISPER_MODEL_TIMEOUT}s)")
                return None
    
    def _ensure_model_loaded(self):
        """Ensure model pool is loaded before use"""
        if not self.is_initialized and not self.load_model():
            raise Exception("Failed to load WhisperX model pool")
    
    def _get_model_from_pool(self):
        """Get available model from pool"""
        self.pool_semaphore.acquire()
        with self._pool_lock:
            if not self.model_pool:
                self.pool_semaphore.release()
                raise Exception("No models available")
            return self.model_pool.pop()
    
    def _return_model_to_pool(self, model):
        """Return model to pool"""
        with self._pool_lock:
            self.model_pool.append(model)
        self.pool_semaphore.release()
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory to prevent CUDA OOM errors"""
        cleanup_utils.cleanup_gpu_memory()
    
    def _aggressive_gpu_cleanup(self):
        """Aggressive GPU memory cleanup before loading models"""
        cleanup_utils.cleanup_aggressive_gpu()
    
    def health_check(self) -> dict:
        """Health check for WhisperX service"""
        try:
            with self._pool_lock:
                available_models = len(self.model_pool)
            
            status = {
                "service_initialized": self.is_initialized,
                "pool_size": self.pool_size,
                "available_models": available_models,
                "busy_models": self.pool_size - available_models if self.is_initialized else 0,
                "device": self.device,
                "model_size": self.model_size,
                "compute_type": self.compute_type,
                "gpu_available": torch.cuda.is_available(),
                "gpu_memory_allocated": 0,
                "gpu_memory_cached": 0
            }
            
            if torch.cuda.is_available():
                status["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
                status["gpu_memory_cached"] = torch.cuda.memory_reserved() / 1024**3  # GB
            
            return status
        except Exception as e:
            return {"error": str(e), "service_initialized": False}
    
    def reset_service(self):
        """Reset the service to recover from stuck state"""
        logger.warning("🔄 Resetting WhisperX service...")
        try:
            # Clear model pool
            with self._pool_lock:
                self.model_pool.clear()
            self.is_initialized = False
            self.preloaded_align_models.clear()
            self._aggressive_gpu_cleanup()
            logger.info("✅ WhisperX service reset complete")
        except Exception as e:
            logger.error(f"❌ Failed to reset WhisperX service: {e}")
    


    def transcribe_audio_file(self, audio_path: str, language: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file and return sentences with timestamps
        
        Args:
            audio_path: Path to audio file to transcribe
            language: Language code (required)
            job_id: Optional job ID for logging
            
        Returns:
            Dict with success, sentences, and metadata
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not os.path.exists(audio_path):
                raise Exception(f"Audio file not found: {audio_path}")
            
            if not language:
                raise Exception("Language code is required")
            
            logger.info(f"Starting WhisperX transcription for: {audio_path}")
            logger.info(f"Language: {language}")
            
            # Ensure models are loaded
            self._ensure_model_loaded()
            
            # Normalize language code (supports 'auto_detect')
            language_code = self._normalize_language_code(language)
            
            # Transcribe audio using WhisperX
            transcription_result = self._transcribe_with_whisperx(audio_path, language_code, job_id)
            
            # Convert to sentences format
            sentences = self._convert_to_sentences_format(transcription_result["segments"]) 
            
            # Calculate audio duration
            import librosa
            duration = librosa.get_duration(path=audio_path)
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "sentences": sentences,
                "metadata": {
                    "audio_path": audio_path,
                    "duration": round(duration, 2),
                    "processing_time": round(processing_time, 2),
                    "language": transcription_result["language"],
                    "segments_count": len(sentences)
                }
            }
            
            logger.info(f"Transcription completed in {processing_time:.2f}s - {len(sentences)} sentences")
                        
            return result
            
        except Exception as e:
            logger.error(f"WhisperX transcription failed: {e}")
            # Skip GPU cleanup even on failure to avoid unnecessary operations
            # self._cleanup_gpu_memory()  # DISABLED per user request
            
            return {
                "success": False,
                "error": str(e),
                "sentences": [],
                "metadata": {
                    "audio_path": audio_path,
                    "processing_time": time.time() - start_time
                }
            }
    

    def _normalize_language_code(self, language: str) -> str:
        """Normalize language code for WhisperX transcription"""
        if not language:
            raise Exception("Language code is required")
        
        # Normalize language using language service
        normalized = language_service.normalize_language_input(language)
        if not normalized:
            raise Exception(f"Unsupported language: {language}")
        
        return normalized
    
    def _transcribe_with_whisperx(self, audio_path: str, language_code: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Core transcription using model pool for concurrent processing"""
        model = self._get_model_from_pool()
        
        try:
            import whisperx
            audio = whisperx.load_audio(audio_path)
            
            if language_code == "auto_detect":
                result = model.transcribe(audio)
            else:
                result = model.transcribe(audio, language=language_code)
        
            used_language = result.get("language", language_code)
            
            # Word-level alignment
            try:
                model_a, metadata = self._get_alignment_model(used_language)
                aligned_result = whisperx.align(
                    result["segments"], model_a, metadata, audio, 
                    self.alignment_device, return_char_alignments=False
                )
                return {
                    "segments": aligned_result.get("segments", result.get("segments", [])),
                    "language": used_language
                }
            except Exception as e:
                logger.warning(f"Alignment failed for '{used_language}': {e}")
                return {"segments": result.get("segments", []), "language": used_language}
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise Exception(f"Transcription failed: {str(e)}")
        finally:
            self._return_model_to_pool(model)
    
    # Preloading removed: alignment models are loaded on-demand only
    
    def _get_alignment_model(self, language_code: str):
        """Get alignment model with cross-platform file locking to prevent race conditions"""
        try:
            import whisperx
            import os
            import platform
            
            # Check if we have it preloaded
            if language_code in self.preloaded_align_models:
                logger.info(f"🚀 Using preloaded alignment model for '{language_code}'")
                cached = self.preloaded_align_models[language_code]
                return cached['model'], cached['metadata']
            
            # Create lock file for this language to prevent concurrent downloads
            lock_dir = os.path.expanduser("~/.cache/whisperx/locks")
            os.makedirs(lock_dir, exist_ok=True)
            lock_file_path = os.path.join(lock_dir, f"{language_code}.lock")
            
            # Cross-platform file locking
            if platform.system() == "Windows":
                # Windows file locking using msvcrt
                import msvcrt
                with open(lock_file_path, 'w') as lock_file:
                    try:
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                        logger.info(f"🔒 Acquired Windows lock for '{language_code}' model download")
                        return self._load_alignment_model_locked(language_code, whisperx)
                    finally:
                        try:
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                        except:
                            pass
            else:
                # Unix file locking using fcntl
                import fcntl
                with open(lock_file_path, 'w') as lock_file:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    logger.info(f"🔒 Acquired Unix lock for '{language_code}' model download")
                    return self._load_alignment_model_locked(language_code, whisperx)
                    
        except Exception as e:
            logger.error(f"❌ Failed to get alignment model for '{language_code}': {e}")
            raise

    def _load_alignment_model_locked(self, language_code: str, whisperx):
        """Load alignment model when file lock is already acquired"""
        # Load model (will download if not exists, or load from cache)
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code, 
            device=self.alignment_device
        )
        
        # Cache alignment model (limit to 3 for memory efficiency)
        if len(self.preloaded_align_models) >= 3:
            oldest_lang = next(iter(self.preloaded_align_models))
            del self.preloaded_align_models[oldest_lang]
        
        self.preloaded_align_models[language_code] = {'model': model_a, 'metadata': metadata}
        
        logger.info(f"✅ Successfully loaded alignment model for '{language_code}'")
        return model_a, metadata
    
    def _convert_to_sentences_format(self, whisperx_segments: List[Dict]) -> List[Dict[str, Any]]:
        """Convert WhisperX segments to optimized sentence format for dubbing"""
        sentences = []
        sentence_id_counter = 0

        for i, segment in enumerate(whisperx_segments):
            text = segment.get("text", "").strip()
            if text:
                start_ms = int(segment.get("start", 0) * 1000)
                end_ms = int(segment.get("end", 0) * 1000)
                duration_ms = end_ms - start_ms
                
                # If segment is too long (>15 seconds), split it
                if duration_ms > 15000:
                    sub_segments = self._split_long_segment(text, start_ms, end_ms, sentence_id_counter)
                    sentences.extend(sub_segments)
                    sentence_id_counter += len(sub_segments)
                else:
                    sentences.append({
                        "text": text,
                        "start": start_ms,
                        "end": end_ms,
                        "id": f"sentence_{sentence_id_counter}"
                    })
                    sentence_id_counter += 1

        return sentences
    
    def _split_long_segment(self, text: str, start_ms: int, end_ms: int, start_id: int) -> List[Dict[str, Any]]:
        """Split long segments into smaller chunks for better dubbing"""
        from app.services.dub.simple_dubbed_api import smart_chunk
        
        chunks = smart_chunk(text, chunk_size=150, min_size=100)
        duration_ms = end_ms - start_ms
        total_chars = len(text)
        
        sub_segments = []
        char_count = 0
        
        for i, chunk in enumerate(chunks):
            chunk_chars = len(chunk)
            # Proportional time allocation based on character count
            chunk_start = start_ms + (duration_ms * char_count // total_chars)
            char_count += chunk_chars
            chunk_end = start_ms + (duration_ms * char_count // total_chars)
            
            sub_segments.append({
                "text": chunk.strip(),
                "start": chunk_start,
                "end": min(chunk_end, end_ms),
                "id": f"sentence_{start_id + i}"
            })
        
        return sub_segments


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
    """Initialize WhisperX transcription service - clean and simple"""
    try:
        logger.info("Initializing WhisperX transcription service...")
        service = get_whisperx_transcription_service()
        
        # Load main model
        if not service.load_model():
            logger.error("Failed to load WhisperX main model")
            return False
        
        logger.info("✅ WhisperX initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ WhisperX initialization failed: {e}")
        return False

def cleanup_whisperx_transcription():
    """Cleanup WhisperX transcription service (called from main.py)"""
    global _whisperx_service
    try:
        if _whisperx_service:
            # Clean up preloaded alignment models
            if hasattr(_whisperx_service, 'preloaded_align_models'):
                for lang_code in list(_whisperx_service.preloaded_align_models.keys()):
                    try:
                        del _whisperx_service.preloaded_align_models[lang_code]
                    except Exception as e:
                        logger.warning(f"Failed to cleanup alignment model for {lang_code}: {e}")
                _whisperx_service.preloaded_align_models.clear()
            
            # Clean up model pool
            if hasattr(_whisperx_service, 'model_pool'):
                with _whisperx_service._pool_lock:
                    for model in _whisperx_service.model_pool:
                        try:
                            del model
                        except Exception as e:
                            logger.warning(f"Failed to cleanup model: {e}")
                    _whisperx_service.model_pool.clear()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            _whisperx_service = None
            logger.info("✅ WhisperX transcription service cleaned up successfully")
    except Exception as e:
        logger.error(f"❌ Error during WhisperX cleanup: {e}")