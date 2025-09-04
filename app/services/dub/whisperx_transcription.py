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
        """Initialize WhisperX service"""
        self.whisperx_model = None
        self.is_initialized = False
        self.preloaded_align_models = {}  # Cache for language alignment models
        

        # Get configuration from settings
        self.model_size = settings.WHISPER_MODEL_SIZE
        # No static preloading; alignment models load on demand only
        self.alignment_device = settings.WHISPER_ALIGNMENT_DEVICE
        
        # Device and compute type setup
        if torch.cuda.is_available():
            self.device = "cuda"
            if settings.WHISPER_COMPUTE_TYPE == "auto":
                self.compute_type = "float16"
            else:
                self.compute_type = settings.WHISPER_COMPUTE_TYPE
            logger.info(f"WhisperX service initialized on GPU (CUDA) with {self.compute_type} precision")
        else:
            self.device = "cpu"
            if settings.WHISPER_COMPUTE_TYPE == "auto":
                self.compute_type = "int8"
            else:
                self.compute_type = settings.WHISPER_COMPUTE_TYPE
            logger.info(f"WhisperX service initialized on CPU with {self.compute_type} precision")
        
        logger.info(f"WhisperX configured - Model: {self.model_size}")
    
    def load_model(self) -> bool:
        """Load WhisperX model with auto-download and aggressive memory management"""
        try:
            if self.is_initialized:
                logger.info("WhisperX model already loaded")
                return True
            
            # 🧹 AGGRESSIVE MEMORY CLEANUP BEFORE LOADING
            self._aggressive_gpu_cleanup()
            
            import whisperx
            self.whisperx_model = whisperx.load_model(
                self.model_size, 
                device=self.device,
                compute_type=self.compute_type
            )
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load WhisperX model: {e}")
            # No CPU fallback - GPU required
            if self.device == "cuda":
                logger.error("❌ GPU failed and CPU fallback is disabled")
            return False
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded before use"""
        if not self.is_initialized:
            if not self.load_model():
                raise Exception("Failed to load WhisperX model")
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory to prevent CUDA OOM errors"""
        cleanup_utils.cleanup_gpu_memory()
    
    def _aggressive_gpu_cleanup(self):
        """Aggressive GPU memory cleanup before loading models"""
        cleanup_utils.cleanup_aggressive_gpu()
    


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
        """Core transcription method using WhisperX"""
        try:
            import whisperx
            
            # Load audio
            audio = whisperx.load_audio(audio_path)
            
            
            # Transcribe with specified language or auto-detect
            if language_code == "auto_detect":
                logger.info("Transcribing with auto language detection")
                result = self.whisperx_model.transcribe(audio)
            else:
                logger.info(f"Transcribing with language: {language_code}")
                result = self.whisperx_model.transcribe(audio, language=language_code)
            
            used_language = result.get("language", language_code)
            logger.info(f"Used language: {used_language}")
            
            # Word-level alignment for better timestamps
            try:
                model_a, metadata = self._get_alignment_model(used_language)
                aligned_result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    self.alignment_device,
                    return_char_alignments=False
                )
                logger.info(f"✅ Word-level alignment completed for '{used_language}'")
                return {
                    "segments": aligned_result.get("segments", result.get("segments", [])),
                    "language": used_language
                }
            except Exception as e:
                logger.warning(f"⚠️ Alignment failed for '{used_language}', using base transcription: {e}")
                return {
                    "segments": result.get("segments", []),
                    "language": used_language
                }
            
        except Exception as e:
            logger.error(f"WhisperX transcription failed: {e}")
            raise Exception(f"Transcription failed: {str(e)}")
    
    # Preloading removed: alignment models are loaded on-demand only
    
    def _get_alignment_model(self, language_code: str):
        """Get alignment model with file locking to prevent race conditions"""
        try:
            import whisperx
            import fcntl
            import os
            import tempfile
            
            # Check if we have it preloaded
            if language_code in self.preloaded_align_models:
                logger.info(f"🚀 Using preloaded alignment model for '{language_code}'")
                cached = self.preloaded_align_models[language_code]
                return cached['model'], cached['metadata']
            
            # Create lock file for this language to prevent concurrent downloads
            lock_dir = os.path.expanduser("~/.cache/whisperx/locks")
            os.makedirs(lock_dir, exist_ok=True)
            lock_file_path = os.path.join(lock_dir, f"{language_code}.lock")
            
            # Use file locking to prevent race conditions
            with open(lock_file_path, 'w') as lock_file:
                try:
                    # Try to acquire exclusive lock (blocks if another worker is downloading)
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    logger.info(f"🔒 Acquired lock for '{language_code}' model download")
                    
                    # Check if model exists after acquiring lock (another worker might have downloaded it)
                    model_cache_path = os.path.expanduser(f"~/.cache/whisperx/models/{language_code}")
                    if os.path.exists(model_cache_path):
                        logger.info(f"✅ Model for '{language_code}' already exists, loading...")
                    else:
                        logger.info(f"🔄 Downloading alignment model for '{language_code}'...")
                    
                    # Load model (will download if not exists, or load from cache)
                    model_a, metadata = whisperx.load_align_model(
                        language_code=language_code, 
                        device=self.alignment_device
                    )
                    
                    # Cache in memory for this worker instance
                    self.preloaded_align_models[language_code] = {
                        'model': model_a,
                        'metadata': metadata
                    }
                    
                    logger.info(f"✅ Successfully loaded alignment model for '{language_code}'")
                    return model_a, metadata
                    
                except Exception as download_error:
                    logger.error(f"❌ Failed to download/load model for '{language_code}': {download_error}")
                    raise
                finally:
                    # Lock is automatically released when file is closed
                    pass
            
        except Exception as e:
            logger.error(f"❌ Failed to get alignment model for '{language_code}': {e}")
            raise
    
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
    """Initialize WhisperX transcription service with optional preloading"""
    try:
        service = get_whisperx_transcription_service()
        
        if not service.load_model():
            return False
        
        common_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "hi", "ar", "bn"]
        logger.info(f"Pre-downloading {len(common_languages)} language models...")
        
        for lang_code in common_languages:
            try:
                import os
                model_cache_path = os.path.expanduser(f"~/.cache/whisperx/models/{lang_code}")
                if os.path.exists(model_cache_path):
                    continue
                
                import whisperx
                whisperx.load_align_model(language_code=lang_code, device="cpu")
                logger.info(f"Downloaded {lang_code}")
                
            except Exception:
                continue
        
        logger.info("WhisperX initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"WhisperX initialization failed: {e}")
        return False

def cleanup_whisperx_transcription():
    """Cleanup WhisperX transcription service (called from main.py)"""
    global _whisperx_service
    try:
        if _whisperx_service:
            # Clean up preloaded models
            if hasattr(_whisperx_service, 'preloaded_align_models'):
                for lang_code in list(_whisperx_service.preloaded_align_models.keys()):
                    try:
                        del _whisperx_service.preloaded_align_models[lang_code]
                    except Exception as e:
                        logger.warning(f"Failed to cleanup alignment model for {lang_code}: {e}")
                _whisperx_service.preloaded_align_models.clear()
            
            # Clean up main model
            if hasattr(_whisperx_service, 'whisperx_model') and _whisperx_service.whisperx_model:
                del _whisperx_service.whisperx_model
                _whisperx_service.whisperx_model = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            _whisperx_service = None
            logger.info("✅ WhisperX transcription service cleaned up successfully")
    except Exception as e:
        logger.error(f"❌ Error during WhisperX cleanup: {e}")