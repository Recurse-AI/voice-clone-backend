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

from app.config.settings import settings
from app.services.language_service import language_service

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
        self.preload_languages = settings.WHISPER_PRELOAD_LANGUAGES
        
        # Device and compute type setup
        if torch.cuda.is_available():
            self.device = "cuda"
            # Use configured compute type or auto-detect
            if settings.WHISPER_COMPUTE_TYPE == "auto":
                self.compute_type = "float16"
            else:
                self.compute_type = settings.WHISPER_COMPUTE_TYPE
            logger.info(f"WhisperX service initialized on GPU (CUDA) with {self.compute_type} precision")
        else:
            self.device = "cpu"
            # Use configured compute type or auto-detect
            if settings.WHISPER_COMPUTE_TYPE == "auto":
                self.compute_type = "int8"
            else:
                self.compute_type = settings.WHISPER_COMPUTE_TYPE
            logger.info(f"WhisperX service initialized on CPU with {self.compute_type} precision")
        
        logger.info(f"WhisperX configured - Model: {self.model_size}, Preload languages: {self.preload_languages}")
    
    def load_model(self) -> bool:
        """Load WhisperX model with auto-download and language model preloading"""
        try:
            if self.is_initialized:
                logger.info("WhisperX model already loaded")
                return True
            
            import whisperx
            logger.info(f"🔄 Loading WhisperX model '{self.model_size}' (auto-downloading if needed)...")
            
            # Load main Whisper model
            self.whisperx_model = whisperx.load_model(
                self.model_size, 
                device=self.device,
                compute_type=self.compute_type
            )
            
            logger.info(f"✅ WhisperX model '{self.model_size}' loaded successfully")
            
            # Preload alignment models for common languages
            if self.preload_languages:
                logger.info(f"🔄 Preloading alignment models for languages: {', '.join(self.preload_languages)}")
                self._preload_alignment_models()
            
            self.is_initialized = True
            logger.info("🎉 WhisperX service fully initialized with all models loaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load WhisperX model: {e}")
            return False
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded before use"""
        if not self.is_initialized:
            if not self.load_model():
                raise Exception("Failed to load WhisperX model")
    
    def transcribe_audio_file(self, audio_path: str, language: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file and return sentences with timestamps
        
        Args:
            audio_path: Path to audio file to transcribe
            language: Language code (required)
            job_id: Optional job ID for cancellation checking
            
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
            
            # Normalize language code
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
            return {
                "success": False,
                "error": str(e),
                "sentences": [],
                "metadata": {
                    "audio_path": audio_path,
                    "processing_time": time.time() - start_time
                }
            }
    
    def get_sentences_and_split_audio(self, audio_url: str, output_dir: str = None, source_video_language: str = None, max_sentences: int = None, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get sentences from WhisperX and split audio into segments"""
        try:
            from .audio_utils import AudioUtils
            audio_utils = AudioUtils()
            
            # Use the exact vocal file path (already downloaded by dub_routes.py)
            vocal_file_path = os.path.join(output_dir, f"vocals_{job_id}.wav")
            
            if not os.path.exists(vocal_file_path):
                raise Exception(f"Vocal file not found at expected path: {vocal_file_path}")
            
            logger.info(f"Using vocal file: {vocal_file_path}")
            
            # Transcribe using WhisperX
            transcription_result = self.transcribe_audio_file(vocal_file_path, source_video_language, job_id)
            if not transcription_result["success"]:
                raise Exception(transcription_result.get("error", "Transcription failed"))
            
            raw_sentences = transcription_result["sentences"]
            if max_sentences:
                raw_sentences = raw_sentences[:max_sentences]
            
            # Split by sentences
            segments_to_split = []
            for sentence in raw_sentences:
                segments_to_split.append({
                    "start": sentence["start"],
                    "end": sentence["end"], 
                    "text": sentence["text"]
                })
            
            # Split audio
            split_result = audio_utils.split_audio_by_timestamps(vocal_file_path, output_dir, segments_to_split)
            if not split_result["success"]:
                raise Exception(f"Failed to split audio: {split_result['error']}")
            
            # Create segments
            enhanced_segments = []
            split_files = split_result.get("split_files", [])
            for i, split_file in enumerate(split_files):
                if i < len(raw_sentences):
                    enhanced_segment = raw_sentences[i].copy()
                    enhanced_segment.update({
                        "output_path": split_file["output_path"],
                        "duration_ms": split_file["duration_ms"]
                    })
                    enhanced_segments.append(enhanced_segment)
            
            return {
                "success": True,
                "transcript_id": f"whisperx_{int(time.time())}",
                "audio_url": audio_url,
                "segments": enhanced_segments
            }
            
        except Exception as e:
            logger.error(f"WhisperX failed: {str(e)}")
            raise Exception(f"WhisperX error: {str(e)}")
    
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
            
            # Check for job cancellation
            if job_id:
                from app.utils.shared_memory import is_job_cancelled
                if is_job_cancelled(job_id):
                    raise Exception("Job cancelled by user")
            
            # Transcribe with specified language
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
                    self.device,
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
    
    def _preload_alignment_models(self):
        """Preload alignment models for common languages to avoid runtime delays"""
        try:
            import whisperx
            
            for lang_code in self.preload_languages:
                try:
                    # Clean language code
                    clean_lang = lang_code.strip().lower()
                    if not clean_lang:
                        continue
                    
                    logger.info(f"🔄 Preloading alignment model for language: {clean_lang}")
                    model_a, metadata = whisperx.load_align_model(
                        language_code=clean_lang, 
                        device=self.device
                    )
                    
                    # Cache the models
                    self.preloaded_align_models[clean_lang] = {
                        'model': model_a,
                        'metadata': metadata
                    }
                    
                    logger.info(f"✅ Alignment model for '{clean_lang}' preloaded successfully")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to preload alignment model for '{clean_lang}': {e}")
                    continue
            
            logger.info(f"✅ Preloaded {len(self.preloaded_align_models)} alignment models")
            
        except Exception as e:
            logger.error(f"❌ Failed to preload alignment models: {e}")
    
    def _get_alignment_model(self, language_code: str):
        """Get alignment model (from cache or load on demand)"""
        try:
            import whisperx
            
            # Check if we have it preloaded
            if language_code in self.preloaded_align_models:
                logger.info(f"🚀 Using preloaded alignment model for '{language_code}'")
                cached = self.preloaded_align_models[language_code]
                return cached['model'], cached['metadata']
            
            # Load on demand if not preloaded
            logger.info(f"🔄 Loading alignment model for '{language_code}' on demand...")
            model_a, metadata = whisperx.load_align_model(
                language_code=language_code, 
                device=self.device
            )
            return model_a, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to get alignment model for '{language_code}': {e}")
            raise
    
    def _convert_to_sentences_format(self, whisperx_segments: List[Dict]) -> List[Dict[str, Any]]:
        """Convert WhisperX segments to simple sentence format"""
        sentences = []
        
        for i, segment in enumerate(whisperx_segments):
            sentence = {
                "text": segment.get("text", "").strip(),
                "start": int(segment.get("start", 0) * 1000),  # Convert to ms
                "end": int(segment.get("end", 0) * 1000),
                "id": f"sentence_{i}"
            }
            
            if sentence["text"]:
                sentences.append(sentence)
        
        return sentences


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
    """Initialize WhisperX transcription service (called from main.py)"""
    try:
        service = get_whisperx_transcription_service()
        return service.load_model()
    except Exception as e:
        logger.error(f"Failed to initialize WhisperX transcription: {e}")
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