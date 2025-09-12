import logging
import os
import threading
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
import torch

# Set PyTorch memory allocation configuration early
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from app.config.settings import settings
from app.services.language_service import language_service
from app.utils.cleanup_utils import cleanup_utils

logger = logging.getLogger(__name__)

class WhisperXTranscriptionService:
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.preloaded_align_models = {}

        self.model_size = settings.WHISPER_MODEL_SIZE
        self.alignment_device = settings.WHISPER_ALIGNMENT_DEVICE
        self.cache_dir = settings.WHISPER_CACHE_DIR
        self.max_seg_seconds = settings.WHISPER_MAX_SEG_SECONDS
        self._setup_device_config()
        self._setup_cache_directory()

        logger.info(f"WhisperX service configured - Model: {self.model_size}, Device: {self.device}, Cache: {self.cache_dir}, Max segment: {self.max_seg_seconds}s")

    def _setup_device_config(self):

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
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_path / "huggingface")
        os.environ['HF_HOME'] = str(cache_path / "huggingface")
        os.environ['TRANSFORMERS_CACHE'] = str(cache_path / "transformers")

        logger.info(f"Cache directory configured: {self.cache_dir}")

    def load_model(self) -> bool:
        if self.is_initialized:
            return True

        try:
            logger.info(f"Loading WhisperX model ({self.model_size}) for fastest transcription")

            import whisperx
            self._setup_optimal_memory()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model = whisperx.load_model(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.cache_dir
            )

            self._optimize_cuda_performance()
            self.is_initialized = True
            logger.info(f"✅ WhisperX model loaded for maximum speed")
            return True

        except Exception as e:
            logger.error(f"Failed to load WhisperX model: {e}")
            return False

    def _setup_optimal_memory(self):
        if self.device == "cuda":
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
            torch.cuda.empty_cache()

    def _optimize_cuda_performance(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _cleanup_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def health_check(self) -> dict:
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
        try:
            logger.info("🎯 Using WhisperX direct transcription")
            return self._transcribe_direct(audio_path, language, job_id)
        except Exception as e:
            logger.error(f"WhisperX direct transcription failed for {audio_path}: {e}")
            raise Exception(f"WhisperX transcription failed: {str(e)}")

    def _transcribe_via_service_worker(self, audio_path: str, language_code: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            normalized_language = language_service.get_language_code_for_transcription(language_code)
            logger.info(f"Service worker transcription: {audio_path} (language: {language_code} -> {normalized_language})")

            request_id = f"whisperx_{job_id}_{uuid.uuid4().hex[:8]}"
            request_data = {
                "request_id": request_id,
                "audio_path": audio_path,
                "language_code": normalized_language,
                "job_id": job_id
            }

            from app.queue.queue_manager import queue_manager
            success = queue_manager.enqueue_with_load_balance(request_data, "whisperx")

            if not success:
                logger.error(f"Failed to enqueue WhisperX request for {job_id}")
                raise Exception("Failed to submit to transcription service worker")

            from app.utils.pipeline_utils import wait_for_service_result, cleanup_service_result
            result = wait_for_service_result("whisperx", request_id, timeout=600)
            if "error" in result:
                raise Exception(f"Transcription service worker error: {result['error']}")

            cleanup_service_result("whisperx", request_id)

            return {
                "success": True,
                "segments": result["segments"],
                "sentences": result["segments"],
                "language": result["language"]
            }

        except Exception as e:
            logger.error(f"Service worker transcription failed: {e}")
            raise

    def _transcribe_direct(self, audio_path: str, language_code: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        import whisperx

        if self.model is None:
            raise Exception("WhisperX model not loaded")

        normalized_language = language_service.get_language_code_for_transcription(language_code)
        audio = whisperx.load_audio(audio_path)
        from app.config.settings import settings
        batch_size = settings.WHISPERX_BATCH_SIZE

        try:
            result = self.model.transcribe(
                audio,
                batch_size=batch_size,
                language=normalized_language if normalized_language != "auto_detect" else None,
                task="transcribe"
            )

            detected_language = result.get("language", normalized_language)
            model_a, metadata = self._get_alignment_model(detected_language)
            if model_a:
                result = whisperx.align(result["segments"], model_a, metadata, audio, self.alignment_device, return_char_alignments=False)

            segments = self._process_segments(result.get("segments", []))

            return {
                "success": True,
                "segments": segments,
                "sentences": segments,
                "language": result.get("language", language_code)
            }

        except Exception as e:
            logger.error(f"Transcription failed (batch_size={batch_size}): {e}")
            raise Exception(f"Transcription failed: {str(e)}")
    
    def _process_segments(self, raw_segments):
        segments = []
        segment_index = 0
        split_count = 0
        
        for seg in raw_segments:
            duration_sec = seg["end"] - seg["start"]
            if duration_sec <= self.max_seg_seconds:
                segments.append(self._create_segment(seg, segment_index))
                segment_index += 1
            else:
                split_segments = self._split_long_segment(seg, duration_sec)
                split_count += len(split_segments) - 1
                logger.info(f"Split {duration_sec:.1f}s segment into {len(split_segments)} parts (max: {self.max_seg_seconds}s)")
                for split_seg in split_segments:
                    segments.append(self._create_segment(split_seg, segment_index))
                    segment_index += 1
        
        if split_count > 0:
            logger.info(f"Created {split_count} additional segments from splitting for optimal dubbing quality")
        
        return segments

    def _split_long_segment(self, segment, total_duration):
        text = segment["text"].strip()
        start_time = segment["start"]
        end_time = segment["end"]
        confidence = segment.get("confidence", 0.9)
        
        num_splits = max(2, int(total_duration // self.max_seg_seconds))
        split_duration = total_duration / num_splits
        words = text.split()
        
        if len(words) <= num_splits:
            words_per_split = 1
        else:
            words_per_split = len(words) // num_splits
        
        result_segments = []
        word_start = 0
        
        for i in range(num_splits):
            split_start = start_time + (i * split_duration)
            split_end = start_time + ((i + 1) * split_duration)
            if i == num_splits - 1:
                split_end = end_time
                text_chunk = " ".join(words[word_start:])
            else:
                word_end = min(word_start + words_per_split, len(words))
                text_chunk = " ".join(words[word_start:word_end])
                word_start = word_end
            
            if text_chunk.strip():
                result_segments.append({
                    "start": split_start,
                    "end": split_end,
                    "text": text_chunk.strip(),
                    "confidence": confidence
                })
        
        return result_segments


    def _create_segment(self, seg, index):
        return {
            "id": f"seg_{index:03d}",
            "segment_index": index,
            "start": int(seg["start"] * 1000),
            "end": int(seg["end"] * 1000),
            "duration_ms": int((seg["end"] - seg["start"]) * 1000),
            "text": seg["text"].strip(),
            "confidence": seg.get("confidence", 0.9)
        }

    def _get_alignment_model(self, language_code: str):
        try:
            import whisperx

            if language_code in self.preloaded_align_models:
                cached = self.preloaded_align_models[language_code]
                return cached['model'], cached['metadata']

            model_a, metadata = whisperx.load_align_model(language_code=language_code, device=self.alignment_device)

            self.preloaded_align_models[language_code] = {
                'model': model_a,
                'metadata': metadata
            }

            logger.info(f"Loaded alignment model for {language_code}")
            return model_a, metadata

        except Exception as e:
            logger.warning(f"Failed to load alignment model for {language_code}: {e}")
            return None, None

_whisperx_service = None
_service_lock = threading.Lock()

def get_whisperx_transcription_service() -> WhisperXTranscriptionService:
    global _whisperx_service

    if _whisperx_service is None:
        with _service_lock:
            if _whisperx_service is None:
                _whisperx_service = WhisperXTranscriptionService()

    return _whisperx_service

def initialize_whisperx_transcription() -> bool:
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
    global _whisperx_service
    try:
        if _whisperx_service:
            if hasattr(_whisperx_service, 'preloaded_align_models'):
                _whisperx_service.preloaded_align_models.clear()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            _whisperx_service = None
            logger.info("✅ WhisperX cleanup complete")
    except Exception as e:
        logger.error(f"❌ WhisperX cleanup error: {e}")