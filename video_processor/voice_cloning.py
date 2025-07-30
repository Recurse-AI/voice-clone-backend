"""
OpenAudio S1-mini Voice Cloning Service - Production Ready
=========================================================

Features:
- Auto-downloads OpenAudio S1-mini models with HF token authentication
- High-quality voice cloning with advanced emotional control
- GPU-optimized inference with torch.compile support
- Clean, simple architecture with no fallback complexity
- 20+ emotional markers support (angry, sad, excited, sarcastic, etc.)

Usage:
- Models download automatically with HF_TOKEN from environment
- Uses only OpenAudio S1-mini (best quality/performance ratio)
- Simple, direct model loading without complex fallbacks
- Provides detailed logging for debugging

Author: Voice Cloning API Team
Version: 3.0 (S1-mini Only)
"""

import logging
import sys
import threading
import os
from pathlib import Path
from typing import Dict, Any

import torch
import soundfile as sf
import numpy as np
from huggingface_hub import hf_hub_download

from config import settings

logger = logging.getLogger(__name__)


class FishSpeechService:
    """OpenAudio S1-mini voice cloning service"""
    
    def __init__(self, device: str = "cuda", use_compile: bool = True):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_compile = use_compile
        self.precision = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        # Model configuration - OpenAudio S1-mini only
        self.model_repo = "fishaudio/openaudio-s1-mini" 
        self.model_path = "checkpoints/openaudio-s1-mini"
        
        # Initialize as None, will be loaded lazily
        self.inference_engine = None
        self.llama_queue = None
        self.decoder_model = None
        self._model_lock = threading.Lock()
        self._is_initialized = False
        
        logger.info(f"OpenAudio S1-mini service initialized with device: {self.device}")
    
    def _ensure_fish_speech_available(self):
        """Verify Fish Speech framework is ready (setup done by runpod_setup.sh)"""
        try:
            # Fish Speech framework should be setup by runpod_setup.sh
            sys.path.insert(0, "./fish-speech")
            from fish_speech.models.text2semantic.llama import BaseTransformer
            logger.debug("Fish Speech framework ready")
            return True
        except ImportError:
            raise Exception(
                "Fish Speech framework not found. Make sure to run ./runpod_setup.sh first "
                "to setup the complete environment including Fish Speech."
            )
    
    def _download_models(self):
        """Download OpenAudio S1-mini models with HF token authentication"""
        logger.info("📥 Downloading OpenAudio S1-mini models...")
        
        model_dir = Path(self.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Get HF token from environment
        hf_token = os.getenv('HF_TOKEN')
        download_kwargs = {
            "repo_id": self.model_repo,
            "local_dir": str(model_dir),
            "local_dir_use_symlinks": False,
            "resume_download": True
        }
        
        if hf_token:
            download_kwargs["token"] = hf_token
            logger.info("🔑 Using HF token for authentication")
        else:
            logger.info("⚠️ No HF token found, downloading without authentication")
        
        # OpenAudio S1-mini files
        model_files = ["config.json", "model.pth", "special_tokens.json", "tokenizer.tiktoken", "codec.pth"]
        
        for file_name in model_files:
            if not (model_dir / file_name).exists():
                try:
                    hf_hub_download(filename=file_name, **download_kwargs)
                    logger.info(f"✅ Downloaded {file_name}")
                except Exception as e:
                    logger.error(f"❌ Failed to download {file_name}: {e}")
                    raise Exception(f"Required model file {file_name} could not be downloaded: {e}")
        
        logger.info("✅ OpenAudio S1-mini models ready")
    
    def _initialize_models(self):
        """Initialize Fish Speech models completely"""
        with self._model_lock:
            if self._is_initialized:
                return
            
            logger.info("🔍 Verifying Fish Speech framework...")
            self._ensure_fish_speech_available()
            
            self._download_models()
            
            logger.info("🚀 Loading OpenAudio S1-mini models...")
            
            # Import Fish Speech modules (manual setup)
            sys.path.insert(0, "./fish-speech")
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.dac.inference import load_model as load_decoder_model
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
            
            # Load Fish Speech language model
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=self.model_path,
                device=self.device,
                precision=self.precision,
                compile=self.use_compile,
            )
            logger.info("✅ Language model loaded")
            
            # Load OpenAudio S1-mini decoder model
            checkpoint_path = f"{self.model_path}/codec.pth"
            
            self.decoder_model = load_decoder_model(
                config_name="modded_dac_vq",  # S1-mini uses modded_dac_vq config
                checkpoint_path=checkpoint_path,
                device=self.device,
            )
            logger.info("✅ OpenAudio S1-mini decoder model loaded")
            
            # Initialize TTS inference engine
            self.inference_engine = TTSInferenceEngine(
                llama_queue=self.llama_queue,
                decoder_model=self.decoder_model,
                compile=self.use_compile,
                precision=self.precision,
            )
            logger.info("✅ TTS inference engine initialized")
            
            # Warm up model
            self._warmup_model()
            logger.info("✅ Model warmup completed")
            
            self._is_initialized = True
            logger.info("✅ OpenAudio S1-mini ready!")
    
    def _warmup_model(self):
        """Warm up model with simple inference"""
        from fish_speech.utils.schema import ServeTTSRequest
        
        warmup_request = ServeTTSRequest(
            text="Ready.",
            references=[],
            max_new_tokens=512,
            format="wav"
        )
        
        list(self.inference_engine.inference(warmup_request))
    
    def clone_voice_for_segment(self, segment_metadata: Dict[str, Any], 
                               reference_audio_path: str, audio_id: str) -> Dict[str, Any]:
        """Clone voice for a specific segment using OpenAudio S1-mini"""
        try:
            # Ensure models are initialized
            if not self._is_initialized:
                self._initialize_models()
            
            from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
            from fish_speech.utils.file import audio_to_bytes
            
            # Get text to synthesize
            text_to_clone = segment_metadata.get("text", "")
            if not text_to_clone.strip():
                return {"success": False, "error": "No text to synthesize"}
            
            # Clean text for TTS (Fish Speech handles emotion markers)
            cleaned_text = self._prepare_text_for_tts(text_to_clone)
            
            # Prepare reference audio
            reference_audio_bytes = audio_to_bytes(reference_audio_path)
            reference_text = segment_metadata.get("original_text", text_to_clone)
            
            # Create reference
            references = [ServeReferenceAudio(
                audio=reference_audio_bytes,
                text=reference_text
            )]
            
            # Create TTS request with optimal settings for voice cloning
            tts_request = ServeTTSRequest(
                text=cleaned_text,
                references=references,
                reference_id=None,
                max_new_tokens=2048,  # Higher for longer segments
                chunk_length=300,
                top_p=0.8,            # Optimal for voice cloning
                repetition_penalty=1.1,
                temperature=0.7,      # Balanced creativity/consistency
                format="wav",
                streaming=False,
                use_memory_cache="on",
                seed=self._get_speaker_seed(segment_metadata.get("speaker", "A"))
            )
            
            # Generate audio
            logger.info(f"Generating voice clone for segment {segment_metadata.get('segment_index', 'unknown')}")
            
            audio_chunks = list(self.inference_engine.inference(tts_request))
            
            if not audio_chunks:
                return {"success": False, "error": "No audio generated"}
            
            # Combine audio chunks
            combined_audio = b''.join(audio_chunks)
            
            # Save cloned audio
            segment_index = segment_metadata.get("segment_index", 1)
            output_filename = f"cloned_segment_{segment_index:03d}.wav"
            output_path = Path(segment_metadata.get("audio_file", "")).parent / output_filename
            
            with open(output_path, 'wb') as f:
                f.write(combined_audio)
            
            # Process and match audio length
            try:
                target_duration = segment_metadata.get("duration", 1.0)
                matched_audio_path = self._match_audio_length(output_path, target_duration)
                
                audio_data, sample_rate = sf.read(str(matched_audio_path))
                final_duration = len(audio_data) / sample_rate
                
                return {
                    "success": True,
                    "cloned_audio_path": str(matched_audio_path),
                    "duration": final_duration,
                    "target_duration": target_duration,
                    "sample_rate": sample_rate,
                    "text_synthesized": cleaned_text,
                    "reference_used": reference_audio_path,
                    "model_used": "openaudio_s1_mini",
                    "length_matched": True
                }
                
            except Exception as e:
                return {"success": False, "error": f"Audio processing failed: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Voice cloning failed for segment: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _prepare_text_for_tts(self, text: str) -> str:
        """Prepare text for OpenAudio S1-mini TTS"""
        # OpenAudio S1-mini handles emotion markers like (happy), (sad), (sarcastic) etc.
        # Just clean up any problematic characters
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Ensure proper sentence endings
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def _get_speaker_seed(self, speaker_id: str) -> int:
        """Get consistent seed for speaker"""
        # Create consistent seed based on speaker ID for voice consistency
        seed_base = hash(speaker_id) % 1000000
        return abs(seed_base)
    
    def _match_audio_length(self, audio_path: Path, target_duration: float) -> Path:
        """Match cloned audio length to exact reference duration"""
        try:
            # Load cloned audio
            audio_data, sample_rate = sf.read(str(audio_path))
            current_duration = len(audio_data) / sample_rate
            
            # If already correct length (within 50ms tolerance), return as is
            if abs(current_duration - target_duration) <= 0.05:
                return audio_path
            
            target_samples = int(target_duration * sample_rate)
            
            # Crop if too long
            if len(audio_data) > target_samples:
                matched_audio = audio_data[:target_samples]
            
            # Pad with silence if too short
            elif len(audio_data) < target_samples:
                padding_needed = target_samples - len(audio_data)
                silence = np.zeros(padding_needed, dtype=audio_data.dtype)
                matched_audio = np.concatenate([audio_data, silence])
            
            else:
                matched_audio = audio_data
            
            # Save matched audio
            matched_path = audio_path.parent / f"matched_{audio_path.name}"
            sf.write(str(matched_path), matched_audio, sample_rate)
            
            # Replace original with matched
            audio_path.unlink()  # Delete original
            matched_path.rename(audio_path)  # Rename matched to original
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Failed to match audio length: {str(e)}")
            return audio_path  # Return original on failure

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.inference_engine:
                del self.inference_engine
            if self.llama_queue:
                del self.llama_queue  
            if self.decoder_model:
                del self.decoder_model
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("OpenAudio S1-mini service cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")


# Global instance - will be initialized when first used
_s1_mini_service = None
_service_lock = threading.Lock()


def get_fish_speech_service(device: str = "cuda") -> FishSpeechService:
    """Get or create OpenAudio S1-mini service instance"""
    global _s1_mini_service
    
    with _service_lock:
        if _s1_mini_service is None:
            _s1_mini_service = FishSpeechService(device=device)
        
        return _s1_mini_service


def initialize_fish_speech_service(device: str = "cuda") -> bool:
    """Initialize OpenAudio S1-mini service and return success status"""
    try:
        service = get_fish_speech_service(device)
        service._initialize_models()
        return service._is_initialized and service.inference_engine is not None
    except Exception as e:
        logger.error(f"Failed to initialize OpenAudio S1-mini service: {e}")
        return False



def set_seed(seed: int):
    """Set random seed for reproducible results"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



