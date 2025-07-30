"""
Fish Speech Voice Cloning Service - Maximum Quality TTS
Auto-downloads models and provides voice cloning functionality
"""

import os
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import tempfile
import shutil

import torch
import soundfile as sf
import numpy as np
from huggingface_hub import hf_hub_download

from config import settings

logger = logging.getLogger(__name__)


class FishSpeechService:
    """Fish Speech TTS service with auto model download and voice cloning"""
    
    def __init__(self, device: str = "cuda", use_compile: bool = True):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_compile = use_compile
        self.precision = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        # Model paths
        self.models_dir = Path("./fish_speech_models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Use full S1 model for maximum quality (not mini)
        self.model_repo = "fishaudio/openaudio-s1"  # Full model, not mini
        self.model_local_path = self.models_dir / "openaudio-s1"
        
        # Initialize as None, will be loaded lazily
        self.inference_engine = None
        self.llama_queue = None
        self.decoder_model = None
        self._model_lock = threading.Lock()
        self._is_initialized = False
        
        logger.info(f"FishSpeechService initialized with device: {self.device}")
    
    def _ensure_fish_speech_available(self):
        """Clone Fish Speech if needed"""
        fish_speech_dir = Path("./fish-speech")
        
        if not fish_speech_dir.exists():
            import subprocess
            result = subprocess.run([
                "git", "clone", "https://github.com/fishaudio/fish-speech.git"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Git clone failed: {result.stderr}")
        
        # Add to Python path
        import sys
        if str(fish_speech_dir) not in sys.path:
            sys.path.insert(0, str(fish_speech_dir))
    
    def _download_models(self):
        """Download Fish Speech models automatically"""
        model_files = [
            "model.pth", "codec.pth", "config.json", 
            "special_tokens.json", "tokenizer.tiktoken"
        ]
        
        self.model_local_path.mkdir(parents=True, exist_ok=True)
        
        for file_name in model_files:
            local_file_path = self.model_local_path / file_name
            
            if not local_file_path.exists():
                hf_hub_download(
                    repo_id=self.model_repo,
                    filename=file_name,
                    local_dir=str(self.model_local_path),
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
    
    def _initialize_models(self):
        """Initialize Fish Speech models completely"""
        with self._model_lock:
            if self._is_initialized:
                return
            
            logger.info("🔄 Cloning Fish Speech repository...")
            self._ensure_fish_speech_available()
            
            logger.info("📥 Downloading OpenAudio S1 models...")
            self._download_models()
            
            logger.info("🚀 Loading Fish Speech models...")
            
            # Import Fish Speech modules
            import pyrootutils
            fish_speech_root = Path("./fish-speech")
            pyrootutils.setup_root(str(fish_speech_root / "__init__.py"), indicator=".project-root", pythonpath=True)
            
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.dac.inference import load_model as load_decoder_model
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
            
            # Load models
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=self.model_local_path,
                device=self.device,
                precision=self.precision,
                compile=self.use_compile,
            )
            
            self.decoder_model = load_decoder_model(
                config_name="modded_dac_vq",
                checkpoint_path=str(self.model_local_path / "codec.pth"),
                device=self.device,
            )
            
            self.inference_engine = TTSInferenceEngine(
                llama_queue=self.llama_queue,
                decoder_model=self.decoder_model,
                compile=self.use_compile,
                precision=self.precision,
            )
            
            # Warm up
            self._warmup_model()
            
            self._is_initialized = True
            logger.info("✅ Fish Speech ready for voice cloning!")
    
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
        """Clone voice for a specific segment using Fish Speech"""
        try:
            
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
            
            # Verify audio file
            try:
                audio_data, sample_rate = sf.read(str(output_path))
                duration = len(audio_data) / sample_rate
                
                return {
                    "success": True,
                    "cloned_audio_path": str(output_path),
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "text_synthesized": cleaned_text,
                    "reference_used": reference_audio_path,
                    "model_used": "fish_speech_openaudio_s1"
                }
                
            except Exception as e:
                return {"success": False, "error": f"Generated audio verification failed: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Voice cloning failed for segment: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _prepare_text_for_tts(self, text: str) -> str:
        """Prepare text for Fish Speech TTS"""
        # Fish Speech already handles emotion markers like (happy), (sad) etc.
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
                
            logger.info("Fish Speech service cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")


# Global instance - will be initialized when first used
_fish_speech_service = None
_service_lock = threading.Lock()


def get_fish_speech_service(device: str = "cuda") -> FishSpeechService:
    """Get or create Fish Speech service instance"""
    global _fish_speech_service
    
    with _service_lock:
        if _fish_speech_service is None:
            _fish_speech_service = FishSpeechService(device=device)
        
        return _fish_speech_service



def set_seed(seed: int):
    """Set random seed for reproducible results"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)



