"""
Voice Cloning Module

Simple and efficient voice cloning using Dia model with improved segment processing.
"""

import torch
import numpy as np
import random
import os
import json
import soundfile as sf
from typing import Optional, Dict, Any, List
from dia.model import Dia
from config import settings
import logging
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VoiceCloningService:
    """Simple and efficient voice cloning service"""
    
    def __init__(self):
        self.dia_model = None
        self.device = settings.DIA_DEVICE
        self.sample_rate = 44100
    
    def load_dia_model(self, repo_id: str = None) -> bool:
        """Load Dia model using official API"""
        try:
            repo_id = repo_id or settings.DIA_MODEL_REPO
            compute_dtype = settings.DIA_COMPUTE_DTYPE if self.device == "cuda" else "float32"
            self.dia_model = Dia.from_pretrained(repo_id, compute_dtype=compute_dtype)
            return True
        except Exception as e:
            raise Exception(f"Error loading Dia model: {e}")
        
    def is_model_loaded(self) -> bool:
        """Check if Dia model is loaded"""
        return self.dia_model is not None 
    
    def clone_voice_segments(self, segments: List[Dict], temperature: float = 1.2,
                           cfg_scale: float = 3.0, top_p: float = 0.95,
                           seed: Optional[int] = None, speed_factor: float = 0.92) -> Dict[str, Any]:
        """Improved voice cloning with proper speed control and reference handling"""
        if not self.is_model_loaded():
            return {"success": False, "error": "Dia model not loaded"}
        
        if not segments:
            return {"success": False, "error": "No segments provided"}
        
        try:
            # Set seed for consistency - use constant from settings
            set_seed(seed or settings.DEFAULT_SEED)
            
            cloned_segments = []
            reference_audio_path = None
            
            # Get reference audio from first segment
            if segments and segments[0].get('reference_audio_path'):
                reference_audio_path = segments[0]['reference_audio_path']
                logger.info(f"Using reference audio: {reference_audio_path}")
            
            # Process each segment individually for better quality
            for segment in segments:
                try:
                    # Get segment text
                    english_text = segment.get('english_text', segment.get('text', ''))
                    if not english_text.strip():
                        continue
                    
                    # Create proper Dia text format
                    speaker = segment.get('speaker', 'A')
                    if not english_text.startswith('[S'):
                        dia_text = f"[S1] {english_text.strip()}"
                    else:
                        dia_text = english_text.strip()
                    
                    # Generate audio with reference
                    cloned_audio = self._generate_single_segment(
                        dia_text, reference_audio_path, temperature, 
                        cfg_scale, top_p, speed_factor
                    )
                    
                    if cloned_audio is not None:
                        # Adjust length to match original duration
                        target_duration = segment.get('duration', 5.0)
                        cloned_audio = self._adjust_audio_length(cloned_audio, target_duration)
                        
                        cloned_segments.append({
                            "success": True,
                            "original_data": segment,
                            "cloned_audio": cloned_audio,
                            "type": "single_segment",
                            "duration": target_duration
                        })
                    
                except Exception as e:
                    logger.warning(f"Failed to clone segment: {str(e)}")
                    continue
            
            return {
                "success": True,
                "cloned_segments": cloned_segments,
                "total_segments": len(segments),
                "successful_clones": len(cloned_segments)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_single_segment(self, text: str, reference_audio_path: Optional[str],
                               temperature: float, cfg_scale: float, top_p: float,
                               speed_factor: float) -> Optional[np.ndarray]:
        """Generate audio for a single segment with proper reference handling"""
        try:
            # Generate audio using Dia model
            with torch.inference_mode():
                if reference_audio_path and os.path.exists(reference_audio_path):
                    # Use reference audio for voice cloning
                    audio = self.dia_model.generate(
                        text,
                        audio_prompt=reference_audio_path,
                        use_torch_compile=False,
                        cfg_scale=cfg_scale,
                        temperature=temperature,
                        top_p=top_p,
                        cfg_filter_top_k=45,
                        max_tokens=3072,
                        verbose=False
                    )
                else:
                    # Generate without reference
                    audio = self.dia_model.generate(
                        text,
                        use_torch_compile=False,
                        cfg_scale=cfg_scale,
                        temperature=temperature,
                        top_p=top_p,
                        cfg_filter_top_k=45,
                        max_tokens=3072,
                        verbose=False
                    )
            
            # Apply speed factor if needed
            if speed_factor != 1.0:
                audio = self._apply_speed_factor(audio, speed_factor)
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio generation failed: {str(e)}")
            return None
    
    def _apply_speed_factor(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
        """Apply speed adjustment to audio"""
        if speed_factor == 1.0:
            return audio
        
        new_length = int(len(audio) / speed_factor)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)
    
    def _adjust_audio_length(self, audio: np.ndarray, target_duration: float) -> np.ndarray:
        """Adjust audio to match target duration"""
        if len(audio) == 0:
            return audio
            
        target_samples = int(target_duration * self.sample_rate)
        current_samples = len(audio)
        
        if current_samples == target_samples:
            return audio
        
        if current_samples > target_samples:
            # Truncate with fade out
            audio = audio[:target_samples]
            # Apply fade out to last 0.1 seconds
            fade_samples = int(0.1 * self.sample_rate)
            if len(audio) > fade_samples:
                fade_curve = np.linspace(1, 0, fade_samples)
                audio[-fade_samples:] *= fade_curve
        else:
            # Pad with silence
            padding = target_samples - current_samples
            audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
        
        return audio 