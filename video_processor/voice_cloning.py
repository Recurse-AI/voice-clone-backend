"""
Voice Cloning Module - Simplified for Dia
"""

import torch
import numpy as np
import random
import os
import gc
import soundfile as sf
from typing import Optional, Dict, Any, List
from dia.model import Dia
from config import settings
import logging
from pathlib import Path
import time
import json
import librosa

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VoiceCloningService:
    """Simplified voice cloning service"""
    
    def __init__(self):
        self.dia_model = None
        self.device = settings.DIA_DEVICE
        self.sample_rate = 44100
    
    def load_dia_model(self, repo_id: str = None) -> bool:
        """Load Dia model"""
        try:
            repo_id = repo_id or settings.DIA_MODEL_REPO
            compute_dtype = settings.DIA_COMPUTE_DTYPE if self.device == "cuda" else "float32"
            print(f"Loading Dia model from repo: {repo_id}")
            print(f"Device: {self.device}, Compute dtype: {compute_dtype}")
            
            self.dia_model = Dia.from_pretrained(repo_id, compute_dtype=compute_dtype)
            print("Dia model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading Dia model: {e}")
            raise Exception(f"Error loading Dia model: {e}")
        
    def is_model_loaded(self) -> bool:
        """Check if Dia model is loaded"""
        is_loaded = self.dia_model is not None
        print(f"Model loaded check: {is_loaded}")
        return is_loaded
    
    def clone_voice_segments(self, segments: List[Dict], temperature: float = 1.2,
                           cfg_scale: float = 3.0, top_p: float = 0.95,
                           seed: Optional[int] = None) -> Dict[str, Any]:
        if not self.is_model_loaded():
            return {"success": False, "error": "Dia model not loaded"}
        
        if not segments:
            return {"success": False, "error": "No segments provided"}
        
        print(f"Starting voice cloning for {len(segments)} segments")
        
        try:
            used_seed = seed or settings.DEFAULT_SEED
            set_seed(used_seed)
            
            reference_audio_path = None
            reference_text = None
            if segments and segments[0].get('reference_audio_path'):
                reference_audio_path = segments[0]['reference_audio_path']
                reference_text = self._load_reference_text(reference_audio_path)
            
            print(f"Reference Audio: {reference_audio_path if reference_audio_path else 'None'}")
            print(f"Reference Text: {reference_text if reference_text else 'None'}")
            
            cloning_start_time = time.time()
            
            cloned_segments = []
            for i, segment in enumerate(segments):
                english_text = segment.get('english_text', segment.get('text', ''))
                if not english_text.strip():
                    print(f"Skipping segment {i+1}: No english_text")
                    continue
                
                combined_display = (reference_text + '\n' + english_text) if reference_text else english_text
                print(f"Combined Text: {combined_display}")
                
                print(f"Processing segment {i+1}...")
                cloned_audio = self._generate_single_segment(
                    english_text, reference_audio_path, reference_text, 
                    temperature, cfg_scale, top_p
                )
                
                if cloned_audio is None:
                    print(f"Skipping segment {i+1}: Failed to generate audio")
                    continue
                
                print(f"Generated audio shape: {cloned_audio.shape if hasattr(cloned_audio, 'shape') else 'No shape'}")
                
                target_duration = segment.get('duration', 5.0)
                cloned_audio = self._adjust_audio_length(
                    cloned_audio, 
                    target_duration, 
                    use_speed_adjustment=settings.USE_SPEED_ADJUSTMENT,
                    speed_factor=settings.AUDIO_SPEED_FACTOR
                )
                
                print(f"Adjusted audio shape: {cloned_audio.shape if hasattr(cloned_audio, 'shape') else 'No shape'}")
                print(f"Successfully generated audio for segment {i+1}")
                
                cloned_segments.append({
                    "success": True,
                    "original_data": segment,
                    "cloned_audio": cloned_audio,
                    "duration": target_duration
                })
                
                self._cleanup_memory()
            
            cloning_end_time = time.time()
            cloning_duration = cloning_end_time - cloning_start_time
            
            print(f"Cloning Time: {cloning_duration:.2f} seconds")
            
            return {
                "success": True,
                "cloned_segments": cloned_segments,
                "total_segments": len(segments),
                "successful_clones": len(cloned_segments),
                "seed_used": used_seed,
                "cloning_duration": cloning_duration
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            self._cleanup_memory()
    
    def _load_reference_text(self, reference_audio_path: str) -> Optional[str]:
        """Load reference text from metadata"""
        try:
            reference_file = Path(reference_audio_path)
            metadata_file = reference_file.parent / f"{reference_file.stem}_metadata.json"
            
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                reference_metadata = json.load(f)
            
            return reference_metadata.get('english_text', '')
            
        except Exception:
            return None
    
    def _generate_single_segment(self, text: str, reference_audio_path: Optional[str], 
                               reference_text: Optional[str], temperature: float, 
                               cfg_scale: float, top_p: float) -> Optional[np.ndarray]:
        try:
            if reference_text and reference_text.strip():
                combined_text = reference_text.strip() + "\n" + text.strip()
            else:
                combined_text = text.strip()
            
            if not combined_text:
                print("Error: Combined text is empty")
                return None
            
            print(f"Generating audio for text: {combined_text[:100]}...")
            
            with torch.inference_mode():
                if reference_audio_path and os.path.exists(reference_audio_path):
                    print(f"Using reference audio: {reference_audio_path}")
                    audio = self.dia_model.generate(
                        text=combined_text,
                        audio_prompt=reference_audio_path,
                        use_torch_compile=False,
                        cfg_scale=cfg_scale,
                        temperature=temperature,
                        top_p=top_p,
                        cfg_filter_top_k=settings.DIA_CFG_FILTER_TOP_K,
                        max_tokens=settings.DIA_MAX_TOKENS,
                        verbose=False
                    )
                else:
                    print("Using standard generation (no reference)")
                    audio = self.dia_model.generate(
                        text=combined_text,
                        use_torch_compile=False,
                        cfg_scale=cfg_scale,
                        temperature=temperature,
                        top_p=top_p,
                        cfg_filter_top_k=settings.DIA_CFG_FILTER_TOP_K,
                        max_tokens=settings.DIA_MAX_TOKENS,
                        verbose=False
                    )
            
            print(f"Audio generation completed, shape: {audio.shape if hasattr(audio, 'shape') else 'No shape'}")
            return audio
            
        except Exception as e:
            print(f"Audio generation failed: {str(e)}")
            return None
    
    def _adjust_audio_length(self, audio: np.ndarray, target_duration: float, 
                          use_speed_adjustment: bool = False, speed_factor: float = 0.75) -> np.ndarray:
        """Simple audio length adjustment - padding/truncation or optional speed adjustment"""
        if audio is None:
            print("Warning: Audio is None")
            return np.zeros(int(target_duration * self.sample_rate), dtype=np.float32)
        
        if not isinstance(audio, np.ndarray):
            print(f"Warning: Audio is not numpy array, type: {type(audio)}")
            try:
                audio = np.array(audio, dtype=np.float32)
            except:
                print("Error: Could not convert audio to numpy array")
                return np.zeros(int(target_duration * self.sample_rate), dtype=np.float32)
        
        if len(audio) == 0:
            print("Warning: Audio is empty")
            return np.zeros(int(target_duration * self.sample_rate), dtype=np.float32)

        # Ensure audio is float32 for processing
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # If audio is 2D, take first channel
        if len(audio.shape) > 1:
            audio = audio[:, 0] if audio.shape[1] > 0 else audio.flatten()

        target_samples = int(target_duration * self.sample_rate)
        current_samples = len(audio)

        print(f"Adjusting audio from {current_samples} samples to {target_samples} samples")
        
        # Option 1: Simple padding/truncation (default - no stretching)
        if not use_speed_adjustment:
            if current_samples > target_samples:
                # Truncate if audio is longer
                adjusted_audio = audio[:target_samples]
                print(f"Truncated audio to {len(adjusted_audio)} samples")
            elif current_samples < target_samples:
                # Pad with zeros if audio is shorter
                padding_needed = target_samples - current_samples
                adjusted_audio = np.pad(audio, (0, padding_needed), mode='constant', constant_values=0)
                print(f"Padded audio with {padding_needed} zero samples")
            else:
                # Already correct length
                adjusted_audio = audio
                print("Audio already correct length")
        
        # Option 2: Speed adjustment (slower) - optional
        else:
            print(f"Using speed adjustment with factor {speed_factor}")
            try:
                # Make audio slower by the speed factor
                adjusted_audio = librosa.effects.time_stretch(y=audio, rate=speed_factor)
                print(f"Applied speed factor: {speed_factor}, new length: {len(adjusted_audio)} samples")
                
                # Still pad/truncate to exact target if needed
                if len(adjusted_audio) > target_samples:
                    adjusted_audio = adjusted_audio[:target_samples]
                elif len(adjusted_audio) < target_samples:
                    padding_needed = target_samples - len(adjusted_audio)
                    adjusted_audio = np.pad(adjusted_audio, (0, padding_needed), mode='constant', constant_values=0)
                    
            except Exception as e:
                print(f"Speed adjustment failed: {e}, falling back to padding")
                # Fallback to padding
                if current_samples > target_samples:
                    adjusted_audio = audio[:target_samples]
                elif current_samples < target_samples:
                    padding_needed = target_samples - current_samples
                    adjusted_audio = np.pad(audio, (0, padding_needed), mode='constant', constant_values=0)
                else:
                    adjusted_audio = audio
        
        return adjusted_audio.astype(np.float32)
    
    def _cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def clear_cache(self):
        """Clear any cached data"""
        self._cleanup_memory() 