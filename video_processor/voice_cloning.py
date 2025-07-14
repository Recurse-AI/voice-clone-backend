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
            self.dia_model = Dia.from_pretrained(repo_id, compute_dtype=compute_dtype)
            return True
        except Exception as e:
            raise Exception(f"Error loading Dia model: {e}")
        
    def is_model_loaded(self) -> bool:
        """Check if Dia model is loaded"""
        return self.dia_model is not None
    
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
                
                print(f"Successfully generated audio for segment {i+1}")
                
                target_duration = segment.get('duration', 5.0)
                cloned_audio = self._adjust_audio_length(cloned_audio, target_duration)
                
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
            
            with torch.inference_mode():
                if reference_audio_path and os.path.exists(reference_audio_path):
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
            
            return audio
            
        except Exception as e:
            print(f"Audio generation failed: {str(e)}")
            return None
    
    def _adjust_audio_length(self, audio: np.ndarray, target_duration: float) -> np.ndarray:
        """Adjust audio to match target duration using time-stretch for small differences"""
        if len(audio) == 0:
            return audio

        target_samples = int(target_duration * self.sample_rate)
        current_samples = len(audio)

        if current_samples == target_samples:
            return audio

        if current_samples < target_samples:
            ratio = target_samples / current_samples
            if ratio <= 1.2:
                audio_float = audio.astype(np.float32)
                stretched = librosa.effects.time_stretch(audio_float, 1/ratio)
                if len(stretched) > target_samples:
                    stretched = stretched[:target_samples]
                else:
                    padding = target_samples - len(stretched)
                    stretched = np.pad(stretched, (0, padding), mode='constant', constant_values=0)
                return stretched
            else:
                padding = target_samples - current_samples
                return np.pad(audio, (0, padding), mode='constant', constant_values=0)

        # audio longer than target: truncate with fade out
        audio = audio[:target_samples]
        fade_samples = int(0.1 * self.sample_rate)
        if len(audio) > fade_samples:
            fade_curve = np.linspace(1, 0, fade_samples)
            audio[-fade_samples:] *= fade_curve
        return audio
    
    def _cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def clear_cache(self):
        """Clear any cached data"""
        self._cleanup_memory() 