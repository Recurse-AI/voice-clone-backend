"""
Voice Cloning Module

Handles Dia model operations and voice cloning functionality.
"""

import torch
import numpy as np
import random
from typing import Optional, Dict, Any, List
from dia.model import Dia
from config import settings


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VoiceCloningService:
    """Service for voice cloning using Dia model"""
    
    def __init__(self):
        self.dia_model = None
        self.device = settings.DIA_DEVICE
    
    def load_dia_model(self, repo_id: str = None) -> bool:
        """Load Dia model using official API"""
        try:
            repo_id = repo_id or settings.DIA_MODEL_REPO
            compute_dtype = settings.DIA_COMPUTE_DTYPE if self.device == "cuda" else "float32"
            self.dia_model = Dia.from_pretrained(repo_id, compute_dtype=compute_dtype)
            return True
        except Exception as e:
            raise Exception(f"Error loading Dia model: {e}")
    
    def generate_with_dia(self, text: str, temperature: float = None, cfg_scale: float = None, 
                         top_p: float = None, audio_prompt_path: Optional[str] = None,
                         reference_text: Optional[str] = None) -> Optional[np.ndarray]:
        """Generate audio with Dia model using official API"""
        try:
            # Use config defaults if parameters not provided
            generation_params = {
                "use_torch_compile": settings.DIA_USE_TORCH_COMPILE,
                "verbose": settings.DIA_VERBOSE,
                "cfg_scale": cfg_scale or settings.DIA_CFG_SCALE,
                "temperature": temperature or settings.DIA_TEMPERATURE,
                "top_p": top_p or settings.DIA_TOP_P,
                "cfg_filter_top_k": settings.DIA_CFG_FILTER_TOP_K,
                "max_tokens": settings.DIA_MAX_TOKENS
            }
            
            if audio_prompt_path and reference_text:
                # Voice cloning: clone_from_text + text_to_generate
                full_text = reference_text + text
                output = self.dia_model.generate(full_text, audio_prompt=audio_prompt_path, **generation_params)
            else:
                # Regular generation without audio prompt
                output = self.dia_model.generate(text, **generation_params)
            
            return output
            
        except Exception as e:
            return None
    
    def clone_voice_segments(self, segments_data: List[Dict[str, Any]], 
                           temperature: float = None, cfg_scale: float = None, 
                           top_p: float = None, seed: Optional[int] = None) -> Dict[str, Any]:
        """Clone voice segments using Dia model"""
        if not self.dia_model:
            return {"success": False, "error": "Dia model not loaded"}
        
        try:
            if seed is not None:
                set_seed(seed)
            
            cloned_segments = []
            
            for i, segment_data in enumerate(segments_data):
                try:
                    # Generate with Dia using proper voice cloning format
                    dia_text = segment_data.get('dia_text', '')
                    reference_text = segment_data.get('reference_text', '')
                    reference_audio_path = segment_data.get('reference_audio_path', None)
                    
                    if not dia_text:
                        cloned_segments.append({
                            "original_data": segment_data,
                            "cloned_audio": None,
                            "text": dia_text,
                            "success": False,
                            "error": "No dia_text found"
                        })
                        continue
                    
                    cloned_audio = self.generate_with_dia(
                        dia_text, temperature, cfg_scale, top_p, 
                        reference_audio_path, reference_text
                    )
                    
                    if cloned_audio is not None:
                        cloned_segments.append({
                            "original_data": segment_data,
                            "cloned_audio": cloned_audio,
                            "text": dia_text,
                            "success": True
                        })
                    else:
                        cloned_segments.append({
                            "original_data": segment_data,
                            "cloned_audio": None,
                            "text": dia_text,
                            "success": False,
                            "error": "Generated audio is None"
                        })
                        
                except Exception as e:
                    cloned_segments.append({
                        "original_data": segment_data,
                        "cloned_audio": None,
                        "text": segment_data.get('dia_text', ''),
                        "success": False,
                        "error": str(e)
                    })
            
            return {
                "success": True,
                "cloned_segments": cloned_segments
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def is_model_loaded(self) -> bool:
        """Check if Dia model is loaded"""
        return self.dia_model is not None 