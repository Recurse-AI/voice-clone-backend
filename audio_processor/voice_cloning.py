"""
Voice Cloning Module

Handles Dia model operations and voice cloning functionality.
"""

import torch
import numpy as np
import random
from typing import Optional, Dict, Any, List
from transformers import AutoProcessor, DiaForConditionalGeneration
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
        self.dia_processor = None
        self.device = settings.DIA_DEVICE
        
        # Dia generation parameters
        self.max_new_tokens = 3072
        self.guidance_scale = 3.0
        self.temperature = 1.8
        self.top_p = 0.90
        self.top_k = 45
    
    def load_dia_model(self, repo_id: str = "nari-labs/Dia-1.6B-0626") -> bool:
        """Load Dia model using transformers"""
        try:
            self.dia_processor = AutoProcessor.from_pretrained(repo_id)
            self.dia_model = DiaForConditionalGeneration.from_pretrained(repo_id).to(self.device)
            return True
        except Exception as e:
            raise Exception(f"Error loading Dia model: {e}")
    
    def generate_with_dia(self, text: str, temperature: float, cfg_scale: float, 
                         top_p: float, audio_prompt_path: Optional[str] = None,
                         reference_text: Optional[str] = None) -> Optional[np.ndarray]:
        """Generate audio with Dia model following official voice cloning pattern"""
        try:
            # Follow official Dia voice cloning format
            if audio_prompt_path and reference_text:
                # Voice cloning: clone_from_text + text_to_generate
                # The model will only return audio from text_to_generate
                full_text = reference_text + text
            else:
                # Regular generation without cloning
                full_text = text
            
            inputs = self.dia_processor(
                text=[full_text], 
                padding=True, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.dia_model.generate(
                    **inputs,
                    audio_prompt=audio_prompt_path,
                    max_new_tokens=self.max_new_tokens,
                    guidance_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=self.top_k,
                    do_sample=True
                )
            
            generated_audio = self.dia_processor.batch_decode(outputs)[0]
            return generated_audio
            
        except Exception as e:
            return None
    
    def clone_voice_segments(self, segments_data: List[Dict[str, Any]], 
                           temperature: float = 1.3, cfg_scale: float = 3.0, 
                           top_p: float = 0.95, seed: Optional[int] = None) -> Dict[str, Any]:
        """Clone voice segments using Dia model"""
        if not self.dia_model:
            return {"success": False, "error": "Dia model not loaded"}
        
        try:
            if seed is not None:
                set_seed(seed)
            
            cloned_segments = []
            
            for segment_data in segments_data:
                # Generate with Dia using proper voice cloning format
                dia_text = segment_data.get('dia_text', '')
                reference_text = segment_data.get('reference_text', '')
                reference_audio_path = segment_data.get('reference_audio_path', None)
                
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
                        "success": False
                    })
            
            return {
                "success": True,
                "cloned_segments": cloned_segments
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def is_model_loaded(self) -> bool:
        """Check if Dia model is loaded"""
        return self.dia_model is not None and self.dia_processor is not None 