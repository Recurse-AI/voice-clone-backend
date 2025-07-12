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
import logging

logger = logging.getLogger(__name__)

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
            logger.info(f"Dia model loaded successfully from {repo_id}")
            return True
        except Exception as e:
            logger.error(f"Error loading Dia model: {e}")
            raise Exception(f"Error loading Dia model: {e}")
    
    def generate_with_dia(self, text: str, temperature: float = None, cfg_scale: float = None, 
                         top_p: float = None, audio_prompt_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Generate audio with Dia model using simple, direct approach"""
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
            
            # Simple text formatting - just ensure it starts with [S1]
            if not text.strip().startswith('[S'):
                text = f"[S1] {text.strip()}"
            
            logger.info(f"Generating audio with text: {text[:100]}...")
            
            if audio_prompt_path:
                # Voice cloning with reference audio
                output = self.dia_model.generate(text, audio_prompt=audio_prompt_path, **generation_params)
            else:
                # Regular generation without audio prompt
                output = self.dia_model.generate(text, **generation_params)
            
            if output is not None and len(output) > 0:
                logger.info(f"Audio generated successfully, length: {len(output)}")
                return output
            else:
                logger.warning("Dia model returned empty audio")
                return None
            
        except Exception as e:
            logger.error(f"Dia generation failed: {str(e)}")
            return None
    
    def clone_voice_segments(self, segments_data: List[Dict[str, Any]], 
                           temperature: float = None, cfg_scale: float = None, 
                           top_p: float = None, seed: Optional[int] = None) -> Dict[str, Any]:
        """Clone voice segments using Dia model with simplified approach"""
        if not self.dia_model:
            return {"success": False, "error": "Dia model not loaded"}
        
        try:
            if seed is not None:
                set_seed(seed)
                logger.info(f"Set random seed to {seed}")
            
            cloned_segments = []
            successful_clones = 0
            
            for i, segment_data in enumerate(segments_data):
                try:
                    # Get the text to clone
                    text = segment_data.get('english_text', segment_data.get('text', ''))
                    reference_audio_path = segment_data.get('reference_audio_path', None)
                    
                    if not text.strip():
                        logger.warning(f"Empty text for segment {i}")
                        cloned_segments.append({
                            "original_data": segment_data,
                            "cloned_audio": None,
                            "text": text,
                            "success": False,
                            "error": "Empty text"
                        })
                        continue
                    
                    # Clean and format text for Dia
                    clean_text = self._clean_text_for_dia(text)
                    
                    logger.info(f"Processing segment {i+1}/{len(segments_data)}: {clean_text[:50]}...")
                    
                    # Generate with Dia
                    cloned_audio = self.generate_with_dia(
                        clean_text, temperature, cfg_scale, top_p, reference_audio_path
                    )
                    
                    if cloned_audio is not None and len(cloned_audio) > 0:
                        cloned_segments.append({
                            "original_data": segment_data,
                            "cloned_audio": cloned_audio,
                            "text": clean_text,
                            "success": True
                        })
                        successful_clones += 1
                        logger.info(f"Successfully cloned segment {i+1}")
                    else:
                        logger.warning(f"Failed to generate audio for segment {i+1}")
                        cloned_segments.append({
                            "original_data": segment_data,
                            "cloned_audio": None,
                            "text": clean_text,
                            "success": False,
                            "error": "Generated audio is None or empty"
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing segment {i+1}: {str(e)}")
                    cloned_segments.append({
                        "original_data": segment_data,
                        "cloned_audio": None,
                        "text": segment_data.get('text', ''),
                        "success": False,
                        "error": str(e)
                    })
            
            logger.info(f"Voice cloning completed: {successful_clones}/{len(segments_data)} segments successful")
            
            return {
                "success": True,
                "cloned_segments": cloned_segments,
                "successful_count": successful_clones,
                "total_count": len(segments_data)
            }
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _clean_text_for_dia(self, text: str) -> str:
        """Clean text for Dia model - keep it simple"""
        # Remove excessive punctuation
        import re
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Remove speaker tags if present and re-add properly
        text = re.sub(r'\[S\d+\]\s*', '', text)
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def is_model_loaded(self) -> bool:
        """Check if Dia model is loaded"""
        return self.dia_model is not None 