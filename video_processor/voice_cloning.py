"""
Voice Cloning Module

Simple voice cloning using Dia model with direct text-to-speech approach.
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
    """Simple voice cloning service using Dia model"""
    
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
        
    def is_model_loaded(self) -> bool:
        """Check if Dia model is loaded"""
        return self.dia_model is not None 
    
    def clone_voice_segments(self, segments: List[Dict], temperature: float = 1.2,
                           cfg_scale: float = 3.0, top_p: float = 0.95,
                           seed: Optional[int] = None) -> Dict[str, Any]:
        """Simple voice cloning for segments"""
        if not self.is_model_loaded():
            return {"success": False, "error": "Dia model not loaded"}
        
        try:
            if seed is not None:
                set_seed(seed)
            
            cloned_segments = []
            
            # Process each segment directly - no complex grouping
            for segment in segments:
                cloned = self._clone_segment_direct(segment, temperature, cfg_scale, top_p)
                if cloned:
                    cloned_segments.append(cloned)
            
            return {
                "success": True,
                "cloned_segments": cloned_segments,
                "total_segments": len(segments),
                "successful_clones": len(cloned_segments)
            }
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _clone_segment_direct(self, segment: Dict, temperature: float,
                            cfg_scale: float, top_p: float) -> Optional[Dict]:
        """Clone a segment directly - clean approach"""
        try:
            # Get reference audio path
            reference_audio_path = segment.get('reference_audio_path')
            if not reference_audio_path or not os.path.exists(reference_audio_path):
                logger.warning(f"No valid reference audio found for segment {segment.get('start', 0):.2f}s-{segment.get('end', 0):.2f}s")
                return None
            
            # Get text for Dia (use dia_text if available, otherwise format from english_text)
            dia_text = segment.get('dia_text', '')
            if not dia_text:
                english_text = segment.get('english_text', segment.get('text', ''))
                dia_text = f"[S1] {english_text}"
            
            # Get reference transcript for voice cloning
            ref_transcript = self._get_reference_transcript(reference_audio_path)
            combined_text = f"{ref_transcript} {dia_text}"
            
            # Generate cloned audio using Dia model
            cloned_audio = self.dia_model.generate(
                combined_text,
                audio_prompt=reference_audio_path,
                use_torch_compile=False,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=45,
                max_tokens=2048,
                verbose=False  # Reduce noise
            )
            
            # Adjust audio length to match original duration
            original_duration = segment.get('duration', 0)
            if original_duration > 0 and len(cloned_audio) > 0:
                cloned_audio = self._adjust_audio_length(cloned_audio, original_duration)
            
            return {
                "success": True,
                "original_data": segment,
                "cloned_audio": cloned_audio,
                "type": "direct",
                "reference_used": reference_audio_path
            }
            
        except Exception as e:
            logger.error(f"Failed to clone segment {segment.get('start', 0):.2f}s-{segment.get('end', 0):.2f}s: {str(e)}")
            return None
    
    def _adjust_audio_length(self, audio: np.ndarray, target_duration: float) -> np.ndarray:
        """Adjust cloned audio to match original duration exactly"""
        if len(audio) == 0:
            return audio
            
        sample_rate = 44100  # Standard sample rate
        target_samples = int(target_duration * sample_rate)
        current_samples = len(audio)
        
        if current_samples == target_samples:
            return audio
        
        # Use interpolation to stretch/compress audio to exact target length
        indices = np.linspace(0, current_samples - 1, target_samples)
        stretched_audio = np.interp(indices, np.arange(current_samples), audio)
        
        return stretched_audio
    
    def _get_reference_transcript(self, reference_path: str) -> str:
        """Get reference transcript from metadata"""
        try:
            # Look for metadata file
            ref_path = Path(reference_path)
            metadata_path = ref_path.parent / f"{ref_path.stem}_metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Use dia_text if available, otherwise format from english_text
                dia_text = metadata.get('dia_text', '')
                if dia_text:
                    return dia_text
                
                english_text = metadata.get('english_text', metadata.get('text', ''))
                if english_text:
                    return f'[S1] {english_text}'
            
            # Fallback to generic reference
            return '[S1] Reference audio.'
            
        except Exception as e:
            logger.warning(f"Failed to get reference transcript: {str(e)}")
            return '[S1] Reference audio.' 