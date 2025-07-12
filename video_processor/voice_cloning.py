"""
Voice Cloning Module

Handles Dia model operations and voice cloning functionality.
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
                         top_p: float = None, audio_prompt_path: Optional[str] = None,
                         audio_prompt_transcript: Optional[str] = None) -> Optional[np.ndarray]:
        """Generate audio with Dia model using official approach with transcript support"""
        try:
            # Use optimized parameters based on official examples
            generation_params = {
                "use_torch_compile": settings.DIA_USE_TORCH_COMPILE,
                "verbose": settings.DIA_VERBOSE,
                "cfg_scale": cfg_scale or 4.0,  # Optimized from official examples
                "temperature": temperature or 1.8,  # Optimized from official examples  
                "top_p": top_p or 0.90,  # Optimized from official examples
                "cfg_filter_top_k": 50,  # Optimized from official examples
                "max_tokens": settings.DIA_MAX_TOKENS
            }
            
            # Prepare text according to official guidelines
            if audio_prompt_path and audio_prompt_transcript:
                # Voice cloning with reference audio - combine transcript + new text
                # This is the key fix: we need to provide transcript of audio prompt
                full_text = f"{audio_prompt_transcript.strip()} {text.strip()}"
                logger.info(f"Voice cloning with transcript: {audio_prompt_transcript[:50]}... + {text[:50]}...")
            else:
                # Regular generation - just ensure proper speaker tags
                full_text = text
                logger.info(f"Regular generation with text: {text[:50]}...")
            
            # Format text according to Dia guidelines
            formatted_text = self._format_text_for_dia(full_text)
            
            logger.info(f"Final formatted text: {formatted_text[:100]}...")
            
            if audio_prompt_path and os.path.exists(audio_prompt_path):
                # Voice cloning with reference audio
                output = self.dia_model.generate(formatted_text, audio_prompt=audio_prompt_path, **generation_params)
            else:
                # Regular generation without audio prompt
                output = self.dia_model.generate(formatted_text, **generation_params)
            
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
        """Clone voice segments using Dia model with proper transcript support"""
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
                    
                    # Get audio prompt transcript (this is the key fix)
                    audio_prompt_transcript = self._get_audio_prompt_transcript(
                        reference_audio_path, segment_data
                    )
                    
                    logger.info(f"Processing segment {i+1}/{len(segments_data)}: {clean_text[:50]}...")
                    if audio_prompt_transcript:
                        logger.info(f"Using audio prompt transcript: {audio_prompt_transcript[:50]}...")
                    
                    # Generate with Dia using proper transcript approach
                    cloned_audio = self.generate_with_dia(
                        clean_text, temperature, cfg_scale, top_p, 
                        reference_audio_path, audio_prompt_transcript
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
    
    def _get_audio_prompt_transcript(self, reference_audio_path: Optional[str], 
                                   segment_data: Dict[str, Any]) -> Optional[str]:
        """Get transcript for audio prompt according to official Dia guidelines"""
        if not reference_audio_path or not os.path.exists(reference_audio_path):
            return None
        
        try:
            # Get reference audio directory and find reference metadata
            ref_audio_path = Path(reference_audio_path)
            ref_dir = ref_audio_path.parent
            
            # Look for reference metadata file
            for metadata_file in ref_dir.glob("*REFERENCE_metadata.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        ref_metadata = json.load(f)
                    
                    # Use the clean English text from reference metadata
                    ref_text = ref_metadata.get('text', ref_metadata.get('original_text', ''))
                    
                    if ref_text.strip():
                        # Format for Dia - simple and clean
                        formatted_ref = self._format_text_for_dia(ref_text)
                        
                        # Keep transcript short and focused (5-10 seconds audio guideline)
                        words = formatted_ref.split()
                        if len(words) > 15:  # Limit to ~10 seconds of speech
                            formatted_ref = ' '.join(words[:15]) + '.'
                        
                        logger.info(f"Created reference transcript from metadata: {formatted_ref[:50]}...")
                        return formatted_ref
                        
                except Exception as e:
                    logger.warning(f"Failed to read reference metadata {metadata_file}: {str(e)}")
                    continue
            
            # Fallback: try to get from current segment data
            # Look for original text in segment data
            original_text = segment_data.get('original_text', 
                                           segment_data.get('text', 
                                                         segment_data.get('english_text', '')))
            
            if original_text.strip():
                # Format for Dia
                formatted_text = self._format_text_for_dia(original_text)
                
                # Keep it short for better cloning
                words = formatted_text.split()
                if len(words) > 15:
                    formatted_text = ' '.join(words[:15]) + '.'
                
                logger.info(f"Created fallback transcript from segment: {formatted_text[:50]}...")
                return formatted_text
            
        except Exception as e:
            logger.warning(f"Failed to get audio prompt transcript: {str(e)}")
        
        return None
    
    def _format_text_for_dia(self, text: str) -> str:
        """Format text according to official Dia guidelines"""
        import re
        
        # Clean text first
        text = self._clean_text_for_dia(text)
        
        # Check if it's a mixed segment (has multiple speakers)
        has_multiple_speakers = '[S1]' in text and '[S2]' in text
        
        if has_multiple_speakers:
            # For mixed segments, ensure proper alternation
            # Split by speaker tags and reconstruct
            parts = re.split(r'(\[S[12]\])', text)
            formatted_parts = []
            current_speaker = None
            
            for part in parts:
                if part in ['[S1]', '[S2]']:
                    current_speaker = part
                    formatted_parts.append(part)
                elif part.strip() and current_speaker:
                    formatted_parts.append(f" {part.strip()}")
            
            return ''.join(formatted_parts)
        else:
            # Single speaker - ensure it starts with [S1]
            text = re.sub(r'\[S\d+\]\s*', '', text)  # Remove existing tags
            return f"[S1] {text.strip()}"
    
    def _clean_text_for_dia(self, text: str) -> str:
        """Clean text for Dia model according to official guidelines"""
        import re
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def is_model_loaded(self) -> bool:
        """Check if Dia model is loaded"""
        return self.dia_model is not None 