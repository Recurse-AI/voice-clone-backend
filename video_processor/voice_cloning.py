"""
Voice Cloning Module

Voice cloning using Dia model with overlapping approach for consistency.
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
    """Voice cloning service with overlapping approach for consistency"""
    
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
                           seed: Optional[int] = None) -> Dict[str, Any]:
        """Voice cloning with overlapping approach for consistency"""
        if not self.is_model_loaded():
            return {"success": False, "error": "Dia model not loaded"}
        
        if not segments:
            return {"success": False, "error": "No segments provided"}
        
        try:
            set_seed(seed or 12345)
            
            cloned_segments = []
            previous_audio = None
            
            # Group segments by chunks for overlapping approach
            chunks = self._group_segments_into_chunks(segments)
            
            for chunk_index, chunk_segments in enumerate(chunks):
                # Generate chunk text with speaker overlap
                chunk_text = self._create_chunk_text(chunk_segments, chunk_index > 0)
                
                # Get reference audio for first chunk (from first segment)
                reference_audio_path = None
                if chunk_index == 0 and chunk_segments:
                    reference_audio_path = chunk_segments[0].get('reference_audio_path')
                
                # Generate audio for this chunk
                chunk_audio = self._generate_chunk_audio(
                    chunk_text, previous_audio, temperature, cfg_scale, top_p, reference_audio_path
                )
                
                if chunk_audio is not None:
                    # Extract individual segment audio from chunk
                    segment_audios = self._extract_segment_audios(chunk_audio, chunk_segments)
                    
                    # Create cloned segment results
                    for segment, audio in zip(chunk_segments, segment_audios):
                        cloned_segments.append({
                            "success": True,
                            "original_data": segment,
                            "cloned_audio": audio,
                            "type": "overlapping",
                            "chunk_index": chunk_index
                        })
                    
                    # Store last part of audio for next chunk overlap
                    previous_audio = self._get_overlap_audio(chunk_audio)
            
            return {
                "success": True,
                "cloned_segments": cloned_segments,
                "total_segments": len(segments),
                "successful_clones": len(cloned_segments)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _group_segments_into_chunks(self, segments: List[Dict]) -> List[List[Dict]]:
        """Group segments into manageable chunks for processing"""
        chunks = []
        current_chunk = []
        max_words_per_chunk = 80  # Increased for longer segments
        current_words = 0
        
        for segment in segments:
            segment_words = segment.get('word_count', 0)
            
            if current_words + segment_words > max_words_per_chunk and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [segment]
                current_words = segment_words
            else:
                current_chunk.append(segment)
                current_words += segment_words
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _create_chunk_text(self, segments: List[Dict], include_overlap: bool) -> str:
        """Create text for chunk with proper speaker formatting"""
        chunk_lines = []
        
        # Add current chunk segments
        for segment in segments:
            speaker = segment.get('speaker', 'A')
            speaker_tag = f"[S{1 if speaker == 'A' else 2}]"
            english_text = segment.get('english_text', segment.get('text', ''))
            chunk_lines.append(f"{speaker_tag} {english_text}")
        
        return " ".join(chunk_lines)
    
    def _generate_chunk_audio(self, text: str, previous_audio: Optional[np.ndarray],
                            temperature: float, cfg_scale: float, top_p: float,
                            reference_audio_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Generate audio for a chunk with optional previous audio or reference as prompt"""
        try:
            audio_prompt_path = None
            
            # Priority: previous_audio > reference_audio > none
            if previous_audio is not None:
                # Use previous chunk audio for overlapping consistency
                audio_prompt_path = self._save_temp_audio(previous_audio)
            elif reference_audio_path and os.path.exists(reference_audio_path):
                # Use reference audio for initial chunk
                audio_prompt_path = reference_audio_path
            
            # Generate audio using Dia model
            with torch.inference_mode():
                audio = self.dia_model.generate(
                    text,
                    audio_prompt=audio_prompt_path,
                    use_torch_compile=False,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=50,
                    max_tokens=3072,
                    verbose=False
                )
            
            # Clean up temporary file (only if we created it)
            if previous_audio is not None and audio_prompt_path and os.path.exists(audio_prompt_path):
                os.unlink(audio_prompt_path)
            
            return audio
            
        except Exception as e:
            return None
    
    def _save_temp_audio(self, audio: np.ndarray) -> str:
        """Save audio to temporary file for use as prompt"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        sf.write(temp_path, audio, self.sample_rate)
        return temp_path
    
    def _extract_segment_audios(self, chunk_audio: np.ndarray, segments: List[Dict]) -> List[np.ndarray]:
        """Extract individual segment audio from chunk audio"""
        if not segments:
            return []
        
        # Simple approach: divide chunk audio equally among segments
        segment_audios = []
        total_duration = sum(seg.get('duration', 0) for seg in segments)
        
        if total_duration > 0:
            audio_length = len(chunk_audio)
            start_sample = 0
            
            for segment in segments:
                segment_duration = segment.get('duration', 0)
                segment_ratio = segment_duration / total_duration
                segment_samples = int(audio_length * segment_ratio)
                
                end_sample = start_sample + segment_samples
                segment_audio = chunk_audio[start_sample:end_sample]
                
                # Adjust length to match original duration
                target_samples = int(segment_duration * self.sample_rate)
                if len(segment_audio) != target_samples and target_samples > 0:
                    segment_audio = self._adjust_audio_length(segment_audio, segment_duration)
                
                segment_audios.append(segment_audio)
                start_sample = end_sample
        
        return segment_audios
    
    def _get_overlap_audio(self, audio: np.ndarray) -> np.ndarray:
        """Get last part of audio for next chunk overlap"""
        overlap_duration = 2.0  # 2 seconds
        overlap_samples = int(overlap_duration * self.sample_rate)
        
        if len(audio) > overlap_samples:
            return audio[-overlap_samples:]
        else:
            return audio
    
    def _adjust_audio_length(self, audio: np.ndarray, target_duration: float) -> np.ndarray:
        """Adjust audio to match target duration"""
        if len(audio) == 0:
            return audio
            
        target_samples = int(target_duration * self.sample_rate)
        current_samples = len(audio)
        
        if current_samples == target_samples:
            return audio
        
        # Use interpolation to stretch/compress audio
        indices = np.linspace(0, current_samples - 1, target_samples)
        stretched_audio = np.interp(indices, np.arange(current_samples), audio)
        
        return stretched_audio 