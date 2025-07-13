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
        
    def is_model_loaded(self) -> bool:
        """Check if Dia model is loaded"""
        return self.dia_model is not None 
    
    def clone_voice_segments(self, segments: List[Dict], temperature: float = 1.3,
                           cfg_scale: float = 3.0, top_p: float = 0.95,
                           seed: Optional[int] = None) -> Dict[str, Any]:
        """Clone voice segments with continuous/non-continuous handling"""
        if not self.is_model_loaded():
            return {"success": False, "error": "Dia model not loaded"}
        
        try:
            if seed is not None:
                set_seed(seed)
            
            # Group segments by type and group_id
            continuous_segments = []
            non_continuous_groups = {}
            
            for segment in segments:
                if segment.get('is_continuous', True):
                    continuous_segments.append(segment)
                else:
                    group_id = segment.get('group_id')
                    if group_id:
                        if group_id not in non_continuous_groups:
                            non_continuous_groups[group_id] = []
                        non_continuous_groups[group_id].append(segment)
            
            cloned_segments = []
            
            # Process continuous segments
            for segment in continuous_segments:
                cloned = self._clone_continuous_segment(
                    segment, temperature, cfg_scale, top_p
                )
                if cloned:
                    cloned_segments.append(cloned)
            
            # Process non-continuous groups
            for group_id, group_segments in non_continuous_groups.items():
                cloned_group = self._clone_non_continuous_group(
                    group_segments, temperature, cfg_scale, top_p
                )
                cloned_segments.extend(cloned_group)
            
            return {
                "success": True,
                "cloned_segments": cloned_segments,
                "total_segments": len(segments),
                "successful_clones": len(cloned_segments)
            }
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _clone_continuous_segment(self, segment: Dict, temperature: float,
                                cfg_scale: float, top_p: float) -> Optional[Dict]:
        """Clone a continuous segment"""
        try:
            # Get reference audio if available
            reference_audio_path = segment.get('reference_audio_path')
            
            # Prepare text for Dia
            dia_text = segment.get('dia_text', '')
            if not dia_text:
                return None
            
            # Clone with Dia
            if reference_audio_path and os.path.exists(reference_audio_path):
                # Load reference transcript
                ref_transcript = self._get_reference_transcript(reference_audio_path)
                
                # Build prompt according to Dia guidelines
                # Reference transcript should come first, then a space, then the target text
                prompt_text = f"{ref_transcript} {dia_text}"
                
                cloned_audio = self.dia_model.generate(
                    prompt_text,
                    audio_prompt=reference_audio_path,
                    use_torch_compile=False,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=50,
                    max_tokens=3072
                )
            else:
                cloned_audio = self.dia_model.generate(
                    dia_text,
                    use_torch_compile=False,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=50,
                    max_tokens=3072
                )
            
            # Length synchronization
            original_duration = segment.get('duration', 0)
            if original_duration > 0:
                cloned_audio = self._synchronize_length(
                    cloned_audio, original_duration, 44100
                )
            
            return {
                "success": True,
                "original_data": segment,
                "cloned_audio": cloned_audio,
                "type": "continuous"
            }
            
        except Exception as e:
            logger.error(f"Failed to clone continuous segment: {str(e)}")
            return None
    
    def _clone_non_continuous_group(self, group_segments: List[Dict], temperature: float,
                                   cfg_scale: float, top_p: float) -> List[Dict]:
        """Clone non-continuous segment group"""
        try:
            # Merge texts from all segments
            merged_text = ' '.join(s.get('dia_text', '') for s in group_segments)
            if not merged_text:
                return []
            
            # Get reference audio
            reference_audio_path = group_segments[0].get('reference_audio_path')
            
            # Clone merged audio
            if reference_audio_path and os.path.exists(reference_audio_path):
                ref_transcript = self._get_reference_transcript(reference_audio_path)
                prompt_text = f"{ref_transcript} {merged_text}"
                
                cloned_audio = self.dia_model.generate(
                    prompt_text,
                    audio_prompt=reference_audio_path,
                    use_torch_compile=False,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=50,
                    max_tokens=3072
                )
            else:
                cloned_audio = self.dia_model.generate(
                    merged_text,
                    use_torch_compile=False,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=50,
                    max_tokens=3072
                )
            
            # Split cloned audio back to original segments
            split_segments = self._split_non_continuous_audio(
                cloned_audio, group_segments, 44100
            )
            
            return split_segments
            
        except Exception as e:
            logger.error(f"Failed to clone non-continuous group: {str(e)}")
            return []
    
    def _split_non_continuous_audio(self, cloned_audio: np.ndarray, 
                                  original_segments: List[Dict],
                                  sample_rate: int) -> List[Dict]:
        """Split non-continuous cloned audio using OpenAI assistance"""
        try:
            from openai import OpenAI
            from .transcription import TranscriptionService
            
            # Transcribe cloned audio
            temp_path = Path(settings.TEMP_DIR) / f"temp_clone_{random.randint(1000, 9999)}.wav"
            sf.write(temp_path, cloned_audio, sample_rate)
            
            transcription_service = TranscriptionService()
            clone_transcript = transcription_service.transcribe_audio(str(temp_path))
            
            # Get cutting instructions from OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            original_info = [
                f"Segment {i+1}: {s.get('duration', 0):.2f}s - \"{s.get('english_text', '')}\""
                for i, s in enumerate(original_segments)
            ]
            
            prompt = f"""Analyze this cloned audio transcription and provide exact cutting points:

Original segments:
{chr(10).join(original_info)}

Cloned audio transcription:
{clone_transcript.get('text', '')}
Duration: {len(cloned_audio) / sample_rate:.2f}s

Provide cutting points in seconds for each segment. Format:
Segment 1: 0.0-X.X
Segment 2: X.X-Y.Y
etc."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            cutting_instructions = response.choices[0].message.content
            
            # Parse cutting points
            import re
            pattern = r'Segment \d+:\s*([\d.]+)-([\d.]+)'
            matches = re.findall(pattern, cutting_instructions)
            
            split_segments = []
            for i, (start, end) in enumerate(matches):
                if i < len(original_segments):
                    start_sample = int(float(start) * sample_rate)
                    end_sample = int(float(end) * sample_rate)
                    
                    segment_audio = cloned_audio[start_sample:end_sample]
                    
                    # Synchronize to original length
                    original_duration = original_segments[i].get('duration', 0)
                    if original_duration > 0:
                        segment_audio = self._synchronize_length(
                            segment_audio, original_duration, sample_rate
                        )
                    
                    split_segments.append({
                        "success": True,
                        "original_data": original_segments[i],
                        "cloned_audio": segment_audio,
                        "type": "non_continuous"
                    })
            
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
            
            return split_segments
            
        except Exception as e:
            logger.error(f"Failed to split non-continuous audio: {str(e)}")
            # Fallback: split proportionally
            return self._split_proportionally(cloned_audio, original_segments, sample_rate)
    
    def _split_proportionally(self, audio: np.ndarray, segments: List[Dict],
                            sample_rate: int) -> List[Dict]:
        """Fallback: Split audio proportionally based on original durations"""
        total_duration = sum(s.get('duration', 0) for s in segments)
        audio_duration = len(audio) / sample_rate
        
        split_segments = []
        current_sample = 0
        
        for segment in segments:
            segment_duration = segment.get('duration', 0)
            proportion = segment_duration / total_duration if total_duration > 0 else 1/len(segments)
            
            segment_samples = int(proportion * len(audio))
            end_sample = min(current_sample + segment_samples, len(audio))
            
            segment_audio = audio[current_sample:end_sample]
            
            # Synchronize to original length
            if segment_duration > 0:
                segment_audio = self._synchronize_length(
                    segment_audio, segment_duration, sample_rate
                )
            
            split_segments.append({
                "success": True,
                "original_data": segment,
                "cloned_audio": segment_audio,
                "type": "non_continuous_fallback"
            })
            
            current_sample = end_sample
        
        return split_segments
    
    def _synchronize_length(self, audio: np.ndarray, target_duration: float,
                          sample_rate: int) -> np.ndarray:
        """Synchronize audio length to target duration"""
        target_samples = int(target_duration * sample_rate)
        current_samples = len(audio)
        
        if current_samples == target_samples:
            return audio
        
        # Calculate deviation
        deviation = abs(current_samples - target_samples) / target_samples
        
        if deviation <= 0.1:  # Within 10% tolerance
            # Simple trim/pad
            if current_samples > target_samples:
                return audio[:target_samples]
            else:
                padding = np.zeros(target_samples - current_samples)
                return np.concatenate([audio, padding])
        else:
            # Time stretching needed
            try:
                import librosa
                stretch_ratio = target_samples / current_samples
                stretched = librosa.effects.time_stretch(audio, rate=1/stretch_ratio)
                
                # Ensure exact length
                if len(stretched) > target_samples:
                    return stretched[:target_samples]
                elif len(stretched) < target_samples:
                    padding = np.zeros(target_samples - len(stretched))
                    return np.concatenate([stretched, padding])
                return stretched
                
            except ImportError:
                # Fallback to simple trim/pad
                if current_samples > target_samples:
                    return audio[:target_samples]
                else:
                    padding = np.zeros(target_samples - current_samples)
                    return np.concatenate([audio, padding])
    
    def _get_reference_transcript(self, reference_path: str) -> str:
        """Get reference transcript from metadata"""
        try:
            # Look for metadata file
            ref_path = Path(reference_path)
            metadata_path = ref_path.parent / f"{ref_path.stem}_metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                dia_text = metadata.get('dia_text', '')
                if dia_text:
                    return dia_text
                return '[S1] ' + metadata.get('english_text', metadata.get('text', 'Reference audio.'))
            
            return '[S1] Reference audio.'
            
        except Exception:
            return '[S1] Reference audio.' 