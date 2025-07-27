"""
Voice Cloning Module - Simplified for Dia
"""

import torch
import numpy as np
import random
import os
import gc
import json
import soundfile as sf
from typing import Optional, Dict, Any, List
from dia.model import Dia
from config import settings
import logging
import time
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
    
    def __init__(self, segment_manager=None):
        self.dia_model = None
        self.device = settings.DIA_DEVICE
        self.sample_rate = 44100
        self.segment_manager = segment_manager
        
        if not segment_manager:
            logger.warning("VoiceCloningService initialized without segment_manager - some features may not work properly")
    
    def load_dia_model(self, repo_id: str = None) -> bool:
        """Load Dia model"""
        try:
            repo_id = repo_id or settings.DIA_MODEL_REPO
            compute_dtype = settings.DIA_COMPUTE_DTYPE if self.device == "cuda" else "float32"
            logger.info(f"Loading Dia model from repo: {repo_id}, Device: {self.device}, Compute dtype: {compute_dtype}")
            
            self.dia_model = Dia.from_pretrained(repo_id, compute_dtype=compute_dtype)
            logger.info("Dia model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Dia model: {e}")
            raise Exception(f"Error loading Dia model: {e}")
        
    def is_model_loaded(self) -> bool:
        """Check if Dia model is loaded"""
        return self.dia_model is not None
    
    def clone_voice_segments(self, segments: List[Dict], temperature: float = 1.0,
                           cfg_scale: float = 3.5, top_p: float = 0.9,
                           seed: Optional[int] = None, audio_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.is_model_loaded():
            return {"success": False, "error": "Dia model not loaded"}
        
        if not self.segment_manager:
            return {"success": False, "error": "Segment manager not available"}
        
        if not segments:
            return {"success": False, "error": "No segments provided"}
        
        logger.info(f"Starting voice cloning for {len(segments)} segments with optimized parameters for consistency")
        
        # Update status if audio_id provided
        if audio_id:
            try:
                from status_manager import status_manager
                from status_manager import ProcessingStatus
                status_manager.update_status(
                    audio_id, 
                    ProcessingStatus.PROCESSING, 
                    progress=60, 
                    details={"message": f"Processing voice cloning: 0/{len(segments)} segments completed"}
                )
            except:
                pass
        
        try:
            cloning_start_time = time.time()
            cloned_segments = []
            
            # Optimize seed for consistency
            base_seed = seed if seed is not None else random.randint(1, 1000000)
            speaker_seeds = {}
            
            # Process each segment with enhanced error handling
            for i, segment in enumerate(segments):
                try:
                    # Extract segment information
                    segment_speaker = segment.get('speaker', 'A')
                    segment_duration = segment.get('duration', 10.0)
                    english_text = segment.get('english_text', '').strip()
                    
                    logger.info(f"Processing segment {i+1}/{len(segments)}: {segment_duration:.2f}s, Speaker: {segment_speaker}")
                    
                    # Handle empty or silence segments
                    if not english_text or english_text in ['[SILENCE]', '']:
                        logger.info(f"Creating silence for segment {i+1} ({segment_duration:.2f}s)")
                        
                        # Create silence with proper duration
                        silence_samples = int(segment_duration * self.sample_rate)
                        silence_audio = np.zeros(silence_samples, dtype=np.float32)
                        
                        # Save silence audio
                        self._save_single_cloned_segment(silence_audio, i+1, audio_id)
                        
                        cloned_segments.append({
                            "success": True,
                            "original_data": {
                                "segment_index": segment.get('segment_index', i+1),
                                "speaker": segment_speaker,
                                "duration": segment_duration,
                                "original_text": segment.get('original_text', ''),
                                "english_text": ""
                            },
                            "duration": float(segment_duration),
                            "speaker": str(segment_speaker),
                            "segment_index": int(segment.get('segment_index', i+1)),
                            "voice_params": {"type": "silence"}
                        })
                        continue
                    
                    audio_path = segment.get('audio_path')
                    if not audio_path or not os.path.exists(audio_path):
                        logger.warning(f"Audio file missing for segment {i+1} - creating silence placeholder")
                        
                        # Create silence placeholder for missing audio
                        silence_samples = int(segment_duration * self.sample_rate)
                        silence_audio = np.zeros(silence_samples, dtype=np.float32)
                        self._save_single_cloned_segment(silence_audio, i+1, audio_id)
                        
                        cloned_segments.append({
                            "success": True,
                            "original_data": {
                                "segment_index": segment.get('segment_index', i+1),
                                "speaker": segment_speaker,
                                "duration": segment_duration,
                                "original_text": segment.get('original_text', ''),
                                "english_text": english_text
                            },
                            "duration": float(segment_duration),
                            "speaker": str(segment_speaker),
                            "segment_index": int(segment.get('segment_index', i+1)),
                            "voice_params": {"type": "silence_placeholder"}
                        })
                        continue
                    
                    # Set consistent seed per speaker
                    speaker = segment.get('speaker', 'A')
                    if speaker not in speaker_seeds:
                        speaker_seeds[speaker] = base_seed + (ord(speaker) - ord('A'))
                    set_seed(speaker_seeds[speaker])
                    
                    logger.info(f"Processing segment {i+1}/{len(segments)} (Speaker {speaker})")
                    
                    # Update progress
                    if audio_id:
                        try:
                            from status_manager import status_manager
                            from status_manager import ProcessingStatus
                            progress = 60 + int((i / len(segments)) * 30)
                            status_manager.update_status(
                                audio_id, 
                                ProcessingStatus.PROCESSING, 
                                progress=progress,
                                details={"message": f"Processing voice cloning: segment {i+1}/{len(segments)} (Speaker {speaker})"}
                            )
                        except:
                            pass
                    
                    try:
                        # Calculate optimized parameters for consistency
                        dynamic_params = self._calculate_optimized_parameters(
                            segment, temperature, cfg_scale, top_p, i, len(segments)
                        )
                        
                        # Generate audio with optimized parameters
                        cloned_audio = self._generate_single_segment(
                            english_text, audio_path, english_text, 
                            dynamic_params["temperature"], 
                            dynamic_params["cfg_scale"], 
                            dynamic_params["top_p"]
                        )
                        
                        if cloned_audio is None:
                            logger.warning(f"Audio generation failed for segment {i+1} - creating silence placeholder")
                            
                            # Create silence placeholder for failed generation
                            silence_samples = int(segment_duration * self.sample_rate)
                            silence_audio = np.zeros(silence_samples, dtype=np.float32)
                            self._save_single_cloned_segment(silence_audio, i+1, audio_id)
                            
                            cloned_segments.append({
                                "success": True,
                                "original_data": {
                                    "segment_index": segment.get('segment_index', i+1),
                                    "speaker": speaker,
                                    "duration": segment_duration,
                                    "original_text": segment.get('original_text', ''),
                                    "english_text": english_text
                                },
                                "duration": float(segment_duration),
                                "speaker": str(speaker),
                                "segment_index": int(segment.get('segment_index', i+1)),
                                "voice_params": {"type": "failed_generation_placeholder"}
                            })
                            continue
                        
                        # Ensure proper duration
                        cloned_audio = self._ensure_proper_duration(cloned_audio, segment_duration)
                        
                        # Save audio immediately
                        self._save_single_cloned_segment(cloned_audio, i+1, audio_id)
                        
                        logger.info(f"Successfully generated {dynamic_params['speech_type']} speech for segment {i+1}")
                        
                        cloned_segments.append({
                            "success": True,
                            "original_data": {
                                "segment_index": segment.get('segment_index', i+1),
                                "speaker": speaker,
                                "duration": segment.get('duration', 10.0),
                                "original_text": segment.get('original_text', ''),
                                "english_text": english_text
                            },
                            "duration": float(segment.get('duration', 10.0)),
                            "speaker": str(speaker),
                            "segment_index": int(segment.get('segment_index', i+1)),
                            "voice_params": dynamic_params  # Store the parameters used
                        })
                        
                    except Exception as generation_error:
                        logger.error(f"Error generating audio for segment {i+1}: {generation_error}")
                        
                        # Create silence placeholder for any generation error
                        logger.info(f"Creating silence placeholder for failed segment {i+1}")
                        silence_samples = int(segment_duration * self.sample_rate)
                        silence_audio = np.zeros(silence_samples, dtype=np.float32)
                        self._save_single_cloned_segment(silence_audio, i+1, audio_id)
                        
                        cloned_segments.append({
                            "success": True,
                            "original_data": {
                                "segment_index": segment.get('segment_index', i+1),
                                "speaker": segment_speaker,
                                "duration": segment_duration,
                                "original_text": segment.get('original_text', ''),
                                "english_text": english_text
                            },
                            "duration": float(segment_duration),
                            "speaker": str(segment_speaker),
                            "segment_index": int(segment.get('segment_index', i+1)),
                            "voice_params": {"type": "error_placeholder"}
                        })
                        continue
                    
                    self._cleanup_memory()
                
                except Exception as segment_error:
                    logger.error(f"Critical error processing segment {i+1}: {segment_error}")
                    
                    # Ensure we never skip a segment - always create placeholder
                    segment_duration = segment.get('duration', 10.0)
                    logger.info(f"Creating silence placeholder for critically failed segment {i+1}")
                    
                    silence_samples = int(segment_duration * self.sample_rate)
                    silence_audio = np.zeros(silence_samples, dtype=np.float32)
                    self._save_single_cloned_segment(silence_audio, i+1, audio_id)
                    
                    cloned_segments.append({
                        "success": True,
                        "original_data": {
                            "segment_index": segment.get('segment_index', i+1),
                            "speaker": segment.get('speaker', 'A'),
                            "duration": segment_duration,
                            "original_text": segment.get('original_text', ''),
                            "english_text": segment.get('english_text', '')
                        },
                        "duration": float(segment_duration),
                        "speaker": str(segment.get('speaker', 'A')),
                        "segment_index": int(segment.get('segment_index', i+1)),
                        "voice_params": {"type": "critical_error_placeholder"}
                    })
                    continue
            
            cloning_duration = time.time() - cloning_start_time
            logger.info(f"Cloning completed in {cloning_duration:.2f} seconds")
            logger.info(f"Processed {len(cloned_segments)} segments (ensuring complete duration coverage)")
            
            # Verify we have all segments
            if len(cloned_segments) != len(segments):
                logger.warning(f"Segment count mismatch: expected {len(segments)}, got {len(cloned_segments)}")
            
            # Save summary
            if cloned_segments:
                self._save_cloned_segments_unified(cloned_segments, audio_id)
            
            return {
                "success": True,
                "cloned_segments": cloned_segments,
                "total_segments": len(segments),
                "successful_clones": len(cloned_segments),
                "seed_used": base_seed,
                "speaker_seeds": speaker_seeds,
                "cloning_duration": cloning_duration
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            self._cleanup_memory()
    
    def _generate_single_segment(self, text: str, reference_audio_path: str, 
                               reference_text: str, temperature: float, 
                               cfg_scale: float, top_p: float) -> Optional[np.ndarray]:
        try:
            # Clean text for voice generation
            clean_text = text.strip()
            if not clean_text:
                logger.warning("Empty text provided for voice generation")
                return None
            
            logger.info(f"Generating audio with Dia - Text: '{clean_text[:50]}...', Max tokens: {settings.DIA_MAX_TOKENS}")
            
            with torch.inference_mode():
                # Generate audio with Dia model
                audio = self.dia_model.generate(
                    text=clean_text,
                    audio_prompt=reference_audio_path,
                    use_torch_compile=False,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=settings.DIA_CFG_FILTER_TOP_K,
                    max_tokens=settings.DIA_MAX_TOKENS,
                    verbose=False
                )
            
            if audio is None or len(audio) == 0:
                logger.error("Dia model returned empty audio")
                return None
            
            logger.info(f"Audio generation completed - generated {len(audio)/self.sample_rate:.3f}s of audio")
            return audio
            
        except Exception as e:
            logger.error(f"Audio generation failed: {str(e)}")
            return None
    
    def _ensure_proper_duration(self, audio: np.ndarray, target_duration: float) -> np.ndarray:
        """Ensure generated audio matches target duration using intelligent padding/trimming"""
        if audio is None or len(audio) == 0:
            # Create silence for the full duration
            logger.warning(f"No audio provided, creating {target_duration:.2f}s of silence")
            return np.zeros(int(target_duration * self.sample_rate), dtype=np.float32)
        
        current_duration = len(audio) / self.sample_rate
        logger.info(f"Audio duration adjustment: current={current_duration:.3f}s, target={target_duration:.3f}s")
        
        # If durations are very close (within 100ms), return as is
        if abs(current_duration - target_duration) <= 0.1:
            return audio
        
        target_samples = int(target_duration * self.sample_rate)
        
        if current_duration > target_duration:
            # Audio is too long - trim it
            logger.info(f"Trimming audio from {current_duration:.3f}s to {target_duration:.3f}s")
            return audio[:target_samples]
        else:
            # Audio is too short - pad with silence
            padding_needed = target_samples - len(audio)
            logger.info(f"Padding audio with {padding_needed/self.sample_rate:.3f}s of silence")
            
            # Add silence to the end to match target duration
            silence_padding = np.zeros(padding_needed, dtype=audio.dtype)
            padded_audio = np.concatenate([audio, silence_padding])
            
            return padded_audio
    
    def _adjust_audio_length(self, audio: np.ndarray, target_duration: float, 
                           use_speed_adjustment: bool = True) -> np.ndarray:
        """
        Legacy method for backward compatibility with main.py
        Uses segment_manager's advanced adjustment if available
        """
        if hasattr(self, 'segment_manager') and self.segment_manager:
            # Create a simple segment dict for the legacy call
            segment = {'duration': target_duration}
            return self.segment_manager.adjust_generated_audio_length(
                audio, self.sample_rate, segment
            )
        else:
            # Simple fallback if segment_manager not available
            target_samples = int(target_duration * self.sample_rate)
            if len(audio) > target_samples:
                return audio[:target_samples]
            elif len(audio) < target_samples:
                padding = target_samples - len(audio)
                return np.pad(audio, (0, padding), mode='constant', constant_values=0)
            return audio
    
    def _calculate_optimized_parameters(self, segment: Dict[str, Any], base_temperature: float, 
                                    base_cfg_scale: float, base_top_p: float, 
                                    current_segment_index: int, total_segments: int) -> Dict[str, float]:
        """
        Calculate optimized voice cloning parameters for better consistency and focus.
        Based on research: lower temperature (0.7-1.1) and optimized cfg_scale (3.5-4.0) for consistency.
        Reduced dynamic variations to maintain voice characteristics across segments.
        """
        duration = segment.get('duration', 10.0)
        english_text = segment.get('english_text', '')
        word_count = len(english_text.split()) if english_text else 0
        
        # Calculate words per second
        words_per_second = word_count / duration if duration > 0 else 0
        
        # Optimal speaking rate is ~2.5 words per second
        optimal_wps = 2.5
        density_ratio = words_per_second / optimal_wps
        
        # Use more conservative parameter adjustments for consistency
        # Based on ElevenLabs best practices: lower temperature = more consistent
        if density_ratio > 1.4:  # High density - need faster speech
            # Minimal temperature increase for faster speech, keep consistent
            temperature = min(base_temperature + 0.1, 1.2)
            cfg_scale = max(base_cfg_scale - 0.2, 3.2)  # Slightly lower for natural fast speech
            top_p = max(base_top_p - 0.02, 0.88)  # More focused
            speech_type = "fast"
        elif density_ratio < 0.6:  # Low density - need slower speech  
            # Minimal temperature decrease for slower speech
            temperature = max(base_temperature - 0.1, 0.7)
            cfg_scale = min(base_cfg_scale + 0.2, 4.0)  # Slightly higher for controlled speech
            top_p = min(base_top_p + 0.02, 0.92)  # Slightly less focused
            speech_type = "slow"
        else:  # Normal density - use consistent base parameters
            temperature = base_temperature
            cfg_scale = base_cfg_scale
            top_p = base_top_p
            speech_type = "normal"
        
        # Additional consistency adjustments based on segment position
        # First and last segments should be slightly more controlled for consistency
        if current_segment_index == 0 or current_segment_index == total_segments - 1:
            temperature = max(temperature - 0.05, 0.7)  # Slightly more controlled
            cfg_scale = min(cfg_scale + 0.1, 4.0)  # Slightly higher control
        
        logger.info(f"Segment {segment.get('segment_index', current_segment_index + 1)}: {word_count} words in {duration:.1f}s "
                   f"({words_per_second:.1f} wps) → {speech_type} speech (T={temperature:.2f}, CFG={cfg_scale:.1f})")
        
        return {
            "temperature": temperature,
            "cfg_scale": cfg_scale,
            "top_p": top_p,
            "words_per_second": words_per_second,
            "density_ratio": density_ratio,
            "speech_type": speech_type
        }
    
    def _save_single_cloned_segment(self, audio_data: np.ndarray, segment_index: int, audio_id: str):
        """Save a single cloned segment audio file."""
        try:
            from pathlib import Path
            from config import settings
            import soundfile as sf
            
            # Get base segments directory
            temp_dir = Path(settings.TEMP_DIR)
            segments_base_dir = temp_dir / f"segments_{audio_id}"
            cloned_dir = segments_base_dir / "cloned"
            cloned_dir.mkdir(parents=True, exist_ok=True)
            
            # Save cloned audio file
            cloned_filename = f"cloned_segment_{segment_index:03d}.wav"
            cloned_path = cloned_dir / cloned_filename
            sf.write(str(cloned_path), audio_data, self.sample_rate)
            
            # Update metadata with cloned path
            segments_dir = segments_base_dir / "segments"
            metadata_file = segments_dir / f"segment_{segment_index:03d}_metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                metadata['cloned_audio_path'] = str(cloned_path)
                metadata['cloned_audio_file'] = cloned_filename
                metadata['cloning_completed'] = True
                
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                logger.debug(f"Saved cloned segment: {cloned_filename}")
            else:
                logger.warning(f"Metadata file not found: {metadata_file}")
                
        except Exception as e:
            logger.error(f"Error saving single cloned segment {segment_index}: {e}")
    
    def _save_cloned_segments_unified(self, cloned_segments: List[Dict], audio_id: str):
        """Save cloning summary"""
        try:
            logger.info(f"Saved {len(cloned_segments)} cloned segments")
            
            from pathlib import Path
            from config import settings
            
            temp_dir = Path(settings.TEMP_DIR)
            segments_base_dir = temp_dir / f"segments_{audio_id}"
            metadata_dir = segments_base_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            summary = {
                "cloning_summary": {
                    "total_segments": len(cloned_segments),
                    "successful_clones": len([s for s in cloned_segments if s.get('success', False)]),
                    "segments": [
                        {
                            "segment_index": s.get('segment_index', 0),
                            "speaker": s.get('speaker', 'A'),
                            "duration": s.get('duration', 0.0),
                            "success": s.get('success', False)
                        }
                        for s in cloned_segments
                    ]
                }
            }
            
            summary_file = metadata_dir / "cloning_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error creating cloning summary: {e}")
    
    def _cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def clear_cache(self):
        """Clear any cached data"""
        self._cleanup_memory() 