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
    
    def __init__(self):
        self.dia_model = None
        self.device = settings.DIA_DEVICE
        self.sample_rate = 44100
    
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
    
    def clone_voice_segments(self, segments: List[Dict], temperature: float = 1.2,
                           cfg_scale: float = 3.0, top_p: float = 0.95,
                           seed: Optional[int] = None, audio_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.is_model_loaded():
            return {"success": False, "error": "Dia model not loaded"}
        
        if not segments:
            return {"success": False, "error": "No segments provided"}
        
        logger.info(f"Starting voice cloning for {len(segments)} segments")
        
        # Update status to voice cloning if audio_id provided
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
            base_seed = seed or settings.DEFAULT_SEED
            cloning_start_time = time.time()
            
            cloned_segments = []
            speaker_seeds = {}
            
            for i, segment in enumerate(segments):
                try:
                    if not segment or not isinstance(segment, dict):
                        logger.warning(f"Skipping segment {i+1}: Invalid segment data")
                        continue
                    
                    english_text = segment.get('english_text', segment.get('text', ''))
                    
                    if not english_text.strip():
                        logger.warning(f"Skipping segment {i+1}: No english_text")
                        continue
                    
                    audio_path = segment.get('audio_path')
                    if not audio_path or not os.path.exists(audio_path):
                        logger.warning(f"Skipping segment {i+1}: No audio file found at {audio_path}")
                        continue
                    
                    # Get speaker and set consistent seed per speaker
                    speaker = segment.get('speaker', 'A')
                    if speaker not in speaker_seeds:
                        speaker_seeds[speaker] = base_seed + (ord(speaker) - ord('A'))
                    
                    set_seed(speaker_seeds[speaker])
                    
                    logger.info(f"Processing segment {i+1}/{len(segments)} (Speaker {speaker})")
                    logger.info(f"Text: {english_text}")
                    
                    # Update status with current segment progress
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
                    
                    # Get target duration (9-11 seconds optimal)
                    target_duration = segment.get('duration', 10.0)
                    
                    try:
                        # Generate audio
                        cloned_audio = self._generate_single_segment(
                            english_text, audio_path, english_text, 
                            temperature, cfg_scale, top_p
                        )
                        
                        if cloned_audio is None:
                            logger.warning(f"Skipping segment {i+1}: Failed to generate audio")
                            continue
                        
                        # Simple length adjustment
                        cloned_audio = self._adjust_audio_length_simple(cloned_audio, target_duration)
                        
                        logger.info(f"Successfully generated audio for segment {i+1}")
                        
                        cloned_segments.append({
                            "success": True,
                            "original_data": segment,
                            "cloned_audio": cloned_audio,
                            "duration": target_duration,
                            "speaker": speaker
                        })
                        
                    except Exception as generation_error:
                        logger.error(f"Error generating audio for segment {i+1}: {generation_error}")
                        continue
                    
                    # Simple memory cleanup
                    self._cleanup_memory()
                
                except Exception as segment_error:
                    logger.error(f"Critical error processing segment {i+1}: {segment_error}")
                    continue
            
            cloning_end_time = time.time()
            cloning_duration = cloning_end_time - cloning_start_time
            
            logger.info(f"Cloning completed in {cloning_duration:.2f} seconds")
            
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
            # Since we're using segment as its own reference, text and reference_text are the same
            # Dia format expects reference text + target text
            combined_text = text + "\n" + text
            
            logger.info(f"Generating audio with Dia format...")
            
            with torch.inference_mode():
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
            
            logger.info(f"Audio generation completed")
            return audio
            
        except Exception as e:
            logger.error(f"Audio generation failed: {str(e)}")
            return None
    
    def _adjust_audio_length_simple(self, audio: np.ndarray, target_duration: float) -> np.ndarray:
        """Simple audio length adjustment"""
        if audio is None or len(audio) == 0:
            return np.zeros(int(target_duration * self.sample_rate), dtype=np.float32)
        
        # Ensure proper format
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        target_samples = int(target_duration * self.sample_rate)
        current_samples = len(audio)
        current_duration = current_samples / self.sample_rate
        
        logger.info(f"Adjusting audio: {current_duration:.2f}s -> {target_duration:.2f}s")
        
        stretch_ratio = target_duration / current_duration if current_duration > 0 else 1.0
        
        # Simple stretch
        adjusted_audio = librosa.effects.time_stretch(audio, rate=1.0/stretch_ratio)
        
        # Handle length differences
        if len(adjusted_audio) > target_samples:
            adjusted_audio = adjusted_audio[:target_samples]
            
            # Adaptive fade based on text quality
            # No text quality analysis, so no fade
            
        elif len(adjusted_audio) < target_samples:
            padding = target_samples - len(adjusted_audio)
            adjusted_audio = np.pad(adjusted_audio, (0, padding), mode='constant', constant_values=0)
        
        # Gentle normalization
        max_val = np.abs(adjusted_audio).max()
        if max_val > 1.0:
            adjusted_audio = adjusted_audio * (0.99 / max_val)
        elif max_val < 0.1:
            adjusted_audio = adjusted_audio * (0.3 / max_val)
        
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