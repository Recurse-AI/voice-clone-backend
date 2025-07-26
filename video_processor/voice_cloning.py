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
                        
                        # Save audio immediately to avoid JSON serialization issues
                        self._save_single_cloned_segment(cloned_audio, i+1, audio_id)
                        
                        logger.info(f"Successfully generated audio for segment {i+1}")
                        
                        cloned_segments.append({
                            "success": True,
                            "original_data": {
                                "segment_index": segment.get('segment_index', i+1),
                                "speaker": speaker,
                                "duration": target_duration,
                                "original_text": segment.get('original_text', ''),
                                "english_text": english_text
                            },
                            "duration": float(target_duration),
                            "speaker": str(speaker),
                            "segment_index": int(segment.get('segment_index', i+1))
                            # Note: cloned_audio excluded to prevent JSON serialization issues
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
            
            # Save cloned segments in unified cloned folder
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
    
    def _remove_trailing_silence(self, audio: np.ndarray, silence_threshold: float = 0.01) -> np.ndarray:
        """Remove silent parts from the tail of audio - DEPRECATED: Use _remove_trailing_silence_conservative instead"""
        logger.warning("Using deprecated _remove_trailing_silence method - consider using conservative version")
        
        if len(audio) == 0:
            return audio
        
        # Find the last non-silent sample
        audio_abs = np.abs(audio)
        non_silent_indices = np.where(audio_abs > silence_threshold)[0]
        
        if len(non_silent_indices) == 0:
            logger.warning("No non-silent audio found, returning original")
            return audio
        
        last_sound_index = non_silent_indices[-1]
        
        # Be more conservative - add longer fade to preserve content
        fade_samples = min(int(0.1 * self.sample_rate), len(audio) - last_sound_index, 2000)  # Max 2000 samples fade
        end_index = min(last_sound_index + fade_samples, len(audio))
        
        original_duration = len(audio) / self.sample_rate
        final_duration = end_index / self.sample_rate
        logger.info(f"Trailing silence removal: {original_duration:.3f}s -> {final_duration:.3f}s (removed {len(audio) - end_index} samples)")
        
        return audio[:end_index]
    
    def _adjust_audio_length_simple(self, audio: np.ndarray, target_duration: float) -> np.ndarray:
        """Simple audio length adjustment that preserves original voice and ensures exact target duration"""
        if audio is None or len(audio) == 0:
            logger.warning(f"Empty audio provided, creating silence for {target_duration:.2f}s")
            return np.zeros(int(target_duration * self.sample_rate), dtype=np.float32)
        
        audio = np.array(audio, dtype=np.float32)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        target_samples = int(target_duration * self.sample_rate)
        current_samples = len(audio)
        current_duration = current_samples / self.sample_rate
        
        logger.info(f"Audio adjustment: {current_duration:.3f}s ({current_samples} samples) -> {target_duration:.3f}s ({target_samples} samples)")
        
        # Ensure we maintain the target duration exactly
        adjusted_audio = audio.copy()
        
        # Strategy: Never trim audio content, only adjust silence/padding
        if len(adjusted_audio) > target_samples:
            # Audio is longer than target - try to remove only trailing silence first
            audio_without_silence = self._remove_trailing_silence_conservative(adjusted_audio, target_samples)
            
            if len(audio_without_silence) <= target_samples:
                # Great! Silence removal worked
                adjusted_audio = audio_without_silence
                logger.info(f"Removed trailing silence: {len(audio_without_silence)} samples (no content loss)")
            else:
                # Audio content is longer than target - preserve content and log warning
                logger.warning(f"Generated audio ({current_duration:.3f}s) longer than target ({target_duration:.3f}s) - preserving content")
                # Keep the generated audio as is, even if longer
                target_samples = len(adjusted_audio)
                target_duration = len(adjusted_audio) / self.sample_rate
                logger.info(f"Updated target duration to preserve content: {target_duration:.3f}s")
        
        # Pad with silence if needed to reach exact target
        if len(adjusted_audio) < target_samples:
            padding = target_samples - len(adjusted_audio)
            adjusted_audio = np.pad(adjusted_audio, (0, padding), mode='constant', constant_values=0)
            logger.info(f"Added {padding} samples of silence padding")
        
        final_duration = len(adjusted_audio) / self.sample_rate
        logger.info(f"Final audio duration: {final_duration:.3f}s ({len(adjusted_audio)} samples)")
        
        # Verify no significant duration loss
        duration_diff = abs(final_duration - target_duration)
        if duration_diff > 0.01:  # More than 10ms difference
            logger.warning(f"Duration difference detected: {duration_diff:.3f}s (target: {target_duration:.3f}s, actual: {final_duration:.3f}s)")
        
        return adjusted_audio.astype(np.float32)
    
    def _remove_trailing_silence_conservative(self, audio: np.ndarray, max_target_samples: int, silence_threshold: float = 0.01) -> np.ndarray:
        """Conservative trailing silence removal that respects target duration"""
        if len(audio) == 0:
            return audio
        
        # Don't remove silence if it would make audio shorter than target
        if len(audio) <= max_target_samples:
            logger.debug("Audio already within target range, skipping silence removal")
            return audio
        
        # Find the last non-silent sample
        audio_abs = np.abs(audio)
        non_silent_indices = np.where(audio_abs > silence_threshold)[0]
        
        if len(non_silent_indices) == 0:
            logger.debug("No non-silent audio found")
            return audio
        
        last_sound_index = non_silent_indices[-1]
        
        # Add small fade-out but respect target duration
        fade_samples = min(int(0.05 * self.sample_rate), 1000)  # Max 1000 samples fade
        natural_end = min(last_sound_index + fade_samples, len(audio))
        
        # Only trim if the natural end is still longer than target
        if natural_end > max_target_samples:
            # Keep target duration exactly
            end_index = max_target_samples
            logger.debug(f"Trimmed to target duration: {end_index} samples")
        else:
            # Keep natural end
            end_index = natural_end
            logger.debug(f"Kept natural end: {end_index} samples")
        
        return audio[:end_index]
    
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
        """Log cloned segments completion - audio already saved individually"""
        try:
            logger.info(f"Saved {len(cloned_segments)} cloned segments in unified structure")
            
            # Optional: Create summary file for debugging
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