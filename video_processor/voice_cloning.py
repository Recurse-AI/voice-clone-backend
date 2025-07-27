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
    
    def clone_voice_segments(self, segments: List[Dict], temperature: float = 1.2,
                           cfg_scale: float = 3.0, top_p: float = 0.95,
                           seed: Optional[int] = None, audio_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.is_model_loaded():
            return {"success": False, "error": "Dia model not loaded"}
        
        if not self.segment_manager:
            return {"success": False, "error": "Segment manager not available"}
        
        if not segments:
            return {"success": False, "error": "No segments provided"}
        
        logger.info(f"Starting voice cloning for {len(segments)} segments")
        
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
                        # Handle silent segment - create silence audio
                        segment_duration = segment.get('duration', 10.0)
                        silence_audio = np.zeros(int(segment_duration * self.sample_rate), dtype=np.float32)
                        
                        logger.info(f"Creating {segment_duration:.2f}s silence for segment {i+1} (no text to clone)")
                        
                        # Save silence audio
                        self._save_single_cloned_segment(silence_audio, i+1, audio_id)
                        
                        cloned_segments.append({
                            "success": True,
                            "original_data": {
                                "segment_index": segment.get('segment_index', i+1),
                                "speaker": speaker,
                                "duration": segment_duration,
                                "original_text": segment.get('original_text', ''),
                                "english_text": ""
                            },
                            "duration": float(segment_duration),
                            "speaker": str(speaker),
                            "segment_index": int(segment.get('segment_index', i+1)),
                            "voice_params": {"type": "silence"}
                        })
                        continue
                    
                    audio_path = segment.get('audio_path')
                    if not audio_path or not os.path.exists(audio_path):
                        logger.warning(f"Skipping segment {i+1}: No audio file found at {audio_path}")
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
                        # Calculate dynamic parameters for this segment
                        dynamic_params = self._calculate_dynamic_parameters(
                            segment, temperature, cfg_scale, top_p
                        )
                        
                        # Generate audio with dynamic parameters
                        cloned_audio = self._generate_single_segment(
                            english_text, audio_path, english_text, 
                            dynamic_params["temperature"], 
                            dynamic_params["cfg_scale"], 
                            dynamic_params["top_p"]
                        )
                        
                        if cloned_audio is None:
                            logger.warning(f"Skipping segment {i+1}: Failed to generate audio")
                            continue
                        
                        # Use segment manager's advanced audio adjustment
                        cloned_audio = self.segment_manager.adjust_generated_audio_length(
                            cloned_audio, self.sample_rate, segment
                        )
                        
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
                        continue
                    
                    self._cleanup_memory()
                
                except Exception as segment_error:
                    logger.error(f"Critical error processing segment {i+1}: {segment_error}")
                    continue
            
            cloning_duration = time.time() - cloning_start_time
            logger.info(f"Cloning completed in {cloning_duration:.2f} seconds")
            
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
    
    def _calculate_dynamic_parameters(self, segment: Dict[str, Any], base_temperature: float, 
                                    base_cfg_scale: float, base_top_p: float) -> Dict[str, float]:
        """
        Calculate dynamic voice cloning parameters based on word density
        
        High density (fast speech): Higher temperature, adjusted cfg_scale
        Low density (slow speech): Lower temperature, adjusted cfg_scale
        """
        duration = segment.get('duration', 10.0)
        english_text = segment.get('english_text', '')
        word_count = len(english_text.split()) if english_text else 0
        
        # Calculate words per second
        words_per_second = word_count / duration if duration > 0 else 0
        
        # Optimal speaking rate is ~2-3 words per second
        optimal_wps = 2.5
        density_ratio = words_per_second / optimal_wps
        
        # Adjust parameters based on density
        if density_ratio > 1.3:  # High density - need faster speech
            # Increase temperature for more dynamic/faster speech
            temperature = min(base_temperature + 0.3, 2.0)
            cfg_scale = max(base_cfg_scale - 0.5, 1.5)  # Lower for more natural fast speech
            top_p = max(base_top_p - 0.05, 0.85)  # Slightly more focused
            speech_type = "fast"
        elif density_ratio < 0.7:  # Low density - need slower speech  
            # Decrease temperature for more controlled/slower speech
            temperature = max(base_temperature - 0.2, 0.8)
            cfg_scale = min(base_cfg_scale + 0.3, 4.0)  # Higher for more controlled speech
            top_p = min(base_top_p + 0.03, 0.98)  # Slightly more diverse
            speech_type = "slow"
        else:  # Normal density
            temperature = base_temperature
            cfg_scale = base_cfg_scale
            top_p = base_top_p
            speech_type = "normal"
        
        logger.info(f"Segment {segment.get('segment_index', 0)}: {word_count} words in {duration:.1f}s "
                   f"({words_per_second:.1f} wps) → {speech_type} speech (T={temperature:.1f}, CFG={cfg_scale:.1f})")
        
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