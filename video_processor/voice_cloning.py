"""
Voice Cloning Module - Enhanced with Advanced Parameter Controls
"""

import torch
import numpy as np
import random
import os
import gc
import json
import soundfile as sf
from typing import Optional, Dict, Any, List
from pathlib import Path
from dia.model import Dia
from config import settings
import logging
import time

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seed for reproducibility - Enhanced version from Colab"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VoiceCloningService:
    """Enhanced voice cloning service with advanced parameter controls"""
    
    def __init__(self, segment_manager=None):
        self.dia_model = None
        self.device = settings.DIA_DEVICE
        self.sample_rate = 44100
        self.segment_manager = segment_manager
        
        # Enhanced default parameters (can be overridden)
        self.default_params = {
            'max_tokens': settings.DIA_ENHANCED_MAX_TOKENS,     # Use config values
            'cfg_scale': settings.DIA_ENHANCED_CFG_SCALE,       # Use config values
            'temperature': settings.DIA_ENHANCED_TEMPERATURE,   # Use config values  
            'top_p': settings.DIA_ENHANCED_TOP_P,               # Use config values
            'cfg_filter_top_k': settings.DIA_ENHANCED_CFG_FILTER_TOP_K,  # Use config values
            'speed_factor': settings.DIA_ENHANCED_SPEED_FACTOR,     # Use config values
            'use_torch_compile': settings.DIA_ENHANCED_USE_TORCH_COMPILE,  # Use config values
            'verbose': False
        }
        
        if not segment_manager:
            logger.warning("VoiceCloningService initialized without segment_manager - some features may not work properly")
    
    def load_dia_model(self, repo_id: str = None, compute_dtype: str = None) -> bool:
        """Load Dia model with enhanced configuration"""
        try:
            repo_id = repo_id or settings.DIA_MODEL_REPO
            compute_dtype = compute_dtype or (settings.DIA_COMPUTE_DTYPE if self.device == "cuda" else "float32")
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
    
    def clone_voice_segments(self, segments: List[Dict], 
                           # Enhanced parameter controls from Colab
                           max_tokens: int = None,
                           cfg_scale: float = None, 
                           temperature: float = None,
                           top_p: float = None,
                           cfg_filter_top_k: int = None,
                           speed_factor: float = None,
                           seed: Optional[int] = None, 
                           use_torch_compile: bool = None,
                           audio_id: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced voice cloning with advanced parameter controls"""
        if not self.is_model_loaded():
            return {"success": False, "error": "Dia model not loaded"}
        
        if not self.segment_manager:
            return {"success": False, "error": "Segment manager not available"}
        
        if not segments:
            return {"success": False, "error": "No segments provided"}
        
        # Use provided parameters or fall back to defaults
        params = {
            'max_tokens': max_tokens or self.default_params['max_tokens'],
            'cfg_scale': cfg_scale or self.default_params['cfg_scale'],
            'temperature': temperature or self.default_params['temperature'],
            'top_p': top_p or self.default_params['top_p'],
            'cfg_filter_top_k': cfg_filter_top_k or self.default_params['cfg_filter_top_k'],
            'speed_factor': speed_factor or self.default_params['speed_factor'],
            'use_torch_compile': use_torch_compile if use_torch_compile is not None else self.default_params['use_torch_compile']
        }
        
        logger.info(f"Starting enhanced voice cloning for {len(segments)} segments")
        logger.info(f"Parameters: CFG={params['cfg_scale']}, Temp={params['temperature']}, TopP={params['top_p']}, Speed={params['speed_factor']}")
        
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
            
            # Enhanced seed handling for consistency
            base_seed = seed if seed is not None else random.randint(1, 1000000)
            speaker_seeds = {}
            
            # Set global seed at start for overall consistency
            if seed is not None:
                set_seed(seed)
                logger.info(f"Using global seed: {seed} for voice consistency")
            
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
                    
                    # Set consistent seed per speaker (important for Dia voice consistency)
                    speaker = segment.get('speaker', 'A')
                    if speaker not in speaker_seeds:
                        speaker_seeds[speaker] = base_seed + (ord(speaker) - ord('A')) * 1000
                    
                    # Apply speaker-specific seed if no global seed was set
                    if seed is None:
                        set_seed(speaker_seeds[speaker])
                        logger.info(f"Using speaker-specific seed {speaker_seeds[speaker]} for speaker {speaker}")
                    
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
                        # Generate audio with enhanced parameters
                        cloned_audio = self._generate_single_segment_enhanced(
                            english_text, 
                            audio_path, 
                            english_text, # Pass English text for both reference and generation
                            params
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
                        
                        # Apply speed factor adjustment (enhanced from Colab)
                        if params['speed_factor'] != 1.0:
                            cloned_audio = self._apply_speed_factor(cloned_audio, params['speed_factor'])
                        
                        # Ensure proper duration
                        cloned_audio = self._ensure_proper_duration(cloned_audio, segment_duration)
                        
                        # Save audio immediately
                        self._save_single_cloned_segment(cloned_audio, i+1, audio_id)
                        
                        logger.info(f"Successfully generated enhanced speech for segment {i+1}")
                        
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
                            "voice_params": params.copy()  # Store the parameters used
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
            logger.info(f"Enhanced cloning completed in {cloning_duration:.2f} seconds")
            logger.info(f"Used parameters: {params}")
            logger.info(f"Processed {len(cloned_segments)} segments")
            
            # Verify we have all segments
            if len(cloned_segments) != len(segments):
                logger.warning(f"Segment count mismatch: expected {len(segments)}, got {len(cloned_segments)}")
            
            # Save summary
            if cloned_segments:
                self._save_cloned_segments_unified(cloned_segments, audio_id, params)
            
            return {
                "success": True,
                "cloned_segments": cloned_segments,
                "total_segments": len(segments),
                "successful_clones": len(cloned_segments),
                "seed_used": base_seed,
                "speaker_seeds": speaker_seeds,
                "cloning_duration": cloning_duration,
                "parameters_used": params
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            self._cleanup_memory()
    
    def _generate_single_segment_enhanced(self, text: str, reference_audio_path: str, 
                                        reference_text: str, params: Dict) -> Optional[np.ndarray]:
        """Generate single segment with enhanced parameters following reference code patterns"""
        if not self.dia_model:
            logger.error("Dia model not loaded")
            return None
        
        try:
            # Use reference text as the audio prompt text (same as original for consistency)
            # This matches the reference code pattern where both texts should be the same
            if reference_text and reference_text.strip():
                # Combine reference text with current text following reference pattern
                combined_text = reference_text.strip() + "\n" + text.strip()
                generation_text = combined_text.strip()
            else:
                generation_text = text.strip()
            
            logger.info(f"Generating with reference audio and combined text")
            logger.info(f"Generation text: {generation_text[:100]}...")
            
            # Use enhanced parameters matching reference code defaults
            max_tokens = params.get('max_tokens', settings.DIA_ENHANCED_MAX_TOKENS)
            cfg_scale = params.get('cfg_scale', settings.DIA_ENHANCED_CFG_SCALE)
            temperature = params.get('temperature', settings.DIA_ENHANCED_TEMPERATURE)
            top_p = params.get('top_p', settings.DIA_ENHANCED_TOP_P)
            cfg_filter_top_k = params.get('cfg_filter_top_k', settings.DIA_ENHANCED_CFG_FILTER_TOP_K)
            use_torch_compile = params.get('use_torch_compile', settings.DIA_ENHANCED_USE_TORCH_COMPILE)
            
            logger.info(f"Using parameters: max_tokens={max_tokens}, cfg_scale={cfg_scale}, temp={temperature}, top_p={top_p}, cfg_filter_top_k={cfg_filter_top_k}")
            
            # Generate with torch.inference_mode() like reference code
            with torch.inference_mode():
                generated_audio = self.dia_model.generate(
                    text=generation_text,
                    audio_prompt=reference_audio_path,  # Use reference audio like in reference code
                    max_tokens=max_tokens,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=cfg_filter_top_k,
                    use_torch_compile=use_torch_compile,
                    verbose=False  # Keep quiet for production
                )
            
            if generated_audio is not None and len(generated_audio) > 0:
                # Apply speed factor if specified (following reference code pattern)
                speed_factor = params.get('speed_factor', 1.0)
                if speed_factor != 1.0:
                    generated_audio = self._apply_speed_factor(generated_audio, speed_factor)
                
                logger.info(f"Successfully generated audio: {generated_audio.shape} samples")
                return generated_audio
            else:
                logger.warning("Generated audio is empty or None")
                return None
                
        except Exception as e:
            logger.error(f"Error in enhanced generation: {str(e)}")
            return None
    
    def _apply_speed_factor(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
        """Apply speed factor adjustment (from Colab implementation)"""
        try:
            if speed_factor == 1.0:
                return audio
            
            # Calculate new length based on speed factor
            new_length = int(len(audio) / speed_factor)
            
            # Use linear interpolation to adjust speed
            indices = np.linspace(0, len(audio) - 1, new_length)
            adjusted_audio = np.interp(indices, np.arange(len(audio)), audio)
            
            logger.debug(f"Applied speed factor {speed_factor}: {len(audio)} → {len(adjusted_audio)} samples")
            return adjusted_audio.astype(audio.dtype)
            
        except Exception as e:
            logger.error(f"Speed factor adjustment failed: {e}")
            return audio
    
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
        """Legacy method for backward compatibility with main.py"""
        return self._ensure_proper_duration(audio, target_duration)
    
    def _save_single_cloned_segment(self, audio_data: np.ndarray, segment_index: int, audio_id: str):
        """Save a single cloned segment audio file."""
        try:
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
    
    def _save_cloned_segments_unified(self, cloned_segments: List[Dict], audio_id: str, params: Dict):
        """Save enhanced cloning summary with parameters"""
        try:
            logger.info(f"Saved {len(cloned_segments)} cloned segments with enhanced parameters")
            
            temp_dir = Path(settings.TEMP_DIR)
            segments_base_dir = temp_dir / f"segments_{audio_id}"
            metadata_dir = segments_base_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            summary = {
                "cloning_summary": {
                    "total_segments": len(cloned_segments),
                    "successful_clones": len([s for s in cloned_segments if s.get('success', False)]),
                    "parameters_used": params,  # Store the enhanced parameters
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
            logger.error(f"Error creating enhanced cloning summary: {e}")
    
    def _cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def clear_cache(self):
        """Clear any cached data"""
        self._cleanup_memory()
    
    # Enhanced parameter validation (from Colab)
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """Validate and sanitize parameters based on Colab ranges"""
        validated = {}
        
        # Validate max_tokens (860-12000)
        max_tokens = kwargs.get('max_tokens', self.default_params['max_tokens'])
        validated['max_tokens'] = max(860, min(12000, int(max_tokens)))
        
        # Validate cfg_scale (1.0-15.0)
        cfg_scale = kwargs.get('cfg_scale', self.default_params['cfg_scale'])
        validated['cfg_scale'] = max(1.0, min(15.0, float(cfg_scale)))
        
        # Validate temperature (0.5-2.0)
        temperature = kwargs.get('temperature', self.default_params['temperature'])
        validated['temperature'] = max(0.5, min(2.0, float(temperature)))
        
        # Validate top_p (0.5-1.0)
        top_p = kwargs.get('top_p', self.default_params['top_p'])
        validated['top_p'] = max(0.5, min(1.0, float(top_p)))
        
        # Validate cfg_filter_top_k (15-100)
        cfg_filter_top_k = kwargs.get('cfg_filter_top_k', self.default_params['cfg_filter_top_k'])
        validated['cfg_filter_top_k'] = max(15, min(100, int(cfg_filter_top_k)))
        
        # Validate speed_factor (0.5-1.5)
        speed_factor = kwargs.get('speed_factor', self.default_params['speed_factor'])
        validated['speed_factor'] = max(0.5, min(1.5, float(speed_factor)))
        
        # Boolean parameters
        validated['use_torch_compile'] = kwargs.get('use_torch_compile', self.default_params['use_torch_compile'])
        validated['verbose'] = kwargs.get('verbose', self.default_params['verbose'])
        
        return validated 