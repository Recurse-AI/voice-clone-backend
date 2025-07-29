"""
Clean Voice Cloning Module - Inspired by Gradio example
Simplified, stable, and consistent voice cloning
"""

import torch
import numpy as np
import soundfile as sf
import warnings
import gc
import time
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dia.model import Dia
from config import settings

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Sets the random seed for reproducibility (from Gradio example)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_silence(audio: np.ndarray, duration_sec: float = 1.0, sample_rate: int = 44100) -> np.ndarray:
    """Add silence to the end of an audio segment (from Gradio example)"""
    silence_samples = int(duration_sec * sample_rate)
    silence = np.zeros(silence_samples, dtype=audio.dtype)
    return np.concatenate([audio, silence])

def apply_speed_adjustment(audio: np.ndarray, speed: float) -> np.ndarray:
    """Apply speed adjustment using interpolation (from Gradio example)"""
    if speed == 1.0:
        return audio
    
    orig_len = len(audio)
    target_len = int(orig_len / speed)
    x_orig = np.arange(orig_len)
    x_new = np.linspace(0, orig_len-1, target_len)
    return np.interp(x_new, x_orig, audio)

class Args:
    """Simple class to hold arguments for the model (from Gradio example)"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class CleanVoiceCloningService:
    """
    Clean, stable voice cloning service inspired by Gradio example
    Simplified architecture with consistent seed handling and retry logic
    """
    
    def __init__(self):
        self.dia_model = None
        self.device = self._detect_device()
        self.sample_rate = 44100
        
        # Stable default parameters (from config)
        self.default_args = Args(
            tokens_per_chunk=settings.DIA_ENHANCED_MAX_TOKENS,
            cfg_scale=settings.DIA_ENHANCED_CFG_SCALE,
            temperature=settings.DIA_ENHANCED_TEMPERATURE,
            top_p=settings.DIA_ENHANCED_TOP_P,
            cfg_filter_top_k=settings.DIA_ENHANCED_CFG_FILTER_TOP_K,
            speed=settings.DIA_ENHANCED_SPEED_FACTOR,
            use_torch_compile=settings.DIA_ENHANCED_USE_TORCH_COMPILE,
            silence=settings.DIA_SILENCE_PADDING,
            seed=settings.DIA_DEFAULT_SEED,
            max_retries=settings.DIA_MAX_RETRIES
        )
        
        logger.info(f"Initialized CleanVoiceCloningService on device: {self.device}")
    
    def _detect_device(self):
        """Detect the best available device for inference (from Gradio example)"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(self, model_name: str = "nari-labs/Dia-1.6B-0626") -> bool:
        """Load Dia model with proper error handling"""
        try:
            logger.info(f"Loading Dia model from {model_name}...")
            start_time = time.time()
            
            compute_dtype = "bfloat16" if self.device.type == "cuda" else "float32"
            self.dia_model = Dia.from_pretrained(
                model_name, 
                compute_dtype=compute_dtype, 
                device=self.device
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Dia model: {str(e)}")
            return False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.dia_model is not None
    
    def generate_with_retry(self, chunk_text: str, audio_prompt: Optional[str] = None, 
                           custom_args: Optional[Args] = None) -> Optional[np.ndarray]:
        """
        Generate audio with retry logic for clamping warnings (from Gradio example)
        """
        if not self.is_model_loaded():
            logger.error("Model not loaded")
            return None
        
        args = custom_args or self.default_args
        retries = 0
        
        while retries <= args.max_retries:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")  # Capture all warnings
                
                try:
                    with torch.inference_mode():
                        audio = self.dia_model.generate(
                            text=chunk_text,
                            max_tokens=args.tokens_per_chunk,
                            cfg_scale=args.cfg_scale,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            cfg_filter_top_k=args.cfg_filter_top_k,
                            use_torch_compile=args.use_torch_compile,
                            audio_prompt=audio_prompt
                        )
                    
                    # Force garbage collection after generation (from Gradio example)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Check for clamping warning (from Gradio example)
                    clamping_warning = any(
                        "Clamping" in str(warning.message)
                        for warning in w
                    )
                    
                    if clamping_warning:
                        logger.warning(f"⚠️ Clamping warning caught. Retrying generation... (attempt {retries + 1})")
                        retries += 1
                        continue  # Retry the loop
                    else:
                        # Success, apply post-processing
                        if audio is not None and len(audio) > 0:
                            # Apply speed adjustment if needed
                            if args.speed != 1.0:
                                audio = apply_speed_adjustment(audio, args.speed)
                            
                            logger.info(f"Successfully generated audio: {len(audio)} samples")
                            return audio
                        else:
                            logger.warning("Generated audio is empty")
                            return None
                
                except Exception as e:
                    logger.error(f"Generation error on attempt {retries + 1}: {str(e)}")
                    retries += 1
                    continue
        
        logger.error(f"⚠️ Max retries ({args.max_retries}) reached. Generation failed.")
        return None
    
    def process_segments_batch(self, segments: List[Dict], audio_id: str, 
                              custom_args: Optional[Args] = None) -> Dict[str, Any]:
        """
        Process multiple segments with consistent voice cloning
        """
        if not self.is_model_loaded():
            return {"success": False, "error": "Model not loaded"}
        
        if not segments:
            return {"success": False, "error": "No segments provided"}
        
        args = custom_args or self.default_args
        
        # Set global seed for consistency (key improvement from Gradio approach)
        if settings.DIA_USE_GLOBAL_SEED:
            set_seed(args.seed)
            logger.info(f"🎯 Using global seed {args.seed} for voice consistency across all segments")
        
        logger.info(f"🚀 Starting batch processing of {len(segments)} segments")
        logger.info(f"📊 Parameters: CFG={args.cfg_scale}, Temp={args.temperature}, Speed={args.speed}")
        
        results = []
        total_start_time = time.time()
        
        # Create output directory
        segments_dir = Path(settings.TEMP_DIR) / f"segments_{audio_id}" / "cloned_segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        for i, segment in enumerate(segments):
            segment_start_time = time.time()
            
            try:
                # Extract segment data
                segment_index = segment.get('segment_index', i + 1)
                english_text = segment.get('english_text', '').strip()
                duration = segment.get('duration', 10.0)
                speaker = segment.get('speaker', 'A')
                audio_path = segment.get('audio_path')
                
                logger.info(f"🎤 Processing segment {segment_index}/{len(segments)}: {duration:.2f}s, Speaker: {speaker}")
                
                # Handle silent/empty segments
                if not english_text or english_text in ['[SILENCE]', '']:
                    logger.info(f"🔇 Creating silence for segment {segment_index} ({duration:.2f}s)")
                    
                    silence_samples = int(duration * self.sample_rate)
                    silence_audio = np.zeros(silence_samples, dtype=np.float32)
                    
                    # Add padding silence if configured
                    if args.silence > 0:
                        silence_audio = add_silence(silence_audio, args.silence, self.sample_rate)
                    
                    # Save silence segment
                    output_path = segments_dir / f"cloned_segment_{segment_index:03d}.wav"
                    sf.write(output_path, silence_audio, self.sample_rate)
                    
                    results.append({
                        "success": True,
                        "segment_index": segment_index,
                        "duration": duration,
                        "speaker": speaker,
                        "output_path": str(output_path),
                        "type": "silence"
                    })
                    continue
                
                # Process speech segment
                generated_audio = self.generate_with_retry(
                    chunk_text=english_text + "\n" + english_text,
                    audio_prompt=audio_path if audio_path and Path(audio_path).exists() else None,
                    custom_args=args
                )
                
                if generated_audio is not None:
                    # Adjust duration to match target
                    target_samples = int(duration * self.sample_rate)
                    if len(generated_audio) != target_samples:
                        generated_audio = self._adjust_audio_duration(generated_audio, target_samples)
                    
                    # Add silence padding if configured
                    if args.silence > 0:
                        generated_audio = add_silence(generated_audio, args.silence, self.sample_rate)
                    
                    # Save cloned segment
                    output_path = segments_dir / f"cloned_segment_{segment_index:03d}.wav"
                    sf.write(output_path, generated_audio, self.sample_rate)
                    
                    segment_time = time.time() - segment_start_time
                    logger.info(f"✅ Segment {segment_index} completed in {segment_time:.2f}s")
                    
                    results.append({
                        "success": True,
                        "segment_index": segment_index,
                        "duration": duration,
                        "speaker": speaker,
                        "output_path": str(output_path),
                        "type": "speech",
                        "processing_time": segment_time
                    })
                else:
                    logger.error(f"❌ Failed to generate audio for segment {segment_index}")
                    results.append({
                        "success": False,
                        "segment_index": segment_index,
                        "error": "Generation failed"
                    })
                
                # Memory cleanup after each segment (key for stability)
                if i % settings.DIA_MEMORY_CLEANUP_FREQUENCY == 0:
                    self._cleanup_memory()
                
            except Exception as e:
                logger.error(f"❌ Error processing segment {segment_index}: {str(e)}")
                results.append({
                    "success": False,
                    "segment_index": segment_index,
                    "error": str(e)
                })
        
        # Final cleanup
        self._cleanup_memory()
        
        total_time = time.time() - total_start_time
        successful_segments = len([r for r in results if r.get("success", False)])
        
        logger.info(f"🎉 Batch processing completed: {successful_segments}/{len(segments)} segments successful in {total_time:.2f}s")
        
        return {
            "success": True,
            "total_segments": len(segments),
            "successful_segments": successful_segments,
            "failed_segments": len(segments) - successful_segments,
            "processing_time": total_time,
            "results": results,
            "output_directory": str(segments_dir)
        }
    
    def _adjust_audio_duration(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        """Adjust audio duration to match target (simplified from Gradio example)"""
        current_samples = len(audio)
        
        if current_samples == target_samples:
            return audio
        elif current_samples > target_samples:
            # Trim audio
            return audio[:target_samples]
        else:
            # Pad with silence
            padding = target_samples - current_samples
            silence = np.zeros(padding, dtype=audio.dtype)
            return np.concatenate([audio, silence])
    
    def _cleanup_memory(self):
        """Clean up memory after processing (from Gradio example)"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def clear_model(self):
        """Clear model from memory"""
        if self.dia_model:
            del self.dia_model
            self.dia_model = None
            self._cleanup_memory()
            logger.info("Model cleared from memory")

    def validate_parameters(self, **kwargs) -> Args:
        """Validate and create Args object with provided parameters"""
        valid_params = {
            'tokens_per_chunk': kwargs.get('max_tokens', self.default_args.tokens_per_chunk),
            'cfg_scale': kwargs.get('cfg_scale', self.default_args.cfg_scale),
            'temperature': kwargs.get('temperature', self.default_args.temperature),
            'top_p': kwargs.get('top_p', self.default_args.top_p),
            'cfg_filter_top_k': kwargs.get('cfg_filter_top_k', self.default_args.cfg_filter_top_k),
            'speed': kwargs.get('speed_factor', self.default_args.speed),
            'use_torch_compile': kwargs.get('use_torch_compile', self.default_args.use_torch_compile),
            'silence': kwargs.get('silence', self.default_args.silence),
            'seed': kwargs.get('seed', self.default_args.seed),
            'max_retries': kwargs.get('max_retries', self.default_args.max_retries)
        }
        
        return Args(**valid_params) 