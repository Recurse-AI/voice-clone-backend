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
                pass  # Don't fail if status update fails
        
        try:
            base_seed = seed or settings.DEFAULT_SEED
            cloning_start_time = time.time()
            
            cloned_segments = []
            speaker_seeds = {}  # Keep same seed per speaker
            
            for i, segment in enumerate(segments):
                try:
                    if not segment or not isinstance(segment, dict):
                        logger.warning(f"Skipping segment {i+1}: Invalid segment data")
                        continue
                    
                    # Get segment data
                    english_text = segment.get('english_text', segment.get('text', ''))
                    
                    if not english_text.strip():
                        logger.warning(f"Skipping segment {i+1}: No english_text")
                        continue
                    
                    # Analyze text for Dia optimization
                    text_analysis = self._analyze_text_for_dia(english_text)
                    logger.info(f"Text analysis: {text_analysis['line_count']} lines, {text_analysis['avg_words_per_line']:.1f} avg words/line, strategy: {text_analysis['optimization_strategy']}")
                    
                    # Use segment's own audio as reference
                    audio_path = segment.get('audio_path')
                    if not audio_path or not os.path.exists(audio_path):
                        logger.warning(f"Skipping segment {i+1}: No audio file found at {audio_path}")
                        continue
                    
                    # Get speaker and set consistent seed per speaker
                    speaker = segment.get('speaker', 'A')
                    if speaker not in speaker_seeds:
                        speaker_seeds[speaker] = base_seed + (ord(speaker) - ord('A'))
                    
                    set_seed(speaker_seeds[speaker])
                    
                    logger.info(f"Processing segment {i+1}/{len(segments)} (Speaker {speaker})...")
                    logger.info(f"Using segment audio as reference: {audio_path}")
                    logger.info(f"Text: {english_text}")
                    
                    # Adjust Dia parameters based on text characteristics
                    optimized_params = self._get_optimized_dia_params(
                        text_analysis, temperature, cfg_scale, top_p
                    )
                    logger.info(f"Optimized Dia params: temp={optimized_params['temperature']:.2f}, cfg={optimized_params['cfg_scale']:.1f}, top_p={optimized_params['top_p']:.2f}")
                    
                    # Update status with current segment progress
                    if audio_id:
                        try:
                            from status_manager import status_manager
                            from status_manager import ProcessingStatus
                            progress = 60 + int((i / len(segments)) * 30)  # 60-90% range for voice cloning
                            status_manager.update_status(
                                audio_id, 
                                ProcessingStatus.PROCESSING, 
                                progress=progress,
                                details={"message": f"Processing voice cloning: segment {i+1}/{len(segments)} (Speaker {speaker})"}
                            )
                        except:
                            pass
                    
                    # Get target duration
                    target_duration = segment.get('duration', 5.0)
                    
                    try:
                        # Generate audio using optimized parameters
                        cloned_audio = self._generate_single_segment(
                            english_text, audio_path, english_text, 
                            optimized_params['temperature'], 
                            optimized_params['cfg_scale'], 
                            optimized_params['top_p']
                        )
                        
                        if cloned_audio is None:
                            logger.warning(f"Skipping segment {i+1}: Failed to generate audio")
                            continue
                        
                        # Apply time adjustment with text-aware parameters
                        cloned_audio = self._adjust_audio_length(
                            cloned_audio, 
                            target_duration, 
                            use_speed_adjustment=settings.USE_SPEED_ADJUSTMENT,
                            speed_factor=settings.AUDIO_SPEED_FACTOR,
                            text_analysis=text_analysis
                        )
                        
                        success_msg = f"Successfully generated audio for segment {i+1}"
                        logger.info(success_msg)
                        
                        cloned_segments.append({
                            "success": True,
                            "original_data": segment,
                            "cloned_audio": cloned_audio,
                            "duration": target_duration,
                            "speaker": speaker,
                            "text_analysis": text_analysis
                        })
                        
                    except Exception as generation_error:
                        logger.error(f"Error generating audio for segment {i+1}: {generation_error}")
                        continue
                    
                    # Clear memory after each segment to prevent crashes
                    self._cleanup_memory()
                    
                    # Additional cleanup for large segments
                    if i > 0 and (i + 1) % 10 == 0:
                        logger.info(f"Processed {i+1} segments, performing deep cleanup...")
                        import gc
                        import torch
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                
                except Exception as segment_error:
                    logger.error(f"Critical error processing segment {i+1}: {segment_error}")
                    continue
            
            cloning_end_time = time.time()
            cloning_duration = cloning_end_time - cloning_start_time
            
            logger.info(f"Cloning Time: {cloning_duration:.2f} seconds")
            
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
                            use_speed_adjustment: bool = False, speed_factor: float = 0.75,
                            text_analysis: Optional[Dict] = None) -> np.ndarray:
        """Adjust audio length with text-aware optimization"""
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
        
        # Text-aware thresholds
        if text_analysis:
            quality = text_analysis.get('estimated_quality', 'fair')
            strategy = text_analysis.get('optimization_strategy', 'acceptable')
            
            if quality == "excellent" or strategy == "optimal_balanced":
                min_ratio, max_ratio, tolerance = 0.92, 1.08, 0.03
            elif quality == "poor" or strategy == "very_short_lines":
                min_ratio, max_ratio, tolerance = 0.88, 1.12, 0.08
            else:
                min_ratio, max_ratio, tolerance = 0.90, 1.10, 0.05
        else:
            min_ratio, max_ratio, tolerance = settings.MIN_STRETCH_RATIO, settings.MAX_STRETCH_RATIO, 0.05
        
        # Apply stretching
        needs_stretching = (
            use_speed_adjustment and 
            abs(1.0 - stretch_ratio) > tolerance and
            min_ratio <= stretch_ratio <= max_ratio
        )
        
        if needs_stretching:
            try:
                if text_analysis:
                    logger.info(f"Text-aware stretching: ratio {stretch_ratio:.2f}, quality: {text_analysis.get('estimated_quality', 'unknown')}")
                else:
                    logger.info(f"Standard stretching: ratio {stretch_ratio:.2f}")
                adjusted_audio = librosa.effects.time_stretch(audio, rate=1.0/stretch_ratio)
            except Exception as e:
                logger.error(f"Time stretching failed: {e}")
                adjusted_audio = audio
        else:
            adjusted_audio = audio
        
        # Handle length differences
        if len(adjusted_audio) > target_samples:
            adjusted_audio = adjusted_audio[:target_samples]
            
            # Adaptive fade based on text quality
            if text_analysis and text_analysis.get('estimated_quality') == 'excellent':
                fade_samples = min(int(0.15 * self.sample_rate), target_samples // 4)
            else:
                fade_samples = min(int(0.1 * self.sample_rate), target_samples // 5)
                
            if fade_samples > 0:
                fade_curve = np.linspace(1.0, 0.0, fade_samples)
                adjusted_audio[-fade_samples:] *= fade_curve
                
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
    
    def _analyze_text_for_dia(self, text: str) -> Dict:
        """Analyze text characteristics for Dia optimization"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        words_per_line = []
        total_words = 0
        
        for line in lines:
            # Remove speaker tags for word counting
            clean_line = line
            if line.startswith('[S') and ']' in line:
                clean_line = line[line.find(']') + 1:].strip()
            
            words = len(clean_line.split()) if clean_line else 0
            words_per_line.append(words)
            total_words += words
        
        line_count = len(lines)
        avg_words_per_line = total_words / line_count if line_count > 0 else 0
        
        # Determine strategy
        if line_count == 1:
            strategy = "single_line"
        elif line_count <= 3 and avg_words_per_line >= 6:
            strategy = "optimal_balanced"
        elif line_count > 5:
            strategy = "too_many_lines"
        elif any(w <= 1 for w in words_per_line):  # Relaxed from <= 2 to <= 1
            strategy = "very_short_lines"
        else:
            strategy = "acceptable"
        
        return {
            'line_count': line_count,
            'total_words': total_words,
            'words_per_line': words_per_line,
            'avg_words_per_line': avg_words_per_line,
            'optimization_strategy': strategy,
            'estimated_quality': self._estimate_dia_quality(line_count, avg_words_per_line, words_per_line)
        }
    
    def _estimate_dia_quality(self, line_count: int, avg_words_per_line: float, words_per_line: List[int]) -> str:
        """Estimate Dia performance quality"""
        has_very_short_lines = any(w <= 1 for w in words_per_line)  # Relaxed
        has_too_many_lines = line_count > 6
        optimal_word_range = 6 <= avg_words_per_line <= 12
        
        if has_very_short_lines:
            return "poor"
        elif has_too_many_lines:
            return "inconsistent"
        elif optimal_word_range and line_count <= 4:
            return "excellent"
        elif optimal_word_range:
            return "good"
        else:
            return "fair"
    
    def _get_optimized_dia_params(self, text_analysis: Dict, base_temp: float, 
                                 base_cfg: float, base_top_p: float) -> Dict:
        """Optimize Dia parameters based on text characteristics"""
        strategy = text_analysis['optimization_strategy']
        quality = text_analysis['estimated_quality']
        
        # Start with base parameters
        temp = base_temp
        cfg = base_cfg  
        top_p = base_top_p
        
        # Adjust based on analysis
        if strategy == "too_many_lines":
            temp = max(base_temp * 0.8, 0.8)
            cfg = min(base_cfg * 1.2, 4.5)
            top_p = max(base_top_p * 0.9, 0.85)
        elif strategy == "very_short_lines":
            temp = min(base_temp * 1.15, 1.8)
            cfg = max(base_cfg * 0.85, 2.5)
            top_p = min(base_top_p * 1.05, 0.98)
        elif strategy == "single_line":
            temp = max(base_temp * 0.9, 0.9) 
        elif quality == "poor":
            temp = min(temp * 1.1, 1.6)
            cfg = max(cfg * 0.9, 2.0)
        
        # Ensure bounds
        temp = max(0.6, min(2.0, temp))
        cfg = max(1.5, min(5.0, cfg))
        top_p = max(0.8, min(0.99, top_p))
        
        return {
            'temperature': temp,
            'cfg_scale': cfg,
            'top_p': top_p,
            'reasoning': f"{strategy}, {quality}, {text_analysis['line_count']} lines"
        }
    
    def _cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def clear_cache(self):
        """Clear any cached data"""
        self._cleanup_memory() 