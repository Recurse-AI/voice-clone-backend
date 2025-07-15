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
from pathlib import Path
import time
import json
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available, using fallback audio adjustment")

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
            print(f"Loading Dia model from repo: {repo_id}")
            print(f"Device: {self.device}, Compute dtype: {compute_dtype}")
            
            self.dia_model = Dia.from_pretrained(repo_id, compute_dtype=compute_dtype)
            print("Dia model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading Dia model: {e}")
            raise Exception(f"Error loading Dia model: {e}")
        
    def is_model_loaded(self) -> bool:
        """Check if Dia model is loaded"""
        is_loaded = self.dia_model is not None
        print(f"Model loaded check: {is_loaded}")
        return is_loaded
    
    def clone_voice_segments(self, segments: List[Dict], temperature: float = 1.2,
                           cfg_scale: float = 3.0, top_p: float = 0.95,
                           seed: Optional[int] = None) -> Dict[str, Any]:
        if not self.is_model_loaded():
            return {"success": False, "error": "Dia model not loaded"}
        
        if not segments:
            return {"success": False, "error": "No segments provided"}
        
        print(f"Starting voice cloning for {len(segments)} segments")
        
        try:
            used_seed = seed or settings.DEFAULT_SEED
            set_seed(used_seed)
            
            # Check for reference audio and text - REQUIRED
            reference_audio_path = None
            reference_text = None
            
            if segments and segments[0].get('reference_audio_path'):
                reference_audio_path = segments[0]['reference_audio_path']
                reference_text = self._load_reference_text(reference_audio_path)
            
            # Throw error if reference is missing
            if not reference_audio_path:
                raise ValueError("Reference audio path is missing. Voice cloning requires a reference audio.")
            
            if not os.path.exists(reference_audio_path):
                raise ValueError(f"Reference audio file not found: {reference_audio_path}")
            
            if not reference_text or not reference_text.strip():
                raise ValueError("Reference text is missing or empty. Voice cloning requires reference text.")
            
            print(f"Reference Audio: {reference_audio_path}")
            print(f"Reference Text: {reference_text}")
            
            cloning_start_time = time.time()
            
            cloned_segments = []
            for i, segment in enumerate(segments):
                english_text = segment.get('english_text', segment.get('text', ''))
                if not english_text.strip():
                    print(f"Skipping segment {i+1}: No english_text")
                    continue
                
                combined_display = reference_text + '\n' + english_text
                print(f"Combined Text: {combined_display}")
                
                print(f"Processing segment {i+1}...")
                
                # Get target duration
                target_duration = segment.get('duration', 5.0)
                
                # Try adaptive generation first if enabled
                cloned_audio = None
                if settings.ADAPTIVE_GENERATION:
                    cloned_audio = self._generate_with_duration_control(
                        english_text, reference_audio_path, reference_text,
                        target_duration, temperature, cfg_scale, top_p
                    )
                
                if cloned_audio is None:
                    # Fallback to standard generation with post-processing
                    cloned_audio = self._generate_single_segment(
                        english_text, reference_audio_path, reference_text, 
                        temperature, cfg_scale, top_p
                    )
                    
                    if cloned_audio is None:
                        print(f"Skipping segment {i+1}: Failed to generate audio")
                        continue
                    
                    # Apply time adjustment
                    cloned_audio = self._adjust_audio_length(
                        cloned_audio, 
                        target_duration, 
                        use_speed_adjustment=settings.USE_SPEED_ADJUSTMENT,
                        speed_factor=settings.AUDIO_SPEED_FACTOR
                    )
                
                print(f"Generated audio shape: {cloned_audio.shape if hasattr(cloned_audio, 'shape') else 'No shape'}")
                print(f"Successfully generated audio for segment {i+1}")
                
                cloned_segments.append({
                    "success": True,
                    "original_data": segment,
                    "cloned_audio": cloned_audio,
                    "duration": target_duration
                })
                
                self._cleanup_memory()
            
            cloning_end_time = time.time()
            cloning_duration = cloning_end_time - cloning_start_time
            
            print(f"Cloning Time: {cloning_duration:.2f} seconds")
            
            return {
                "success": True,
                "cloned_segments": cloned_segments,
                "total_segments": len(segments),
                "successful_clones": len(cloned_segments),
                "seed_used": used_seed,
                "cloning_duration": cloning_duration
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            self._cleanup_memory()
    
    def _load_reference_text(self, reference_audio_path: str) -> Optional[str]:
        """Load reference text from metadata"""
        try:
            reference_file = Path(reference_audio_path)
            metadata_file = reference_file.parent / f"{reference_file.stem}_metadata.json"
            
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                reference_metadata = json.load(f)
            
            return reference_metadata.get('english_text', '')
            
        except Exception:
            return None
    
    def _generate_single_segment(self, text: str, reference_audio_path: str, 
                               reference_text: str, temperature: float, 
                               cfg_scale: float, top_p: float) -> Optional[np.ndarray]:
        try:
            # Both reference text and audio are required
            if not reference_text or not reference_text.strip():
                raise ValueError("Reference text is required for voice cloning")
            
            if not reference_audio_path or not os.path.exists(reference_audio_path):
                raise ValueError(f"Reference audio file is required and must exist: {reference_audio_path}")
            
            # Combine reference text with target text
            combined_text = reference_text.strip() + "\n" + text.strip()
            
            if not combined_text:
                raise ValueError("Combined text is empty")
            
            print(f"Generating audio for text: {combined_text[:100]}...")
            print(f"Using reference audio: {reference_audio_path}")
            
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
            
            print(f"Audio generation completed, shape: {audio.shape if hasattr(audio, 'shape') else 'No shape'}")
            return audio
            
        except Exception as e:
            print(f"Audio generation failed: {str(e)}")
            return None
    
    def _generate_with_duration_control(self, text: str, reference_audio_path: str,
                                      reference_text: str, target_duration: float,
                                      temperature: float, cfg_scale: float, 
                                      top_p: float) -> Optional[np.ndarray]:
        """Generate audio with duration control by adjusting generation parameters"""
        try:
            # Calculate approximate tokens needed for target duration
            # Dia generates ~10-15 tokens per second of audio
            tokens_per_second = 12  # Average
            target_tokens = int(target_duration * tokens_per_second)
            
            # Adjust max_tokens but keep it within reasonable bounds
            max_tokens = min(max(target_tokens, 256), settings.DIA_MAX_TOKENS)
            
            # Adjust temperature slightly based on duration needs
            # Lower temperature for shorter segments, higher for longer
            duration_factor = target_duration / 5.0  # Normalize around 5 seconds
            adjusted_temperature = temperature * (0.9 + 0.2 * min(duration_factor, 1.5))
            
            print(f"Adaptive generation: target {target_duration}s, using max_tokens={max_tokens}, temp={adjusted_temperature:.2f}")
            
            # Combine reference text with target text
            combined_text = reference_text.strip() + "\n" + text.strip()
            
            with torch.inference_mode():
                audio = self.dia_model.generate(
                    text=combined_text,
                    audio_prompt=reference_audio_path,
                    use_torch_compile=False,
                    cfg_scale=cfg_scale,
                    temperature=adjusted_temperature,
                    top_p=top_p,
                    cfg_filter_top_k=settings.DIA_CFG_FILTER_TOP_K,
                    max_tokens=max_tokens,
                    verbose=False
                )
            
            # Check if generated audio is close to target duration
            if audio is not None and len(audio) > 0:
                generated_duration = len(audio) / self.sample_rate
                duration_ratio = generated_duration / target_duration
                
                # If within 20% of target, use it with minor adjustment
                if 0.8 <= duration_ratio <= 1.2:
                    print(f"Generated duration {generated_duration:.2f}s is close to target, applying minor adjustment")
                    return self._adjust_audio_length(audio, target_duration, use_speed_adjustment=True)
                else:
                    print(f"Generated duration {generated_duration:.2f}s is too far from target {target_duration}s")
                    return None
            
            return None
            
        except Exception as e:
            print(f"Adaptive generation failed: {str(e)}")
            return None
    
    def _adjust_audio_length(self, audio: np.ndarray, target_duration: float, 
                          use_speed_adjustment: bool = False, speed_factor: float = 0.75) -> np.ndarray:
        """Adjust audio length to match target duration with pitch preservation"""
        if audio is None or len(audio) == 0:
            return np.zeros(int(target_duration * self.sample_rate), dtype=np.float32)
        
        # Ensure audio is float32 numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        target_samples = int(target_duration * self.sample_rate)
        current_samples = len(audio)
        current_duration = current_samples / self.sample_rate
        
        print(f"Adjusting audio: {current_duration:.2f}s -> {target_duration:.2f}s")
        
        # Calculate the ratio for time stretching
        stretch_ratio = target_duration / current_duration if current_duration > 0 else 1.0
        
        # Only apply time stretching for significant differences (>10%)
        if use_speed_adjustment and abs(1.0 - stretch_ratio) > 0.1:
            try:
                if LIBROSA_AVAILABLE:
                    # Use librosa for high-quality pitch-preserving time stretching
                    print(f"Using librosa time-stretch with ratio {stretch_ratio:.2f}")
                    
                    # Apply time stretching while preserving pitch
                    adjusted_audio = librosa.effects.time_stretch(audio, rate=1.0/stretch_ratio)
                    
                    # Ensure we have exactly the target length
                    if len(adjusted_audio) > target_samples:
                        adjusted_audio = adjusted_audio[:target_samples]
                    elif len(adjusted_audio) < target_samples:
                        padding = target_samples - len(adjusted_audio)
                        adjusted_audio = np.pad(adjusted_audio, (0, padding), mode='constant', constant_values=0)
                    
                else:
                    # Fallback: Use phase vocoder-like approach with FFT
                    print(f"Using FFT-based time stretching with ratio {stretch_ratio:.2f}")
                    adjusted_audio = self._fft_time_stretch(audio, stretch_ratio)
                    
                    # Ensure exact target length
                    if len(adjusted_audio) > target_samples:
                        adjusted_audio = adjusted_audio[:target_samples]
                    elif len(adjusted_audio) < target_samples:
                        padding = target_samples - len(adjusted_audio)
                        adjusted_audio = np.pad(adjusted_audio, (0, padding), mode='constant', constant_values=0)
                        
            except Exception as e:
                print(f"Time stretching failed: {e}, falling back to simple adjustment")
                use_speed_adjustment = False
        
        # Simple padding or truncation (fallback or for small adjustments)
        if not use_speed_adjustment or abs(1.0 - stretch_ratio) <= 0.1:
            if current_samples > target_samples:
                # Truncate with fade-out
                adjusted_audio = audio[:target_samples]
                
                # Apply longer fade-out (100ms) to avoid clicks
                fade_samples = min(int(0.1 * self.sample_rate), target_samples // 5)
                if fade_samples > 0:
                    fade_curve = np.linspace(1.0, 0.0, fade_samples)
                    adjusted_audio[-fade_samples:] *= fade_curve
                    
            elif current_samples < target_samples:
                # Pad with silence (no noise to avoid artifacts)
                padding_needed = target_samples - current_samples
                adjusted_audio = np.pad(audio, (0, padding_needed), mode='constant', constant_values=0)
                
            else:
                adjusted_audio = audio
        
        # Ensure float32 output
        adjusted_audio = adjusted_audio.astype(np.float32)
        
        # Apply gentle normalization to prevent clipping while preserving dynamics
        max_val = np.abs(adjusted_audio).max()
        if max_val > 0.95:
            # Use soft limiting instead of hard clipping
            adjusted_audio = np.tanh(adjusted_audio * 0.95 / max_val) * 0.95
        
        # Apply very subtle high-frequency enhancement to compensate for any dulling
        if LIBROSA_AVAILABLE and use_speed_adjustment:
            try:
                # Gentle high-shelf filter to restore clarity
                adjusted_audio = self._apply_high_shelf(adjusted_audio, gain_db=1.5)
            except:
                pass
        
        return adjusted_audio
    
    def _fft_time_stretch(self, audio: np.ndarray, stretch_ratio: float) -> np.ndarray:
        """FFT-based time stretching as fallback when librosa is not available"""
        try:
            # Use overlapping windows for smoother results
            window_size = 2048
            hop_size = window_size // 4
            
            # Apply window function
            window = np.hanning(window_size)
            
            # Calculate number of frames
            n_frames = (len(audio) - window_size) // hop_size + 1
            
            # Initialize output
            output_hop = int(hop_size * stretch_ratio)
            output_length = int(len(audio) * stretch_ratio)
            output = np.zeros(output_length)
            
            # Process each frame
            for i in range(n_frames):
                # Extract frame
                start = i * hop_size
                end = start + window_size
                if end > len(audio):
                    break
                
                frame = audio[start:end] * window
                
                # FFT
                spectrum = np.fft.rfft(frame)
                
                # Inverse FFT
                stretched_frame = np.fft.irfft(spectrum, n=window_size)
                
                # Overlap-add
                out_start = int(i * output_hop)
                out_end = out_start + window_size
                if out_end > output_length:
                    out_end = output_length
                    stretched_frame = stretched_frame[:out_end - out_start]
                
                output[out_start:out_end] += stretched_frame * window[:len(stretched_frame)]
            
            return output
            
        except Exception as e:
            print(f"FFT time stretch failed: {e}")
            # Fallback to simple resampling
            return np.interp(
                np.linspace(0, len(audio) - 1, int(len(audio) * stretch_ratio)),
                np.arange(len(audio)),
                audio
            )
    
    def _apply_high_shelf(self, audio: np.ndarray, gain_db: float = 2.0, freq: float = 8000) -> np.ndarray:
        """Apply high-shelf filter to restore high frequencies"""
        if not LIBROSA_AVAILABLE:
            return audio
        
        try:
            # Design a gentle high-shelf filter
            nyquist = self.sample_rate / 2
            normalized_freq = freq / nyquist
            
            # Simple high-pass filter as approximation
            b = [1.0 + gain_db/20, -1.0]
            a = [1.0, -0.95]
            
            # Apply filter
            from scipy import signal
            filtered = signal.lfilter(b, a, audio)
            
            # Mix with original (parallel processing)
            return audio * 0.8 + filtered * 0.2
            
        except:
            return audio
    
    def _cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def clear_cache(self):
        """Clear any cached data"""
        self._cleanup_memory() 