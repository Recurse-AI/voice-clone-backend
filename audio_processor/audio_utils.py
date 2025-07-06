"""
Audio Utilities Module

Handles audio manipulation operations like silence detection, artifact detection, 
and audio length adjustment.
"""

import numpy as np
import librosa
from typing import List, Tuple


class AudioUtils:
    """Utilities for audio processing and manipulation"""
    
    @staticmethod
    def detect_silent_parts(audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Detect silent parts in audio"""
        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(y=audio, hop_length=hop_length, frame_length=frame_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        silence_threshold = np.percentile(rms, 10)
        silent_frames = rms < silence_threshold
        
        # Find consecutive silent periods
        silent_periods = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and not in_silence:
                in_silence = True
                silence_start = times[i]
            elif not is_silent and in_silence:
                in_silence = False
                silence_duration = times[i] - silence_start
                if silence_duration > 0.5:  # Only keep silences longer than 0.5s
                    silent_periods.append((silence_start, times[i]))
        
        # Handle case where audio ends in silence
        if in_silence:
            silence_duration = times[-1] - silence_start
            if silence_duration > 0.5:
                silent_periods.append((silence_start, times[-1]))
        
        return silent_periods
    
    @staticmethod
    def adjust_audio_length(audio: np.ndarray, target_duration: float, 
                           sample_rate: int) -> np.ndarray:
        """Adjust audio length to match target duration"""
        current_duration = len(audio) / sample_rate
        target_samples = int(target_duration * sample_rate)
        
        if len(audio) == target_samples:
            return audio
        
        if len(audio) > target_samples:
            # Truncate audio
            return audio[:target_samples]
        else:
            # Pad with silence
            padding = target_samples - len(audio)
            return np.pad(audio, (0, padding), mode='constant', constant_values=0)
    
    @staticmethod
    def detect_audio_artifacts(audio: np.ndarray, sample_rate: int) -> bool:
        """Detect audio artifacts (clipping, excessive silence, etc.)"""
        # Check for clipping
        clipping_threshold = 0.95
        if np.max(np.abs(audio)) > clipping_threshold:
            return True
        
        # Check for excessive silence
        silence_threshold = 0.01
        silent_samples = np.sum(np.abs(audio) < silence_threshold)
        if silent_samples / len(audio) > 0.8:  # More than 80% silence
            return True
        
        # Check for DC offset
        dc_offset = np.mean(audio)
        if abs(dc_offset) > 0.1:
            return True
        
        return False
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, target_loudness: float = -23.0) -> np.ndarray:
        """Normalize audio to target loudness"""
        if len(audio) == 0:
            return audio
        
        # Simple peak normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            normalized = audio / max_val
            # Apply target loudness scaling
            target_scale = 10 ** (target_loudness / 20)
            return normalized * target_scale
        
        return audio
    
    @staticmethod
    def extract_audio_features(audio: np.ndarray, sr: int) -> dict:
        """Extract basic audio features for quality assessment"""
        if len(audio) == 0:
            return {"error": "Empty audio"}
        
        # Basic metrics
        rms = librosa.feature.rms(y=audio)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        return {
            "duration": len(audio) / sr,
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "spectral_centroid_mean": float(np.mean(spectral_centroids)),
            "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
            "peak_amplitude": float(np.max(np.abs(audio))),
            "dynamic_range": float(np.max(rms) - np.min(rms)) if len(rms) > 0 else 0.0
        }
    
    @staticmethod
    def fade_in_out(audio: np.ndarray, fade_duration: float, sample_rate: int) -> np.ndarray:
        """Apply fade in and fade out to audio"""
        fade_samples = int(fade_duration * sample_rate)
        
        if fade_samples >= len(audio) // 2:
            return audio
        
        # Create fade curves
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        # Apply fades
        audio_faded = audio.copy()
        audio_faded[:fade_samples] *= fade_in
        audio_faded[-fade_samples:] *= fade_out
        
        return audio_faded 