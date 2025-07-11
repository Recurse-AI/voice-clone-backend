"""
Audio Utilities Module

Handles audio manipulation operations like silence detection, artifact detection, 
and audio length adjustment.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple, Dict, Any
import subprocess
import tempfile
import os
from pathlib import Path


class AudioUtils:
    """Utilities for audio processing and manipulation"""
    
    def __init__(self):
        pass
    
    def extract_audio_from_video(self, video_path: str, output_path: str) -> Dict[str, Any]:
        """Extract audio from video file using ffmpeg"""
        try:
            # Get FFmpeg executable path
            ffmpeg_path = self._get_ffmpeg_path()
            
            cmd = [
                ffmpeg_path, '-y',
                '-i', video_path,
                '-ac', '2',  # Stereo
                '-ar', '44100',  # Sample rate
                '-acodec', 'pcm_s16le',  # Audio codec
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {"success": True, "audio_path": output_path}
            else:
                return {"success": False, "error": f"FFmpeg error: {result.stderr}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def download_audio_file(self, url: str, output_path: str) -> Dict[str, Any]:
        """Download audio file from URL"""
        try:
            import requests
            
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify file was downloaded
            if not os.path.exists(output_path):
                return {"success": False, "error": "File not found after download"}
            
            if os.path.getsize(output_path) == 0:
                return {"success": False, "error": "Downloaded file is empty"}
            
            return {"success": True, "file_path": output_path}
            
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Download failed: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def _get_ffmpeg_path(self):
        """Get FFmpeg executable path based on platform"""
        # Check if ffmpeg is in PATH
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return 'ffmpeg'
        except FileNotFoundError:
            pass
        
        # Check local Windows installation
        import platform
        if platform.system() == 'Windows':
            local_ffmpeg = Path('./ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe')
            if local_ffmpeg.exists():
                return str(local_ffmpeg)
        
        # Default to system ffmpeg
        return 'ffmpeg'
    
    @staticmethod
    def detect_silent_parts(audio: np.ndarray, sr: int, 
                           silence_threshold: float = 0.01, 
                           min_silence_duration: float = 0.5) -> List[Tuple[float, float]]:
        """Detect silent parts in audio"""
        # Calculate RMS energy
        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(y=audio, hop_length=hop_length, frame_length=frame_length)[0]
        
        # Convert to time
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Find silent frames
        silent_frames = rms < silence_threshold
        
        # Group consecutive silent frames
        silent_parts = []
        start_time = None
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and start_time is None:
                start_time = times[i]
            elif not is_silent and start_time is not None:
                duration = times[i] - start_time
                if duration >= min_silence_duration:
                    silent_parts.append((start_time, times[i]))
                start_time = None
        
        # Check if the last segment is silent
        if start_time is not None:
            duration = times[-1] - start_time
            if duration >= min_silence_duration:
                silent_parts.append((start_time, times[-1]))
        
        return silent_parts
    
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
    def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to target dB level"""
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2))
        
        if rms == 0:
            return audio
        
        # Calculate target RMS from dB
        target_rms = 10**(target_db / 20.0)
        
        # Apply gain
        gain = target_rms / rms
        normalized_audio = audio * gain
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized_audio))
        if max_val > 1.0:
            normalized_audio = normalized_audio / max_val
        
        return normalized_audio
    
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
    
    def apply_fade(self, audio: np.ndarray, fade_in: float = 0.1, fade_out: float = 0.1, sr: int = 44100) -> np.ndarray:
        """Apply fade in/out to audio"""
        fade_in_samples = int(fade_in * sr)
        fade_out_samples = int(fade_out * sr)
        
        # Apply fade in
        if fade_in_samples > 0 and len(audio) > fade_in_samples:
            fade_in_curve = np.linspace(0, 1, fade_in_samples)
            audio[:fade_in_samples] *= fade_in_curve
        
        # Apply fade out
        if fade_out_samples > 0 and len(audio) > fade_out_samples:
            fade_out_curve = np.linspace(1, 0, fade_out_samples)
            audio[-fade_out_samples:] *= fade_out_curve
        
        return audio
    
    def mix_audio_tracks(self, track1: np.ndarray, track2: np.ndarray, 
                        ratio1: float = 0.8, ratio2: float = 0.2) -> np.ndarray:
        """Mix two audio tracks with specified ratios"""
        # Match lengths
        min_length = min(len(track1), len(track2))
        track1 = track1[:min_length]
        track2 = track2[:min_length]
        
        # Mix with ratios
        mixed = track1 * ratio1 + track2 * ratio2
        
        # Prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val
        
        return mixed 