"""
Audio Utilities Module

Handles audio manipulation operations like silence detection, artifact detection, 
and audio length adjustment.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any

import librosa
import numpy as np
import soundfile as sf
from app.config.settings import settings


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
            
            # Memory cleanup after download
            del response
            import gc
            gc.collect()
            
            return {"success": True, "audio_path": output_path}
         
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Download failed: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def _get_ffmpeg_path(self):
        """Get FFmpeg executable path based on platform"""
        from app.utils.ffmpeg_helper import get_ffmpeg_path
        return get_ffmpeg_path()
    
    
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
    
    def split_audio_by_timestamps(self, input_audio_path: str, output_dir: str, 
                                 segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Split audio file into segments based on timestamps using ffmpeg"""
        try:
            # Get FFmpeg executable path
            ffmpeg_path = self._get_ffmpeg_path()
            if not ffmpeg_path:
                return {"success": False, "error": "FFmpeg not found"}
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            split_files = []
            
            for i, segment in enumerate(segments):
                start_ms = segment.get('start', 0)
                end_ms = segment.get('end', 0)
                text = segment.get('text', f'segment_{i}')
                
                # Convert milliseconds to seconds
                start_sec = start_ms / 1000.0
                duration_sec = (end_ms - start_ms) / 1000.0
                
                # Create simple English filename - no text content
                output_filename = f"segment_{i:03d}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                # FFmpeg command to extract segment
                cmd = [
                    ffmpeg_path, '-y',
                    '-i', input_audio_path,
                    '-ss', str(start_sec),
                    '-t', str(duration_sec),
                    '-ac', '2',  # Stereo
                    '-ar', '44100',  # Sample rate
                    '-acodec', 'pcm_s16le',  # Audio codec
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    split_files.append({
                        "index": i,
                        "text": text,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_ms": end_ms - start_ms,
                        "output_path": output_path
                    })
                else:
                    return {"success": False, "error": f"FFmpeg error for segment {i}: {result.stderr}"}
            
            return {
                "success": True,
                "segments_count": len(split_files),
                "split_files": split_files,
                "output_directory": output_dir
            }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def mix_audio_files(file1: str, file2: str, out_path: str, ratio1: float = 0.75, ratio2: float = 0.25):
        """Mix two audio files with given ratios and write to out_path"""
        import soundfile as sf
        audio1, sr1 = sf.read(file1)
        audio2, sr2 = sf.read(file2)
        if sr1 != sr2:
            raise Exception("Sample rate mismatch")
        min_len = min(len(audio1), len(audio2))
        mixed = ratio1 * audio1[:min_len] + ratio2 * audio2[:min_len]
        sf.write(out_path, mixed, sr1)
        return out_path

    @staticmethod
    def remove_temp_dir(job_id: str = None, folder_path: str = None) -> bool:
        """Remove temporary directory by job_id or explicit folder_path.

        Either job_id or folder_path must be provided. If job_id is given, the
        directory is resolved under settings.TEMP_DIR.
        Returns True if deletion attempted (regardless of existing), False if
        parameters were invalid.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not job_id and not folder_path:
            logger.warning("AudioUtils.remove_temp_dir: No job_id or folder_path provided")
            return False
        if folder_path is None:
            folder_path = os.path.join(settings.TEMP_DIR, job_id)
        
        try:
            import shutil
            import os
            
            if os.path.exists(folder_path):
                # Count files before deletion for logging
                file_count = 0
                for root, dirs, files in os.walk(folder_path):
                    file_count += len(files)
                
                shutil.rmtree(folder_path, ignore_errors=True)
                logger.info(f"🧹 Successfully removed temp directory: {folder_path} ({file_count} files)")
                return True
            else:

                return True
        except Exception as e:
            logger.error(f"🧹 Failed to remove temp directory {folder_path}: {e}")
            return False
    