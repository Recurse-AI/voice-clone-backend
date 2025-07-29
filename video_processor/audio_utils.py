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
import os
from pathlib import Path


class AudioUtils:
    """Utilities for audio processing and manipulation"""
    
    def __init__(self):
        pass
    
    def extract_audio_from_video(self, video_path: str, output_path: str) -> Dict[str, Any]:
        """Extract audio from video file using ffmpeg with format compatibility handling"""
        try:
            # Get FFmpeg executable path
            ffmpeg_path = self._get_ffmpeg_path()
            
            # First, try to probe the video file
            probe_result = self._probe_video_file(ffmpeg_path, video_path)
            if not probe_result["success"]:
                return {"success": False, "error": f"Cannot read video file: {probe_result['error']}"}
            
            # Try multiple extraction strategies for maximum compatibility
            strategies = [
                # Strategy 1: Standard extraction
                {
                    "name": "standard",
                    "cmd": [
                        ffmpeg_path, '-y', '-i', video_path,
                        '-ac', '2',  # Stereo
                        '-ar', '44100',  # Sample rate
                        '-acodec', 'pcm_s16le',  # Audio codec
                        output_path
                    ]
                },
                # Strategy 2: More compatible extraction with format specification
                {
                    "name": "compatible",
                    "cmd": [
                        ffmpeg_path, '-y', '-f', 'mp4', '-i', video_path,
                        '-vn',  # No video
                        '-acodec', 'pcm_s16le',
                        '-ac', '2',
                        '-ar', '44100',
                        '-f', 'wav',
                        output_path
                    ]
                },
                # Strategy 3: Auto-detect input format, force WAV output
                {
                    "name": "auto_detect",
                    "cmd": [
                        ffmpeg_path, '-y', '-i', video_path,
                        '-map', '0:a:0',  # Select first audio stream
                        '-c:a', 'pcm_s16le',
                        '-ac', '2',
                        '-ar', '44100',
                        output_path
                    ]
                },
                # Strategy 4: Last resort - copy audio stream and convert
                {
                    "name": "copy_convert", 
                    "cmd": [
                        ffmpeg_path, '-y', '-i', video_path,
                        '-vn', '-acodec', 'copy', '-f', 'mp3',
                        f"{output_path}.mp3"
                    ]
                }
            ]
            
            for strategy in strategies:
                try:
                    result = subprocess.run(strategy["cmd"], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        # Check if output file exists and is valid
                        if strategy["name"] == "copy_convert":
                            # Convert MP3 to WAV
                            mp3_path = f"{output_path}.mp3"
                            if os.path.exists(mp3_path):
                                convert_result = self._convert_mp3_to_wav(ffmpeg_path, mp3_path, output_path)
                                os.unlink(mp3_path)  # Clean up MP3
                                if convert_result["success"]:
                                    return {"success": True, "audio_path": output_path}
                        else:
                            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                                return {"success": True, "audio_path": output_path}
                    
                    # Log the attempt for debugging
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Audio extraction strategy '{strategy['name']}' failed: {result.stderr[:200]}")
                    
                except subprocess.TimeoutExpired:
                    logger.warning(f"Audio extraction strategy '{strategy['name']}' timed out")
                    continue
                except Exception as e:
                    logger.warning(f"Audio extraction strategy '{strategy['name']}' error: {str(e)}")
                    continue
            
            return {"success": False, "error": "All audio extraction strategies failed. Video format may not be supported."}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _probe_video_file(self, ffmpeg_path: str, video_path: str) -> Dict[str, Any]:
        """Probe video file to check if it's readable"""
        try:
            cmd = [ffmpeg_path, '-v', 'quiet', '-i', video_path, '-f', 'null', '-']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Video probe timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _convert_mp3_to_wav(self, ffmpeg_path: str, mp3_path: str, wav_path: str) -> Dict[str, Any]:
        """Convert MP3 to WAV format"""
        try:
            cmd = [
                ffmpeg_path, '-y', '-i', mp3_path,
                '-acodec', 'pcm_s16le',
                '-ac', '2',
                '-ar', '44100',
                wav_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}
                
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
            local_ffmpeg = Path('./ffmpeg-master-latest-win64-gpl/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe')
            if local_ffmpeg.exists():
                return str(local_ffmpeg)
        
        # Default to system ffmpeg
        return 'ffmpeg'
    
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
    
    @staticmethod
    def detect_silence_boundaries(audio: np.ndarray, sr: int, threshold: float = 0.01, 
                                 min_silence_duration: float = 0.5) -> List[Tuple[int, int]]:
        """Detect silence boundaries in audio"""
        # Calculate frame-based energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = frame_length // 4
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find silent frames
        silent_frames = rms < threshold
        
        # Convert frame indices to sample indices
        silence_boundaries = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silent_frames):
            sample_index = i * hop_length
            
            if is_silent and not in_silence:
                # Start of silence
                silence_start = sample_index
                in_silence = True
            elif not is_silent and in_silence:
                # End of silence
                silence_duration = (sample_index - silence_start) / sr
                if silence_duration >= min_silence_duration:
                    silence_boundaries.append((silence_start, sample_index))
                in_silence = False
        
        # Handle silence at the end
        if in_silence:
            silence_duration = (len(audio) - silence_start) / sr
            if silence_duration >= min_silence_duration:
                silence_boundaries.append((silence_start, len(audio)))
        
        return silence_boundaries
    
    @staticmethod
    def remove_tail_silence(audio: np.ndarray, sr: int, threshold: float = 0.01, 
                           min_silence_duration: float = 0.3) -> np.ndarray:
        """Remove silence from the tail (end) of audio after voice cloning"""
        if len(audio) == 0:
            return audio
        
        # Work backwards from the end to find where audio content actually ends
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = frame_length // 4
        
        # Calculate RMS energy in small windows
        window_size = int(0.1 * sr)  # 100ms windows
        step_size = int(0.05 * sr)   # 50ms steps
        
        # Start from the end and work backwards
        for i in range(len(audio) - window_size, 0, -step_size):
            if i < 0:
                break
                
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window**2))
            
            # If we find a window with sufficient energy, this is where content ends
            if rms > threshold:
                # Add a small buffer to avoid cutting off speech
                buffer = int(0.1 * sr)  # 100ms buffer
                end_point = min(i + window_size + buffer, len(audio))
                
                # Only trim if we're actually removing a significant amount of silence
                silence_duration = (len(audio) - end_point) / sr
                if silence_duration >= min_silence_duration:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Removed {silence_duration:.2f}s of tail silence")
                    return audio[:end_point]
                else:
                    return audio  # Not enough silence to justify trimming
        
        # If no significant content found, return original audio
        return audio
    
    @staticmethod
    def remove_head_silence(audio: np.ndarray, sr: int, threshold: float = 0.01,
                           min_silence_duration: float = 0.3) -> np.ndarray:
        """Remove silence from the head (beginning) of audio"""
        if len(audio) == 0:
            return audio
        
        # Work forwards from the beginning to find where audio content starts
        window_size = int(0.1 * sr)  # 100ms windows
        step_size = int(0.05 * sr)   # 50ms steps
        
        # Start from the beginning and work forwards
        for i in range(0, len(audio) - window_size, step_size):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window**2))
            
            # If we find a window with sufficient energy, this is where content starts
            if rms > threshold:
                # Add a small buffer to avoid cutting off speech
                buffer = int(0.05 * sr)  # 50ms buffer
                start_point = max(i - buffer, 0)
                
                # Only trim if we're actually removing a significant amount of silence
                silence_duration = start_point / sr
                if silence_duration >= min_silence_duration:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Removed {silence_duration:.2f}s of head silence")
                    return audio[start_point:]
                else:
                    return audio  # Not enough silence to justify trimming
        
        # If no significant content found, return original audio
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