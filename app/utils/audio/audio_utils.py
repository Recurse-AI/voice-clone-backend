"""
Audio Utilities Module

Handles audio manipulation operations like silence detection, artifact detection, 
and audio length adjustment.
"""

import os
import re
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import librosa
import numpy as np
import soundfile as sf
from app.config.settings import settings

logger = logging.getLogger(__name__)


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
    def trim_silence(audio: np.ndarray, top_db: float = 40.0) -> np.ndarray:
        """Trim leading/trailing silence using energy threshold."""
        try:
            if audio.size == 0:
                return audio
            import librosa
            trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            return trimmed if trimmed.size > 0 else audio
        except Exception:
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
    
    @staticmethod
    def crossfade_arrays(a: np.ndarray, b: np.ndarray, fade_ms: float, sample_rate: int) -> np.ndarray:
        if a.size == 0:
            return b.astype(np.float32)
        if b.size == 0:
            return a.astype(np.float32)
        
        fade_samples = max(1, int((fade_ms / 1000.0) * sample_rate))
        min_chunk_size = min(a.size, b.size)
        
        if fade_samples >= min_chunk_size:
            if min_chunk_size < 100:
                return np.concatenate([a.astype(np.float32), b.astype(np.float32)])
            fade_samples = min_chunk_size // 3
        
        overlap_a = a[-fade_samples:].astype(np.float32)
        overlap_b = b[:fade_samples].astype(np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        fade_in = 1.0 - fade_out
        cross = overlap_a * fade_out + overlap_b * fade_in
        
        return np.concatenate([
            a[:-fade_samples].astype(np.float32), 
            cross, 
            b[fade_samples:].astype(np.float32)
        ])
    
    def split_audio_by_timestamps(self, input_audio_path: str, output_dir: str, 
                                 segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Split audio file into segments based on timestamps using ffmpeg (parallel)"""
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Get FFmpeg executable path
            ffmpeg_path = self._get_ffmpeg_path()
            if not ffmpeg_path:
                return {"success": False, "error": "FFmpeg not found"}
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            def split_segment(i, segment):
                """Split single segment"""
                start_ms = segment.get('start', 0)
                end_ms = segment.get('end', 0)
                text = segment.get('text', f'segment_{i}')
                
                start_sec = start_ms / 1000.0
                duration_sec = (end_ms - start_ms) / 1000.0
                
                output_filename = f"segment_{i:03d}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                cmd = [
                    ffmpeg_path, '-y',
                    '-i', input_audio_path,
                    '-ss', str(start_sec),
                    '-t', str(duration_sec),
                    '-ac', '2',
                    '-ar', '44100',
                    '-acodec', 'pcm_s16le',
                    output_path
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        return (i, {
                            "index": i,
                            "text": text,
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "duration_ms": end_ms - start_ms,
                            "output_path": output_path
                        }, None)
                    else:
                        return (i, None, f"FFmpeg error for segment {i}: {result.stderr}")
                except subprocess.TimeoutExpired:
                    return (i, None, f"FFmpeg timeout (30s) for segment {i}")
                except Exception as e:
                    return (i, None, f"Segment {i} error: {str(e)}")
            
            # Process segments in parallel
            max_workers = min(10, len(segments))
            results = [None] * len(segments)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(split_segment, i, seg): i for i, seg in enumerate(segments)}
                
                for future in as_completed(futures):
                    idx, result, error = future.result()
                    if error:
                        return {"success": False, "error": error}
                    results[idx] = result
            
            return {
                "success": True,
                "segments_count": len(results),
                "split_files": results,
                "output_directory": output_dir
            }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def mix_audio_files(file1: str, file2: str, out_path: str, ratio1: float = 1.0, ratio2: float = 0.3):
        """Mix two audio files with given ratios and write to out_path - file1 (dubbed) always at 1.0, file2 (instrument) configurable"""
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
    def remove_temp_dir(folder_path: str) -> bool:
        """Remove temporary directory safely"""
        try:
            import shutil
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                shutil.rmtree(folder_path, ignore_errors=True)
                return True
            return False
        except Exception as e:
            # Log error but don't raise - cleanup should be non-blocking
            logger.warning(f"Failed to remove temp directory {folder_path}: {e}")
            return False

    @staticmethod
    def reconstruct_final_audio(segments: list, original_audio_path: str,
                              job_id: str = None, process_temp_dir: str = None) -> str:
        """Reconstruct final audio from dubbed segments"""
        if not segments:
            return None

        try:
            if original_audio_path and os.path.exists(original_audio_path):
                info = sf.info(original_audio_path)
                target_sample_rate = info.samplerate
                duration_samples = info.frames
            else:
                target_sample_rate = 44100
                max_end_ms = max(s.get("end", 0) for s in segments)
                duration_samples = int((max_end_ms / 1000.0) * target_sample_rate)
            
            final_audio = np.zeros(duration_samples, dtype=np.float32)

            for segment in segments:
                cloned_path = segment.get("cloned_audio_path")
                if not cloned_path or not os.path.exists(cloned_path):
                    continue

                cloned_audio, segment_sample_rate = sf.read(cloned_path)
                if len(cloned_audio.shape) > 1:
                    cloned_audio = np.mean(cloned_audio, axis=1)

                if segment_sample_rate != target_sample_rate:
                    from scipy import signal
                    cloned_audio = signal.resample(cloned_audio, 
                                                 int(len(cloned_audio) * target_sample_rate / segment_sample_rate))

                start_ms = segment.get("start", 0)
                end_ms = segment.get("end", 0)
                if start_ms < 0 or end_ms <= start_ms:
                    continue

                start_sample = int((start_ms / 1000.0) * target_sample_rate)
                end_sample = int((end_ms / 1000.0) * target_sample_rate)
                expected_samples = end_sample - start_sample

                if len(cloned_audio) != expected_samples:
                    expected_duration = expected_samples / target_sample_rate
                    current_duration = len(cloned_audio) / target_sample_rate
                    
                    if abs(current_duration - expected_duration) > 0.01:
                        temp_input = os.path.join(process_temp_dir, f"stretch_input_{segment.get('index', 0)}.wav")
                        temp_output = os.path.join(process_temp_dir, f"stretch_output_{segment.get('index', 0)}.wav")
                        
                        sf.write(temp_input, cloned_audio, target_sample_rate)
                        
                        atempo = current_duration / expected_duration
                        atempo = max(0.5, min(2.0, atempo))
                        
                        ffmpeg_path = AudioUtils()._get_ffmpeg_path()
                        result = subprocess.run([
                            ffmpeg_path, '-y', '-i', temp_input,
                            '-filter:a', f'atempo={atempo}',
                            temp_output
                        ], capture_output=True)
                        
                        if result.returncode == 0 and os.path.exists(temp_output):
                            cloned_audio, _ = sf.read(temp_output)
                            os.remove(temp_input)
                            os.remove(temp_output)
                    
                    if len(cloned_audio) != expected_samples:
                        if len(cloned_audio) > expected_samples:
                            cloned_audio = cloned_audio[:expected_samples]
                        else:
                            cloned_audio = np.pad(cloned_audio, (0, expected_samples - len(cloned_audio)), mode="constant")

                cloned_audio = AudioUtils.fade_in_out(cloned_audio.astype(np.float32), 
                                                    fade_duration=0.003, sample_rate=target_sample_rate)

                actual_end = min(len(final_audio), start_sample + len(cloned_audio))
                if start_sample < len(final_audio) and actual_end > start_sample:
                    segment_slice = cloned_audio[:actual_end - start_sample]
                    final_audio[start_sample:actual_end] = segment_slice

            final_audio = AudioUtils.apply_final_audio_processing(final_audio)
            
            final_path = os.path.join(process_temp_dir, f"final_{job_id}.wav")
            sf.write(final_path, final_audio, target_sample_rate)


            logger.info(f"Final audio: {final_path} ({len(final_audio)/target_sample_rate:.2f}s)")
            return final_path

        except Exception as e:
            logger.error(f"Audio reconstruction failed: {e}")
            return None

    @staticmethod
    def optimize_voice_audio(audio: np.ndarray) -> np.ndarray:
        """Apply voice-specific optimizations to reduce file size while maintaining quality"""
        try:
            max_val = np.max(np.abs(audio))
            if max_val > 1e-10:
                audio = audio / max_val
                
                threshold = 0.6
                ratio = 0.3
                compressed = np.where(
                    np.abs(audio) > threshold,
                    np.sign(audio) * (threshold + (np.abs(audio) - threshold) * ratio),
                    audio
                )
                
                compressed = compressed * 0.8
                return compressed.astype(np.float32)
            
            return audio.astype(np.float32)
        except Exception:
            return audio.astype(np.float32)

    def segment_audio_for_transcription(self, audio_path: str, segment_duration_minutes: float = 3.0) -> List[str]:
        """Split audio into segments for memory-efficient transcription"""
        try:
            import librosa
            
            # Load audio to get duration
            audio, sr = librosa.load(audio_path, sr=None)
            total_duration = len(audio) / sr
            segment_duration_seconds = segment_duration_minutes * 60
            
            # If audio is shorter than segment duration, return original file
            if total_duration <= segment_duration_seconds:
                return [audio_path]
            
            # Create temp directory for segments  
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="audio_segments_")
            ffmpeg_path = self._get_ffmpeg_path()
            
            segments = []
            num_segments = int(np.ceil(total_duration / segment_duration_seconds))
            
            for i in range(num_segments):
                start_time = i * segment_duration_seconds
                segment_path = os.path.join(temp_dir, f"segment_{i:03d}.wav")
                
                # Use FFmpeg to extract segment
                cmd = [
                    ffmpeg_path, '-y',
                    '-i', audio_path,
                    '-ss', str(start_time),
                    '-t', str(segment_duration_seconds),
                    '-acodec', 'pcm_s16le',
                    segment_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    segments.append(segment_path)
                else:
                    logger.error(f"Failed to create audio segment {i}: {result.stderr}")
            
            return segments
            
        except Exception as e:
            logger.error(f"Audio segmentation failed: {e}")
            return [audio_path]  # Return original if segmentation fails

    @staticmethod
    def apply_final_audio_processing(audio: np.ndarray) -> np.ndarray:
        """Apply final processing to the complete audio for optimal compression"""
        try:
            audio = audio - np.mean(audio)
            
            max_val = np.max(np.abs(audio))
            if max_val > 1e-10:
                target_level = 0.7
                audio = audio * (target_level / max_val)
            
            return audio.astype(np.float32)
        except Exception:
            return audio.astype(np.float32)