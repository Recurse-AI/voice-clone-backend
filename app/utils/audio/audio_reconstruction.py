import os
import subprocess
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class AudioReconstruction:
    """Simplified audio reconstruction using only FFmpeg"""
    
    @staticmethod
    def reconstruct_final_audio(
        segments: List[Dict[str, Any]], 
        original_audio_path: Optional[str],
        job_id: str,
        process_temp_dir: str
    ) -> Optional[str]:
        """
        Reconstruct final audio from dubbed segments using FFmpeg only.
        Much simpler than numpy/scipy approach.
        """
        if not segments:
            logger.warning("No segments to reconstruct")
            return None
        
        try:
            from app.utils.ffmpeg_helper import get_ffmpeg_path
            ffmpeg_path = get_ffmpeg_path()
            
            if not ffmpeg_path:
                logger.error("FFmpeg not found")
                return None
            
            target_sr = AudioReconstruction._get_sample_rate(original_audio_path, ffmpeg_path)
            filter_complex = AudioReconstruction._build_filter_complex(segments, target_sr)
            
            if not filter_complex:
                logger.error("Failed to build filter complex")
                return None
            
            input_files = []
            for seg in segments:
                cloned_path = seg.get("cloned_audio_path")
                if cloned_path and os.path.exists(cloned_path):
                    input_files.append(cloned_path)
            
            if not input_files:
                logger.warning("No valid audio files found in segments")
                return None
            
            final_path = os.path.join(process_temp_dir, f"final_{job_id}.wav")
            cmd = [ffmpeg_path, "-y"]
            
            for audio_file in input_files:
                cmd.extend(["-i", audio_file])
            
            cmd.extend([
                "-filter_complex", filter_complex,
                "-ar", str(target_sr),
                "-ac", "1",
                "-acodec", "pcm_s16le",
                final_path
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg reconstruction failed: {result.stderr}")
                return None
            
            if os.path.exists(final_path):
                logger.info(f"✅ Audio reconstructed successfully: {final_path}")
                return final_path
            
            return None
            
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg reconstruction timeout (300s)")
            return None
        except Exception as e:
            logger.error(f"Audio reconstruction error: {e}")
            return None
    
    @staticmethod
    def _get_sample_rate(audio_path: Optional[str], ffmpeg_path: str) -> int:
        """Get sample rate from audio file using FFmpeg"""
        if not audio_path or not os.path.exists(audio_path):
            return 44100  # Default
        
        try:
            cmd = [
                ffmpeg_path, "-i", audio_path,
                "-hide_banner", "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            for line in result.stderr.split('\n'):
                if 'Hz' in line and 'Audio:' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'Hz' and i > 0:
                            return int(parts[i-1])
            
            return 44100
        except Exception:
            return 44100
    
    @staticmethod
    def _build_filter_complex(segments: List[Dict[str, Any]], target_sr: int) -> str:
        """
        Build FFmpeg filter complex for segment reconstruction.
        Handles: delay, time-stretch, fade, concatenation
        """
        filters = []
        valid_segments = []
        
        for idx, seg in enumerate(segments):
            cloned_path = seg.get("cloned_audio_path")
            if not cloned_path or not os.path.exists(cloned_path):
                continue
            
            start_ms = seg.get("start", 0)
            end_ms = seg.get("end", 0)
            expected_duration = (end_ms - start_ms) / 1000.0
            
            if expected_duration <= 0:
                continue
            
            filter_parts = [f"[{idx}:a]"]
            filter_parts.append(f"aresample={target_sr}")
            filter_parts.append("pan=mono|c0=c0")
            filter_parts.append("afade=t=in:d=0.003,afade=t=out:d=0.003")
            
            delay_ms = start_ms
            if delay_ms > 0:
                filter_parts.append(f"adelay={delay_ms}|{delay_ms}")
            
            filter_chain = ",".join(filter_parts)
            filters.append(f"{filter_chain}[a{idx}]")
            valid_segments.append(idx)
        
        if not valid_segments:
            return ""
        
        input_labels = "".join([f"[a{i}]" for i in valid_segments])
        mix_filter = f"{input_labels}amix=inputs={len(valid_segments)}:duration=longest:normalize=0[out]"
        full_filter = ";".join(filters) + ";" + mix_filter
        
        return full_filter


def reconstruct_final_audio_ffmpeg(
    segments: List[Dict[str, Any]], 
    original_audio_path: Optional[str],
    job_id: str,
    process_temp_dir: str
) -> Optional[str]:
    """Convenience function for FFmpeg-based reconstruction"""
    return AudioReconstruction.reconstruct_final_audio(
        segments, original_audio_path, job_id, process_temp_dir
    )

