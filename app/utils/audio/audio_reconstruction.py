import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class AudioReconstruction:
    @staticmethod
    def reconstruct_final_audio(
        segments: List[Dict[str, Any]], 
        original_audio_path: Optional[str],
        job_id: str,
        process_temp_dir: str,
        video_duration_seconds: Optional[float] = None
    ) -> Optional[str]:
        if not segments:
            return None
        
        try:
            import soundfile as sf
            from app.utils.ffmpeg_helper import get_ffmpeg_path
            
            target_sr = 44100
            max_end_ms = max(seg.get("end", 0) for seg in segments)
            
            total_duration_ms = max_end_ms
            if video_duration_seconds:
                video_duration_ms = int(video_duration_seconds * 1000)
                if video_duration_ms > max_end_ms:
                    total_duration_ms = video_duration_ms
                    logger.info(f"📏 Extending audio from {max_end_ms/1000:.2f}s to {video_duration_seconds:.2f}s to match video")
            
            total_samples = int((total_duration_ms / 1000.0) * target_sr)
            timeline_audio = np.zeros(total_samples, dtype=np.float32)
            
            logger.info(f"Building {total_duration_ms/1000:.1f}s timeline with {len(segments)} segments")
            
            for idx, seg in enumerate(segments):
                audio_path = seg.get("cloned_audio_path")
                if not audio_path or not os.path.exists(audio_path):
                    continue
                
                start_ms = seg.get("start", 0)
                end_ms = seg.get("end", 0)
                expected_duration_ms = end_ms - start_ms
                
                if expected_duration_ms <= 0:
                    continue
                
                audio_data, sr = sf.read(audio_path)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                if sr != target_sr:
                    from scipy import signal
                    audio_data = signal.resample(audio_data, int(len(audio_data) * target_sr / sr))
                
                actual_duration_ms = (len(audio_data) / target_sr) * 1000
                expected_samples = int((expected_duration_ms / 1000.0) * target_sr)
                duration_ratio = actual_duration_ms / expected_duration_ms
                
                if abs(duration_ratio - 1.0) > 0.05:
                    if duration_ratio < 1.0:
                        slowdown_factor = max(duration_ratio, 0.85)
                        audio_data = AudioReconstruction._apply_tempo_change(audio_data, target_sr, slowdown_factor, process_temp_dir, f"{job_id}_seg{idx}")
                        if audio_data is not None:
                            actual_duration_ms = (len(audio_data) / target_sr) * 1000
                            logger.info(f"Seg {idx}: applied {slowdown_factor:.2f}x slowdown, {actual_duration_ms:.0f}ms -> {expected_duration_ms:.0f}ms")
                    else:
                        speedup_factor = min(duration_ratio, 1.45)
                        audio_data = AudioReconstruction._apply_tempo_change(audio_data, target_sr, speedup_factor, process_temp_dir, f"{job_id}_seg{idx}")
                        if audio_data is not None:
                            actual_duration_ms = (len(audio_data) / target_sr) * 1000
                            logger.info(f"Seg {idx}: applied {speedup_factor:.2f}x speedup, {actual_duration_ms:.0f}ms -> {expected_duration_ms:.0f}ms")
                
                if len(audio_data) < expected_samples:
                    audio_data = np.pad(audio_data, (0, expected_samples - len(audio_data)))
                elif len(audio_data) > expected_samples:
                    audio_data = audio_data[:expected_samples]
                
                start_sample = int((start_ms / 1000.0) * target_sr)
                end_sample = min(start_sample + len(audio_data), total_samples)
                timeline_audio[start_sample:end_sample] = audio_data[:end_sample - start_sample]
            
            final_path = os.path.join(process_temp_dir, f"final_{job_id}.wav")
            sf.write(final_path, timeline_audio, target_sr, subtype='PCM_16')
            
            logger.info(f"Audio reconstructed: {final_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            return None
    
    @staticmethod
    def _apply_tempo_change(audio_data: np.ndarray, sr: int, tempo_factor: float, temp_dir: str, file_id: str) -> np.ndarray:
        try:
            import soundfile as sf
            import subprocess
            from app.utils.ffmpeg_helper import get_ffmpeg_path
            
            ffmpeg = get_ffmpeg_path()
            if not ffmpeg:
                return audio_data
            
            input_path = os.path.join(temp_dir, f"temp_in_{file_id}.wav")
            output_path = os.path.join(temp_dir, f"temp_out_{file_id}.wav")
            
            sf.write(input_path, audio_data, sr)
            
            cmd = [ffmpeg, "-y", "-i", input_path, "-af", f"atempo={tempo_factor:.4f}", "-ar", str(sr), "-ac", "1", output_path]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(output_path):
                adjusted_audio, _ = sf.read(output_path)
                os.remove(input_path)
                os.remove(output_path)
                return adjusted_audio
            
            return audio_data
        except Exception:
            return audio_data


def reconstruct_final_audio_ffmpeg(
    segments: List[Dict[str, Any]], 
    original_audio_path: Optional[str],
    job_id: str,
    process_temp_dir: str,
    video_duration_seconds: Optional[float] = None
) -> Optional[str]:
    return AudioReconstruction.reconstruct_final_audio(
        segments, original_audio_path, job_id, process_temp_dir, video_duration_seconds
    )

