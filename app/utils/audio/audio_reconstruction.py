import os
import subprocess
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
        process_temp_dir: str
    ) -> Optional[str]:
        if not segments:
            return None
        
        try:
            import soundfile as sf
            from app.utils.ffmpeg_helper import get_ffmpeg_path
            
            target_sr = 44100
            max_end_ms = max(seg.get("end", 0) for seg in segments)
            total_samples = int((max_end_ms / 1000.0) * target_sr)
            timeline_audio = np.zeros(total_samples, dtype=np.float32)
            
            logger.info(f"Building {max_end_ms/1000:.1f}s timeline with {len(segments)} segments")
            
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
                tempo_ratio = actual_duration_ms / expected_duration_ms
                
                if abs(tempo_ratio - 1.0) > 0.01:
                    if tempo_ratio > 1.0:
                        audio_data = AudioReconstruction._trim_silence(audio_data, target_sr)
                        actual_duration_ms = (len(audio_data) / target_sr) * 1000
                        tempo_ratio = actual_duration_ms / expected_duration_ms
                    
                    clamped_tempo = max(0.8, min(1.7, tempo_ratio))
                    
                    adjusted_audio = AudioReconstruction._apply_tempo_ffmpeg(
                        audio_data, target_sr, clamped_tempo, process_temp_dir, f"{job_id}_seg{idx}"
                    )
                    
                    if adjusted_audio is not None:
                        audio_data = adjusted_audio
                        logger.info(f"Seg {idx}: {actual_duration_ms:.0f}ms -> {expected_duration_ms:.0f}ms (tempo={clamped_tempo:.2f})")
                
                expected_samples = int((expected_duration_ms / 1000.0) * target_sr)
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
    def _apply_tempo_ffmpeg(audio_data: np.ndarray, sr: int, tempo: float, temp_dir: str, file_id: str) -> Optional[np.ndarray]:
        try:
            import soundfile as sf
            from app.utils.ffmpeg_helper import get_ffmpeg_path
            
            ffmpeg = get_ffmpeg_path()
            if not ffmpeg:
                return None
            
            input_path = os.path.join(temp_dir, f"temp_in_{file_id}.wav")
            output_path = os.path.join(temp_dir, f"temp_out_{file_id}.wav")
            
            sf.write(input_path, audio_data, sr)
            
            cmd = [
                ffmpeg, "-y", "-i", input_path,
                "-af", f"atempo={tempo:.6f}",
                "-ar", str(sr), "-ac", "1",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(output_path):
                adjusted_audio, _ = sf.read(output_path)
                os.remove(input_path)
                os.remove(output_path)
                return adjusted_audio
            
            return None
            
        except Exception:
            return None
    
    @staticmethod
    def _trim_silence(audio_data: np.ndarray, sr: int, threshold: float = 0.01) -> np.ndarray:
        try:
            abs_audio = np.abs(audio_data)
            
            start_idx = 0
            for i in range(len(audio_data)):
                if abs_audio[i] > threshold:
                    start_idx = i
                    break
            
            end_idx = len(audio_data)
            for i in range(len(audio_data) - 1, -1, -1):
                if abs_audio[i] > threshold:
                    end_idx = i + 1
                    break
            
            if start_idx < end_idx:
                return audio_data[start_idx:end_idx]
            return audio_data
            
        except Exception:
            return audio_data


def reconstruct_final_audio_ffmpeg(
    segments: List[Dict[str, Any]], 
    original_audio_path: Optional[str],
    job_id: str,
    process_temp_dir: str
) -> Optional[str]:
    return AudioReconstruction.reconstruct_final_audio(
        segments, original_audio_path, job_id, process_temp_dir
    )

