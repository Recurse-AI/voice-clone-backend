import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class AudioReconstruction:
    @staticmethod
    def _calculate_gaps(segments: List[Dict[str, Any]]) -> List[float]:
        gaps = []
        for i in range(len(segments) - 1):
            gap_ms = segments[i + 1].get("start", 0) - segments[i].get("end", 0)
            gaps.append(max(0, gap_ms))
        gaps.append(0)
        return gaps
    
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
            
            target_sr = 44100
            max_end_ms = max(seg.get("end", 0) for seg in segments)
            
            total_duration_ms = max_end_ms
            if video_duration_seconds:
                video_duration_ms = int(video_duration_seconds * 1000)
                if video_duration_ms > max_end_ms:
                    total_duration_ms = video_duration_ms
            
            total_samples = int((total_duration_ms / 1000.0) * target_sr)
            timeline_audio = np.zeros(total_samples, dtype=np.float32)
            
            gaps = AudioReconstruction._calculate_gaps(segments)
            current_position_ms = 0.0
            
            logger.info(f"Building {total_duration_ms/1000:.1f}s timeline with {len(segments)} segments")
            
            for idx, seg in enumerate(segments):
                audio_path = seg.get("cloned_audio_path")
                if not audio_path or not os.path.exists(audio_path):
                    current_position_ms = seg.get("end", 0) + gaps[idx]
                    continue
                
                expected_start_ms = seg.get("start", 0)
                expected_end_ms = seg.get("end", 0)
                expected_duration_ms = expected_end_ms - expected_start_ms
                gap_ms = gaps[idx]
                
                if expected_duration_ms <= 0:
                    continue
                
                audio_data, sr = sf.read(audio_path)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                if sr != target_sr:
                    from scipy import signal
                    audio_data = signal.resample(audio_data, int(len(audio_data) * target_sr / sr))
                
                actual_duration_ms = (len(audio_data) / target_sr) * 1000
                difference_ms = actual_duration_ms - expected_duration_ms
                is_last = idx == len(segments) - 1
                
                if difference_ms > 50:
                    if gap_ms >= difference_ms:
                        gap_ms -= difference_ms
                        logger.info(f"Seg {idx}: {actual_duration_ms:.0f}ms (+{difference_ms:.0f}ms), absorbed by gap")
                    else:
                        if is_last and current_position_ms + actual_duration_ms > total_duration_ms:
                            overshoot = current_position_ms + actual_duration_ms - total_duration_ms
                            speedup = 1.45 if overshoot > 500 else 1.2
                        else:
                            speedup = 1.2
                        
                        audio_data = AudioReconstruction._apply_tempo_change(
                            audio_data, target_sr, speedup, process_temp_dir, f"{job_id}_seg{idx}"
                        )
                        actual_duration_ms = (len(audio_data) / target_sr) * 1000
                        logger.info(f"Seg {idx}: applied {speedup:.2f}x speedup -> {actual_duration_ms:.0f}ms")
                
                elif difference_ms < -50:
                    gap_ms += abs(difference_ms)
                    if expected_duration_ms < 1000:
                        audio_data = AudioReconstruction._apply_tempo_change(
                            audio_data, target_sr, 0.9, process_temp_dir, f"{job_id}_seg{idx}"
                        )
                        actual_duration_ms = (len(audio_data) / target_sr) * 1000
                        logger.info(f"Seg {idx}: applied 0.9x slowdown -> {actual_duration_ms:.0f}ms")
                
                start_sample = int((current_position_ms / 1000.0) * target_sr)
                
                if start_sample >= total_samples:
                    logger.warning(f"Seg {idx}: start_sample ({start_sample}) >= total_samples ({total_samples}), skipping")
                    current_position_ms += actual_duration_ms + gap_ms
                    continue
                
                end_sample = min(start_sample + len(audio_data), total_samples)
                audio_length = end_sample - start_sample
                
                if audio_length <= 0:
                    logger.warning(f"Seg {idx}: audio_length <= 0, skipping")
                    current_position_ms += actual_duration_ms + gap_ms
                    continue
                
                timeline_audio[start_sample:end_sample] = audio_data[:audio_length]
                
                current_position_ms += actual_duration_ms + gap_ms
            
            final_path = os.path.join(process_temp_dir, f"final_{job_id}.wav")
            sf.write(final_path, timeline_audio, target_sr, subtype='PCM_16')
            
            logger.info(f"Audio reconstructed: {final_path}, final position: {current_position_ms/1000:.2f}s")
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

