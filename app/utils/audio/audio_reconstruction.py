import os
import subprocess
import logging
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
                "-map", "[out]",
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
        if not audio_path or not os.path.exists(audio_path):
            return 44100
        
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
        filters = []
        concat_inputs = []
        
        current_time_ms = 0
        
        for idx, seg in enumerate(segments):
            cloned_path = seg.get("cloned_audio_path")
            if not cloned_path or not os.path.exists(cloned_path):
                continue
            
            start_ms = seg.get("start", 0)
            end_ms = seg.get("end", 0)
            expected_duration_ms = end_ms - start_ms
            
            if expected_duration_ms <= 0:
                continue
            
            if start_ms > current_time_ms:
                silence_duration_ms = start_ms - current_time_ms
                silence_duration_s = silence_duration_ms / 1000.0
                silence_filter = f"aevalsrc=0:d={silence_duration_s}:s={target_sr}:c=mono[silence{idx}]"
                filters.append(silence_filter)
                concat_inputs.append(f"[silence{idx}]")
            
            audio_filters = [f"aresample={target_sr}", "aformat=channel_layouts=mono"]
            
            audio_path = seg.get("cloned_audio_path")
            if audio_path and os.path.exists(audio_path):
                try:
                    import soundfile as sf
                    audio_data, sr = sf.read(audio_path)
                    actual_duration_ms = (len(audio_data) / sr) * 1000
                    
                    tempo_ratio = actual_duration_ms / expected_duration_ms
                    
                    if abs(tempo_ratio - 1.0) > 0.01:
                        if tempo_ratio < 1.0:
                            clamped_tempo = max(0.8, tempo_ratio)
                        else:
                            clamped_tempo = min(1.7, tempo_ratio)
                        
                        audio_filters.append(f"atempo={clamped_tempo:.6f}")
                        result_duration_ms = actual_duration_ms / clamped_tempo
                        
                        if result_duration_ms < expected_duration_ms:
                            pad_duration_s = (expected_duration_ms - result_duration_ms) / 1000.0
                            audio_filters.append(f"apad=pad_dur={pad_duration_s:.6f}")
                            logger.info(f"Segment {idx}: {actual_duration_ms:.0f}ms -> {expected_duration_ms:.0f}ms (tempo={clamped_tempo:.3f}, pad={pad_duration_s:.3f}s)")
                        elif result_duration_ms > expected_duration_ms:
                            trim_duration_s = expected_duration_ms / 1000.0
                            audio_filters.append(f"atrim=0:{trim_duration_s:.6f}")
                            logger.info(f"Segment {idx}: {actual_duration_ms:.0f}ms -> {expected_duration_ms:.0f}ms (tempo={clamped_tempo:.3f}, trim={trim_duration_s:.3f}s)")
                        else:
                            logger.info(f"Segment {idx}: {actual_duration_ms:.0f}ms -> {expected_duration_ms:.0f}ms (tempo={clamped_tempo:.3f})")
                except Exception as e:
                    logger.warning(f"Duration check failed for segment {idx}: {e}")
            
            audio_filter = f"[{idx}:a]{','.join(audio_filters)}[a{idx}]"
            filters.append(audio_filter)
            concat_inputs.append(f"[a{idx}]")
            
            current_time_ms = end_ms
        
        if not concat_inputs:
            return ""
        
        filter_str = ";".join(filters)
        concat_str = "".join(concat_inputs) + f"concat=n={len(concat_inputs)}:v=0:a=1[out]"
        full_filter = filter_str + ";" + concat_str if filter_str else concat_str
        
        return full_filter


def reconstruct_final_audio_ffmpeg(
    segments: List[Dict[str, Any]], 
    original_audio_path: Optional[str],
    job_id: str,
    process_temp_dir: str
) -> Optional[str]:
    return AudioReconstruction.reconstruct_final_audio(
        segments, original_audio_path, job_id, process_temp_dir
    )

