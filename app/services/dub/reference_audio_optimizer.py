import logging
import os
import numpy as np
import soundfile as sf
from typing import List, Dict, Any, Optional
from io import BytesIO

logger = logging.getLogger(__name__)

class ReferenceAudioOptimizer:
    
    @staticmethod
    def optimize_for_elevenlabs(segments: List[Dict[str, Any]], speaker: str, process_temp_dir: str) -> Optional[bytes]:
        return ReferenceAudioOptimizer._optimize_reference(
            segments, speaker, process_temp_dir, max_duration_seconds=27, model_name="ElevenLabs"
        )
    
    @staticmethod
    def optimize_for_fish(segments: List[Dict[str, Any]], speaker: str, process_temp_dir: str) -> Optional[bytes]:
        return ReferenceAudioOptimizer._optimize_reference(
            segments, speaker, process_temp_dir, max_duration_seconds=27, model_name="Fish Audio"
        )
    
    @staticmethod
    def _optimize_reference(segments: List[Dict[str, Any]], speaker: str, process_temp_dir: str, 
                           max_duration_seconds: float, model_name: str) -> Optional[bytes]:
        try:
            speaker_segments = [s for s in segments if s.get("speaker") == speaker and s.get("original_audio_file")]
            
            if not speaker_segments:
                logger.warning(f"No segments found for speaker {speaker}")
                return None
            
            sorted_segments = sorted(speaker_segments, key=lambda x: x.get("duration_ms", 0), reverse=True)
            
            selected_segments = []
            total_duration_ms = 0
            max_duration_ms = max_duration_seconds * 1000
            
            for seg in sorted_segments:
                seg_duration = seg.get("duration_ms", 0)
                if total_duration_ms + seg_duration <= max_duration_ms:
                    selected_segments.append(seg)
                    total_duration_ms += seg_duration
                
                if total_duration_ms >= max_duration_ms * 0.9:
                    break
            
            if not selected_segments:
                selected_segments = [sorted_segments[0]]
            
            selected_segments.sort(key=lambda x: x.get("start", 0))
            
            logger.info(f"{model_name} reference: Selected {len(selected_segments)} segments ({total_duration_ms/1000:.1f}s) for {speaker}")
            
            audio_segments = []
            for seg in selected_segments:
                audio_path = os.path.join(process_temp_dir, seg["original_audio_file"])
                if os.path.exists(audio_path):
                    data, sr = sf.read(audio_path)
                    audio_segments.append((data, sr))
            
            if not audio_segments:
                logger.warning(f"No valid audio files found for {speaker}")
                return None
            
            sample_rate = audio_segments[0][1]
            concatenated = np.concatenate([audio for audio, _ in audio_segments])
            
            buffer = BytesIO()
            sf.write(buffer, concatenated, sample_rate, format='WAV')
            buffer.seek(0)
            
            return buffer.read()
            
        except Exception as e:
            logger.error(f"Failed to optimize reference audio for {speaker}: {e}")
            return None


def get_reference_audio_optimizer() -> ReferenceAudioOptimizer:
    return ReferenceAudioOptimizer()

