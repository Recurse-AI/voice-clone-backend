import os
import logging
from typing import List, Dict, Any
from app.utils.audio import AudioUtils
from app.utils.audio.audio_reconstruction import reconstruct_final_audio_ffmpeg
from app.services.dub.context import DubbingContext

logger = logging.getLogger(__name__)

class AudioHandler:
    @staticmethod
    def split_audio_segments(context: DubbingContext) -> List[str]:
        vocal_file_path = os.path.join(context.process_temp_dir, f"vocal_{context.job_id}.wav")
        
        if not os.path.exists(vocal_file_path):
            raise Exception(f"Vocal file not found for segmentation")
        
        audio_utils = AudioUtils()
        segments_to_split = []
        
        for seg in context.segments:
            if seg.get("original_text", "").strip():
                segments_to_split.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["original_text"]
                })
        
        split_result = audio_utils.split_audio_by_timestamps(
            vocal_file_path, context.process_temp_dir, segments_to_split
        )
        
        if not split_result["success"]:
            raise Exception(f"Audio segmentation failed: {split_result['error']}")
        
        return split_result.get("split_files", [])
    
    @staticmethod
    def reconstruct_final_audio(context: DubbingContext) -> str:
        orig_vocal_path = os.path.join(context.process_temp_dir, f"vocal_{context.job_id}.wav")
        orig_vocal_path = orig_vocal_path if os.path.exists(orig_vocal_path) else None
        
        return reconstruct_final_audio_ffmpeg(
            context.segments,
            orig_vocal_path,
            context.job_id,
            context.process_temp_dir
        )

