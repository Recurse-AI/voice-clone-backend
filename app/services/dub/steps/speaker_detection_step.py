import os
import logging
from app.services.dub.context import DubbingContext

logger = logging.getLogger(__name__)

class SpeakerDetectionStep:
    @staticmethod
    def execute(context: DubbingContext, tag_segments: bool = True):
        if context.manifest:
            logger.info("Resume/Redub: using manifest speaker info")
            return
        
        try:
            from app.services.dub.speaker_detection_service import speaker_detection_service
            
            vocal_path = os.path.join(context.process_temp_dir, f"vocal_{context.job_id}.wav")
            if not os.path.exists(vocal_path):
                logger.warning(f"Vocal file not found at {vocal_path}")
                return
            
            num_speakers = context.num_of_speakers if context.num_of_speakers else None
            logger.info(f"Running speaker detection (expected speakers: {num_speakers or 'auto'})")
            
            speaker_timeline = speaker_detection_service.detect_speakers(
                vocal_path, context.job_id, num_speakers
            )
            
            if tag_segments and speaker_timeline:
                for seg in context.transcription_result.get("segments", []):
                    seg_start_ms = seg.get("start", 0)
                    seg_end_ms = seg.get("end", 0)
                    seg["speaker"] = SpeakerDetectionStep._get_segment_speaker(
                        seg_start_ms, seg_end_ms, speaker_timeline
                    )
                logger.info(f"Tagged {len(context.transcription_result.get('segments', []))} segments with speakers")
        
        except Exception as e:
            logger.error(f"Speaker detection failed: {e}")
    
    @staticmethod
    def _get_segment_speaker(seg_start_ms: int, seg_end_ms: int, speaker_timeline: list) -> str:
        seg_start_s = seg_start_ms / 1000.0
        seg_end_s = seg_end_ms / 1000.0
        
        if not speaker_timeline:
            return None
        
        timeline = sorted(speaker_timeline, key=lambda s: (s["start"], s["end"]))
        overlap_by_speaker = {}
        
        for t in timeline:
            if t["end"] <= seg_start_s:
                continue
            if t["start"] >= seg_end_s:
                break
            overlap = min(seg_end_s, t["end"]) - max(seg_start_s, t["start"])
            if overlap > 0:
                spk = t["speaker"]
                overlap_by_speaker[spk] = overlap_by_speaker.get(spk, 0.0) + overlap
        
        if overlap_by_speaker:
            return max(overlap_by_speaker.items(), key=lambda kv: kv[1])[0]
        
        seg_mid = (seg_start_s + seg_end_s) / 2.0
        nearest = min(
            timeline,
            key=lambda t: 0 if (t["start"] <= seg_mid <= t["end"]) 
            else min(abs(seg_mid - t["start"]), abs(seg_mid - t["end"]))
        )
        return nearest["speaker"] if nearest else None

