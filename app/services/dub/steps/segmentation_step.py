import logging
from typing import Dict, Any
from app.services.dub.context import DubbingContext
from app.services.dub.utils.progress_tracker import ProgressTracker
from app.services.dub.ai_segmentation_service import get_ai_segmentation_service
from app.services.language_service import language_service

logger = logging.getLogger(__name__)

class SegmentationStep:
    @staticmethod
    def execute(context: DubbingContext):
        ProgressTracker.update_phase_progress(
            context.job_id, "dubbing", 0.0, "AI creating segments and dubbing"
        )
        
        edited_map = SegmentationStep._get_edited_text_map(context) if context.manifest else {}
        
        if edited_map:
            logger.info(f"RESUME MODE: Using existing segments and translations ({len(edited_map)} segments)")
            context.segments = SegmentationStep._use_manifest_segments_for_resume(
                context.transcription_result, edited_map
            )
            ProgressTracker.update_phase_progress(
                context.job_id, "dubbing", 1.0, "Using reviewed segments and dubbing"
            )
        else:
            ai_service = get_ai_segmentation_service()
            preserve_segments = bool(context.manifest)
            context.segments = ai_service.create_optimal_segments_and_dub(
                context.transcription_result,
                context.target_language_code,
                preserve_segments=preserve_segments,
                num_speakers=len(context.reference_ids) if context.reference_ids else None
            )
            
            if preserve_segments:
                logger.info(f"REDUB MODE: Preserved {len(context.segments)} segments with translation")
                ProgressTracker.update_phase_progress(
                    context.job_id, "dubbing", 1.0, f"Redub completed {len(context.segments)} segments"
                )
            else:
                logger.info(f"AI created {len(context.segments)} optimal segments")
                ProgressTracker.update_phase_progress(
                    context.job_id, "dubbing", 1.0, f"AI completed {len(context.segments)} segments"
                )
    
    @staticmethod
    def _get_edited_text_map(context: DubbingContext) -> Dict[str, str]:
        if not context.manifest:
            return {}
        
        current_target_lang = language_service.normalize_language_input(context.target_language)
        manifest_target_lang = context.manifest.get("target_language")
        
        if manifest_target_lang:
            manifest_target_lang = language_service.normalize_language_input(manifest_target_lang)
            if manifest_target_lang != current_target_lang:
                logger.info(f"Language changed: {manifest_target_lang} → {current_target_lang}, forcing AI translation")
                return {}
        
        edited_map = {}
        for seg in context.manifest.get("segments", []):
            if seg.get("id") and seg.get("dubbed_text"):
                edited_map[seg["id"]] = seg["dubbed_text"]
        logger.info(f"RESUME: Using {len(edited_map)} existing translations")
        return edited_map
    
    @staticmethod
    def _use_manifest_segments_for_resume(transcription_result: Dict[str, Any], edited_map: Dict[str, str]) -> list:
        segments = transcription_result.get("segments", [])
        formatted_segments = []
        
        logger.info(f"RESUME: Processing {len(segments)} manifest segments directly")
        
        for idx, seg in enumerate(segments):
            seg_id = seg.get("id", f"seg_{idx+1:03d}")
            start_ms = int(seg.get("start", 0))
            end_ms = int(seg.get("end", 0))
            
            if formatted_segments:
                prev_end_ms = formatted_segments[-1]["end"]
                if start_ms < prev_end_ms:
                    gap = (prev_end_ms - start_ms) // 2
                    formatted_segments[-1]["end"] = prev_end_ms - gap
                    formatted_segments[-1]["duration_ms"] = formatted_segments[-1]["end"] - formatted_segments[-1]["start"]
                    start_ms = prev_end_ms - gap
            
            duration_ms = end_ms - start_ms
            original_text = seg.get("original_text", seg.get("text", "")).strip()
            dubbed_text = seg.get("dubbed_text", "").strip()
            
            if seg_id in edited_map:
                dubbed_text = edited_map[seg_id]
            elif not dubbed_text:
                dubbed_text = original_text
            
            formatted_segments.append({
                "id": seg_id,
                "segment_index": idx,
                "start": start_ms,
                "end": end_ms,
                "duration_ms": duration_ms,
                "original_text": original_text,
                "dubbed_text": dubbed_text,
                "voice_cloned": seg.get("voice_cloned", False),
                "original_audio_file": seg.get("original_audio_file"),
                "cloned_audio_file": seg.get("cloned_audio_file"),
                "speaker": seg.get("speaker"),
                "reference_id": seg.get("reference_id")
            })
        
        logger.info(f"RESUME: Prepared {len(formatted_segments)} segments for voice cloning")
        return formatted_segments

