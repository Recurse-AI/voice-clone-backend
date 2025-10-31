import os
import logging
from typing import Dict, Any
from app.services.dub.context import DubbingContext
from app.services.dub.steps.voice_cloning_step import VoiceCloningStep
from app.services.dub.steps.finalization_step import FinalizationStep
from app.services.dub.handlers.review_handler import ReviewHandler
from app.services.dub.handlers.audio_handler import AudioHandler
from app.services.language_service import language_service

logger = logging.getLogger(__name__)

class BaseFlow:
    @staticmethod
    def split_audio_for_references(context: DubbingContext):
        split_files = AudioHandler.split_audio_segments(context)
        
        for i, segment in enumerate(context.transcription_result.get("segments", [])):
            if i < len(split_files):
                segment["original_audio_file"] = os.path.basename(split_files[i]["output_path"])
        
        context.audio_already_split = True
    
    @staticmethod
    def assign_reference_ids_to_segments(context: DubbingContext):
        if not context.reference_ids or context.voice_type != 'ai_voice':
            return
        
        logger.info(f"Assigning {len(context.reference_ids)} user-provided reference_ids to {len(context.segments)} segments")
        
        for segment in context.segments:
            speaker = segment.get("speaker")
            if speaker:
                speaker_index = int(speaker.split("_")[-1]) if "_" in speaker else 0
                reference_id = context.reference_ids[speaker_index] if speaker_index < len(context.reference_ids) else context.reference_ids[-1]
                segment["reference_id"] = reference_id
    
    @staticmethod
    def handle_review_mode(context: DubbingContext) -> Dict[str, Any]:
        logger.info("Preparing for review mode")
        return ReviewHandler.prepare_for_review(context)
    
    @staticmethod
    def execute_voice_cloning_and_finalize(context: DubbingContext) -> Dict[str, Any]:
        source_lang = context.transcription_result.get("language", "auto_detect") if context.transcription_result else "auto_detect"
        context.source_language_code = language_service.normalize_language_input(source_lang)
        
        VoiceCloningStep().execute(context)
        return FinalizationStep.execute(context)

