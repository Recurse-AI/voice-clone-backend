import os
import logging
from typing import Dict, Any
from app.services.dub.context import DubbingContext
from app.services.dub.steps.reference_creation_step import ReferenceCreationStep
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
    def handle_review_mode(context: DubbingContext) -> Dict[str, Any]:
        if context.reference_ids:
            logger.info(f"Assigning user-provided reference_ids before review")
            ReferenceCreationStep.assign_to_segments(context)
        else:
            logger.info("No reference_ids needed - will create after resume/approve")
        
        return ReviewHandler.prepare_for_review(context)
    
    @staticmethod
    def execute_voice_cloning_and_finalize(context: DubbingContext) -> Dict[str, Any]:
        source_lang = context.transcription_result.get("language", "auto_detect") if context.transcription_result else "auto_detect"
        context.source_language_code = language_service.normalize_language_input(source_lang)
        
        VoiceCloningStep().execute(context)
        return FinalizationStep.execute(context)

