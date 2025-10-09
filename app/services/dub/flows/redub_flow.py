import logging
from typing import Dict, Any
from app.services.dub.context import DubbingContext
from app.services.dub.flows.base_flow import BaseFlow
from app.services.dub.handlers.manifest_handler import ManifestHandler
from app.services.dub.steps.transcription_step import TranscriptionStep
from app.services.dub.steps.segmentation_step import SegmentationStep
from app.services.dub.steps.reference_creation_step import ReferenceCreationStep

logger = logging.getLogger(__name__)

class RedubFlow(BaseFlow):
    def execute(self, context: DubbingContext) -> Dict[str, Any]:
        logger.info(f"Starting redub flow for job {context.job_id}")
        
        ManifestHandler.load_and_restore(context)
        TranscriptionStep.execute(context)
        SegmentationStep.execute(context)
        
        for segment in context.segments:
            segment["reference_id"] = None
        
        if context.review_mode:
            return self.handle_review_mode(context)
        
        if not context.reference_ids and context.model_type in ["best", "medium"]:
            self.split_audio_for_references(context)
            ReferenceCreationStep.execute(context)
        
        ReferenceCreationStep.assign_to_segments(context)
        
        return self.execute_voice_cloning_and_finalize(context)

