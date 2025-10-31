import logging
from typing import Dict, Any
from app.services.dub.context import DubbingContext
from app.services.dub.flows.base_flow import BaseFlow
from app.services.dub.handlers.manifest_handler import ManifestHandler
from app.services.dub.steps.transcription_step import TranscriptionStep
from app.services.dub.steps.segmentation_step import SegmentationStep

logger = logging.getLogger(__name__)

class ResumeFlow(BaseFlow):
    def execute(self, context: DubbingContext) -> Dict[str, Any]:
        logger.info(f"Starting resume flow for job {context.job_id}")
        
        ManifestHandler.load_and_restore(context)
        TranscriptionStep.execute(context)
        SegmentationStep.execute(context)
        
        self.assign_reference_ids_to_segments(context)
        
        return self.execute_voice_cloning_and_finalize(context)

