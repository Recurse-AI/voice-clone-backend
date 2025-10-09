import logging
from typing import Dict, Any
from app.services.dub.context import DubbingContext
from app.services.dub.flows.base_flow import BaseFlow
from app.services.dub.steps.transcription_step import TranscriptionStep
from app.services.dub.steps.speaker_detection_step import SpeakerDetectionStep
from app.services.dub.steps.segmentation_step import SegmentationStep
from app.services.dub.steps.reference_creation_step import ReferenceCreationStep

logger = logging.getLogger(__name__)

class VideoDubFlow(BaseFlow):
    def execute(self, context: DubbingContext) -> Dict[str, Any]:
        logger.info(f"Starting fresh video dub flow for job {context.job_id}")
        
        TranscriptionStep.execute(context)
        SpeakerDetectionStep.execute(context, tag_segments=True)
        SegmentationStep.execute(context)
        
        context.vocal_url, context.instrument_url = self._get_audio_urls(context.separation_urls)
        
        if context.review_mode:
            return self.handle_review_mode(context)
        
        if not context.reference_ids and context.model_type in ["best", "medium"]:
            logger.info("Creating reference_ids for voice cloning")
            ReferenceCreationStep.execute(context)
        
        ReferenceCreationStep.assign_to_segments(context)
        
        return self.execute_voice_cloning_and_finalize(context)
    
    @staticmethod
    def _get_audio_urls(separation_urls: dict) -> tuple:
        if separation_urls:
            vocal_url = separation_urls.get("vocal_audio")
            instrument_url = separation_urls.get("instrument_audio")
        else:
            vocal_url = None
            instrument_url = None
        
        if not vocal_url:
            logger.warning("No vocal audio URL available")
        
        return vocal_url, instrument_url

