import logging
from app.services.dub.context import DubbingContext
from app.services.dub.flows.video_dub_flow import VideoDubFlow
from app.services.dub.flows.resume_flow import ResumeFlow
from app.services.dub.flows.redub_flow import RedubFlow
from app.services.language_service import language_service

logger = logging.getLogger(__name__)

class FlowCoordinator:
    @staticmethod
    def determine_and_execute(context: DubbingContext) -> dict:
        flow = FlowCoordinator._determine_flow(context)
        logger.info(f"Executing {flow.__class__.__name__} for job {context.job_id}")
        return flow.execute(context)
    
    @staticmethod
    def _determine_flow(context: DubbingContext):
        if not context.manifest:
            logger.info("Fresh video dub detected (no manifest)")
            return VideoDubFlow()
        
        if context.is_redub():
            logger.info(f"Redub detected (parent_job_id: {context.manifest.get('parent_job_id')})")
            return RedubFlow()
        
        current_target_lang = language_service.normalize_language_input(context.target_language)
        manifest_target_lang = context.manifest.get("target_language")
        
        if manifest_target_lang:
            manifest_target_lang = language_service.normalize_language_input(manifest_target_lang)
            if manifest_target_lang == current_target_lang:
                logger.info("Resume detected (same target language)")
                return ResumeFlow()
        
        logger.info("Redub with language change detected")
        return RedubFlow()

