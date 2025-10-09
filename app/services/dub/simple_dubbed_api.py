"""
Backward compatibility wrapper for SimpleDubbedAPI
Delegates to the new DubbingOrchestrator
"""
import logging
from typing import Optional, Dict, Any, List
from app.services.dub.orchestrators.dubbing_orchestrator import DubbingOrchestrator

logger = logging.getLogger(__name__)

class SimpleDubbedAPI:
    def __init__(self):
        self._orchestrator = DubbingOrchestrator()
        logger.info("SimpleDubbedAPI initialized (using new orchestrator)")
    
    def process_dubbed_audio(
        self,
        job_id: str,
        target_language: str,
        source_video_language: str = None,
        output_dir: str = None,
        review_mode: bool = False,
        manifest_override: Optional[Dict[str, Any]] = None,
        separation_urls: Optional[Dict[str, str]] = None,
        video_subtitle: bool = False,
        model_type: str = "normal",
        voice_type: Optional[str] = None,
        reference_ids: Optional[List[str]] = None,
        add_subtitle_to_video: bool = False,
        num_of_speakers: int = 1
    ) -> dict:
        return self._orchestrator.process_dubbed_audio(
            job_id=job_id,
            target_language=target_language,
            source_video_language=source_video_language,
            output_dir=output_dir,
            review_mode=review_mode,
            manifest_override=manifest_override,
            separation_urls=separation_urls,
            video_subtitle=video_subtitle,
            model_type=model_type,
            voice_type=voice_type,
            reference_ids=reference_ids,
            add_subtitle_to_video=add_subtitle_to_video,
            num_of_speakers=num_of_speakers
        )

def get_simple_dubbed_api() -> SimpleDubbedAPI:
    return SimpleDubbedAPI()

