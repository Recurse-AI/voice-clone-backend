import logging
from typing import Optional, Dict, Any, List
from app.services.dub.context import DubbingContext
from app.services.dub.orchestrators.flow_coordinator import FlowCoordinator
from app.services.dub.utils.voice_cleanup import VoiceCleanup
from app.services.dub.utils.progress_tracker import ProgressTracker
from app.services.simple_status_service import JobStatus
from app.services.language_service import language_service
from app.utils.audio import AudioUtils

logger = logging.getLogger(__name__)

class DubbingOrchestrator:
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
        valid, error, target_code = self._validate_target_language(target_language, model_type)
        if not valid:
            logger.error(error)
            return {"success": False, "error": error}
        
        try:
            context = self._build_context(
                job_id, target_language, target_code, source_video_language,
                output_dir, review_mode, manifest_override, separation_urls,
                video_subtitle, model_type, voice_type, reference_ids,
                add_subtitle_to_video, num_of_speakers
            )
            
            ProgressTracker.update_status(
                job_id, JobStatus.PROCESSING, 60,
                {"message": "Starting dubbing", "phase": "dubbing"}
            )
            
            result = FlowCoordinator.determine_and_execute(context)
            
            VoiceCleanup.cleanup_voices(context.created_voice_ids, context.model_type)
            
            return result
        
        except Exception as e:
            logger.error(f"Dubbed processing failed: {e}")
            if output_dir:
                AudioUtils.remove_temp_dir(folder_path=output_dir)
            if 'context' in locals():
                VoiceCleanup.cleanup_voices(locals()['context'].created_voice_ids, model_type)
            return {"success": False, "error": str(e)}
    
    def _validate_target_language(self, target_language: str, model_type: str = "normal") -> tuple:
        if not language_service.is_dubbing_supported(target_language, model_type):
            supported = language_service.get_supported_dubbing_languages(model_type)
            model_names = {'best': 'ElevenLabs', 'medium': 'Fish API', 'normal': 'Local'}
            error = f"Unsupported target language '{target_language}' for {model_names.get(model_type, model_type)} model. Supported: {len(supported)} languages"
            return False, error, None
        
        code = language_service.normalize_language_input(target_language)
        logger.info(f"Processing with target language: {target_language} -> {code} (model: {model_type})")
        return True, None, code
    
    def _build_context(
        self,
        job_id: str,
        target_language: str,
        target_language_code: str,
        source_video_language: Optional[str],
        output_dir: str,
        review_mode: bool,
        manifest_override: Optional[Dict[str, Any]],
        separation_urls: Optional[Dict[str, str]],
        video_subtitle: bool,
        model_type: str,
        voice_type: Optional[str],
        reference_ids: Optional[List[str]],
        add_subtitle_to_video: bool,
        num_of_speakers: int
    ) -> DubbingContext:
        
        if manifest_override:
            model_type = manifest_override.get("model_type", model_type)
            voice_type = voice_type or manifest_override.get("voice_type")
            reference_ids = reference_ids or manifest_override.get("reference_ids", [])
            add_subtitle_to_video = manifest_override.get("add_subtitle_to_video", add_subtitle_to_video)
            num_of_speakers = manifest_override.get("num_of_speakers", num_of_speakers)
        
        return DubbingContext(
            job_id=job_id,
            target_language=target_language,
            target_language_code=target_language_code,
            source_video_language=source_video_language,
            model_type=model_type,
            voice_type=voice_type,
            reference_ids=reference_ids or [],
            num_of_speakers=num_of_speakers,
            review_mode=review_mode,
            add_subtitle_to_video=add_subtitle_to_video,
            video_subtitle=video_subtitle,
            process_temp_dir=output_dir,
            manifest=manifest_override,
            separation_urls=separation_urls
        )

def get_dubbing_orchestrator() -> DubbingOrchestrator:
    return DubbingOrchestrator()

