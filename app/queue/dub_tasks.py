from app.schemas import VideoDubRequest

def process_video_dub_task(request_dict: dict, user_id: str):
    from app.routes.video.dub_routes import process_video_dub_background
    request = VideoDubRequest(**request_dict)
    process_video_dub_background(request, user_id)

def process_redub_task(redub_job_id: str, target_language: str, source_video_language: str, 
                      redub_job_dir: str, manifest: dict, human_review: bool):
    from app.services.dub.simple_dubbed_api import get_simple_dubbed_api
    from app.utils.unified_status_manager import get_unified_status_manager, ProcessingStatus
    import logging
    
    logger = logging.getLogger(__name__)
    
    def _update_status_non_blocking(job_id: str, status: ProcessingStatus, progress: int, details: dict):
        manager = get_unified_status_manager()
        manager.update_job_status(job_id, status, progress, details)
    
    try:
        _update_status_non_blocking(redub_job_id, ProcessingStatus.PROCESSING, 45, {
            "message": f"Redubbing to {target_language}", 
            "phase": "initialization"
        })

        api = get_simple_dubbed_api()
        result = api.process_dubbed_audio(
            job_id=redub_job_id,
            target_language=target_language,
            source_video_language=source_video_language,
            output_dir=redub_job_dir,
            review_mode=human_review,
            manifest_override=manifest,
        )
        
        if not result["success"]:
            _update_status_non_blocking(redub_job_id, ProcessingStatus.FAILED, 0, {
                "message": "Redub failed",
                "error": result.get("error")
            })
            return
        
        if human_review:
            if result.get("review"):
                logger.info(f"Redub job {redub_job_id} reached awaiting_review")
                return
        
        result_url = result.get("result_url") or (result.get("result_urls", {}) or {}).get("final_video")
        folder_upload = result.get("folder_upload", {})
        
        _update_status_non_blocking(redub_job_id, ProcessingStatus.COMPLETED, 100, {
            "message": "Redub completed successfully",
            "result_url": result_url,
            "details": result.get("details"),
            "folder_upload": folder_upload,
            "result_urls": result.get("result_urls")
        })
        
        logger.info(f"Redub job {redub_job_id} completed")
        
    except Exception as e:
        logger.error(f"Redub task failed: {e}")
        _update_status_non_blocking(redub_job_id, ProcessingStatus.FAILED, 0, {
            "message": "Redub processing failed",
            "error": str(e)
        })

def process_audio_separation_task(job_id: str, runpod_request_id: str, user_id: str, duration_seconds: float):
    from app.routes.audio_processing import process_audio_separation_background
    process_audio_separation_background(job_id, runpod_request_id, user_id, duration_seconds)
