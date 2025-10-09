import logging
from typing import Dict, Any
from app.services.dub.context import DubbingContext
from app.services.dub.utils.progress_tracker import ProgressTracker
from app.services.simple_status_service import JobStatus
from app.services.dub.manifest_service import build_manifest, save_manifest_to_dir, upload_process_dir_to_r2
from app.services.r2_service import R2Service

logger = logging.getLogger(__name__)

class ReviewHandler:
    @staticmethod
    def prepare_for_review(context: DubbingContext) -> Dict[str, Any]:
        ProgressTracker.update_phase_progress(
            context.job_id, "review_prep", 0.5, "Preparing segments for review"
        )
        
        if context.manifest:
            manifest = context.manifest.copy()
            manifest["segments"] = context.segments
            manifest["target_language"] = context.target_language
            manifest["model_type"] = context.model_type
            manifest["add_subtitle_to_video"] = context.add_subtitle_to_video
        else:
            manifest = build_manifest(
                context.job_id,
                context.transcript_id,
                context.target_language,
                context.segments,
                context.vocal_url,
                context.instrument_url,
                context.model_type,
                context.voice_type,
                context.reference_ids,
                add_subtitle_to_video=context.add_subtitle_to_video
            )
        
        save_manifest_to_dir(manifest, context.process_temp_dir, context.job_id)
        
        exclude_files = []
        if context.vocal_url:
            exclude_files.append(f"vocal_{context.job_id}.wav")
        if context.instrument_url:
            exclude_files.append(f"instrument_{context.job_id}.wav")
        
        r2_storage = R2Service()
        folder_upload, manifest_url, manifest_key = upload_process_dir_to_r2(
            context.job_id, context.process_temp_dir, r2_storage, exclude_files=exclude_files
        )
        
        if not manifest_url:
            return {"success": False, "error": "Failed to generate manifest URL"}
        
        ReviewHandler._charge_review_credits(context.job_id)
        
        ProgressTracker.update_status(
            context.job_id, JobStatus.AWAITING_REVIEW, 80, {
                "message": "Awaiting human review",
                "segments_manifest_url": manifest_url,
                "segments_manifest_key": manifest_key,
                "segments_count": len(context.segments),
                "transcript_id": context.transcript_id
            }
        )
        
        # Cleanup immediately - files already uploaded to R2 and can be restored from manifest
        try:
            from app.utils.audio import AudioUtils
            AudioUtils.remove_temp_dir(folder_path=context.process_temp_dir)
            logger.info(f"Cleaned up temp files for job {context.job_id} after review prep")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files for job {context.job_id}: {e}")
        
        return {
            "success": True,
            "job_id": context.job_id,
            "review": {
                "segments_manifest_url": manifest_url,
                "segments_manifest_key": manifest_key,
                "segments_count": len(context.segments),
                "transcript_id": context.transcript_id
            },
            "folder_upload": folder_upload
        }
    
    @staticmethod
    def _charge_review_credits(job_id: str):
        from app.utils.job_utils import job_utils
        from app.config.database import sync_client
        from app.config.settings import settings
        
        try:
            job_doc = sync_client[settings.DB_NAME].dub_jobs.find_one({"job_id": job_id})
            if job_doc and job_doc.get("user_id"):
                job_utils.complete_job_billing_sync(job_id, "dub", job_doc["user_id"], 0.75)
                logger.info(f"Charged 75% credits for job {job_id} ready for review")
        except Exception as e:
            logger.error(f"Failed to charge 75% credits for job {job_id}: {e}")

