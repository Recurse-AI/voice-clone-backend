"""
Comprehensive Job Cancellation Service
Handles cancellation of jobs across all systems: RunPod, threads, local storage, and state
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from app.utils.runpod_service import runpod_service
from app.utils.shared_memory import mark_job_cancelled, delete_upload_status_async
from app.utils.db_sync_operations import cleanup_separation_files
from app.services.separation_job_service import separation_job_service
from app.services.dub_job_service import dub_job_service
from app.services.credit_service import credit_service, JobType
from app.services.dub.audio_utils import AudioUtils
from app.config.settings import settings
import os

logger = logging.getLogger(__name__)


class JobCancellationService:
    """Comprehensive job cancellation service"""
    
    @staticmethod
    async def cancel_separation_job(job_id: str, user_id: str, hard_delete: bool = True) -> Dict[str, Any]:
        """Cancel separation job comprehensively"""
        try:
            # 1. Get job details first
            job = await separation_job_service.get_job(job_id)
            if not job:
                return {"success": False, "error": "Separation job not found"}
            
            if job.user_id != user_id:
                return {"success": False, "error": "Access denied"}
            
            runpod_request_id = getattr(job, 'runpod_request_id', None)
            
            # 2. Mark job as cancelled (signals background threads to stop)
            mark_job_cancelled(job_id)
            logger.info(f"🛑 Marked separation job {job_id} as cancelled")
            
            # 3. Cancel RunPod job if exists
            runpod_cancelled = False
            if runpod_request_id:
                try:
                    runpod_cancelled = runpod_service.cancel_job(runpod_request_id)
                    if runpod_cancelled:
                        logger.info(f"🛑 Cancelled RunPod job {runpod_request_id}")
                    else:
                        logger.warning(f"Failed to cancel RunPod job {runpod_request_id}")
                except Exception as e:
                    logger.warning(f"RunPod cancellation failed for {runpod_request_id}: {e}")
            
            # 4. Update job status to cancelled
            await separation_job_service.update_job_status(
                job_id, 
                "cancelled", 
                0,
                cancelled_at=datetime.now(timezone.utc),
                error="Job cancelled by user"
            )
            
            # 5. Cleanup files asynchronously
            try:
                def cleanup_files():
                    cleanup_separation_files(job_id)
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, cleanup_files)
                logger.info(f"🧹 Cleaned up files for separation job {job_id}")
            except Exception as e:
                logger.warning(f"File cleanup failed for separation job {job_id}: {e}")
            
            # 6. Refund credits
            try:
                refund_result = await credit_service.refund_reserved_credits(
                    job_id, JobType.SEPARATION, "user_cancelled"
                )
                if refund_result["success"]:
                    logger.info(f"💰 Refunded {refund_result['credits_refunded']} credits for cancelled separation {job_id}")
            except Exception as e:
                logger.warning(f"Credit refund failed for separation {job_id}: {e}")
            
            # 7. Clear upload status
            try:
                await delete_upload_status_async(job_id)
                logger.info(f"🧹 Cleared upload status for job {job_id}")
            except Exception as e:
                logger.warning(f"Upload status cleanup failed for {job_id}: {e}")
            
            # 8. Database handling based on hard_delete option
            if hard_delete:
                # Complete database deletion (default)
                success = await separation_job_service.delete_job(job_id, user_id)
                db_action = "deleted"
            else:
                # Soft delete - keep record with cancelled status
                await separation_job_service.update_job_status(
                    job_id, 
                    "cancelled", 
                    0,
                    cancelled_at=datetime.now(timezone.utc),
                    error="Job cancelled by user"
                )
                success = True
                db_action = "marked_cancelled"
            
            return {
                "success": success,
                "message": "Separation job cancelled successfully",
                "job_id": job_id,
                "runpod_cancelled": runpod_cancelled,
                "cleanup_completed": True,
                "database_action": db_action
            }
            
        except Exception as e:
            logger.error(f"Failed to cancel separation job {job_id}: {e}")
            return {
                "success": False, 
                "error": str(e),
                "message": "Failed to cancel separation job",
                "job_id": job_id,
                "runpod_cancelled": False,
                "cleanup_completed": False,
                "database_action": "none"
            }
    
    @staticmethod  
    async def cancel_dub_job(job_id: str, user_id: str, hard_delete: bool = True) -> Dict[str, Any]:
        """Cancel dub job comprehensively"""
        try:
            # 1. Get job details first
            job = await dub_job_service.get_job(job_id)
            if not job:
                return {"success": False, "error": "Dub job not found"}
                
            if job.user_id != user_id:
                return {"success": False, "error": "Access denied"}
            
            # 2. Mark job as cancelled (signals background threads to stop)
            mark_job_cancelled(job_id)
            logger.info(f"🛑 Marked dub job {job_id} as cancelled")
            
            # 3. Update job status to cancelled
            await dub_job_service.update_job_status(
                job_id,
                "cancelled", 
                0,
                cancelled_at=datetime.now(timezone.utc),
                error="Job cancelled by user"
            )
            
            # 4. Cleanup dub files 
            try:
                def cleanup_dub_files():
                    # Clean up dub temp directories - comprehensive patterns
                    temp_patterns = [
                        f"dub_{job_id}",                    # Main job folder  
                        f"voice_clone_{job_id}",           # Voice cloning temp
                        f"separation_{job_id}",            # Separation temp
                        f"audio_{job_id}",                 # Audio processing temp
                        f"processing_{job_id}",            # General processing temp
                        f"voice_cloning/dub_job_{job_id}", # Nested voice cloning
                        f"transcription_{job_id}",         # Transcription temp
                        f"segments_{job_id}"               # Segments temp
                    ]
                    
                    for pattern in temp_patterns:
                        temp_dir = os.path.join(settings.TEMP_DIR, pattern)
                        if os.path.exists(temp_dir):
                            AudioUtils.remove_temp_dir(folder_path=temp_dir)
                            logger.info(f"🧹 Removed dub temp directory: {temp_dir}")
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, cleanup_dub_files)
                logger.info(f"🧹 Cleaned up files for dub job {job_id}")
            except Exception as e:
                logger.warning(f"File cleanup failed for dub job {job_id}: {e}")
            
            # 5. Refund credits
            try:
                refund_result = await credit_service.refund_reserved_credits(
                    job_id, JobType.DUB, "user_cancelled"
                )
                if refund_result["success"]:
                    logger.info(f"💰 Refunded {refund_result['credits_refunded']} credits for cancelled dub {job_id}")
            except Exception as e:
                logger.warning(f"Credit refund failed for dub {job_id}: {e}")
            
            # 6. Clear upload status 
            try:
                await delete_upload_status_async(job_id)
                logger.info(f"🧹 Cleared upload status for job {job_id}")
            except Exception as e:
                logger.warning(f"Upload status cleanup failed for {job_id}: {e}")
            
            # 7. Database handling based on hard_delete option
            if hard_delete:
                # Complete database deletion (default)
                success = await dub_job_service.delete_job(job_id, user_id)
                db_action = "deleted"
            else:
                # Soft delete - keep record with cancelled status
                await dub_job_service.update_job_status(
                    job_id,
                    "cancelled", 
                    0,
                    cancelled_at=datetime.now(timezone.utc),
                    error="Job cancelled by user"
                )
                success = True
                db_action = "marked_cancelled"
            
            return {
                "success": success,
                "message": "Dub job cancelled successfully", 
                "job_id": job_id,
                "cleanup_completed": True,
                "database_action": db_action
            }
            
        except Exception as e:
            logger.error(f"Failed to cancel dub job {job_id}: {e}")
            return {
                "success": False, 
                "error": str(e),
                "message": "Failed to cancel dub job",
                "job_id": job_id,
                "cleanup_completed": False,
                "database_action": "none"
            }


# Global service instance
job_cancellation_service = JobCancellationService()
