"""
Dub Service - Clean business logic for video dubbing
Replaces complex route logic with clean service layer
"""
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from app.repositories.dub_repository import dub_repo
from app.services.simple_status_service import status_service, JobStatus
from app.services.credit_service import credit_service
from app.config.credit_constants import JobType as CreditJobType
from app.config.settings import settings

logger = logging.getLogger(__name__)


class DubService:
    """
    Clean dub service with single responsibility:
    - Coordinate dub job creation and processing
    - Handle credit reservations
    - Manage file validation and uploads
    - Update job status through simple status service
    """
    
    def __init__(self):
        from app.services.r2_service import R2Service
        self.r2_service = R2Service()
    
    async def create_dub_job(self, job_id: str, user_id: str, 
                           target_language: str, source_video_language: str,
                           project_title: str, duration: float,
                           human_review: bool = False) -> Dict[str, Any]:
        """Create and start a dub job"""
        try:
            # Validate duration
            if duration is None or duration <= 0:
                return {
                    "success": False,
                    "error": "Duration is required and must be greater than 0 seconds"
                }
            
            # Build job data
            job_data = {
                "job_id": job_id,
                "user_id": user_id,
                "target_language": target_language,
                "original_filename": project_title,
                "source_video_language": source_video_language,
                "duration": duration,
                "human_review": human_review
            }
            
            # Reserve credits and create job atomically
            credit_result = await credit_service.reserve_credits_and_create_job(
                user_id=user_id,
                job_data=job_data,
                job_type=CreditJobType.DUB,
                duration_seconds=duration
            )
            
            if not credit_result["success"]:
                return {
                    "success": False,
                    "error": credit_result.get("error", "Credit reservation failed")
                }
            
            # Set initial status
            status_service.update_status(
                job_id, "dub", JobStatus.PENDING, 0,
                {
                    "message": "Job queued for processing",
                    "phase": "queued",
                    "user_id": user_id,
                    "target_language": target_language
                }
            )
            
            logger.info(f"✅ Dub job created: {job_id}")
            
            return {
                "success": True,
                "job_id": job_id,
                "message": "Video dub started successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to create dub job {job_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to create dub job: {str(e)}"
            }
    
    def validate_uploaded_files(self, job_id: str) -> Dict[str, Any]:
        """Validate uploaded audio files for processing"""
        try:
            job_dir = os.path.join(settings.TEMP_DIR, job_id)
            
            if not os.path.exists(job_dir):
                return {
                    "success": False,
                    "error": "Upload directory not found"
                }
            
            # Find audio file
            allowed_formats = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}
            audio_files = [
                f for f in os.listdir(job_dir) 
                if any(f.lower().endswith(ext) for ext in allowed_formats)
            ]
            
            if not audio_files:
                return {
                    "success": False,
                    "error": "No audio file found in upload directory"
                }
            
            audio_path = os.path.join(job_dir, audio_files[0])
            
            return {
                "success": True,
                "audio_path": audio_path,
                "job_dir": job_dir
            }
            
        except Exception as e:
            logger.error(f"File validation failed for {job_id}: {e}")
            return {
                "success": False,
                "error": f"File validation failed: {str(e)}"
            }
    
    def upload_audio_to_r2(self, job_id: str, audio_path: str) -> Dict[str, Any]:
        """Upload audio file to R2 storage"""
        try:
            # Generate R2 key
            r2_key = self.r2_service.generate_file_path(job_id, "", f"{job_id}.wav")
            
            # Upload file
            upload_result = self.r2_service.upload_file(audio_path, r2_key)
            
            if not upload_result.get("success"):
                return {
                    "success": False,
                    "error": f"Audio upload failed: {upload_result.get('error')}"
                }
            
            return {
                "success": True,
                "audio_url": upload_result["url"],
                "r2_key": r2_key
            }
            
        except Exception as e:
            logger.error(f"R2 upload failed for {job_id}: {e}")
            return {
                "success": False,
                "error": f"R2 upload failed: {str(e)}"
            }
    
    def start_processing(self, job_id: str, audio_url: str, 
                             target_language: str, source_video_language: str,
                             human_review: bool = False) -> Dict[str, Any]:
        """Start background processing for dub job"""
        try:
            # Update status to processing
            status_service.update_status(
                job_id, "dub", JobStatus.PROCESSING, 5,
                {
                    "message": "Starting dubbing process",
                    "phase": "initialization",
                    "audio_url": audio_url
                }
            )
            
            # The actual processing will be handled by the worker
            return {
                "success": True,
                "message": "Processing started"
            }
            
        except Exception as e:
            logger.error(f"Failed to start processing for {job_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to start processing: {str(e)}"
            }
    
    def complete_job(self, job_id: str, result_url: str = None, 
                         details: Dict[str, Any] = None, credit_percentage: float = 1.0) -> bool:
        """Mark job as completed with results"""
        try:
            # Get job to extract user_id for billing
            from app.utils.db_sync_operations import get_dub_job_sync, SyncDBOperations
            job_data = get_dub_job_sync(job_id)
            user_id = job_data.get("user_id") if job_data else None
            
            # Update database sync
            SyncDBOperations.update_dub_job_status(job_id, "completed", 100, {
                "result_url": result_url,
                "completed_at": datetime.now().isoformat(),
                **(details or {})
            })
            
            # Update simple status
            status_service.update_status(
                job_id, "dub", JobStatus.COMPLETED, 100,
                {
                    "message": "Video dubbing completed successfully",
                    "result_url": result_url,
                    **(details or {})
                }
            )
            
            # Complete credit billing with specified percentage
            from app.utils.job_utils import job_utils
            job_utils.complete_job_billing_sync(job_id, "dub", user_id, credit_percentage)
            
            # Send completion email
            self._send_completion_email(job_id, user_id, result_url, details)
            
            logger.info(f"✅ Dub job completed: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete dub job {job_id}: {e}")
            return False
    
    def fail_job(self, job_id: str, error: str, refund_reason: str = "job_failed") -> bool:
        """Mark job as failed and refund credits"""
        try:
            # Update database sync
            from app.utils.db_sync_operations import SyncDBOperations
            SyncDBOperations.update_dub_job_status(job_id, "failed", 0, {"error": error})
            
            # Update simple status
            status_service.update_status(
                job_id, "dub", JobStatus.FAILED, 0,
                {"message": "Dubbing failed", "error": error}
            )
            
            # Refund credits
            from app.utils.job_utils import job_utils
            job_utils.refund_job_credits_sync(job_id, "dub", refund_reason)
            
            logger.info(f"✅ Dub job failed and refunded: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fail dub job {job_id}: {e}")
            return False
    
    def _send_completion_email(self, job_id: str, user_id: str, result_url: str = None, details: Dict[str, Any] = None):
        """Send completion email notification"""
        try:
            # Get user synchronously to avoid async conflicts
            from app.utils.db_sync_operations import get_user_sync
            user = get_user_sync(user_id)

            if not user:
                logger.warning(f"User {user_id} not found, skipping email")
                return

            logger.info(f"Sending completion email to user {user_id} ({user.get('email')}) for dub job {job_id}")

            # Prepare download URLs with dynamic frontend URLs
            download_urls = {}
            if details and details.get("result_urls"):
                result_urls = details["result_urls"]
                if result_urls.get("audio_url"):
                    download_urls["audio_url"] = f"{settings.FRONTEND_URL}/workspace/dubbing/audio-download/{job_id}"
                if result_urls.get("video_url"):
                    download_urls["video_url"] = f"{settings.FRONTEND_URL}/workspace/dubbing/video-download/{job_id}"

            # Send email directly without BackgroundTasks to avoid async issues
            from app.utils.email_helper import send_email, create_job_completion_template
            from app.config.settings import settings

            # Create email content
            html_body = create_job_completion_template(
                user.get('name', 'User'), "dub", job_id, download_urls
            )

            subject = f"🎬 Your Video Dubbing is Ready - ClearVocals"

            # Check if email credentials are configured
            if not settings.EMAIL_HOST_USER or not settings.EMAIL_HOST_PASSWORD:
                logger.warning(f"⚠️ Email credentials not configured - skipping email for dub job {job_id}")
                return

            # Send email directly and handle errors gracefully
            email_sent = send_email(
                sender_email=settings.EMAIL_HOST_USER,
                receiver_email=user.get('email'),
                subject=subject,
                body=html_body,
                password=settings.EMAIL_HOST_PASSWORD,
                is_html=True,
                raise_on_error=False  # Don't raise exceptions in worker context
            )
            
            if email_sent:
                logger.info(f"✅ Completion email sent for dub job {job_id}")
            else:
                logger.error(f"❌ Email failed for dub job {job_id}")
                # Don't raise exception, just log the error so job completion isn't affected

        except Exception as e:
            logger.error(f"❌ Failed to send completion email for dub job {job_id}: {e}")
            logger.error(f"Email error details: user_id={user_id}, error={str(e)}")
    
    async def update_review_status(self, job_id: str, review_status: str,
                                 manifest_url: str = None) -> bool:
        """Update job review status"""
        try:
            await dub_repo.update_review_status(job_id, review_status, manifest_url)
            
            # Update simple status based on review status
            if review_status == "awaiting":
                job_status = JobStatus.AWAITING_REVIEW
                progress = 80
                message = "Awaiting human review"
            elif review_status == "approved":
                job_status = JobStatus.REVIEWING
                progress = 81
                message = "Applying human edits"
            else:
                job_status = JobStatus.PROCESSING
                progress = 75
                message = "Processing review feedback"
            
            status_service.update_status(
                job_id, "dub", job_status, progress,
                {
                    "message": message,
                    "review_status": review_status,
                    "segments_manifest_url": manifest_url
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update review status for {job_id}: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive job status"""
        try:
            # Get from database using sync method
            from app.utils.db_sync_operations import get_dub_job_sync
            job_data = get_dub_job_sync(job_id)
            if not job_data:
                return None
            
            # Get current status from simple service
            status_data = status_service.get_status(job_id, "dub")
            
            # Combine data
            return {
                **job_data,
                "current_status": status_data.get("status") if status_data else job_data.get("status"),
                "current_progress": status_data.get("progress") if status_data else job_data.get("progress", 0),
                "last_updated": status_data.get("updated_at") if status_data else job_data.get("updated_at")
            }
            
        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {e}")
            return None


# Global service instance
dub_service = DubService()
