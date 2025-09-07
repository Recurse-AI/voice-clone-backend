"""
Separation Service - Business logic for audio separation
Clean service layer with single responsibility
"""
import logging
import os
from typing import Dict, Any, Optional
from app.repositories.separation_repository import separation_repo
from app.services.simple_status_service import status_service, JobStatus
from app.services.credit_service import credit_service
from app.config.credit_constants import JobType as CreditJobType
from app.config.settings import settings
from app.utils.runpod_service import runpod_service
from app.services.r2_service import get_r2_service

logger = logging.getLogger(__name__)


class SeparationService:
    """
    Clean separation service with single responsibility:
    - Coordinate separation job creation and processing
    - Handle credit reservations
    - Manage RunPod submissions
    - Update job status through simple status service
    """
    
    def __init__(self):
        self.r2_service = get_r2_service()
    
    async def create_separation_job(self, job_id: str, user_id: str, audio_url: str, 
                                  original_filename: str, duration: float, 
                                  caller_info: str = None) -> Dict[str, Any]:
        """Create and start a separation job"""
        try:
            # Build job data
            job_data = {
                "job_id": job_id,
                "user_id": user_id,
                "audio_url": audio_url,
                "original_filename": original_filename,
                "caller_info": caller_info or "audio_separation_api",
                "duration": duration
            }
            
            # Reserve credits and create job atomically
            credit_result = await credit_service.reserve_credits_and_create_job(
                user_id=user_id,
                job_data=job_data,
                job_type=CreditJobType.SEPARATION,
                duration_seconds=duration
            )
            
            if not credit_result["success"]:
                return {
                    "success": False,
                    "error": credit_result.get("error", "Credit reservation failed")
                }
            
            # Submit to RunPod
            try:
                runpod_request_id = runpod_service.submit_separation_request(
                    audio_url, caller_info=caller_info
                )
                
                # Update job with RunPod request ID
                await separation_repo.set_runpod_request_id(job_id, runpod_request_id)
                
                # Set initial status
                status_service.update_status(
                    job_id, "separation", JobStatus.PENDING, 0,
                    {"message": "Job queued for processing", "phase": "queued"}
                )
                
                logger.info(f"✅ Separation job created: {job_id} (RunPod: {runpod_request_id})")
                
                return {
                    "success": True,
                    "job_id": job_id,
                    "runpod_request_id": runpod_request_id,
                    "message": "Separation job started successfully"
                }
                
            except Exception as runpod_error:
                # Rollback credits if RunPod fails
                await credit_service.refund_job_credits(job_id, CreditJobType.SEPARATION, "runpod_request_failed")
                logger.error(f"RunPod submission failed for {job_id}: {runpod_error}")
                
                return {
                    "success": False,
                    "error": f"Audio separation request failed: {str(runpod_error)}"
                }
                
        except Exception as e:
            logger.error(f"Failed to create separation job {job_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to create separation job: {str(e)}"
            }
    
    async def process_separation_results(self, job_id: str, runpod_output: Dict[str, Any]) -> bool:
        """Process RunPod separation results"""
        try:
            from app.utils.separation_utils import separation_utils
            
            # Extract URLs from RunPod response
            runpod_urls = separation_utils.extract_urls_from_clearvocals_response(runpod_output)
            vocal_url = runpod_urls.get('vocal_audio')
            instrument_url = runpod_urls.get('instrument_audio')
            
            # Download files to local storage
            job_dir = os.path.join(settings.TEMP_DIR, job_id)
            download_success, file_paths = separation_utils.download_separation_files(
                job_id=job_id,
                job_dir=job_dir,
                runpod_urls=runpod_urls,
                on_error_callback=None
            )
            
            if not download_success:
                logger.warning(f"Some files failed to download for job {job_id}")
            
            # Update job with results
            await separation_repo.update_results(
                job_id=job_id,
                vocal_url=vocal_url,
                instrument_url=instrument_url,
                details={
                    "processing_time": "completed",
                    "vocal_file": file_paths.get('vocal'),
                    "instrument_file": file_paths.get('instrument'),
                    "result_data": runpod_output
                }
            )
            
            # Update status to completed
            status_service.update_status(
                job_id, "separation", JobStatus.COMPLETED, 100,
                {
                    "message": "Separation completed successfully",
                    "vocal_url": vocal_url,
                    "instrument_url": instrument_url
                }
            )

            # Send completion email
            # Get user_id from job data
            from app.utils.db_sync_operations import get_separation_job_sync
            job_data = get_separation_job_sync(job_id)
            user_id = job_data.get("user_id") if job_data else None
            if user_id:
                self._send_completion_email(job_id, user_id, vocal_url, instrument_url)

            logger.info(f"✅ Separation results processed for job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process separation results for {job_id}: {e}")
            return False
    
    def fail_separation_job(self, job_id: str, error: str, refund_reason: str = "job_failed") -> bool:
        """Mark separation job as failed and refund credits"""
        try:
            # Update job status sync
            from app.utils.db_sync_operations import SyncDBOperations
            SyncDBOperations.update_separation_job_status(job_id, "failed", 0, error=error)
            
            # Update simple status
            status_service.update_status(
                job_id, "separation", JobStatus.FAILED, 0,
                {"message": "Separation failed", "error": error}
            )
            
            # Refund credits
            from app.utils.job_utils import job_utils
            job_utils.refund_job_credits_sync(job_id, "separation", refund_reason)
            
            logger.info(f"✅ Separation job failed and refunded: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fail separation job {job_id}: {e}")
            return False
    
    def _send_completion_email(self, job_id: str, user_id: str, vocal_url: str = None, instrument_url: str = None):
        """Send completion email notification for separation jobs"""
        try:
            # Get user synchronously to avoid async conflicts
            from app.utils.db_sync_operations import get_user_sync
            user = get_user_sync(user_id)

            if not user:
                logger.warning(f"User {user_id} not found, skipping email")
                return

            logger.info(f"Sending completion email to user {user_id} ({user.get('email')}) for separation job {job_id}")

            # Prepare download URLs
            download_urls = {}
            if vocal_url:
                download_urls["separation_url"] = vocal_url
            if instrument_url:
                download_urls["instrument_url"] = instrument_url

            # Send email directly without BackgroundTasks to avoid async issues
            from app.utils.email_helper import send_email, create_job_completion_template
            from app.config.settings import settings

            # Create email content
            html_body = create_job_completion_template(
                user.get('name', 'User'), "separation", job_id, download_urls
            )

            subject = f"🎵 Your Audio Separation is Ready - ClearVocals"

            # Check if email credentials are configured
            if not settings.EMAIL_HOST_USER or not settings.EMAIL_HOST_PASSWORD:
                logger.warning(f"⚠️ Email credentials not configured - skipping email for separation job {job_id}")
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
                logger.info(f"✅ Completion email sent for separation job {job_id}")
            else:
                logger.error(f"❌ Email failed for separation job {job_id}")
                # Don't raise exception, just log the error so job completion isn't affected

        except Exception as e:
            logger.error(f"❌ Failed to send completion email for separation job {job_id}: {e}")
            logger.error(f"Email error details: user_id={user_id}, error={str(e)}")

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get separation job status"""
        return await separation_repo.get_by_id(job_id)


# Global service instance
separation_service = SeparationService()
