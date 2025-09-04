"""
Reusable job utility functions for better code organization
"""
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
from app.config.settings import settings
from app.services.credit_service import credit_service
from app.config.credit_constants import JobType

logger = logging.getLogger(__name__)


class JobUtils:
    """Utility class for common job operations"""
    
    @staticmethod
    async def validate_job_for_redub(job) -> Dict[str, Any]:
        """
        Validate if a job is suitable for redub operation
        Returns: {"valid": bool, "message": str, "manifest_url": str}
        """
        if not job:
            return {"valid": False, "message": "Job not found"}
        
        # Check job status
        if job.status not in ["completed", "awaiting_review"]:
            return {
                "valid": False, 
                "message": f"Job must be completed or awaiting review. Current status: {job.status}"
            }
        
        # Check manifest availability
        manifest_url = job.segments_manifest_url or (job.details or {}).get("segments_manifest_url")
        if not manifest_url:
            return {"valid": False, "message": "No manifest available for this job"}
        
        return {"valid": True, "message": "Job is valid for redub", "manifest_url": manifest_url}
    
    @staticmethod
    def validate_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate manifest integrity
        Returns: {"valid": bool, "message": str}
        """
        if not manifest:
            return {"valid": False, "message": "Manifest is empty"}
        
        if not manifest.get("segments") or len(manifest["segments"]) == 0:
            return {"valid": False, "message": "No segments found in manifest"}
        
        # Check for required fields in segments
        for i, segment in enumerate(manifest["segments"]):
            if not segment.get("id"):
                return {"valid": False, "message": f"Segment {i} missing id"}
            if not segment.get("original_text") and not segment.get("dubbed_text"):
                return {"valid": False, "message": f"Segment {i} missing text content"}
        
        return {"valid": True, "message": "Manifest is valid"}
    

    @staticmethod
    async def setup_job_directory(source_job_id: str, target_job_id: str) -> str:
        """
        Setup job directory by copying from source or creating new
        Returns: path to the new job directory
        """
        source_dir = os.path.join(settings.TEMP_DIR, source_job_id)
        target_dir = os.path.join(settings.TEMP_DIR, target_job_id)
        
        try:
            if os.path.exists(source_dir):
                shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
                logger.info(f"Copied job files from {source_dir} to {target_dir}")
            else:
                os.makedirs(target_dir, exist_ok=True)
                logger.warning(f"Source directory not found: {source_dir}. Created new directory: {target_dir}")
            
            return target_dir
            
        except Exception as e:
            # Clean up partial state
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir, ignore_errors=True)
            raise Exception(f"Failed to setup job directory: {str(e)}")
    
    @staticmethod
    def prepare_manifest_for_redub(manifest: Dict[str, Any], redub_job_id: str, target_language: str, parent_job_id: str) -> Dict[str, Any]:
        """
        Prepare manifest for redub with updated metadata
        Returns: updated manifest
        
        Note: We preserve the original target_language for proper override comparison.
        The new target language is passed via the API call parameters.
        """
        updated_manifest = manifest.copy()
        updated_manifest.update({
            "job_id": redub_job_id,
            # Don't update target_language - preserve original for override comparison
            "redub_target_language": target_language,  # Store new target language separately
            "parent_job_id": parent_job_id,
            "version": 1,  # Reset version for new job
            "redub_timestamp": datetime.now().isoformat()
        })
        return updated_manifest
    

    # ===== CREDIT BILLING UTILITIES =====
    
    @staticmethod
    async def complete_job_billing(job_id: str, job_type: str, user_id: str, billing_percentage: float = 1.0) -> bool:
        """
        Complete credit billing for job completion (async context)
        Centralized utility for reusability across all routes
        Supports 75%/25% billing split
        """
        try:
            job_type_enum = JobType.DUB if job_type.lower() == "dub" else JobType.SEPARATION
            
            billing_result = await credit_service.complete_job_billing(job_id, job_type_enum, user_id, billing_percentage)
            
            if billing_result.get("success"):
                logger.info(f"Credit billing completed for {job_type} job {job_id} ({billing_percentage*100}%)")
                return True
            else:
                logger.warning(f"⚠️ Credit billing failed for {job_type} job {job_id}: {billing_result.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Credit billing error for {job_type} job {job_id}: {e}")
            return False
    
    @staticmethod 
    def complete_job_billing_sync(job_id: str, job_type: str, user_id: str, billing_percentage: float = 1.0) -> bool:
        """
        Complete credit billing for job completion (sync context)
        For use in thread pools and background tasks
        Supports 75%/25% billing split
        """
        try:
            import asyncio
            from app.utils.event_loop_manager import loop_manager
            
            job_type_enum = JobType.DUB if job_type.lower() == "dub" else JobType.SEPARATION
            
            # Get the main event loop
            main_loop = loop_manager.get_main_loop()
            
            if main_loop and main_loop.is_running():
                # Schedule the coroutine on the main loop from this thread
                future = asyncio.run_coroutine_threadsafe(
                    credit_service.complete_job_billing(job_id, job_type_enum, user_id, billing_percentage),
                    main_loop
                )
                # Wait for the result with timeout
                try:
                    billing_result = future.result(timeout=30)
                except Exception as e:
                    logger.error(f"Async billing failed for {job_id}: {e}")
                    billing_result = {"success": False, "error": str(e)}
            else:
                # Fallback: No main loop available (shouldn't happen in production)
                logger.warning(f"Main event loop not available for job {job_id}, skipping billing")
                return False
            
            if billing_result.get("success"):
                logger.info(f"Credit billing completed for {job_type} job {job_id} ({billing_percentage*100}%)")
                return True
            else:
                logger.warning(f"⚠️ Credit billing failed for {job_type} job {job_id}: {billing_result.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Credit billing error for {job_type} job {job_id}: {e}")
            return False
    
    @staticmethod
    async def refund_job_credits(job_id: str, job_type: str, reason: str = "job_failed") -> bool:
        """
        Refund credits for failed jobs (async context)
        Centralized utility for reusability across all routes
        """
        try:
            job_type_enum = JobType.DUB if job_type.lower() == "dub" else JobType.SEPARATION
            
            billing_result = await credit_service.refund_job_credits(job_id, job_type_enum, reason)
            
            if billing_result.get("success"):
                logger.info(f"💰 Credits refunded for {job_type} job {job_id} (reason: {reason})")
                return True
            else:
                logger.warning(f"⚠️ Credit refund failed for {job_type} job {job_id}: {billing_result.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Credit refund error for {job_type} job {job_id}: {e}")
            return False
    
    @staticmethod
    def refund_job_credits_sync(job_id: str, job_type: str, reason: str = "job_failed") -> bool:
        """
        Refund credits for failed jobs (sync context)  
        For use in thread pools and background tasks
        """
        try:
            import asyncio
            from app.utils.event_loop_manager import loop_manager
            
            job_type_enum = JobType.DUB if job_type.lower() == "dub" else JobType.SEPARATION
            
            # Get the main event loop
            main_loop = loop_manager.get_main_loop()
            
            if main_loop and main_loop.is_running():
                # Schedule the coroutine on the main loop from this thread
                future = asyncio.run_coroutine_threadsafe(
                    credit_service.refund_job_credits(job_id, job_type_enum, reason),
                    main_loop
                )
                # Wait for the result with timeout
                billing_result = future.result(timeout=30)
            else:
                # Fallback: No main loop available (shouldn't happen in production)
                logger.warning(f"Main event loop not available for job {job_id}, skipping refund")
                return False
            
            if billing_result.get("success"):
                logger.info(f"💰 Credits refunded for {job_type} job {job_id} (reason: {reason})")
                return True
            else:
                logger.warning(f"⚠️ Credit refund failed for {job_type} job {job_id}: {billing_result.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Credit refund error for {job_type} job {job_id}: {e}")
            return False


# Create global instance for easy access
job_utils = JobUtils()
