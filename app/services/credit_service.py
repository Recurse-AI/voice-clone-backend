import logging
from typing import Optional, Dict, Any
from enum import Enum
from app.config.database import users_collection
from app.services.dub_job_service import dub_job_service
from app.services.separation_job_service import separation_job_service
from bson import ObjectId
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class JobType(Enum):
    """Job types with their credit rates"""
    DUB = "dub"           # 0.05 credits per second
    SEPARATION = "separation"  # 1 credit per minute (60 seconds)

class CreditCalculator:
    """Credit calculation logic for different job types"""
    
    @staticmethod
    def calculate_dub_credits(duration_seconds: float) -> float:
        """Calculate credits for dubbing job (0.05 credits per second)"""
        if duration_seconds < 0:
            logger.warning(f"Negative duration for dub credit calculation: {duration_seconds}")
            return 0.0
        return round(duration_seconds * 0.05, 2)
    
    @staticmethod
    def calculate_separation_credits(duration_seconds: float) -> float:
        """Calculate credits for separation job (1 credit per minute)"""
        if duration_seconds < 0:
            logger.warning(f"Negative duration for separation credit calculation: {duration_seconds}")
            return 0.0
        duration_minutes = duration_seconds / 60.0
        return round(duration_minutes * 1.0, 2)
    
    @staticmethod
    def calculate_credits(job_type: JobType, duration_seconds: float) -> float:
        """Calculate credits based on job type and duration"""
        if job_type == JobType.DUB:
            return CreditCalculator.calculate_dub_credits(duration_seconds)
        elif job_type == JobType.SEPARATION:
            return CreditCalculator.calculate_separation_credits(duration_seconds)
        else:
            raise ValueError(f"Unknown job type: {job_type}")

class CreditService:
    """Common credit management service for all job types"""
    
    @staticmethod
    async def check_sufficient_credits(user_id: str, job_type: JobType, duration_seconds: float) -> Dict[str, Any]:
        """
        Check if user has sufficient credits before starting a job
        Returns: {"sufficient": bool, "required": float, "available": float, "message": str}
        """
        try:
            required_credits = CreditCalculator.calculate_credits(job_type, duration_seconds)
            
            # Get user's current credits
            user = await users_collection.find_one({"_id": ObjectId(user_id)})
            if not user:
                return {
                    "sufficient": False,
                    "required": required_credits,
                    "available": 0,
                    "message": "User not found"
                }
            
            available_credits = user.get("credits", 0)
            sufficient = available_credits >= required_credits
            
            return {
                "sufficient": sufficient,
                "required": required_credits,
                "available": available_credits,
                "message": "Sufficient credits" if sufficient else "Insufficient credits"
            }
            
        except Exception as e:
            logger.error(f"Failed to check credits for user {user_id}: {e}")
            return {
                "sufficient": False,
                "required": 0,
                "available": 0,
                "message": f"Credit check failed: {str(e)}"
            }
    
    @staticmethod
    async def deduct_credits_on_completion(
        user_id: str, 
        job_id: str, 
        job_type: JobType, 
        duration_seconds: float
    ) -> Dict[str, Any]:
        """
        Deduct credits after successful job completion
        Returns: {"success": bool, "deducted": float, "remaining": float, "message": str}
        """
        try:
            credits_to_deduct = CreditCalculator.calculate_credits(job_type, duration_seconds)
            
            # Get user's current credits
            user = await users_collection.find_one({"_id": ObjectId(user_id)})
            if not user:
                return {
                    "success": False,
                    "deducted": 0,
                    "remaining": 0,
                    "message": "User not found"
                }
            
            current_credits = user.get("credits", 0)
            
            # Double-check credits (safety net)
            if current_credits < credits_to_deduct:
                logger.warning(f"Insufficient credits during deduction for user {user_id}")
                return {
                    "success": False,
                    "deducted": 0,
                    "remaining": current_credits,
                    "message": "Insufficient credits during deduction"
                }
            
            # Deduct credits
            new_credits = current_credits - credits_to_deduct
            await users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"credits": new_credits}}
            )
            
            # Update job record with credit deduction info
            credit_info = {
                "credits_deducted": True,
                "deducted_amount": credits_to_deduct,
                "deducted_at": f"{__import__('datetime').datetime.now(timezone.utc).isoformat()}",
                "job_type": job_type.value,
                "duration": duration_seconds
            }
            
            if job_type == JobType.DUB:
                job = await dub_job_service.get_job(job_id)
                if job and job.details:
                    updated_details = job.details.copy()
                    updated_details.update(credit_info)
                    await dub_job_service.update_details(job_id, updated_details)
                    
            elif job_type == JobType.SEPARATION:
                job = await separation_job_service.get_job(job_id) 
                if job and job.details:
                    updated_details = job.details.copy()
                    updated_details.update(credit_info)
                    await separation_job_service.update_details(job_id, updated_details)
            
            logger.info(f"Deducted {credits_to_deduct} credits for {job_type.value} job {job_id} (duration: {duration_seconds}s)")
            
            return {
                "success": True,
                "deducted": credits_to_deduct,
                "remaining": new_credits,
                "message": "Credits deducted successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to deduct credits for job {job_id}: {e}")
            return {
                "success": False,
                "deducted": 0,
                "remaining": 0,
                "message": f"Credit deduction failed: {str(e)}"
            }

# Global instance
credit_service = CreditService()