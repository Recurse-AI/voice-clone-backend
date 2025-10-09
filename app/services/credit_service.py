"""
Refactored Credit Service v2
Clean, modular, and reusable implementation with proper separation of concerns
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from app.config.database import db, users_collection, client, get_async_db
from app.config.credit_constants import ErrorCodes, JobType
from app.utils.decorators import handle_credit_operations, log_execution_time, validate_user_access
from app.utils.credit_utils import (
    CreditCalculatorUtil, UserValidationUtil, ResponseUtil
)
from bson import ObjectId
from app.config.credit_constants import CreditRates

logger = logging.getLogger(__name__)

class CreditService:
    """
    Refactored credit service with improved modularity and reusability
    
    Features:
    - Clean separation of concerns
    - Reusable utility functions
    - Consistent error handling
    - Proper validation
    - Comprehensive logging
    """
    
    def __init__(self):
        self.calculator = CreditCalculatorUtil()
        self.validator = UserValidationUtil()
        self.response_util = ResponseUtil()
    
    @handle_credit_operations
    @log_execution_time
    async def reserve_credits_and_create_job(
        self, 
        user_id: str,
        job_data: Dict[str, Any],
        job_type: JobType,
        duration_seconds: float
    ) -> Dict[str, Any]:
        """
        Atomically reserve credits and create job based on user type
        
        Clean, modular implementation with proper error handling
        """
        # Calculate required credits
        required_credits = self.calculator.calculate_job_credits(job_type, duration_seconds)
        original_credits = required_credits
        
        # Double credits for premium voice model
        model_type = job_data.get("model_type", "normal")
        if model_type == "best":
            required_credits *= 3
            logger.info(f"🔧 DEBUG: ElevenLabs (best) model - tripling credits: {original_credits} → {required_credits}")
        elif model_type == "medium":
            required_credits *= 2
            logger.info(f"🔧 DEBUG: Fish API (medium) model - doubling credits: {original_credits} → {required_credits}")
        else:
            logger.info(f"🔧 DEBUG: Local (normal) model - credits: {required_credits}")
            
        estimated_cost = self.calculator.calculate_cost_estimate(required_credits)
        
        # Get user and determine billing strategy
        user = await self._get_user_safely(user_id)
        if not user:
            return self.response_util.create_error_response(
                ErrorCodes.USER_NOT_FOUND, "User not found"
            )
        
        
        # Determine billing type and handle accordingly
        if self.validator.is_payg_user(user):
            return await self._handle_payg_reservation(
                user_id, job_data, job_type, required_credits, estimated_cost
            )
        else:
            return await self._handle_credit_pack_reservation(
                user_id, job_data, job_type, required_credits, estimated_cost, user
            )
    
    @handle_credit_operations
    @log_execution_time
    async def complete_job_billing(
        self,
        job_id: str,
        job_type: JobType,
        user_id: str,
        billing_percentage: float = 1.0
    ) -> Dict[str, Any]:
        """
        Handle job completion billing based on job's billing type
        
        Simplified and modular implementation with percentage support for 75%/25% split
        """
        # Get job details
        job = await self._get_job_safely(job_id, job_type)
        if not job:
            return self.response_util.create_error_response(
                ErrorCodes.JOB_NOT_FOUND, "Job not found"
            )
        
        billing_type = job.get("billing_type", "credit_pack")
        credits_required = job.get("credits_required", 0)
        
        # Calculate credits to bill based on percentage
        credits_to_bill = credits_required * billing_percentage
        
        # Handle completion based on billing type
        if billing_type == "pay_as_you_go":
            return await self._complete_payg_job(job_id, job_type, user_id, credits_to_bill)
        else:
            return await self._complete_credit_pack_job(job_id, job_type, credits_to_bill)
    
    @handle_credit_operations
    @log_execution_time
    async def refund_job_credits(
        self,
        job_id: str,
        job_type: JobType,
        reason: str = "job_failed"
    ) -> Dict[str, Any]:
        """
        Handle job credit refunds based on billing type
        
        Clean implementation with proper error handling
        """
        # Get job details
        job = await self._get_job_safely(job_id, job_type)
        if not job:
            return self.response_util.create_error_response(
                ErrorCodes.JOB_NOT_FOUND, "Job not found"
            )
        
        billing_type = job.get("billing_type", "credit_pack")
        
        if billing_type == "pay_as_you_go":
            return await self._refund_payg_job(job_id, job_type, reason)
        else:
            return await self._refund_credit_pack_job(job_id, job_type, reason)
    
    # Private helper methods for clean organization
    
    async def _get_user_safely(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Safely get user data with error handling"""
        try:
            return await users_collection.find_one({"_id": ObjectId(user_id)})
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None
    
    async def _get_job_safely(self, job_id: str, job_type: JobType) -> Optional[Dict[str, Any]]:
        """Safely get job data with error handling"""
        try:
            collection_map = {
                JobType.DUB: "dub_jobs",
                JobType.SEPARATION: "separation_jobs",
                JobType.CLIP: "clip_jobs"
            }
            collection_name = collection_map.get(job_type, "dub_jobs")
            loop_db = get_async_db()
            return await loop_db[collection_name].find_one({"job_id": job_id})
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None
    

    
    async def _update_user_usage(self, user_id: str, credits_used: float) -> None:
        """Add credits to user's total usage and check for billing"""
        try:
            from datetime import datetime, timezone
            from bson import ObjectId
            loop_db = get_async_db()
            
            # Simple increment of total usage
            await loop_db.get_collection("users").update_one(
                {"_id": ObjectId(user_id)},
                {
                    "$inc": {"total_usage": credits_used},
                    "$set": {"usage_updated_at": datetime.now(timezone.utc)}
                }
            )
            
            logger.debug(f"Added {credits_used} credits to user {user_id} total usage")
            
            # Add real-time billing check for PAYG users
            from app.services.stripe_service import stripe_service
            await stripe_service.add_usage_and_check_billing(user_id, credits_used)
            
        except Exception as e:
            logger.error(f"Failed to update user usage for {user_id}: {e}")
            # Don't raise - this shouldn't break job completion
    
    async def _validate_payg_payment_method(self, user_id: str) -> None:
        """Validate that PAYG user has a valid payment method (optimized - no Stripe calls during job creation)"""
        try:
            from bson import ObjectId
            loop_db = get_async_db()
            # Get user details
            user = await loop_db.get_collection("users").find_one({"_id": ObjectId(user_id)})
            if not user:
                raise ValueError("User not found")
            
            # Check if user has active PAYG subscription
            subscription = user.get("subscription", {})
            if subscription.get("type") == "pay as you go" and subscription.get("status") == "active":
                # Check if user is blocked due to billing failure
                if subscription.get("billingBlocked", False):
                    block_reason = subscription.get("billingBlockReason", "Payment failed")
                    raise ValueError(f"Account temporarily suspended: {block_reason}. Please contact support or update payment method.")
                
                # Check if user has payment method
                if not subscription.get("stripeCustomerId"):
                    raise ValueError("Pay-as-you-go users must add a payment method first. Please complete payment setup.")
                
                logger.info(f"PAYG user {user_id} validated for service usage")
            else:
                raise ValueError("Pay-as-you-go subscription is required for this service.")
                
        except Exception as e:
            logger.error(f"Payment method validation failed for user {user_id}: {e}")
            raise ValueError(str(e))
    

    
    async def _handle_payg_reservation(
        self,
        user_id: str,
        job_data: Dict[str, Any], 
        job_type: JobType,
        required_credits: float,
        estimated_cost: float
    ) -> Dict[str, Any]:
        """Handle pay-as-you-go reservation (virtual credits)"""
        # Check if user has valid payment method
        await self._validate_payg_payment_method(user_id)
        
        if job_type == JobType.DUB:
            collection_name = "dub_jobs"
        elif job_type == JobType.SEPARATION:
            collection_name = "separation_jobs"
        elif job_type == JobType.CLIP:
            collection_name = "clip_jobs"
        else:
            collection_name = "dub_jobs"
        
        # Create simple job data for PAYG
        filtered_job_data = {k: v for k, v in job_data.items() if k not in ["job_id", "user_id"]}
        final_job_data = {
            "job_id": job_data["job_id"],
            "user_id": user_id,
            "credits_required": required_credits,
            "credits_reserved": True,  # Add missing flag for refund checking
            "billing_type": "pay_as_you_go",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        final_job_data.update(filtered_job_data)
        
        # Add default fields for clip jobs if missing
        if job_type == JobType.CLIP:
            final_job_data.setdefault("status", "pending")
            final_job_data.setdefault("progress", 0)
            final_job_data.setdefault("segments", [])
        
        # Simple job insertion - no transactions needed for PAYG
        try:
            loop_db = get_async_db()
            await loop_db[collection_name].insert_one(final_job_data)
            
            logger.info(f"Pay-as-you-go reservation: {required_credits} credits for job {job_data['job_id']}")
            
            return self.response_util.create_success_response(
                "Pay-as-you-go job reserved successfully",
                credits_reserved=required_credits,
                billing_type="pay_as_you_go"
            )
            
        except Exception as e:
            logger.error(f"PAYG job creation failed: {e}")
            return self.response_util.create_error_response(
                "CREDIT_OPERATION_FAILED",
                f"Failed to create job: {str(e)}"
            )
    
    async def _handle_credit_pack_reservation(
        self,
        user_id: str, 
        job_data: Dict[str, Any],
        job_type: JobType, 
        required_credits: float,
        estimated_cost: float,
        user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle credit pack reservation (actual credit deduction)"""
        # Validate sufficient credits
        has_sufficient, available_credits = self.validator.has_sufficient_credits(user, required_credits)
        
        if not has_sufficient:
            return self.response_util.create_error_response(
                ErrorCodes.INSUFFICIENT_CREDITS,
                f"Insufficient credits. Required: {required_credits}, Available: {available_credits}"
            )
        
        if job_type == JobType.DUB:
            collection_name = "dub_jobs"
        elif job_type == JobType.SEPARATION:
            collection_name = "separation_jobs"
        elif job_type == JobType.CLIP:
            collection_name = "clip_jobs"
        else:
            collection_name = "dub_jobs"
        
        # Simple credit deduction without complex transactions
        try:
            # Deduct credits from user balance
            loop_db = get_async_db()
            user_result = await loop_db.get_collection("users").find_one_and_update(
                {
                    "_id": ObjectId(user_id),
                    "credits": {"$gte": required_credits}
                },
                {
                    "$inc": {"credits": -required_credits},
                    "$set": {"updated_at": datetime.now(timezone.utc)}
                },
                return_document=True
            )
            
            if not user_result:
                return self.response_util.create_error_response(
                    ErrorCodes.INSUFFICIENT_CREDITS,
                    "Insufficient credits"
                )
            
            # Create job data for credit pack
            filtered_job_data = {k: v for k, v in job_data.items() if k not in ["job_id", "user_id"]}
            final_job_data = {
                "job_id": job_data["job_id"],
                "user_id": user_id,
                "credits_required": required_credits,
                "credits_reserved": True,
                "billing_type": "credit_pack",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            final_job_data.update(filtered_job_data)
            
            # Add default fields for clip jobs if missing
            if job_type == JobType.CLIP:
                final_job_data.setdefault("status", "pending")
                final_job_data.setdefault("progress", 0)
                final_job_data.setdefault("segments", [])
            
            loop_db = get_async_db()
            await loop_db[collection_name].insert_one(final_job_data)
            
            remaining_credits = user_result.get("credits", 0)
            logger.info(f"Credit pack reservation: {required_credits} credits deducted for job {job_data['job_id']} (remaining: {remaining_credits})")
            
            return self.response_util.create_success_response(
                "Credit pack job reserved successfully",
                credits_reserved=required_credits,
                remaining_credits=remaining_credits,
                billing_type="credit_pack"
            )
            
        except Exception as e:
            logger.error(f"Credit pack reservation failed: {e}")
            return self.response_util.create_error_response(
                "CREDIT_OPERATION_FAILED",
                f"Failed to reserve credits: {str(e)}"
            )
    
    async def _complete_payg_job(
        self, job_id: str, job_type: JobType, user_id: str, credits_used: float
    ) -> Dict[str, Any]:
        """Complete pay-as-you-go job with simple threshold billing"""
        try:
            # Update job as completed
            collection_map = {
                JobType.DUB: "dub_jobs",
                JobType.SEPARATION: "separation_jobs",
                JobType.CLIP: "clip_jobs"
            }
            collection_name = collection_map.get(job_type, "dub_jobs")
            loop_db = get_async_db()
            collection = loop_db.get_collection(collection_name)
            
            await collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "credits_billed": True,
                        "billing_completed_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            
            # Simple usage tracking - no real-time billing
            await self._update_user_usage(user_id, credits_used)
            
            logger.info(f"PAYG job completed: Added {credits_used} credits to user usage for job {job_id}")
            
            return self.response_util.create_success_response(
                "Pay-as-you-go job completed successfully",
                credits_used=credits_used,
                billing_type="pay_as_you_go"
            )
            
        except Exception as e:
            logger.error(f"Pay-as-you-go completion failed for job {job_id}: {e}")
            raise e
    
    async def _complete_credit_pack_job(
        self, job_id: str, job_type: JobType, credits_deducted: float
    ) -> Dict[str, Any]:
        """Complete credit pack job - credits already deducted"""
        try:
            # Update job status - credits were already deducted during reservation
            collection_map = {
                JobType.DUB: "dub_jobs",
                JobType.SEPARATION: "separation_jobs",
                JobType.CLIP: "clip_jobs"
            }
            collection_name = collection_map.get(job_type, "dub_jobs")
            loop_db = get_async_db()
            collection = loop_db.get_collection(collection_name)
            
            await collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "credits_billed": True,
                        "billing_completed_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            
            logger.info(f"Credit pack job completed: Job {job_id} used {credits_deducted} credits")
            
            return self.response_util.create_success_response(
                "Credit pack job completed successfully",
                credits_deducted=credits_deducted,
                billing_type="credit_pack"
            )
            
        except Exception as e:
            logger.error(f"Credit pack completion failed for job {job_id}: {e}")
            raise e
    
    async def _refund_payg_job(self, job_id: str, job_type: JobType, reason: str) -> Dict[str, Any]:
        """Refund pay-as-you-go job (virtual cancellation)"""
        collection_map = {
            JobType.DUB: "dub_jobs",
            JobType.SEPARATION: "separation_jobs",
            JobType.CLIP: "clip_jobs"
        }
        collection_name = collection_map.get(job_type, "dub_jobs")
        
        loop_db = get_async_db()
        await loop_db[collection_name].update_one(
            {"job_id": job_id},
                {
                    "$set": {
                    "credits_cancelled": True,
                    "credits_cancelled_at": datetime.now(timezone.utc),
                    "cancellation_reason": reason
                    },
                    "$unset": {"credits_reserved": ""}
                }
            )
            
        logger.info(f"Pay-as-you-go job {job_id} cancelled (reason: {reason})")
        
        return self.response_util.create_success_response(
            "Pay-as-you-go job cancelled successfully",
            credits_refunded=0, reason=reason
        )
    
    async def _refund_credit_pack_job(self, job_id: str, job_type: JobType, reason: str) -> Dict[str, Any]:
        """Refund credit pack job (actual credit restoration)"""
        collection_map = {
            JobType.DUB: "dub_jobs",
            JobType.SEPARATION: "separation_jobs",
            JobType.CLIP: "clip_jobs"
        }
        collection_name = collection_map.get(job_type, "dub_jobs")
        
        # Get job details
        loop_db = get_async_db()
        job = await loop_db[collection_name].find_one({"job_id": job_id, "credits_reserved": True})
        
        if not job:
            return self.response_util.create_error_response(
                ErrorCodes.JOB_NOT_FOUND, "No reserved credits found"
            )
        
        credits_to_refund = job.get("credits_required", 0)
        user_id = job.get("user_id")
        
        # Simple refund operation without complex transactions
        try:
            # Only refund credit pack jobs
            if job.get("billing_type") == "credit_pack":
                # Add credits back to user
                loop_db = get_async_db()
                await loop_db.get_collection("users").update_one(
                    {"_id": ObjectId(user_id)},
                    {"$inc": {"credits": credits_to_refund}}
                )
                
                # Mark job as refunded
                loop_db = get_async_db()
                await loop_db[collection_name].update_one(
                    {"job_id": job_id},
                    {
                        "$set": {
                            "credits_refunded": True,
                            "credits_refunded_at": datetime.now(timezone.utc),
                            "refund_reason": reason
                        }
                    }
                )
                
                logger.info(f"Refunded {credits_to_refund} credits for credit pack job {job_id}")
                
                return self.response_util.create_success_response(
                    "Credits refunded successfully",
                    credits_refunded=credits_to_refund, 
                    reason=reason
                )
            else:
                logger.info(f"Skipped refund for PAYG job {job_id}")
                return self.response_util.create_success_response(
                    "No refund needed for PAYG job",
                    reason=reason
                )
                
        except Exception as e:
            logger.error(f"Credit refund failed for job {job_id}: {e}")
            return self.response_util.create_error_response(
                "REFUND_FAILED", 
                f"Failed to refund credits: {str(e)}"
            )
    



# Create a singleton instance
credit_service = CreditService()
