"""
Refactored Credit Service v2
Clean, modular, and reusable implementation with proper separation of concerns
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from app.config.database import db, users_collection, client
from app.config.credit_constants import BillingType, ErrorCodes, JobType
from app.utils.decorators import handle_credit_operations, log_execution_time, validate_user_access
from app.utils.credit_utils import (
    CreditCalculatorUtil, UserValidationUtil, DatabaseUtil, 
    ResponseUtil, SpendingLimitUtil
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
        self.db_util = DatabaseUtil()
        self.response_util = ResponseUtil()
        self.spending_util = SpendingLimitUtil()
    
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
        estimated_cost = self.calculator.calculate_cost_estimate(required_credits)
        
        # Get user and determine billing strategy
        user = await self._get_user_safely(user_id)
        if not user:
            return self.response_util.create_error_response(
                ErrorCodes.USER_NOT_FOUND, "User not found"
            )
        
        # Validate spending limits
        spending_check = await self._validate_spending_limits(user_id, estimated_cost)
        if not spending_check["allowed"]:
            return self.response_util.create_error_response(
                ErrorCodes.SPENDING_LIMIT_EXCEEDED,
                "Spending limit exceeded",
                spending_check.get("message")
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
        user_id: str
    ) -> Dict[str, Any]:
        """
        Handle job completion billing based on job's billing type
        
        Simplified and modular implementation
        """
        # Get job details
        job = await self._get_job_safely(job_id, job_type)
        if not job:
            return self.response_util.create_error_response(
                ErrorCodes.JOB_NOT_FOUND, "Job not found"
            )
        
        billing_type = job.get("billing_type", BillingType.CREDIT_PACK)
        credits_required = job.get("credits_required", 0)
        
        # Handle completion based on billing type
        if billing_type == BillingType.PAY_AS_YOU_GO:
            return await self._complete_payg_job(job_id, job_type, user_id, credits_required)
        else:
            return await self._complete_credit_pack_job(job_id, job_type, credits_required)
    
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
        
        billing_type = job.get("billing_type", BillingType.CREDIT_PACK)
        
        if billing_type == BillingType.PAY_AS_YOU_GO:
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
            collection_name = self.db_util.get_collection_name(job_type)
            return await db[collection_name].find_one({"job_id": job_id})
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None
    
    async def _validate_spending_limits(self, user_id: str, estimated_cost: float) -> Dict[str, Any]:
        """Validate spending limits with proper error handling"""
        try:
            from app.services.stripe_service import stripe_service
            
            spending_allowed = await stripe_service.check_spending_limit(user_id, estimated_cost)
            
            return {
                "allowed": spending_allowed,
                "message": "Spending limit exceeded" if not spending_allowed else "OK"
            }
        except Exception as e:
            logger.warning(f"Spending limit check failed for user {user_id}: {e}")
            # Allow by default to avoid blocking users
            return {"allowed": True, "message": "Spending limit check skipped"}
    
    async def _update_user_usage(self, user_id: str, credits_used: float) -> None:
        """Simple: Add credits to user's total usage"""
        try:
            from app.config.database import users_collection
            from datetime import datetime, timezone
            from bson import ObjectId
            
            # Simple increment of total usage
            await users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {
                    "$inc": {"total_usage": credits_used},
                    "$set": {"usage_updated_at": datetime.now(timezone.utc)}
                }
            )
            
            logger.debug(f"Added {credits_used} credits to user {user_id} total usage")
            
        except Exception as e:
            logger.error(f"Failed to update user usage for {user_id}: {e}")
            # Don't raise - this shouldn't break job completion
    
    async def _validate_payg_payment_method(self, user_id: str) -> None:
        """Validate that PAYG user has a valid payment method (optimized - no Stripe calls during job creation)"""
        try:
            from app.config.database import users_collection
            from bson import ObjectId
            
            # Get user details
            user = await users_collection.find_one({"_id": ObjectId(user_id)})
            if not user:
                raise ValueError("User not found")
            
            # Check if user has active PAYG subscription
            subscription = user.get("subscription", {})
            if subscription.get("type") == "pay as you go" and subscription.get("status") == "active":
                # Check if user has Stripe customer ID (indicates payment setup completed)
                if not subscription.get("stripeCustomerId") or not subscription.get("stripeSubscriptionId"):
                    raise ValueError("Pay-as-you-go users must complete payment setup before using services. Please add a card first.")
                
                # Quick check for outstanding bills (optional - can be done later)
                outstanding_amount = await self._get_outstanding_billing_amount(user_id)
                if outstanding_amount > 0:
                    raise ValueError(f"Please clear your outstanding bill of ${outstanding_amount:.2f} before using more services.")
            else:
                raise ValueError("Pay-as-you-go subscription is required for this service.")
                
        except Exception as e:
            logger.error(f"Payment method validation failed for user {user_id}: {e}")
            raise ValueError(str(e))
    
    async def _get_outstanding_billing_amount(self, user_id: str) -> float:
        """Get user's outstanding billing amount from unpaid PAYG usage"""
        try:
            # Simple approach: Check if user has PAYG jobs that were completed but not reported to Stripe
            # This prevents abuse while keeping it simple
            
            total_outstanding = 0.0
            
            # Check recent PAYG jobs (last 7 days - weekly billing cycle) that might need billing
            from datetime import timedelta
            from app.config.credit_constants import BillingPeriod
            recent_date = datetime.now(timezone.utc) - timedelta(days=BillingPeriod.PAYG_CYCLE_DAYS)
            
            for collection_name in ["dub_jobs", "separation_jobs"]:
                collection = db.get_collection(collection_name)
                
                # Find completed PAYG jobs
                cursor = collection.find({
                    "userId": user_id,
                    "billingType": BillingType.PAY_AS_YOU_GO,
                    "status": {"$in": ["completed", "success"]},
                    "completedAt": {"$gte": recent_date}
                })
                
                async for job in cursor:
                    credits_used = job.get("creditsUsed", 0)
                    if credits_used > 0:
                        cost = credits_used * CreditRates.COST_PER_CREDIT_USD
                        total_outstanding += cost
                        
            return round(total_outstanding, 2)
            
        except Exception as e:
            logger.warning(f"Failed to check outstanding billing for user {user_id}: {e}")
            return 0.0
    
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
        
        collection_name = self.db_util.get_collection_name(job_type)
        
        # Create job data with virtual reservation
        final_job_data = self.db_util.create_job_data(
            job_id=job_data["job_id"], user_id=user_id, credits_required=required_credits, 
            billing_type=BillingType.PAY_AS_YOU_GO, **{k: v for k, v in job_data.items() if k not in ["job_id", "user_id"]}
        )
        
        # Insert job
        async with await client.start_session() as session:
            async with session.start_transaction():
                try:
                    await db[collection_name].insert_one(final_job_data, session=session)
                    await session.commit_transaction()
                    
                    # Update spending tracking
                    await self._update_spending_tracking(user_id, estimated_cost)
                    
                    logger.info(f"Pay-as-you-go reservation: {required_credits} credits for job {job_data['job_id']}")
                    
                    return self.response_util.create_credit_operation_response(
                        True, "Pay-as-you-go job reserved successfully",
                        credits_reserved=required_credits,
                        billing_type=BillingType.PAY_AS_YOU_GO,
                        virtual_reservation=True
                    )
                    
                except Exception as e:
                    await session.abort_transaction()
                    raise e
    
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
        
        collection_name = self.db_util.get_collection_name(job_type)
        
        # Atomic operation: deduct credits and create job
        async with await client.start_session() as session:
            async with session.start_transaction():
                try:
                    # Deduct credits from user balance
                    user_result = await users_collection.find_one_and_update(
                        {
                            "_id": ObjectId(user_id),
                            "credits": {"$gte": required_credits}
                        },
                        {
                            "$inc": {"credits": -required_credits},
                            "$set": {"updated_at": datetime.now(timezone.utc)}
                        },
                        session=session,
                        return_document=True
                    )
                    
                    if not user_result:
                        raise ValueError("Insufficient credits (race condition)")
                    
                    # Create job data with actual deduction
                    final_job_data = self.db_util.create_job_data(
                        job_id=job_data["job_id"], user_id=user_id, credits_required=required_credits,
                        billing_type=BillingType.CREDIT_PACK, **{k: v for k, v in job_data.items() if k not in ["job_id", "user_id"]}
                    )
                    
                    await db[collection_name].insert_one(final_job_data, session=session)
                    await session.commit_transaction()
                    
                    # Update spending tracking
                    await self._update_spending_tracking(user_id, estimated_cost)
                    
                    remaining_credits = user_result.get("credits", 0)
                    logger.info(f"Credit pack reservation: {required_credits} credits deducted for job {job_data['job_id']} (remaining: {remaining_credits})")
                    
                    return self.response_util.create_credit_operation_response(
                        True, "Credit pack job reserved successfully",
                        credits_reserved=required_credits,
                        remaining_credits=remaining_credits,
                        billing_type=BillingType.CREDIT_PACK
                    )
                    
                except Exception as e:
                    await session.abort_transaction()
                    raise e
    
    async def _complete_payg_job(
        self, job_id: str, job_type: JobType, user_id: str, credits_used: float
    ) -> Dict[str, Any]:
        """Complete pay-as-you-go job by tracking usage locally (no immediate Stripe sync)"""
        try:
            # Track usage in database only - no immediate Stripe sync
            collection_name = self.db_util.get_collection_name(job_type)
            credit_info = self.db_util.create_credit_info(
                job_type, credits_used, stripe_usage_reported=False  # Will sync later
            )
            
            await self._update_job_details(job_id, job_type, credit_info)
            
            # Update user's total usage for billing
            await self._update_user_usage(user_id, credits_used)
            
            logger.info(f"Pay-as-you-go completion: Added {credits_used} credits to user usage for job {job_id}")
            
            return self.response_util.create_credit_operation_response(
                True, "Pay-as-you-go job completed successfully",
                credits_used=credits_used,
                billing_type=BillingType.PAY_AS_YOU_GO
            )
            
        except Exception as e:
            logger.error(f"Pay-as-you-go completion failed for job {job_id}: {e}")
            raise e
    
    async def _complete_credit_pack_job(
        self, job_id: str, job_type: JobType, credits_deducted: float
    ) -> Dict[str, Any]:
        """Complete credit pack job (credits already deducted)"""
        try:
            # Update job record to confirm completion
            credit_info = self.db_util.create_credit_info(
                job_type, credits_deducted, deducted_amount=credits_deducted
            )
            
            await self._update_job_details(job_id, job_type, credit_info)
            
            logger.info(f"Credit pack completion: Confirmed {credits_deducted} credits for job {job_id}")
            
            return self.response_util.create_credit_operation_response(
                True, "Credit pack job completed successfully",
                credits_confirmed=credits_deducted,
                billing_type=BillingType.CREDIT_PACK
            )
            
        except Exception as e:
            logger.error(f"Credit pack completion failed for job {job_id}: {e}")
            raise e
    
    async def _refund_payg_job(self, job_id: str, job_type: JobType, reason: str) -> Dict[str, Any]:
        """Refund pay-as-you-go job (virtual cancellation)"""
        collection_name = self.db_util.get_collection_name(job_type)
        
        await db[collection_name].update_one(
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
        
        return self.response_util.create_credit_operation_response(
            True, "Pay-as-you-go job cancelled successfully",
            credits_refunded=0, reason=reason
        )
    
    async def _refund_credit_pack_job(self, job_id: str, job_type: JobType, reason: str) -> Dict[str, Any]:
        """Refund credit pack job (actual credit restoration)"""
        collection_name = self.db_util.get_collection_name(job_type)
        
        # Get job details
        job = await db[collection_name].find_one({"job_id": job_id, "credits_reserved": True})
        
        if not job:
            return self.response_util.create_error_response(
                ErrorCodes.JOB_NOT_FOUND, "No reserved credits found"
            )
        
        credits_to_refund = job.get("credits_required", 0)
        user_id = job.get("user_id")
        
        # Atomic refund operation
        async with await client.start_session() as session:
            async with session.start_transaction():
                try:
                    # Only refund if credits were actually deducted
                    if job.get("credits_deducted", False):
                        await users_collection.update_one(
                        {"_id": ObjectId(user_id)},
                        {
                            "$inc": {"credits": credits_to_refund},
                            "$set": {"updated_at": datetime.now(timezone.utc)}
                        },
                        session=session
                    )
                    
                    # Mark job as refunded
                    await db[collection_name].update_one(
                        {"job_id": job_id},
                        {
                            "$set": {
                                "credits_refunded": True,
                                "credits_refunded_at": datetime.now(timezone.utc),
                                "refund_reason": reason
                            },
                            "$unset": {"credits_reserved": ""}
                        },
                        session=session
                    )
                    
                    await session.commit_transaction()
                    
                    logger.info(f"Refunded {credits_to_refund} credits for credit pack job {job_id}")
                    
                    return self.response_util.create_credit_operation_response(
                        True, "Credits refunded successfully",
                        credits_refunded=credits_to_refund, reason=reason
                    )
                    
                except Exception as e:
                    await session.abort_transaction()
                    raise e
    
    async def _update_spending_tracking(self, user_id: str, amount: float):
        """Update spending tracking for spending limits"""
        try:
            from app.services.stripe_service import stripe_service
            await stripe_service.update_current_spending(user_id, amount)
        except Exception as e:
            logger.warning(f"Failed to update spending tracking for user {user_id}: {e}")
    
    async def _update_job_details(self, job_id: str, job_type: JobType, credit_info: Dict[str, Any]):
        """Update job details with credit information"""
        try:
            from app.services.dub_job_service import dub_job_service
            from app.services.separation_job_service import separation_job_service
            
            job_service = dub_job_service if job_type == JobType.DUB else separation_job_service
            job = await job_service.get_job(job_id)
            
            if job and job.details:
                updated_details = job.details.copy()
                updated_details.update(credit_info)
                await job_service.update_details(job_id, updated_details)
        except Exception as e:
            logger.warning(f"Failed to update job details for {job_id}: {e}")
    


# Create a singleton instance
credit_service = CreditService()
