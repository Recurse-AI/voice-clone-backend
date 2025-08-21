import logging
from typing import Optional, Dict, Any
from enum import Enum
from app.config.database import users_collection, client, db
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
    async def atomic_reserve_and_create_job(
        user_id: str, 
        job_data: Dict[str, Any],
        job_type: JobType, 
        duration_seconds: float,
        collection_name: str
    ) -> Dict[str, Any]:
        """Atomically reserve credits and create job (transaction-safe)"""
        required_credits = CreditCalculator.calculate_credits(job_type, duration_seconds)
        job_id = job_data["job_id"]
        
        async with await client.start_session() as session:
            async with session.start_transaction():
                try:
                    # Check and reserve credits atomically
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
                        raise ValueError("Insufficient credits")
                    
                    # Create job with credit info
                    job_data.update({
                        "credits_required": required_credits,
                        "credits_reserved": True,
                        "created_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc)
                    })
                    
                    await db[collection_name].insert_one(job_data, session=session)
                    await session.commit_transaction()
                    
                    logger.info(f"Reserved {required_credits} credits for {job_type.value} job {job_id}")
                    return {
                        "success": True,
                        "credits_reserved": required_credits,
                        "remaining_credits": user_result.get("credits", 0) - required_credits
                    }
                    
                except Exception as e:
                    await session.abort_transaction()
                    logger.error(f"Credit reservation failed for job {job_id}: {e}")
                    return {"success": False, "error": str(e)}
    
    @staticmethod
    async def confirm_credit_usage(job_id: str, job_type: JobType) -> Dict[str, Any]:
        """Mark credits as confirmed (no refund after this)"""
        try:
            collection_name = "dub_jobs" if job_type == JobType.DUB else "separation_jobs"
            result = await db[collection_name].update_one(
                {"job_id": job_id, "credits_reserved": True},
                {
                    "$set": {
                        "credits_confirmed": True,
                        "credits_confirmed_at": datetime.now(timezone.utc)
                    },
                    "$unset": {"credits_reserved": ""}
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Confirmed credits for {job_type.value} job {job_id}")
                return {"success": True}
            else:
                return {"success": False, "error": "Job not found or credits already confirmed"}
                
        except Exception as e:
            logger.error(f"Credit confirmation failed for job {job_id}: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def refund_reserved_credits(job_id: str, job_type: JobType, reason: str = "job_failed") -> Dict[str, Any]:
        """Refund reserved credits (only if not confirmed)"""
        async with await client.start_session() as session:
            async with session.start_transaction():
                try:
                    collection_name = "dub_jobs" if job_type == JobType.DUB else "separation_jobs"
                    
                    # Get job with reserved credits
                    job = await db[collection_name].find_one(
                        {"job_id": job_id, "credits_reserved": True},
                        session=session
                    )
                    
                    if not job:
                        return {"success": False, "error": "No reserved credits found"}
                    
                    credits_to_refund = job.get("credits_required", 0)
                    user_id = job.get("user_id")
                    
                    # Refund credits to user
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
                    
                    logger.info(f"Refunded {credits_to_refund} credits for {job_type.value} job {job_id} (reason: {reason})")
                    return {"success": True, "credits_refunded": credits_to_refund}
                    
                except Exception as e:
                    await session.abort_transaction()
                    logger.error(f"Credit refund failed for job {job_id}: {e}")
                    return {"success": False, "error": str(e)}
    
    @staticmethod
    async def check_credits_simple(user_id: str, job_type: JobType, duration_seconds: float) -> Dict[str, Any]:
        """Simple credit check (for backward compatibility)"""
        required_credits = CreditCalculator.calculate_credits(job_type, duration_seconds)
        try:
            user = await users_collection.find_one({"_id": ObjectId(user_id)}, {"credits": 1})
            if not user:
                return {"sufficient": False, "error": "User not found"}
            
            available_credits = user.get("credits", 0)
            return {
                "sufficient": available_credits >= required_credits,
                "required": required_credits,
                "available": available_credits
            }
        except Exception as e:
            return {"sufficient": False, "error": str(e)}
    
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

    @staticmethod
    def run_async_safely(async_func, timeout: int = 30):
        """Safely run async function in any thread context"""
        import asyncio
        import concurrent.futures
        
        try:
            try:
                # Try to get existing loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, use thread executor
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, async_func)
                        return future.result(timeout=timeout)
                else:
                    return loop.run_until_complete(async_func)
            except RuntimeError:
                # No event loop in current thread, create new one
                return asyncio.run(async_func)
        except Exception as e:
            logger.error(f"Async function execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def confirm_credit_usage_sync(job_id: str, job_type: JobType):
        """Sync wrapper for confirm_credit_usage"""
        result = CreditService.run_async_safely(
            CreditService.confirm_credit_usage(job_id, job_type)
        )
        if not result.get("success"):
            logger.warning(f"Credit confirmation failed for {job_type.value} job {job_id}: {result.get('error')}")
        else:
            logger.info(f"Credit confirmed for {job_type.value} job {job_id}")
        return result
    
    @staticmethod
    def refund_reserved_credits_sync(job_id: str, job_type: JobType, reason: str = "job_failed"):
        """Sync wrapper for refund_reserved_credits"""
        result = CreditService.run_async_safely(
            CreditService.refund_reserved_credits(job_id, job_type, reason)
        )
        if result.get("success"):
            credits_refunded = result.get("credits_refunded", 0)
            logger.info(f"Refunded {credits_refunded} credits for {job_type.value} job {job_id} (reason: {reason})")
        else:
            logger.warning(f"Credit refund failed for {job_type.value} job {job_id}: {result.get('error')}")
        return result

# Global instance
credit_service = CreditService()