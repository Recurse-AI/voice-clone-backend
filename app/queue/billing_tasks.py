"""
Billing tasks for RQ queue processing
Handles async billing operations in main process
"""
import logging
import asyncio
from typing import Dict, Any
from app.config.credit_constants import JobType

logger = logging.getLogger(__name__)


async def _complete_billing_async(job_id: str, job_type: str, user_id: str, billing_percentage: float) -> Dict[str, Any]:
    """Complete billing operation async"""
    # Lazy import to ensure DB client binds to the active event loop
    from app.services.credit_service import credit_service
    job_type_map = {"dub": JobType.DUB, "separation": JobType.SEPARATION, "clip": JobType.CLIP}
    job_type_enum = job_type_map.get(job_type.lower(), JobType.DUB)
    
    result = await credit_service.complete_job_billing(job_id, job_type_enum, user_id, billing_percentage)
    
    if result.get("success"):
        logger.info(f"✅ Credit billing completed for {job_type} job {job_id} ({billing_percentage*100}%)")
    else:
        logger.warning(f"⚠️ Credit billing failed for {job_type} job {job_id}: {result.get('message')}")
    
    return result


async def _refund_credits_async(job_id: str, job_type: str, reason: str) -> Dict[str, Any]:
    """Refund credits operation async"""
    # Lazy import to ensure DB client binds to the active event loop
    from app.services.credit_service import credit_service
    job_type_map = {"dub": JobType.DUB, "separation": JobType.SEPARATION, "clip": JobType.CLIP}
    job_type_enum = job_type_map.get(job_type.lower(), JobType.DUB)
    
    result = await credit_service.refund_job_credits(job_id, job_type_enum, reason)
    
    if result.get("success"):
        logger.info(f"💰 Credits refunded for {job_type} job {job_id} (reason: {reason})")
    else:
        logger.warning(f"⚠️ Credit refund failed for {job_type} job {job_id}: {result.get('message')}")
    
    return result


def process_billing_task(operation: str, kwargs: Dict[str, Any]) -> bool:
    """
    Process billing task with a clean event loop lifecycle.
    Uses asyncio.run to avoid reusing a closed loop and to bind Motor client correctly.
    """
    try:
        if operation == 'complete_billing':
            result = asyncio.run(_complete_billing_async(
                job_id=kwargs['job_id'],
                job_type=kwargs['job_type'],
                user_id=kwargs['user_id'],
                billing_percentage=kwargs.get('billing_percentage', 1.0)
            ))
        elif operation == 'refund_credits':
            result = asyncio.run(_refund_credits_async(
                job_id=kwargs['job_id'],
                job_type=kwargs['job_type'],
                reason=kwargs.get('reason', 'job_failed')
            ))
        else:
            raise ValueError(f"Unknown billing operation: {operation}")

        success = result.get("success", False)
        if not success:
            # Raise to signal failure to RQ (enables visibility/retry policies)
            raise RuntimeError(result.get("message", "Billing operation failed"))
        return True

    except Exception as e:
        logger.error(f"Billing task {operation} failed: {e}")
        # Re-raise so RQ marks the job as failed instead of silently succeeding
        raise
