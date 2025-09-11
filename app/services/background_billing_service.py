"""
Simple Background Billing Service
Bill users when they cross $10 threshold
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timezone
from app.config.database import get_async_db
from app.config.credit_constants import CreditRates
from app.services.stripe_service import stripe_service
from bson import ObjectId

logger = logging.getLogger(__name__)

class BackgroundBillingService:
    """Simple billing: $10 threshold, 5x max bill, 3 retries"""
    
    async def process_all_payg_users(self) -> Dict[str, Any]:
        """Process billing for all PAYG users"""
        try:
            payg_users = await self._get_active_payg_users()
            results = {"total_users": len(payg_users), "billed": 0, "blocked": 0, "errors": 0}
            
            for user in payg_users:
                try:
                    result = await self._process_user_billing(str(user["_id"]), user)
                    if "billed" in result: results["billed"] += 1
                    elif "blocked" in result: results["blocked"] += 1
                except Exception as e:
                    logger.error(f"Billing failed for {user['_id']}: {e}")
                    results["errors"] += 1
            
            return results
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_active_payg_users(self) -> List[Dict[str, Any]]:
        """Get PAYG users with usage >= $10"""
        db = get_async_db()
        cursor = db.users.find({
            "subscription.type": "pay as you go",
            "subscription.status": "active", 
            "subscription.stripeCustomerId": {"$exists": True},
            "total_usage": {"$gte": 250}  # 250 credits = $10
        })
        return await cursor.to_list(length=None)
    
    async def _process_user_billing(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple billing: $10 chunks, track consecutive failures"""
        try:
            total_usage = user_data.get("total_usage", 0.0)
            cost = total_usage * CreditRates.COST_PER_CREDIT_USD
            
            if cost < 10:
                return {"skipped": f"${cost:.2f} below $10"}
            
            # Check current failed attempts
            subscription = user_data.get("subscription", {})
            failed_attempts = subscription.get("billingFailedAttempts", 0)
            
            # If already blocked, skip
            if subscription.get("billingBlocked", False):
                return {"skipped": "User already blocked"}
            
            # Calculate billing amount in $10 chunks
            chunks = int(cost // 10)
            amount_to_bill = chunks * 10
            
            # Try billing once
            success = await self._try_billing(user_id, amount_to_bill)
            
            if success:
                # Success - reset failed attempts and unblock if needed
                await self._reset_failed_attempts(user_id)
                await self._unblock_user(user_id)
                
                # Update usage
                credits_billed = amount_to_bill / CreditRates.COST_PER_CREDIT_USD
                new_usage = max(0, total_usage - credits_billed)
                db = get_async_db()
                await db.users.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$set": {"total_usage": new_usage}}
                )
                return {"billed": amount_to_bill, "remaining": new_usage}
            else:
                # Failed - increment counter
                new_failed_attempts = failed_attempts + 1
                await self._increment_failed_attempts(user_id, new_failed_attempts)
                
                # Block after 3 consecutive failures
                if new_failed_attempts >= 3:
                    await self._block_user(user_id, f"3 consecutive billing failures")
                    return {"blocked": f"Blocked after {new_failed_attempts} failures"}
                else:
                    return {"failed": f"Attempt {new_failed_attempts}/3 failed"}
                
        except Exception as e:
            return {"error": str(e)}
    
    async def _try_billing(self, user_id: str, amount: float) -> bool:
        """Try billing once"""
        try:
            result = await stripe_service.process_threshold_billing(user_id, amount)
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Billing error for user {user_id}: {e}")
            return False
    
    async def _increment_failed_attempts(self, user_id: str, count: int):
        """Track consecutive billing failures"""
        db = get_async_db()
        await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"subscription.billingFailedAttempts": count}}
        )
    
    async def _reset_failed_attempts(self, user_id: str):
        """Reset failed attempts counter on success"""
        db = get_async_db()
        await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$unset": {"subscription.billingFailedAttempts": ""}}
        )
    
    async def _block_user(self, user_id: str, reason: str):
        """Block user"""
        db = get_async_db()
        await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "subscription.billingBlocked": True,
                "subscription.billingBlockReason": reason,
                "subscription.billingBlockedAt": datetime.now(timezone.utc)
            }}
        )
    
    async def _unblock_user(self, user_id: str):
        """Auto unblock user on successful payment"""
        db = get_async_db()
        await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$unset": {
                "subscription.billingBlocked": "",
                "subscription.billingBlockReason": "",
                "subscription.billingBlockedAt": ""
            }}
        )

# Singleton instance
background_billing_service = BackgroundBillingService()
