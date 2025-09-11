"""
Clean Stripe Service
Modular implementation with improved error handling and reusability
"""

import logging
from typing import Dict, Any, List, Optional
import stripe
from fastapi import HTTPException

from app.config.database import users_collection
from app.config.settings import settings  
from app.config.credit_constants import StripeConfig, ErrorCodes
from app.schemas.user import TokenUser
from app.utils.decorators import handle_stripe_errors, log_execution_time
from bson import ObjectId
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

class StripeService:
    """Simple Stripe service for $10 threshold billing"""
    
    def __init__(self):
        pass
    
    # Customer Management
    
    @handle_stripe_errors
    @log_execution_time
    async def get_or_create_customer(self, user, user_id: str) -> str:
        """Get existing Stripe customer or create new one"""
        # Try to get existing customer ID - handle both TokenUser and dict
        if hasattr(user, "subscription"):
            customer_id = getattr(getattr(user, "subscription", None), "stripeCustomerId", None)
        else:
            customer_id = user.get("subscription", {}).get("stripeCustomerId")

        if customer_id:
            # Verify customer exists in Stripe
            if await self._verify_stripe_customer(customer_id):
                return customer_id
        
        # Create new customer
        return await self._create_new_customer(user, user_id)
    
    async def _verify_stripe_customer(self, customer_id: str) -> bool:
        """Verify customer exists in Stripe"""
        try:
            stripe.Customer.retrieve(customer_id)
            return True
        except stripe.error.StripeError:
            logger.warning(f"Stripe customer {customer_id} not found")
            return False
    

    
    async def _create_new_customer(self, user, user_id: str) -> str:
        """Create new Stripe customer and save to database"""
        # Handle both TokenUser and dict objects
        email = user.email if hasattr(user, 'email') else user.get('email', '')
        name = getattr(user, "name", "") if hasattr(user, 'name') else user.get('name', '')
        
        stripe_customer = stripe.Customer.create(
            email=email,
            name=name,
            metadata={"userId": user_id}
        )
        
        # Save customer ID to database
        await users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "subscription.stripeCustomerId": stripe_customer.id,
                    "updatedAt": datetime.now(timezone.utc)
                }
            }
        )
        
        logger.info(f"Created Stripe customer {stripe_customer.id} for user {user_id} and saved to database")
        return stripe_customer.id
    
    # Payment Methods Management
    
    @handle_stripe_errors
    @log_execution_time
    async def create_setup_intent(self, user: TokenUser, user_id: str) -> Dict[str, str]:
        """Create checkout setup session for adding payment methods"""
        customer_id = await self.get_or_create_customer(user, user_id)
        
        # Create a Checkout Session in setup mode
        checkout_session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="setup",
            payment_method_types=['card'],
            success_url=f"{settings.FRONTEND_URL}/subscription/manage?setup_success=true",
            cancel_url=f"{settings.FRONTEND_URL}/subscription/manage?setup_error=cancelled"
        )
        
        return {
            "url": checkout_session.url,  # This gives the correct checkout.stripe.com URL
            "session_id": checkout_session.id
        }
    
    @handle_stripe_errors
    @log_execution_time  
    async def get_payment_methods(self, user, user_id: str) -> List[Dict[str, Any]]:
        """Get user's saved payment methods"""
        customer_id = await self.get_or_create_customer(user, user_id)
        payment_methods = stripe.PaymentMethod.list(customer=customer_id, type='card')
        
        return [self._format_payment_method(pm) for pm in payment_methods.data]
    
    def _format_payment_method(self, payment_method: stripe.PaymentMethod) -> Dict[str, Any]:
        """Format payment method for API response"""
        return {
            "id": payment_method.id,
            "brand": payment_method.card.brand,
            "last4": payment_method.card.last4,
            "expiryMonth": payment_method.card.exp_month,
            "expiryYear": payment_method.card.exp_year,
            "isDefault": False
        }
    

    
    @handle_stripe_errors
    @log_execution_time
    async def remove_payment_method(
        self, payment_method_id: str, user: TokenUser, user_id: str
    ) -> bool:
        """Remove a payment method with ownership verification"""
        customer_id = await self.get_or_create_customer(user, user_id)
        
        # Verify ownership
        payment_method = stripe.PaymentMethod.retrieve(payment_method_id)
        if payment_method.customer != customer_id:
            raise HTTPException(
                status_code=403, 
                detail="Payment method does not belong to user"
            )
        
        # Detach payment method
        stripe.PaymentMethod.detach(payment_method_id)
        
        logger.info(f"Removed payment method {payment_method_id} for user {user_id}")
        return True
    
    # Simple Threshold Management
    
    @log_execution_time
    async def get_threshold_status(self, user_id: str) -> Dict[str, Any]:
        """Get simple threshold billing status"""
        try:
            from app.config.database import users_collection
            from app.config.credit_constants import CreditRates
            from bson import ObjectId
            
            user = await users_collection.find_one({"_id": ObjectId(user_id)})
            if not user:
                return {"threshold": 10.0, "current_usage": 0.0, "current_cost": 0.0}
            
            subscription = user.get("subscription", {})
            threshold = subscription.get("thresholdAmount", 10.0)
            total_credits = user.get("total_usage", 0.0)
            current_cost = total_credits * CreditRates.COST_PER_CREDIT_USD
            
            return {
                "threshold": threshold,
                "current_usage": total_credits,
                "current_cost": current_cost,
                "ready_to_bill": current_cost >= threshold
            }
            
        except Exception as e:
            logger.error(f"Failed to get threshold status for user {user_id}: {e}")
            return {"threshold": 10.0, "current_usage": 0.0, "current_cost": 0.0}
    
    # Simple Usage Tracking and Auto-Billing
    
    @log_execution_time
    async def check_and_process_billing(self, user_id: str) -> Dict[str, Any]:
        """Check if user reached $10 threshold and auto-bill if needed"""
        try:
            from app.config.database import get_async_db
            from app.config.credit_constants import CreditRates
            from bson import ObjectId
            
            db = get_async_db()
            user = await db.users.find_one({"_id": ObjectId(user_id)})
            if not user:
                return {"success": False, "message": "User not found"}
            
            # Only process for PAYG users
            subscription = user.get("subscription", {})
            if subscription.get("type") != "pay as you go" or subscription.get("status") != "active":
                return {"success": True, "message": "Not a PAYG user"}
            
            # Calculate current usage cost
            total_credits = user.get("total_usage", 0.0)
            cost_usd = total_credits * CreditRates.COST_PER_CREDIT_USD
            threshold = subscription.get("thresholdAmount", 10.0)
            
            # Auto-bill if threshold reached
            if cost_usd >= threshold:
                return await self.process_threshold_billing(user_id, threshold)
            
            return {"success": True, "message": f"Usage ${cost_usd:.2f} below threshold ${threshold}"}
            
        except Exception as e:
            logger.error(f"Auto-billing check failed for user {user_id}: {e}")
            return {"success": False, "message": str(e)}
    
    @log_execution_time
    async def add_usage_and_check_billing(self, user_id: str, credits_used: float) -> Dict[str, Any]:
        """Check if billing needed after usage has been added"""
        try:
            # Don't update usage here - it's already updated by _update_user_usage
            # Just check if billing needed
            billing_result = await self.check_and_process_billing(user_id)
            
            return {
                "usage_added": credits_used,
                "billing_result": billing_result
            }
            
        except Exception as e:
            logger.error(f"Billing check failed for user {user_id}: {e}")
            return {"success": False, "message": str(e)}
    


    
    async def process_threshold_billing(self, user_id: str, threshold_usd: float = 10.0) -> Dict[str, Any]:
        """Charge the actual usage amount when threshold is reached"""
        try:
            from app.config.database import get_async_db
            from app.config.credit_constants import CreditRates
            from bson import ObjectId
            
            # Get user's current usage
            db = get_async_db()
            user = await db.users.find_one(
                {"_id": ObjectId(user_id)}, 
                {"total_usage": 1, "subscription": 1, "email": 1, "name": 1}
            )
            
            if not user:
                return {"success": False, "message": "User not found"}
            
            total_credits = user.get("total_usage", 0.0)
            cost_usd = total_credits * CreditRates.COST_PER_CREDIT_USD
            
            # Only bill if threshold is crossed
            if cost_usd < threshold_usd:
                return {"success": True, "message": f"Usage ${cost_usd:.2f} below threshold ${threshold_usd}"}
            
            # Create payment intent for $10 charge
            customer_id = await self.get_or_create_customer(user, user_id)
            
            # Get user's default payment method
            payment_methods = await self.get_payment_methods(user, user_id)
            if not payment_methods:
                return {"success": False, "message": "No payment method available"}
            actual_amount = cost_usd
            # Create payment intent
            payment_intent = stripe.PaymentIntent.create(
                amount=int(actual_amount * 100),  # Charge actual amount
                currency='usd',
                customer=customer_id,
                payment_method=payment_methods[0]["id"],
                confirm=True,
                return_url=f"{settings.FRONTEND_URL}/dashboard",
                receipt_email=user.get("email"),
                description=f"Usage payment for {total_credits:.2f} credits",
                metadata={
                    'userId': user_id,
                    'credits': str(total_credits),
                    'billing_type': 'threshold'
                }
            )
            
            if payment_intent.status == 'succeeded':
                # Reset usage proportionally
                credits_charged = actual_amount / CreditRates.COST_PER_CREDIT_USD
                new_usage = max(0, total_credits - credits_charged)
                
                await db.users.update_one(
                    {"_id": ObjectId(user_id)},
                    {
                        "$set": {"total_usage": 0.0}
                    }
                )
                
                return {
                    "success": True, 
                    "charged_amount": actual_amount, 
                    "credits": total_credits,
                    "payment_intent_id": payment_intent.id
                }
            
        except Exception as e:
            logger.error(f"Threshold billing failed for user {user_id}: {e}")
            return {"success": False, "message": str(e)}

# Create singleton instance
stripe_service = StripeService()
