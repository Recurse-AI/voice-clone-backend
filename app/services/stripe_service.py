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
from app.utils.decorators import handle_stripe_errors, log_execution_time, validate_user_access
from app.utils.credit_utils import CreditCalculatorUtil, ResponseUtil, SpendingLimitUtil
from bson import ObjectId
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

class StripeService:
    """Clean Stripe service with improved modularity and consistency"""
    
    def __init__(self):
        self.calculator = CreditCalculatorUtil()
        self.response_util = ResponseUtil()
        self.spending_util = SpendingLimitUtil()
    
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
        """Create setup intent for adding payment methods"""
        customer_id = await self.get_or_create_customer(user, user_id)
        
        setup_intent = stripe.SetupIntent.create(
            customer=customer_id,
            usage=StripeConfig.SETUP_INTENT_USAGE,
            payment_method_types=['card']  # Only allow card to avoid Link issues
        )
        
        return {
            "url": f"{StripeConfig.CHECKOUT_BASE_URL}/setup/{setup_intent.client_secret}",
            "client_secret": setup_intent.client_secret
        }
    
    @handle_stripe_errors
    @log_execution_time  
    async def get_payment_methods(self, user, user_id: str) -> List[Dict[str, Any]]:
        """Get user's saved payment methods and sync with database"""
        customer_id = await self.get_or_create_customer(user, user_id)
        
        # First try to get existing payment methods
        payment_methods = stripe.PaymentMethod.list(
            customer=customer_id,
            type='card'
        )
        
        payment_methods_data = [
            self._format_payment_method(pm) 
            for pm in payment_methods.data
        ]
        
        # Auto-sync payment method status with database (fallback for webhook issues)
        await self._sync_payment_method_status(user_id, len(payment_methods_data) > 0)
        
        return payment_methods_data
    
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
    
    async def _sync_payment_method_status(self, user_id: str, has_payment_methods: bool):
        """Sync payment method status with database (fallback for webhook issues)"""
        try:
            # Get current user data
            user_data = await users_collection.find_one({"_id": ObjectId(user_id)})
            if not user_data:
                return
            
            current_status = user_data.get("hasPaymentMethod", False)
            
            # Only update if status has changed
            if current_status != has_payment_methods:
                update_data = {
                    "hasPaymentMethod": has_payment_methods,
                    "updatedAt": datetime.now(timezone.utc)
                }
                
                # Set payment method added timestamp if this is the first time
                if has_payment_methods and not user_data.get("paymentMethodAddedAt"):
                    update_data["paymentMethodAddedAt"] = datetime.now(timezone.utc)
                
                await users_collection.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$set": update_data}
                )
                
                logger.info(f"Synced payment method status for user {user_id}: {has_payment_methods}")
                
        except Exception as e:
            logger.error(f"Failed to sync payment method status for user {user_id}: {e}")
    
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
    
    # Spending Limits Management
    
    @log_execution_time
    async def get_spending_limit(self, user_id: str) -> Dict[str, Any]:
        """Get user's spending limit with safety defaults for PAYG users"""
        try:
            user = await users_collection.find_one({"_id": ObjectId(user_id)})
            if not user:
                return {"amount": 0, "period": "monthly", "currentSpent": 0}
            
            spending_limit = user.get("spendingLimit", {})
            
            # Apply default safety limit for PAYG users if no limit is set
            from app.config.credit_constants import DefaultLimits
            subscription = user.get("subscription", {})
            is_payg_user = subscription.get("type") == "pay as you go"
            
            default_amount = DefaultLimits.PAYG_WEEKLY_LIMIT_USD if is_payg_user else 0
            default_period = "weekly" if is_payg_user else "monthly"
            
            # Calculate total current spending: Stripe billed + local usage  
            from app.config.credit_constants import CreditRates
            stripe_spent = spending_limit.get("currentSpent", 0)
            local_usage = user.get("total_usage", 0.0)
            local_usage_usd = local_usage * CreditRates.COST_PER_CREDIT_USD
            total_current_spent = stripe_spent + local_usage_usd
            
            return {
                "amount": spending_limit.get("amount", default_amount),
                "period": spending_limit.get("period", default_period), 
                "currentSpent": total_current_spent
            }
            
        except Exception as e:
            logger.error(f"Failed to get spending limit for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve spending limit")
    
    @log_execution_time
    async def update_spending_limit(self, user_id: str, amount: float, period: str) -> bool:
        """Update user's spending limit"""
        # Validate period
        from app.utils.credit_utils import UserValidationUtil
        if not UserValidationUtil.validate_spending_period(period):
            raise HTTPException(status_code=400, detail=f"Invalid period: {period}")
        
        try:
            # Create spending limit data
            spending_data = self.spending_util.create_spending_limit_data(amount, period)
            
            # Update database
            update_data = {
                f"spendingLimit.{key}": value 
                for key, value in spending_data.items()
            }
            update_data["updatedAt"] = datetime.now(timezone.utc)
            
            await users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_data}
            )
            
            logger.info(f"Updated spending limit for user {user_id}: {amount} {period}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update spending limit for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to update spending limit")
    
    @log_execution_time
    async def check_spending_limit(self, user_id: str, amount: float) -> bool:
        """Check if user can spend the amount without exceeding limit"""
        try:
            user = await users_collection.find_one({"_id": ObjectId(user_id)})
            if not user:
                return True  # Allow if user not found
            
            # Apply safety limits for PAYG users
            from app.config.credit_constants import DefaultLimits
            subscription = user.get("subscription", {})
            is_payg_user = subscription.get("type") == "pay as you go"
            
            spending_limit = user.get("spendingLimit")
            
            # Calculate total current spending: Stripe billed + local usage
            from app.config.credit_constants import CreditRates
            stripe_spent = spending_limit.get("currentSpent", 0) if spending_limit else 0
            local_usage = user.get("total_usage", 0.0)
            local_usage_usd = local_usage * CreditRates.COST_PER_CREDIT_USD
            total_current_spent = stripe_spent + local_usage_usd
            
            # For PAYG users, apply default safety limit if none set
            if is_payg_user and (not spending_limit or not spending_limit.get("amount")):
                limit_amount = DefaultLimits.PAYG_WEEKLY_LIMIT_USD
                return self.spending_util.is_within_limit(total_current_spent, limit_amount, amount)
            
            # For non-PAYG users or users with custom limits
            if not spending_limit or not spending_limit.get("amount"):
                return True  # No limit set for credit pack users
            
            limit_amount = spending_limit.get("amount", 0)
            return self.spending_util.is_within_limit(total_current_spent, limit_amount, amount)
            
        except Exception as e:
            logger.error(f"Failed to check spending limit for user {user_id}: {e}")
            return True  # Allow on error to avoid blocking users
    
    @log_execution_time
    async def update_current_spending(self, user_id: str, amount: float):
        """Update current spending amount"""
        try:
            await users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {
                    "$inc": {"spendingLimit.currentSpent": amount},
                    "$set": {"updatedAt": datetime.now(timezone.utc)}
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to update current spending for user {user_id}: {e}")
    
    # Pay-as-you-go Usage Reporting
    
    @handle_stripe_errors
    @log_execution_time
    async def report_usage_to_stripe(self, subscription_item_id: str, usage_quantity: int):
        """Report usage to Stripe for metered billing"""
        usage_record = stripe.SubscriptionItem.create_usage_record(
            subscription_item_id,
            quantity=usage_quantity,
            timestamp=int(datetime.now().timestamp()),
            action='increment'
        )
        
        logger.info(f"Reported {usage_quantity} credits to Stripe item {subscription_item_id}")
        return usage_record
    

    async def _get_user_subscription(self, user_id: str) -> Dict[str, Any]:
        """Get user subscription details for usage reporting"""
        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return {"is_payg": False}
        
        subscription = user.get("subscription", {})
        subscription_id = subscription.get("stripeSubscriptionId")
        
        if (not subscription_id or 
            subscription.get("type") != "pay as you go" or 
            subscription.get("status") != "active"):
            return {"is_payg": False}
        
        # Get Stripe subscription details
        try:
            stripe_subscription = stripe.Subscription.retrieve(subscription_id, expand=['items'])
            # Access subscription items correctly
            if hasattr(stripe_subscription, 'items') and stripe_subscription.items:
                # Try different ways to access items data
                if hasattr(stripe_subscription.items, 'data'):
                    subscription_items = stripe_subscription.items.data
                elif hasattr(stripe_subscription.items, '__iter__'):
                    subscription_items = list(stripe_subscription.items)
                else:
                    subscription_items = []
                    
                if subscription_items:
                    return {
                        "is_payg": True,
                        "subscription_item_id": subscription_items[0].id
                    }
            
            return {"is_payg": False}
                
        except stripe.error.StripeError as e:
            logger.error(f"Failed to retrieve Stripe subscription {subscription_id}: {e}")
            return {"is_payg": False}
    
    async def process_weekly_billing(self, user_id: str) -> Dict[str, Any]:
        """Process weekly billing for PAYG users - bill accumulated usage"""
        try:
            from app.config.database import users_collection
            from app.config.credit_constants import CreditRates
            from bson import ObjectId
            
            # Get user's total accumulated usage
            user = await users_collection.find_one(
                {"_id": ObjectId(user_id)}, 
                {"total_usage": 1, "subscription": 1}
            )
            
            if not user or user.get("subscription", {}).get("type") != "pay as you go":
                return {"success": False, "message": "User is not PAYG"}
            
            total_credits = user.get("total_usage", 0.0)
            if total_credits <= 0:
                return {"success": True, "message": "No usage to bill"}
            
            # Get subscription details for billing
            subscription_details = await self._get_user_subscription(user_id)
            if not subscription_details["is_payg"]:
                return {"success": False, "message": "No active PAYG subscription"}
            
            # Bill to Stripe (convert credits to usage units)
            subscription_item_id = subscription_details["subscription_item_id"]
            usage_units = int(total_credits * 100)  # Convert to billing units (e.g., cents)
            await self.report_usage_to_stripe(subscription_item_id, usage_units)
            
            # Reset user's usage counter and update currentSpent
            cost_usd = total_credits * CreditRates.COST_PER_CREDIT_USD
            await users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {
                    "$set": {"total_usage": 0.0},
                    "$inc": {"spendingLimit.currentSpent": cost_usd}
                }
            )
            
            logger.info(f"Weekly billing completed: User {user_id}, ${cost_usd} for {total_credits} credits")
            return {"success": True, "billed_amount": cost_usd, "credits": total_credits}
            
        except Exception as e:
            logger.error(f"Weekly billing failed for user {user_id}: {e}")
            return {"success": False, "message": str(e)}

# Create singleton instance
stripe_service = StripeService()
