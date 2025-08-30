"""
Essential Stripe Routes for Frontend
Only the necessary API endpoints
"""

from fastapi import APIRouter, Depends, Security, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging

from app.schemas.user import TokenUser
from app.dependencies.auth import get_current_user
from app.services.user_service import get_user_id
from app.services.stripe_service import stripe_service
from app.services.pricing_service import PricingService
from app.services.transaction_service import TransactionService
from app.config.credit_constants import ErrorCodes
from app.utils.response_helper import success_response, error_response
from app.utils.decorators import log_execution_time
from pydantic import BaseModel, field_validator, model_validator
import stripe
from app.config.settings import settings
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Request models
class SpendingLimitRequest(BaseModel):
    amount: float
    period: str
    
    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > 10000:
            raise ValueError('Amount cannot exceed $10,000')
        return v
    
    @field_validator('period')
    @classmethod
    def validate_period(cls, v):
        from app.config.credit_constants import SpendingPeriod
        if v not in SpendingPeriod.get_all():
            raise ValueError(f'Period must be one of: {SpendingPeriod.get_all()}')
        return v

class CheckoutRequest(BaseModel):
    planId: str = None
    planName: str = None
    
    @model_validator(mode='before')
    @classmethod
    def validate_plan(cls, values):
        if isinstance(values, dict):
            plan_id = values.get('planId')
            plan_name = values.get('planName')
            
            # Get the plan value from either field
            plan_value = plan_id or plan_name
            
            if not plan_value:
                raise ValueError('Either planId or planName is required')
            
            # Validate against known credit pack plan names
            valid_plans = ['medium', 'special', 'limited']
            if plan_value not in valid_plans:
                raise ValueError(f'Plan must be one of: {valid_plans}')
            
            # Set both fields to ensure consistency
            values['planId'] = plan_value
            values['planName'] = plan_value
        
        return values

# Router setup
stripe_route = APIRouter(prefix="/api/stripe", tags=["stripe"])

# Initialize services
pricing_service = PricingService()
transaction_service = TransactionService()

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

# Utility functions
def handle_api_error(e: Exception, operation: str) -> JSONResponse:
    """Centralized error handling for API endpoints"""
    if isinstance(e, HTTPException):
        return error_response(str(e.detail), f"{operation} Error", e.status_code)
    
    logger.error(f"{operation} error: {str(e)}")
    return error_response(
        f"Failed to {operation.lower()}",
        "Internal Server Error",
        500
    )

async def get_authenticated_user(current_user: TokenUser) -> Dict[str, Any]:
    """Get authenticated user with error handling"""
    user = await get_user_id(current_user.id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ========== ESSENTIAL ENDPOINTS ==========

# 1. Pricing Plans
@stripe_route.get("/plans")
@log_execution_time
async def get_pricing_plans():
    """Get all active pricing plans for frontend"""
    try:
        plans_result = await pricing_service.get_all_plans()
        
        if isinstance(plans_result, dict) and plans_result.get("status_code") == 200:
            return JSONResponse(
                status_code=200,
                content=plans_result["content"]
            )
        else:
            return success_response({"plans": plans_result, "count": len(plans_result)})
        
    except Exception as e:
        return handle_api_error(e, "Get Pricing Plans")

# 2. Setup Intent (Add Payment Methods)
@stripe_route.post("/setup-intent")
@log_execution_time
async def create_setup_intent(current_user: TokenUser = Security(get_current_user)):
    """Create setup intent for adding payment methods"""
    try:
        user = await get_authenticated_user(current_user)
        result = await stripe_service.create_setup_intent(user, str(current_user.id))
        return success_response({"message": {"url": result["url"]}})
        
    except Exception as e:
        return handle_api_error(e, "Setup Intent Creation")

# 3. Payment Methods Management
@stripe_route.get("/payment-methods")
@log_execution_time
async def get_payment_methods(current_user: TokenUser = Security(get_current_user)):
    """Get user's saved payment methods (auto-syncs database status)"""
    try:
        user = await get_authenticated_user(current_user)
        # This call will automatically sync database status via _sync_payment_method_status
        payment_methods = await stripe_service.get_payment_methods(user, str(current_user.id))
        return success_response({"payment_methods": payment_methods})
        
    except Exception as e:
        return handle_api_error(e, "Get Payment Methods")

@stripe_route.delete("/payment-methods/{payment_method_id}")
@log_execution_time
async def remove_payment_method(
    payment_method_id: str,
    current_user: TokenUser = Security(get_current_user)
):  
    """Remove a payment method"""
    try:
        user = await get_authenticated_user(current_user)
        
        # Check for outstanding bills first
        from app.services.credit_service import credit_service
        outstanding_amount = await credit_service._get_outstanding_billing_amount(str(current_user.id))
        if outstanding_amount > 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot remove payment method with outstanding bill of ${outstanding_amount:.2f}. Please clear your bill first."
            )
        
        await stripe_service.remove_payment_method(payment_method_id, user, str(current_user.id))
        return success_response({"success": True})
        
    except Exception as e:
        return handle_api_error(e, "Remove Payment Method")

# 4. Spending Limits
@stripe_route.get("/spending-limit")
@log_execution_time
async def get_spending_limit(current_user: TokenUser = Security(get_current_user)):
    """Get user's spending limit"""
    try:
        spending_limit = await stripe_service.get_spending_limit(str(current_user.id))
        return success_response(spending_limit)
        
    except Exception as e:
        return handle_api_error(e, "Get Spending Limit")

@stripe_route.put("/spending-limit")
@log_execution_time
async def update_spending_limit(
    request: SpendingLimitRequest,
    current_user: TokenUser = Security(get_current_user)
):
    """Update user's spending limit"""
    try:
        await stripe_service.update_spending_limit(
            str(current_user.id), 
            request.amount, 
            request.period
        )
        return success_response({"success": True})
        
    except Exception as e:
        return handle_api_error(e, "Update Spending Limit")

# 5. Checkout Session
@stripe_route.post("/create-checkout-session")
@log_execution_time
async def create_checkout_session(
    request: CheckoutRequest,
    current_user: TokenUser = Security(get_current_user)
):  
    """Create checkout session for predefined plans"""
    try:
        user = await get_authenticated_user(current_user)
        
        # Validate planId
        if not request.planId or not isinstance(request.planId, str):
            raise HTTPException(status_code=400, detail="Invalid planId")
        
        plan = await pricing_service.get_plan_by_name(request.planId)
        if not plan:
            raise HTTPException(status_code=404, detail=f"Plan '{request.planId}' not found")
        
        # Validate credit pack exists
        if 'creditPack' not in plan or not plan['creditPack']:
            raise HTTPException(status_code=400, detail="Plan does not have credit pack information")
        
        customer_id = await stripe_service.get_or_create_customer(user, str(current_user.id))
        
        # Get plan details for dynamic pricing
        credit_pack = plan['creditPack']
        credits = credit_pack.get('credits', 0)
        price = credit_pack.get('discountedPrice', credit_pack.get('originalPrice', 0))
        
        # Validate price
        if price <= 0:
            raise HTTPException(status_code=400, detail="Invalid price for credit pack")
        
        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=['card'],
            locale='auto',
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': f"{credit_pack.get('name', 'Credit Pack')}",
                        'description': f"{credits} credits - {plan.get('description', 'Credit pack for AI services')}"
                    },
                    'unit_amount': int(price * 100),  # Convert to cents
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f"{settings.FRONTEND_URL}/payment/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{settings.FRONTEND_URL}/payment/cancel",
            metadata={
                'userId': str(current_user.id),
                'credit': str(credits),
                'planId': request.planId
            }
        )
        
        return success_response({"sessionId": session.id, "url": session.url})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Checkout session creation failed: {e}")
        return handle_api_error(e, "Create Checkout Session")

# 6. Payment Verification
@stripe_route.get("/verify-payment/{session_id}")
@log_execution_time
async def verify_payment(session_id: str, current_user: TokenUser = Security(get_current_user)):
    """Verify payment and get session details"""
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        
        if session.metadata.get('userId') != str(current_user.id):
            raise HTTPException(status_code=403, detail="Access denied")
        
        result = {
            "status": session.status,
            "payment_status": session.payment_status,
            "session_id": session_id,
            "mode": session.mode
        }
        
        if session.status == "complete":
            if session.mode == "setup":
                # This was a card add session - update database
                await update_user_payment_method_status(str(current_user.id), session.metadata.get('purpose'))
                
                result["card_added"] = True
                result["message"] = "Payment method added successfully"
                
                # Check if this was for PAYG - subscription is now automatically created
                if session.metadata.get('purpose') == 'payg_setup':
                    result["subscription"] = {
                        "type": "pay as you go",
                        "status": "active"
                    }
                    result["message"] = "Card added and Pay-as-You-Go subscription activated successfully."
            elif session.mode == "subscription":
                # Handle subscription completion
                subscription_result = await handle_subscription_completion(session, str(current_user.id))
                if subscription_result:
                    result["subscription"] = subscription_result
                    result["message"] = "Subscription activated successfully"
            else:
                # Regular payment (credit pack purchase)
                transaction_result = await transaction_service.handle_checkout_completed(session)
                if transaction_result:
                    result["transaction"] = transaction_result
        
        return success_response(result)
        
    except Exception as e:
        return handle_api_error(e, "Verify Payment")



# 7. Purchase History
@stripe_route.get("/purchase-history")
@log_execution_time
async def get_purchase_history(current_user: TokenUser = Security(get_current_user)):
    """Get user's purchase history"""
    try:
        history = await transaction_service.get_purchase_history(str(current_user.id))
        # TransactionService already returns formatted response, so use it directly
        return JSONResponse(status_code=history["status_code"], content=history["content"])
        
    except Exception as e:
        return handle_api_error(e, "Get Purchase History")

# 8. Customer Portal
@stripe_route.get("/customer-portal")
@log_execution_time
async def create_customer_portal(current_user: TokenUser = Security(get_current_user)):
    """Create Stripe customer portal session"""
    try:
        user = await get_authenticated_user(current_user)
        customer_id = await stripe_service.get_or_create_customer(user, str(current_user.id))
        
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{settings.FRONTEND_URL}/dashboard"
        )
        
        return success_response({"url": portal_session.url})
        
    except Exception as e:
        return handle_api_error(e, "Create Customer Portal")




# 9. Pay-as-you-go Subscription with Card Setup
@stripe_route.post("/create-checkout-asug")
@log_execution_time
async def create_payg_checkout_session(current_user: TokenUser = Security(get_current_user)):
    """Create pay-as-you-go subscription with payment method setup"""
    try:
        user = await get_authenticated_user(current_user)
        customer_id = await stripe_service.get_or_create_customer(user, str(current_user.id))
        
        # Check if user already has payment methods
        payment_methods = await stripe_service.get_payment_methods(user, str(current_user.id))
        
        # If no payment method, redirect to card add widget  
        if not payment_methods:
            # Create card add checkout session instead of setup intent
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=['card'],
                mode='setup',
                locale='auto',
                success_url=f"{settings.FRONTEND_URL}/subscription/success?session_id={{CHECKOUT_SESSION_ID}}&action=card_added",
                cancel_url=f"{settings.FRONTEND_URL}/subscription/cancel",
                metadata={
                    'userId': str(current_user.id),
                    'purpose': 'payg_setup',
                    'next_action': 'create_subscription'
                }
            )
            
            return success_response({
                "sessionId": session.id,
                "url": session.url,
                "requiresPaymentMethod": True,
                "message": "Please add a payment method first to subscribe to Pay-as-You-Go"
            })
        
        # Create subscription with existing payment method
        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=['card'],
            locale='auto',
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': 'Pay-as-You-Go Subscription',
                        'description': 'Weekly subscription with usage-based billing (billed every 7 days)'
                    },
                    'unit_amount': 0,  # No upfront cost
                    'recurring': {
                        'interval': 'week'  # Changed from 'month' to 'week' for 7-day billing cycles
                    }
                },
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f"{settings.FRONTEND_URL}/subscription/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{settings.FRONTEND_URL}/subscription/cancel",
            metadata={
                'userId': str(current_user.id),
                'subscriptionType': 'pay as you go'
            }
        )
        
        return success_response({"sessionId": session.id, "url": session.url})
        
    except Exception as e:
        return handle_api_error(e, "Create PAYG Checkout Session")

# 10. Cancel Subscription
@stripe_route.post("/cancel-subscription")
@log_execution_time
async def cancel_subscription(current_user: TokenUser = Security(get_current_user)):
    """Cancel user's pay-as-you-go subscription"""
    try:
        user = await get_authenticated_user(current_user)
        
        # Check for outstanding bills first
        from app.services.credit_service import credit_service
        outstanding_amount = await credit_service._get_outstanding_billing_amount(str(current_user.id))
        if outstanding_amount > 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot cancel subscription with outstanding bill of ${outstanding_amount:.2f}. Please clear your bill first."
            )
        
        # Get user's subscription ID from database
        from app.config.database import users_collection
        from bson import ObjectId
        
        user_doc = await users_collection.find_one({"_id": ObjectId(current_user.id)})
        if not user_doc or not user_doc.get("subscription", {}).get("stripeSubscriptionId"):
            raise HTTPException(status_code=404, detail="No active subscription found")
        
        subscription_id = user_doc["subscription"]["stripeSubscriptionId"]
        
        # Cancel subscription in Stripe
        subscription = stripe.Subscription.modify(
            subscription_id,
            cancel_at_period_end=True
        )
        
        # Update database
        await users_collection.update_one(
            {"_id": ObjectId(current_user.id)},
            {
                "$set": {
                    "subscription.status": "cancelled",
                    "subscription.cancelledAt": datetime.now(timezone.utc)
                }
            }
        )
        
        return success_response({
            "success": True,
            "message": "Subscription will be cancelled at the end of current period",
            "periodEnd": subscription.current_period_end
        })
        
    except Exception as e:
        return handle_api_error(e, "Cancel Subscription")

# 11. Billing History (reuse existing transaction service)
@stripe_route.get("/billing-history")
@log_execution_time
async def get_billing_history(current_user: TokenUser = Security(get_current_user)):
    """Get user's billing and purchase history"""
    try:
        # Use existing transaction service
        history = await transaction_service.get_purchase_history(str(current_user.id))
        return JSONResponse(status_code=history["status_code"], content=history["content"])
        
    except Exception as e:
        return handle_api_error(e, "Get Billing History")

# 12. Webhook (No auth required)
@stripe_route.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks"""
    try:
        body = await request.body()
        sig_header = request.headers.get('stripe-signature')
        
        event = stripe.Webhook.construct_event(
            body, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
        
        # Handle different event types
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            
            # Check mode and handle accordingly
            if session.get('mode') == 'subscription':
                await handle_subscription_checkout(session)
            elif session.get('mode') == 'setup':
                await handle_setup_completion(session)
            else:
                await transaction_service.handle_checkout_completed(session)
                
            logger.info(f"Processed checkout completion for session: {session['id']}")
            
        elif event['type'] == 'customer.subscription.created':
            subscription = event['data']['object']
            await handle_subscription_created(subscription)
            logger.info(f"Processed subscription created: {subscription['id']}")
        
        elif event['type'] == 'customer.subscription.updated':
            subscription = event['data']['object']
            await handle_subscription_updated(subscription)
            logger.info(f"Processed subscription updated: {subscription['id']}")
            
        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            await handle_subscription_deleted(subscription)
            logger.info(f"Processed subscription deleted: {subscription['id']}")
            
        elif event['type'] == 'setup_intent.succeeded':
            setup_intent = event['data']['object']
            await handle_setup_intent_succeeded(setup_intent)
            logger.info(f"Processed setup intent succeeded: {setup_intent['id']}")
        
        return success_response({"received": True})
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse(status_code=400, content={"error": "Webhook failed"})

# ========== HELPER FUNCTIONS ==========

async def handle_subscription_completion(session: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Handle subscription completion in verify_payment endpoint"""
    try:
        logger.info(f"Handling subscription completion for user {user_id}, session {session['id']}")
        
        # Get subscription from Stripe
        subscription_id = session.get('subscription')
        if not subscription_id:
            logger.error(f"No subscription ID in session {session['id']}")
            return None
            
        subscription = stripe.Subscription.retrieve(subscription_id)
        subscription_type = session['metadata'].get('subscriptionType', 'pay as you go')
        
        # Update user subscription in database
        from app.config.database import users_collection
        from bson import ObjectId
        
        # Safely handle current_period_end
        current_period_end = None
        if subscription.get('current_period_end'):
            current_period_end = datetime.fromtimestamp(subscription.current_period_end, tz=timezone.utc)
        
        update_result = await users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "subscription.type": subscription_type,
                    "subscription.status": subscription.status,
                    "subscription.stripeSubscriptionId": subscription_id,
                    "subscription.currentPeriodEnd": current_period_end,
                    "updatedAt": datetime.now(timezone.utc)
                }
            }
        )
        
        if update_result.modified_count > 0:
            logger.info(f"Successfully updated {subscription_type} subscription for user {user_id}")
            return {
                "subscription_id": subscription_id,
                "type": subscription_type,
                "status": subscription.status,
                "current_period_end": subscription.current_period_end
            }
        else:
            logger.warning(f"Failed to update subscription for user {user_id}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to handle subscription completion: {e}")
        return None

# ========== WEBHOOK HELPER FUNCTIONS ==========

async def handle_subscription_checkout(session: Dict[str, Any]):
    """Handle completed subscription checkout"""
    try:
        user_id = session['metadata'].get('userId')
        if not user_id:
            logger.error(f"No userId in subscription session metadata: {session['id']}")
            return
        
        # Get subscription from Stripe
        subscription_id = session.get('subscription')
        if subscription_id:
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            # Update user subscription in database
            from app.config.database import users_collection
            from bson import ObjectId
            
            # Safely handle current_period_end
            current_period_end = None
            if subscription.get('current_period_end'):
                current_period_end = datetime.fromtimestamp(subscription.current_period_end, tz=timezone.utc)
                
            await users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {
                    "$set": {
                        "subscription.type": "pay as you go",
                        "subscription.status": subscription.status,
                        "subscription.stripeSubscriptionId": subscription_id,
                        "subscription.currentPeriodEnd": current_period_end,
                        "updatedAt": datetime.now(timezone.utc)
                    }
                }
            )
            
            logger.info(f"Updated PAYG subscription for user {user_id}")
        
    except Exception as e:
        logger.error(f"Failed to handle subscription checkout: {e}")

async def handle_subscription_created(subscription: Dict[str, Any]):
    """Handle subscription creation"""
    try:
        customer_id = subscription.get('customer')
        if not customer_id:
            return
            
        # Find user by stripe customer ID
        from app.config.database import users_collection
        
        user = await users_collection.find_one({
            "subscription.stripeCustomerId": customer_id
        })
        
        if user:
            # Safely handle current_period_end
            current_period_end = None
            if subscription.get('current_period_end'):
                current_period_end = datetime.fromtimestamp(subscription['current_period_end'], tz=timezone.utc)
                
            await users_collection.update_one(
                {"_id": user["_id"]},
                {
                    "$set": {
                        "subscription.status": subscription['status'],
                        "subscription.stripeSubscriptionId": subscription['id'],
                        "subscription.currentPeriodEnd": current_period_end,
                        "updatedAt": datetime.now(timezone.utc)
                    }
                }
            )
            
            logger.info(f"Updated subscription created for user {user['_id']}")
            
    except Exception as e:
        logger.error(f"Failed to handle subscription created: {e}")

async def handle_subscription_updated(subscription: Dict[str, Any]):
    """Handle subscription status updates"""
    try:
        subscription_id = subscription.get('id')
        if not subscription_id:
            return
            
        # Find user by subscription ID
        from app.config.database import users_collection
        
        user = await users_collection.find_one({
            "subscription.stripeSubscriptionId": subscription_id
        })
        
        if user:
            # Safely handle current_period_end
            current_period_end = None
            if subscription.get('current_period_end'):
                current_period_end = datetime.fromtimestamp(subscription['current_period_end'], tz=timezone.utc)
                
            await users_collection.update_one(
                {"_id": user["_id"]},
                {
                    "$set": {
                        "subscription.status": subscription['status'],
                        "subscription.currentPeriodEnd": current_period_end,
                        "updatedAt": datetime.now(timezone.utc)
                    }
                }
            )
            
            logger.info(f"Updated subscription status to {subscription['status']} for user {user['_id']}")
            
    except Exception as e:
        logger.error(f"Failed to handle subscription updated: {e}")

async def handle_subscription_deleted(subscription: Dict[str, Any]):
    """Handle subscription cancellation"""
    try:
        subscription_id = subscription.get('id')
        if not subscription_id:
            return
            
        # Find user by subscription ID
        from app.config.database import users_collection
        
        user = await users_collection.find_one({
            "subscription.stripeSubscriptionId": subscription_id
        })
        
        if user:
            await users_collection.update_one(
                {"_id": user["_id"]},
                {
                    "$set": {
                        "subscription.status": "cancelled",
                        "subscription.cancelledAt": datetime.now(timezone.utc),
                        "updatedAt": datetime.now(timezone.utc)
                    }
                }
            )
            
            logger.info(f"Marked subscription as cancelled for user {user['_id']}")
            
    except Exception as e:
        logger.error(f"Failed to handle subscription deleted: {e}")

async def handle_setup_completion(session: Dict[str, Any]):
    """Handle completed setup session (card add)"""
    try:
        user_id = session['metadata'].get('userId')
        purpose = session['metadata'].get('purpose')
        
        if not user_id:
            logger.error(f"No userId in setup session metadata: {session['id']}")
            return

        # Get setup intent from session
        setup_intent_id = session.get('setup_intent')
        if setup_intent_id:
            setup_intent = stripe.SetupIntent.retrieve(setup_intent_id)
            customer_id = setup_intent.get('customer')
            payment_method_id = setup_intent.get('payment_method')
            
            # Ensure payment method is attached to customer
            if customer_id and payment_method_id:
                try:
                    stripe.PaymentMethod.attach(
                        payment_method_id,
                        customer=customer_id,
                    )
                    logger.info(f"Payment method {payment_method_id} attached to customer {customer_id}")
                except stripe.error.InvalidRequestError as e:
                    if "already attached" in str(e):
                        logger.info(f"Payment method {payment_method_id} already attached to customer {customer_id}")
                    else:
                        logger.error(f"Failed to attach payment method: {e}")
            
        # Update user's payment method status
        await update_user_payment_method_status(user_id, purpose)
            
    except Exception as e:
        logger.error(f"Failed to handle setup completion: {e}")

async def handle_setup_intent_succeeded(setup_intent: Dict[str, Any]):
    """Handle setup intent succeeded event"""
    try:
        customer_id = setup_intent.get('customer')
        payment_method_id = setup_intent.get('payment_method')
        
        if not customer_id or not payment_method_id:
            logger.error(f"Missing customer or payment method in setup intent: {setup_intent.get('id')}")
            return
            
        # Ensure payment method is attached to customer
        try:
            stripe.PaymentMethod.attach(
                payment_method_id,
                customer=customer_id,
            )
            logger.info(f"Payment method {payment_method_id} attached to customer {customer_id}")
        except stripe.error.InvalidRequestError as e:
            if "already attached" in str(e):
                logger.info(f"Payment method {payment_method_id} already attached to customer {customer_id}")
            else:
                logger.error(f"Failed to attach payment method: {e}")
                return
        
        # Find user by customer ID
        from app.config.database import users_collection
        
        user = await users_collection.find_one({
            "subscription.stripeCustomerId": customer_id
        })
        
        if user:
            await update_user_payment_method_status(str(user['_id']), 'setup_intent')
            logger.info(f"Updated payment method status for user {user['_id']}")
            
    except Exception as e:
        logger.error(f"Failed to handle setup intent succeeded: {e}")

async def update_user_payment_method_status(user_id: str, purpose: str):
    """Helper function to update user payment method status and create PAYG subscription if needed"""
    try:
        from app.config.database import users_collection
        from bson import ObjectId
        
        # Base update for payment method status
        update_data = {
            "hasPaymentMethod": True,
            "paymentMethodAddedAt": datetime.now(timezone.utc),
            "updatedAt": datetime.now(timezone.utc)
        }
        
        # If this is for PAYG setup, also update subscription to PAYG
        if purpose == 'payg_setup':
            update_data.update({
                "subscription.type": "pay as you go",
                "subscription.status": "active"
            })
            logger.info(f"Payment method added and PAYG subscription activated for user {user_id}")
        else:
            logger.info(f"Payment method successfully added for user {user_id}")
        
        await users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
            
    except Exception as e:
        logger.error(f"Failed to update user payment method status: {e}")