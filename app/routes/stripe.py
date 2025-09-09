"""
Essential Stripe Routes for Frontend
Only the necessary API endpoints
"""

from fastapi import APIRouter, Security, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging

from app.schemas.user import TokenUser
from app.dependencies.auth import get_current_user
from app.services.user_service import get_user_id
from app.services.stripe_service import stripe_service
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

# ========== SIMPLE ENDPOINTS ==========

# 1. Setup Intent (Add Payment Methods)
@stripe_route.post("/setup-intent")
@log_execution_time
async def create_setup_intent(current_user: TokenUser = Security(get_current_user)):
    """Create setup intent for adding payment methods"""
    try:
        user = await get_authenticated_user(current_user)
        result = await stripe_service.create_setup_intent(user, str(current_user.id))
        return success_response("Setup intent created successfully", {"url": result["url"], "sessionId": result["session_id"]})
        
    except Exception as e:
        return handle_api_error(e, "Setup Intent Creation")

# 2. Payment Methods Management
@stripe_route.get("/payment-methods")
@log_execution_time
async def get_payment_methods(current_user: TokenUser = Security(get_current_user)):
    """Get user's saved payment methods (auto-syncs database status)"""
    try:
        user = await get_authenticated_user(current_user)
        # This call will automatically sync database status via _sync_payment_method_status
        payment_methods = await stripe_service.get_payment_methods(user, str(current_user.id))
        return success_response("Payment methods retrieved successfully", {"payment_methods": payment_methods})
        
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
        payment_methods = await stripe_service.get_payment_methods(user, str(current_user.id))
        # if len(payment_methods) <= 1:
        #     raise HTTPException(
        #         status_code=400,
        #         detail=f"you have only one card saved. So you can't remove it. Add a new card then you can remove this card"
        #     )
        # Check for outstanding bills first
        from app.services.credit_service import credit_service
        # Calculate current usage in USD
        from app.config.credit_constants import CreditRates, ThresholdBilling
        
        
        total_credits = user.total_usage or 0.0
        cost_usd = total_credits * CreditRates.COST_PER_CREDIT_USD
        current_usage_usd = round(cost_usd, 2)

        # outstanding_amount = await credit_service._get_outstanding_billing_amount(str(current_user.id))
        if current_usage_usd > 0:
            return JSONResponse(
                status_code=200,
                content={
                    "success": False,
                    "error": "OUTSTANDING_BILL",
                    "message": f"Cannot remove payment method with outstanding bill of ${current_usage_usd:.2f}. Please clear your bill first.",
                    "outstanding_amount": current_usage_usd,
                    "usage_credits": total_credits,
                    "details": "You must clear your outstanding bill before removing payment methods to ensure proper billing."
                }
            )
        
        await stripe_service.remove_payment_method(payment_method_id, user, str(current_user.id))
        return success_response("Payment method removed successfully", {"success": True})
        
    except Exception as e:
        return handle_api_error(e, "Remove Payment Method")

# Removed - use /billing-status for comprehensive threshold and usage info

# 3. Checkout Session (Credit Packs Only)
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
        
        # Simple validation - just check against valid plans
        valid_plans = ['medium', 'special', 'limited']
        if request.planId not in valid_plans:
            raise HTTPException(status_code=404, detail=f"Plan '{request.planId}' not found")
        
        customer_id = await stripe_service.get_or_create_customer(user, str(current_user.id))
        
        # Simple hardcoded plan pricing
        plan_prices = {
            'medium': {'credits': 250, 'price': 10.0, 'name': 'Medium Pack'},
            'special': {'credits': 500, 'price': 19.0, 'name': 'Special Pack'},
            'limited': {'credits': 1000, 'price': 30.0, 'name': 'Limited Pack'}
        }
        
        plan_data = plan_prices.get(request.planId)
        if not plan_data:
            raise HTTPException(status_code=400, detail="Invalid plan")
        
        credits = plan_data['credits']
        price = plan_data['price']
        
        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=['card'],
            locale='auto',
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': plan_data['name'],
                        'description': f"{credits} credits - Credit pack for AI services"
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
        
        return success_response("Checkout session created successfully", {"sessionId": session.id, "url": session.url})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Checkout session creation failed: {e}")
        return handle_api_error(e, "Create Checkout Session")

# 4. Payment Verification
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
                await update_user_payment_method_status(str(current_user.id))
                result["card_added"] = True
                result["message"] = "Payment method added successfully"

                if session.metadata.get('purpose') == 'payg_setup':
                    # Check if subscription was properly created via webhook
                    from app.config.database import users_collection
                    from bson import ObjectId
                    user_doc = await users_collection.find_one({"_id": ObjectId(current_user.id)}, {"subscription": 1})
                    subscription_doc = (user_doc or {}).get("subscription", {})
                    if subscription_doc.get("stripeSubscriptionId"):
                        result["subscription"] = {
                            "type": subscription_doc.get("type"),
                            "status": subscription_doc.get("status"),
                            "current_period_end": subscription_doc.get("currentPeriodEnd")
                        }
                        result["message"] = "Card added and Pay-as-You-Go subscription activated successfully."
            elif session.mode == "subscription":
                # Simple PAYG activation (no complex subscription handling)
                result["subscription"] = {"type": "pay as you go", "status": "active"}
                result["message"] = "Subscription activated successfully"
            else:
                # Regular payment (credit pack purchase)
                transaction_result = await transaction_service.handle_checkout_completed(session)
                if transaction_result:
                    result["transaction"] = transaction_result
        
        return success_response(result)
        
    except Exception as e:
        return handle_api_error(e, "Verify Payment")



# 5. Purchase History
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

# 6. Customer Portal
@stripe_route.get("/customer-portal")
@log_execution_time
async def create_customer_portal(current_user: TokenUser = Security(get_current_user)):
    """Create Stripe customer portal session"""
    try:
        user = await get_authenticated_user(current_user)
        customer_id = await stripe_service.get_or_create_customer(user, str(current_user.id))
        
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{settings.FRONTEND_URL}/profile"
        )
        
        return success_response("Customer portal session created successfully", {"url": portal_session.url})
        
    except Exception as e:
        return handle_api_error(e, "Create Customer Portal")




# 7. Add Payment Method Only
@stripe_route.post("/add-payment-method")
@log_execution_time
async def add_payment_method(current_user: TokenUser = Security(get_current_user)):
    """Add payment method without activating any subscription"""
    try:
        user = await get_authenticated_user(current_user)
        customer_id = await stripe_service.get_or_create_customer(user, str(current_user.id))
        
        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=['card'],
            mode='setup',
            locale='auto',
            success_url=f"{settings.FRONTEND_URL}/subscription/success?session_id={{CHECKOUT_SESSION_ID}}&action=card_add",
            cancel_url=f"{settings.FRONTEND_URL}/subscription/cancel",
            metadata={
                'userId': str(current_user.id),
                'purpose': 'card_add'
            }
        )
        
        return success_response("Payment method setup session created", {
            "sessionId": session.id,
            "url": session.url,
            "message": "Please add your payment method"
        })
        
    except Exception as e:
        return handle_api_error(e, "Add Payment Method")

# 8. Activate Pay-as-You-Go 
@stripe_route.post("/activate-payg")
@log_execution_time
async def activate_payg(current_user: TokenUser = Security(get_current_user)):
    """Activate pay-as-you-go for users who have payment methods"""
    try:
        user = await get_authenticated_user(current_user)
        
        # Check if user has payment methods (use cached status)
        has_payment_method = user.hasPaymentMethod
        if not has_payment_method:
            return error_response("Please add a payment method first", 400)
        
        # Activate PAYG
        from app.config.database import users_collection
        from bson import ObjectId
        customer_id = await stripe_service.get_or_create_customer(user, str(current_user.id))
        
        await users_collection.update_one(
            {"_id": ObjectId(current_user.id)},
            {
                "$set": {
                    "subscription.type": "pay as you go",
                    "subscription.status": "active",
                    "subscription.stripeCustomerId": customer_id,
                    "subscription.thresholdAmount": 10.0,
                    "updatedAt": datetime.now(timezone.utc)
                }
            }
        )
        
        return success_response("Pay-as-You-Go activated successfully", {
            "type": "pay as you go",
            "status": "active",
            "thresholdAmount": 10.0
        })
        
    except Exception as e:
        return handle_api_error(e, "Activate Pay-as-You-Go")

# 9. Deactivate Pay-as-You-Go (Auto switch to Credit Pack)
@stripe_route.post("/deactivate-payg")
@log_execution_time
async def deactivate_payg(current_user: TokenUser = Security(get_current_user)):
    """Deactivate PAYG and auto-switch to credit pack mode"""
    try:
        from app.config.database import users_collection
        from bson import ObjectId
        
        # Switch to credit pack mode
        await users_collection.update_one(
            {"_id": ObjectId(current_user.id)},
            {
                "$set": {
                    "subscription.type": "credit pack",
                    "subscription.status": "active",
                    "updatedAt": datetime.now(timezone.utc)
                },
                "$unset": {
                    "subscription.thresholdAmount": "",
                }
            }
        )
        
        return success_response("Switched to Credit Pack mode successfully", {
            "type": "credit pack",
            "status": "active",
            "message": "Pay-as-You-Go deactivated. Now using Credit Pack billing."
        })
        
    except Exception as e:
        return handle_api_error(e, "Deactivate PAYG")

# Removed duplicate - use /purchase-history instead

# 9. Comprehensive Billing Status (includes threshold info)
@stripe_route.get("/billing-status")
@log_execution_time
async def get_billing_status(current_user: TokenUser = Security(get_current_user)):
    """Get comprehensive billing status including threshold info and usage tracking"""
    try:
        from app.config.database import users_collection
        from app.config.credit_constants import CreditRates, ThresholdBilling
        from bson import ObjectId
        
        user = await users_collection.find_one({"_id": ObjectId(current_user.id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        subscription = user.get("subscription", {})
        total_credits = user.get("total_usage", 0.0)
        cost_usd = total_credits * CreditRates.COST_PER_CREDIT_USD
        threshold = subscription.get("thresholdAmount", ThresholdBilling.DEFAULT_THRESHOLD_USD)
        
        # Calculate when next billing will happen
        credits_until_billing = max(0, (threshold - cost_usd) / CreditRates.COST_PER_CREDIT_USD)
        billing_progress_percentage = min(100, (cost_usd / threshold) * 100)
        
        # Check if billing is blocked
        billing_blocked = subscription.get("billingBlocked", False)
        failed_attempts = subscription.get("billingFailedAttempts", 0)
        
        # Handle datetime serialization
        last_updated = user.get("usage_updated_at") or user.get("updatedAt")
        last_updated_iso = last_updated.isoformat() if last_updated else None
        
        return success_response("Billing status retrieved successfully", {
            # Basic subscription info  
            "subscription_type": subscription.get("type", "none"),
            "subscription_status": subscription.get("status", "none"),
            "has_payment_method": user.get("hasPaymentMethod", False),
            "stripe_customer_id": subscription.get("stripeCustomerId", "not-set"),
            
            # Usage tracking
            "current_usage_credits": total_credits,
            "current_usage_usd": round(cost_usd, 2),
            "credits_until_billing": round(credits_until_billing, 2),
            "billing_progress_percentage": round(billing_progress_percentage, 1),
            
            # Threshold info
            "billing_threshold_usd": threshold,
            "will_bill_at_usd": threshold,
            "cost_per_credit": CreditRates.COST_PER_CREDIT_USD,
            
            # Billing status
            "billing_blocked": billing_blocked,
            "billing_failed_attempts": failed_attempts,
            "billing_block_reason": subscription.get("billingBlockReason", None),
            
            # Timing info
            "last_updated": last_updated_iso
        })
        
    except Exception as e:
        return handle_api_error(e, "Get Billing Status")


@stripe_route.post("/manual-billing")
@log_execution_time
async def trigger_manual_billing(current_user: TokenUser = Security(get_current_user)):
    """
    Allow users to manually trigger billing when they want to pay
    This will charge the user's current usage amount if it's above the threshold
    """
    try:
        user_id = str(current_user.id)
        
        # Get user's current billing status
        billing_status = await stripe_service.get_threshold_status(user_id)
        
        # Check if user has any usage
        current_usage = billing_status.get("current_usage", 0.0)
        current_cost = billing_status.get("current_cost", 0.0)
        threshold = billing_status.get("threshold", 10.0)
        
        if current_usage == 0:
            return success_response("No usage to bill", {
                "current_usage": current_usage,
                "current_cost": current_cost,
                "threshold": threshold,
                "ready_to_bill": False
            })
        
        if current_cost < threshold:
            return success_response(f"Usage ${current_cost:.2f} below threshold ${threshold}", {
                "current_usage": current_usage,
                "current_cost": current_cost,
                "threshold": threshold,
                "ready_to_bill": False,
                "message": f"You need ${threshold - current_cost:.2f} more usage to reach the billing threshold"
            })
        
        # Trigger the billing process
        billing_result = await stripe_service.process_threshold_billing(user_id)
        
        if billing_result.get("success"):
            return success_response("Payment processed successfully", {
                "charged_amount": billing_result.get("charged_amount"),
                "credits_used": billing_result.get("credits"),
                "payment_intent_id": billing_result.get("payment_intent_id"),
                "new_usage": 0.0,
                "message": f"Successfully charged ${billing_result.get('charged_amount', 0):.2f}"
            })
        else:
            return error_response(
                billing_result.get("message", "Payment failed"), 
                400,
                {
                    "current_usage": current_usage,
                    "current_cost": current_cost,
                    "threshold": threshold,
                    "error": billing_result.get("message")
                }
            )
            
    except Exception as e:
        logger.error(f"Manual billing failed for user {current_user.id}: {e}")
        return error_response("Manual billing failed", 500, {"error": str(e)})
 

# 10. Webhook (No auth required)
@stripe_route.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks"""
    try:
        body = await request.body()
        sig_header = request.headers.get('stripe-signature')
        
        event = stripe.Webhook.construct_event(
            body, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
        
        # Handle essential events only
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            
            if session.get('mode') == 'setup':
                await handle_setup_completion(session)
            else:
                # Regular payment completion
                await transaction_service.handle_checkout_completed(session)
                
            logger.info(f"Processed checkout completion for session: {session['id']}")
            
        elif event['type'] == 'payment_intent.succeeded':
            payment_intent = event['data']['object']
            await handle_payment_intent_succeeded(payment_intent)
            logger.info(f"Processed payment intent succeeded: {payment_intent['id']}")
            
        elif event['type'] == 'setup_intent.succeeded':
            setup_intent = event['data']['object']
            await handle_setup_intent_succeeded(setup_intent)
            logger.info(f"Processed setup intent succeeded: {setup_intent['id']}")
        
        return success_response("Webhook processed successfully", {"received": True})
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse(status_code=400, content={"error": "Webhook failed"})

# ========== HELPER FUNCTIONS ==========

async def handle_payment_intent_succeeded(payment_intent: Dict[str, Any]):
    """Handle successful payment intent (our $10 charges)"""
    try:
        user_id = payment_intent.get('metadata', {}).get('userId')
        billing_type = payment_intent.get('metadata', {}).get('billing_type')
        
        if billing_type == 'threshold' and user_id:
            logger.info(f"Threshold billing payment succeeded for user {user_id}")
            # Payment already processed in stripe_service.process_threshold_billing
        
    except Exception as e:
        logger.error(f"Failed to handle payment intent succeeded: {e}")

# ========== SIMPLE WEBHOOK HELPERS ==========

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
            
            # If this setup was for PAYG, just activate simple threshold billing
            if purpose == 'payg_setup' and customer_id:
                try:
                    from app.config.database import users_collection
                    from bson import ObjectId
                    
                    await users_collection.update_one(
                        {"_id": ObjectId(user_id)},
                        {
                            "$set": {
                                "subscription.type": "pay as you go",
                                "subscription.status": "active",
                                "subscription.stripeCustomerId": customer_id,
                                "subscription.thresholdAmount": 10.0,
                                "updatedAt": datetime.now(timezone.utc)
                            }
                        }
                    )
                    logger.info(f"Simple PAYG activated for user {user_id} with $10 threshold")
                except Exception as e:
                    logger.error(f"Failed to activate PAYG for user {user_id}: {e}")
            
        # Update user's payment method status
        await update_user_payment_method_status(user_id)
            
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
            await update_user_payment_method_status(str(user['_id']))
            logger.info(f"Updated payment method status for user {user['_id']}")
            
    except Exception as e:
        logger.error(f"Failed to handle setup intent succeeded: {e}")

async def update_user_payment_method_status(user_id: str):
    """Update user payment method status only"""
    try:
        from app.config.database import users_collection
        from bson import ObjectId
        
        # Base update for payment method status
        update_data = {
            "hasPaymentMethod": True,
            "paymentMethodAddedAt": datetime.now(timezone.utc),
            "updatedAt": datetime.now(timezone.utc)
        }
        logger.info(f"Payment method successfully added for user {user_id}")
        
        await users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
            
    except Exception as e:
        logger.error(f"Failed to update user payment method status: {e}")

@stripe_route.get("/payment-details/{payment_intent_id}")
@log_execution_time
async def get_payment_details(
    payment_intent_id: str, 
    current_user: TokenUser = Security(get_current_user)
):
    """Get payment details and invoice information"""
    try:
        # Retrieve payment intent from Stripe
        payment_intent = stripe.PaymentIntent.retrieve(payment_intent_id)
        
        # Verify the payment belongs to the current user
        if payment_intent.metadata.get('userId') != str(current_user.id):
            return error_response("Payment not found", 404)
        
        # Get invoice if exists
        invoice_info = None
        if payment_intent.invoice:
            invoice = stripe.Invoice.retrieve(payment_intent.invoice)
            invoice_info = {
                "invoice_id": invoice.id,
                "invoice_pdf": invoice.invoice_pdf,
                "hosted_invoice_url": invoice.hosted_invoice_url,
                "invoice_number": invoice.number,
                "status": invoice.status
            }
        
        return success_response("Payment details retrieved", {
            "payment_intent_id": payment_intent.id,
            "amount": payment_intent.amount / 100,
            "currency": payment_intent.currency,
            "status": payment_intent.status,
            "created": payment_intent.created,
            "customer_id": payment_intent.customer,
            "metadata": payment_intent.metadata,
            "invoice": invoice_info
        })
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error retrieving payment {payment_intent_id}: {e}")
        return error_response("Failed to retrieve payment details", 400, {"error": str(e)})
    except Exception as e:
        logger.error(f"Error retrieving payment details: {e}")
        return error_response("Failed to retrieve payment details", 500, {"error": str(e)})

@stripe_route.get("/payment-history")
@log_execution_time
async def get_payment_history(current_user: TokenUser = Security(get_current_user)):
    """Get all payment history for the current user"""
    try:
        # Get user's Stripe customer ID
        from app.config.database import users_collection
        from bson import ObjectId
        
        user = await users_collection.find_one({"_id": ObjectId(current_user.id)})
        if not user:
            return error_response("User not found", 404)
        
        customer_id = user.get("subscription", {}).get("stripeCustomerId")
        if not customer_id:
            return success_response("No payment history", {"payments": []})
        
        # Get payment intents for this customer
        payment_intents = stripe.PaymentIntent.list(
            customer=customer_id,
            limit=50
        )
        
        payments = []
        for pi in payment_intents.data:
            # Simple payment info without invoice complexity
            payment_data = {
                "payment_intent_id": pi.id,
                "amount": pi.amount / 100,
                "currency": pi.currency,
                "status": pi.status,
                "created": pi.created,
                "metadata": pi.metadata
            }
            
            # Safely check for invoice without retrieving it
            try:
                payment_data["has_invoice"] = bool(getattr(pi, 'invoice', None))
            except:
                payment_data["has_invoice"] = False
            
            payments.append(payment_data)
        
        return success_response("Payment history retrieved", {
            "payments": payments,
            "total_count": len(payments)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving payment history: {e}")
        return error_response("Failed to retrieve payment history", 500, {"error": str(e)})