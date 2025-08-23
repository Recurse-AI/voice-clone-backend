from typing import List, Optional, Dict, Any
from fastapi import HTTPException
from datetime import datetime, timedelta
from app.config.database import db, users_collection
from app.models.pricing import Pricing, CreditPack
from app.utils.logger import logger
from fastapi.encoders import jsonable_encoder
from bson import ObjectId
from app.config.settings import settings
import stripe
from app.schemas.user import TokenUser

stripe.api_key = settings.STRIPE_SECRET_KEY
FRONTEND_URL = settings.FRONTEND_URL
STRIPE_WEBHOOK_SECRET = settings.STRIPE_WEBHOOK_SECRET

class StripeService:
    def __init__(self):
        self.collection = db.get_collection("pricings")
        self.users_collection = users_collection

    async def verify_webhook_signature(self, payload: bytes, signature: str) -> stripe.Event:
        """Verify Stripe webhook signature"""
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
            return event
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid webhook signature: {e}")
            raise HTTPException(status_code=400, detail="Invalid webhook signature")


    async def handle_stripe_customer(self, user: TokenUser, user_id: str) -> str:
        """Create or retrieve Stripe customer"""
        try:
            customer_id = getattr(getattr(user, "subscription", None), "stripeCustomerId", None)

            if customer_id:
                try:
                    stripe.Customer.retrieve(customer_id)
                    return customer_id
                except stripe.error.StripeError:
                    customer_id = None

            stripe_customer = stripe.Customer.create(
                email=user.email,
                name=getattr(user, "name", ""),
                metadata={"userId": user_id}
            )
            return stripe_customer.id

        except Exception as e:
            logger.error(f"Error handling Stripe customer: {e}")
            raise HTTPException(status_code=400, detail="Failed to create/retrieve Stripe customer")

    async def get_session_info(self, user: TokenUser, user_id: str, price: float, final_price: float, discount_percentage: float, credits: int, name: str) -> stripe.checkout.Session:
        try:
            customer_id = await self.handle_stripe_customer(user, user_id)
            if not customer_id:
                raise HTTPException(status_code=400, detail="Failed to create/retrieve Stripe customer")

            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=["card"],
                line_items=[{
                    "price_data": {
                        "currency": "usd",
                        "product_data": {   
                            "name": f"{name} - {credits} Credits", 
                            "description": f"Get {credits} processing credits with {discount_percentage}% savings"
                        },
                        "unit_amount": int(final_price * 100),
                    },
                    "quantity": 1,
                }],
                allow_promotion_codes=True,
                mode="payment",
                success_url=f"{settings.FRONTEND_URL}/payment/success?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{settings.FRONTEND_URL}/pricing",
                metadata={
                    "userId": user_id,
                    "credit": credits,
                    "price": final_price,
                    "originalPrice": price,
                    "discountPercentage": discount_percentage,
                    "packName": name
                }
            )
            return session
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error creating checkout session: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create checkout session")
        
    async def handle_usage_billing(self, session: stripe.checkout.Session):
        """Handle usage billing for pay-as-you-go subscriptions"""
        try:
            # Get the subscription
            subscription = stripe.Subscription.retrieve(session.subscription)
            
            try:
                # Get subscription items
                subscription_items = subscription.items.data
                if subscription_items:
                    subscription_item = subscription_items[0]
                    subscription_item_id = subscription_item.id
                else:
                    raise ValueError("No items in subscription.items.data")
                    
            except (AttributeError, ValueError):
                subscription_items = stripe.SubscriptionItem.list(subscription=session.subscription)
                if not subscription_items.data:
                    raise HTTPException(
                        status_code=400, 
                        detail="No subscription items found"
                    )
                subscription_item = subscription_items.data[0]
                subscription_item_id = subscription_item.id
            
            # For new subscriptions, start with 0 usage
            current_usage = 0
            
            logger.info(f"Current usage for subscription {session.subscription}: {current_usage}")

            if subscription.status == 'active':
                await self.save_subscription_details(session)
                logger.info(f"Active subscription saved for user {session.metadata.get('userId')}")
            else:
                logger.info(f"Subscription {session.subscription} is not active")

            # Get current period end from subscription level (more reliable)
            current_period_end = getattr(subscription, 'current_period_end', None)
            
            return {
                "subscription_id": session.subscription,
                "subscription_item_id": subscription_item_id,
                "current_usage": current_usage,
                "status": subscription.status,
                "current_period_end": current_period_end
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error in usage billing: {e}")
            raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
        except Exception as e:
            logger.error(f"Error handling usage billing: {e}")
            raise HTTPException(status_code=500, detail="Failed to handle usage billing")

    async def create_customer_portal_session(self, user: TokenUser, user_id: str):
        customer_id = await self.handle_stripe_customer(user, user_id)
        
        user_data = await self.users_collection.find_one({"_id": ObjectId(user_id)})
        subscription_id = user_data.get("subscription", {}).get("stripeSubscriptionId") if user_data else None
        
        portal_params = {
            "customer": customer_id,
            "return_url": f"{settings.FRONTEND_URL}/account",
        }
        
        if subscription_id:
            try:
                subscription = stripe.Subscription.retrieve(subscription_id)
                if subscription.customer == customer_id:
                    portal_params["flow_data"] = {
                        "type": "subscription_update",
                        "subscription_update": {
                            "subscription": subscription_id
                        }
                    }
                else:
                    print(f"Warning: Subscription {subscription_id} belongs to customer {subscription.customer}, not {customer_id}")
            except stripe.error.StripeError as e:
                print(f"Error retrieving subscription: {str(e)}")
        
        session = stripe.billing_portal.Session.create(**portal_params)
        return session.url

    async def save_subscription_details(self, session: stripe.checkout.Session):
        """Save subscription details to database"""
        try:
            user_id = session.metadata.get('userId')
            subscription_id = session.subscription
            customer_id = session.customer
            
            # Get subscription details
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            # Safely get current_period_end with fallback
            current_period_end = None
            try:
                if hasattr(subscription, 'current_period_end') and subscription.current_period_end:
                    current_period_end = datetime.fromtimestamp(subscription.current_period_end)
            except (AttributeError, TypeError, ValueError):
                current_period_end = datetime.now() + timedelta(days=settings.TIME_CYCLE)
                logger.warning(f"Using fallback current_period_end for subscription {subscription_id}")
            
            # Update user subscription in database
            update_data = {
                "subscription.type": "pay as you go",
                "subscription.stripeCustomerId": customer_id,
                "subscription.stripeSubscriptionId": subscription_id,
                "subscription.status": "active",
                "updatedAt": datetime.now()
            }
            
            if current_period_end:
                update_data["subscription.currentPeriodEnd"] = current_period_end
            
            await self.users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_data}
            )
            
            logger.info(f"Subscription details saved for user {user_id}")
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error saving subscription details: {e}")
            raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to save subscription details: {e}")
            raise HTTPException(status_code=500, detail="Failed to save subscription details")

    async def update_subscription_status(self, subscription: stripe.Subscription):
        """Update subscription status in database"""
        try:
            # Find user by subscription ID
            user = await self.users_collection.find_one({
                "subscription.stripeSubscriptionId": subscription.id
            })
            
            if user:
                await self.users_collection.update_one(
                    {"_id": user["_id"]},
                    {
                        "$set": {
                            "subscription.status": subscription.status,
                            "subscription.currentPeriodEnd": datetime.fromtimestamp(
                                subscription.current_period_end
                            ),
                            "updatedAt": datetime.now()
                        }
                    }
                )
                
                logger.info(f"Subscription status updated for user {user['_id']}")
                
        except Exception as e:
            logger.error(f"Failed to update subscription status: {e}")

    async def report_usage_to_stripe(self, subscription_item_id: str, usage_quantity: int):
        """Report usage to Stripe for metered billing"""
        try:
            usage_record = stripe.SubscriptionItem.create_usage_record(
                subscription_item_id,
                quantity=usage_quantity,
                timestamp=int(datetime.now().timestamp()),
                action='increment'  # Add to existing usage
            )
            
            logger.info(f"Usage reported to Stripe: {usage_quantity} credits for item {subscription_item_id}")
            return usage_record
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to report usage to Stripe: {e}")
            raise HTTPException(status_code=400, detail=f"Usage reporting failed: {str(e)}")

    async def get_subscription_usage(self, subscription_id: str):
        """Get detailed usage information for a subscription"""
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            usage_data = []
            for item in subscription.items.data:
                if item.price.recurring.usage_type == 'metered':
                    # Get usage records for current period
                    usage_records = stripe.SubscriptionItem.list_usage_records(
                        item.id,
                        limit=100
                    )
                    
                    # Get usage summary
                    usage_summary = stripe.SubscriptionItem.list_usage_record_summaries(
                        item.id,
                        limit=1
                    )
                    
                    usage_data.append({
                        'subscription_item_id': item.id,
                        'price_id': item.price.id,
                        'usage_records': usage_records.data,
                        'current_usage': usage_summary.data[0].total_usage if usage_summary.data else 0,
                        'period_start': usage_summary.data[0].period.start if usage_summary.data else None,
                        'period_end': usage_summary.data[0].period.end if usage_summary.data else None
                    })
            
            return usage_data
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to get subscription usage: {e}")
            return None


    async def verify_session(self, session_id: str, user_id: str) -> dict:
        """Verify Stripe session status and return session details"""
        try:
            # Retrieve the session
            session = stripe.checkout.Session.retrieve(session_id)
            
            if not session:
                raise HTTPException(
                    status_code=404,
                    detail="Payment session not found"
                )
            
            # Verify user ownership
            if session.metadata.get("userId") != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="Unauthorized access to payment session"
                )

            # Handle different session modes
            if session.mode == "subscription":
                # Handle subscription mode (pay-as-you-go)
                if session.metadata.get("billingType") == "usage-based":
                    # Handle usage-based subscription
                    usage_info = await self.handle_usage_billing(session)
                    
                    return {
                        "session": session,
                        "credits": 0,  # No credits for usage-based
                        "amount": 0,    # No upfront payment
                        "pack_name": "pay as you go",
                        "discount_percentage": 0,
                        "original_price": 0,
                        "subscription_info": usage_info,
                        "session_type": "subscription",
                        "billing_type": "usage-based"
                    }
                else:
                    # Handle regular subscription
                    return {
                        "session": session,
                        "credits": 0,
                        "amount": 0,
                        "pack_name": "subscription",
                        "discount_percentage": 0,
                        "original_price": 0,
                        "session_type": "subscription",
                        "billing_type": "regular"
                    }
            elif session.mode == "payment":
                # Check payment status
                if session.payment_status != "paid":
                    raise HTTPException(
                        status_code=400,
                        detail="Payment not completed"
                    )

                # Get credit amount from metadata
                credits = float(session.metadata.get("credit", 0))
                if not credits or credits <= 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid credits in payment session"
                    )

                return {
                    "session": session,
                    "credits": credits,
                    "amount": session.amount_total / 100.00,  # Convert from cents to dollars
                    "pack_name": session.metadata.get("packName"),
                    "discount_percentage": session.metadata.get("discountPercentage"),
                    "original_price": session.metadata.get("originalPrice"),
                    "session_type": "payment",
                    "billing_type": "credit_pack"
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported session mode"
                )

        except stripe.error.StripeError as e:
            logger.error(f"Stripe verification error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Session verification error: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Failed to verify payment session"
            )

    def custom_credit_amount(self, amount: float) -> float:
        custom_price = amount

        # Set discount based on price
        if custom_price >= 40.00:
            discount = 85.00
        elif custom_price >= 20.00:
            discount = 70.00
        elif custom_price >= 10.00:
            discount = 40.00
        elif custom_price >= 5.00:
            discount = 20.00
        else:
            discount = 0.00

        credit_per_unit = 25
        custom_credits = credit_per_unit * custom_price
        total_credits = custom_credits + (custom_credits * discount / 100)

        return total_credits

    async def get_psug_session(self, user: TokenUser, user_id: str):
        customer_id = await self.handle_stripe_customer(user, user_id)
        if not customer_id:
            raise HTTPException(status_code=400, detail="Failed to create/retrieve Stripe customer")
            
        checkout_params = {
            "customer": customer_id,
            "mode": "subscription",
            "line_items": [
                {
                    "price": settings.PAY_AS_YOU_GO_PRICE_ID,
                },
            ],
            "success_url": f"{settings.FRONTEND_URL}/payment/success?session_id={{CHECKOUT_SESSION_ID}}",
            "cancel_url": f"{settings.FRONTEND_URL}/pricing",
            "metadata": {
                "userId": user_id,
                "billingType": "usage-based",
                "subscriptionType": "pay-as-you-go"
            },
            "payment_method_collection": "always",
            "payment_method_types": ["card"],
            "saved_payment_method_options": {
                "allow_redisplay_filters": ["always"],
                "payment_method_types": ["card"]
            },
            "subscription_data": {
                "metadata": {
                    "userId": user_id,
                    "billingType": "usage-based"
                }
            }
        }

        session = stripe.checkout.Session.create(**checkout_params)
        return session

        

stripe_service = StripeService()
