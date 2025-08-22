from typing import List, Optional, Dict, Any
from fastapi import HTTPException
from datetime import datetime
from app.config.database import db
from app.models.pricing import Pricing, CreditPack
import logging
from fastapi.encoders import jsonable_encoder

logger = logging.getLogger(__name__)
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
                "original_price": session.metadata.get("originalPrice")
            }

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



stripe_service = StripeService()
