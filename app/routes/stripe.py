from fastapi import APIRouter, Security, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from fastapi.security import HTTPBearer
from app.schemas.user import *
from app.config.database import db 
from app.config.settings import settings
from app.config.constants import MIN_PAYMENT_AMOUNT_USD, ERROR_USER_NOT_FOUND
from app.services.user_service import *
from app.services.pricing_service import pricing_service
from app.services.stripe_service import stripe_service
from app.services.transaction_service import transaction_service
from app.utils.user_helper import *
from app.utils.response_helper import error_response, success_response, not_found_response
from app.dependencies.auth import get_current_user
from app.utils.logger import logger
from app.utils.email_helper import *
from app.utils.token_helper import *
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import stripe
from bson import ObjectId
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException

stripe_route = APIRouter()

# Define the security scheme
security = HTTPBearer(
    bearerFormat="JWT",  
    description="Enter your Bearer token"
)

def convert_objectids(obj):
    if isinstance(obj, list):
        return [convert_objectids(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_objectids(v) for k, v in obj.items()}
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj


stripe.api_key = settings.STRIPE_SECRET_KEY
FRONTEND_URL = settings.FRONTEND_URL
STRIPE_WEBHOOK_SECRET = settings.STRIPE_WEBHOOK_SECRET

class Checkout(BaseModel):
    planName: str

class CustomCheckout(BaseModel):
    price: float

def mongo_json(data):
    return jsonable_encoder(
        data,
        custom_encoder={
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }
    )

def auth_protect():
    pass

@stripe_route.get("/plans")
async def get_pricing_plans(request: Request):
    try:
        response: List[Dict[str, Any]] = await pricing_service.get_all_plans()
        if not response:
            return JSONResponse(
                status_code=404,
                content=pricing_service.mongo_json({
                    "error": "No pricing plans found",
                    "message": "No active pricing plans available"
                })
            )
        return JSONResponse(
            status_code=response["status_code"],
            content=response["content"]
        )
    except Exception as error:
        logger.error(f"Get pricing plans error: {str(error)}")
        from os import getenv
        message = "An error occurred while retrieving pricing plans" if getenv("NODE_ENV") == "production" else str(error)
        return JSONResponse(
            status_code=500,
            content=pricing_service.mongo_json({
                "error": "Server error",
                "message": message
            })
        )

@stripe_route.post("/create-checkout-session-custom", dependencies=[Depends(auth_protect)])
async def create_checkout_session_custom(
    request: CustomCheckout, 
    current_user: TokenUser = Security(get_current_user)
):  
    try:
        user = await get_user_id(current_user.id)
        if not user:
            return error_response(ERROR_USER_NOT_FOUND, "User not found")
        
        # Create Stripe session
        price = request.price
        final_price = price
        discount_percentage = 0
        credits = stripe_service.custom_credit_amount(price)
        name = "custom"

        if not credits and not price:
            return error_response(
                "Failed to create checkout session",
                "Need credits and price",
                500
            )

        session = await stripe_service.get_session_info(user, str(current_user.id), price, final_price, discount_percentage, credits, name)
        
        return success_response({
            "sessionId": session.id,
            "url": session.url,
            "credits": credits,
            "price": final_price,
            "originalPrice": price,
            "discount": discount_percentage,
            "name": name
        })
    except HTTPException as he:
        return error_response(str(he.detail), "Checkout Error", he.status_code)
    except Exception as e:
        logger.error(f"Checkout session error: {str(e)}")
        return error_response(
            "Failed to create checkout session",
            "Internal Server Error",
            500
        )
  

@stripe_route.post("/create-checkout-session", dependencies=[Depends(auth_protect)])
async def create_checkout_session(
    request: Checkout, 
    current_user: TokenUser = Security(get_current_user)
):
    try:
        user = await get_user_id(current_user.id)
        if not user:
            return error_response(ERROR_USER_NOT_FOUND, "User not found")
        # Get credit pack details
        pack_details = await pricing_service.get_credit_pack_details_by_name(request.planName)
        if not pack_details:
            return error_response(
                "Credit pack not found", 
                "The requested pricing plan was not found or is inactive"
            )
        # Create Stripe session
        price = pack_details["original_price"]
        final_price = pack_details["final_price"]
        discount_percentage = pack_details["discount_percentage"]
        credits = pack_details["credits"]
        name = pack_details["name"]  

        session = await stripe_service.get_session_info(user, str(current_user.id), price, final_price, discount_percentage, credits, name)
        
        return success_response({
            "sessionId": session.id,
            "url": session.url,
            "credits": pack_details["credits"],
            "price": pack_details["final_price"],
            "originalPrice": pack_details["original_price"],
            "discount": pack_details["discount_percentage"],
            "name": pack_details["name"]
        })
    except HTTPException as he:
        return error_response(str(he.detail), "Checkout Error", he.status_code)
    except Exception as e:
        logger.error(f"Checkout session error: {str(e)}")
        return error_response(
            "Failed to create checkout session",
            "Internal Server Error",
            500
        )

@stripe_route.get("/customer-portal", dependencies=[Depends(auth_protect)])
def get_customer_portal():
    pass

@stripe_route.post("/webhook")
async def webhook(request: Request):
    sig = request.headers.get('stripe-signature')
    logger.info(f"Received webhook with signature: {'present' if sig else 'missing'}")
    
    try:
        payload = await request.body()
        event = await stripe_service.verify_webhook_signature(payload, sig)
        
        logger.info(f"Processing webhook event: {event['type']} with ID: {event['id']}")
        
        if event['type'] == 'checkout.session.completed':
            result = await transaction_service.handle_checkout_completed(event['data']['object'])
            if result:
                logger.info(f"Successfully processed checkout session: {result['message']}")
            else:
                logger.warning("Checkout session processed with no result")
                
        elif event['type'] == 'customer.created':
            logger.info(f"Stripe customer created: {event['data']['object'].get('id')}")
        
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "Webhook processed successfully"}
        )

    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": str(he.detail)})
    except Exception as error:
        logger.error(f"Webhook processing error: {str(error)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Server error", "message": str(error)}
        )

@stripe_route.get("/purchase-history", dependencies=[Depends(auth_protect)])
async def get_purchase_history(user: TokenUser = Security(get_current_user)):
    try:
        response = await transaction_service.get_purchase_history(str(user.id))
        return JSONResponse(
            status_code=response["status_code"],
            content=response["content"]
        )
    except Exception as error:
        logger.error(f"Error retrieving purchase history: {str(error)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Server error",
                "message": str(error)
            }
        )


@stripe_route.get("/verify-payment/{sessionId}", dependencies=[Depends(auth_protect)])
async def verify_payment(
    sessionId: str, 
    current_user: TokenUser = Security(get_current_user)
):
    try:
        user = await get_user_id(current_user.id)
        user_id = str(user.id)

        if not sessionId:
            return error_response("Session ID required", "Stripe session ID is required")

        # Check existing transaction
        existing = await transaction_service.get_transaction(sessionId, user_id)
        if existing:
            return success_response({
                "message": "Payment already processed",
                "transaction": existing  # Already serialized
            })

        # Verify Stripe session
        session_data = await stripe_service.verify_session(sessionId, user_id)
        
        # Create transaction and update user credits
        transaction = await transaction_service.create_transaction(
            user_id=user_id,
            credits=session_data["credits"],
            amount=session_data["amount"],
            session_id=sessionId,
            description=f"Purchase of {session_data['credits']} credits ({session_data['pack_name']})"
        )

        # Update user credits
        user_update = await transaction_service.update_user_credits(
            user_id, 
            session_data["credits"]
        )
        
        return success_response({
            "message": "Payment verified and processed successfully",
            "transaction": transaction,  # Already serialized
            "user": user_update,  # Already serialized
            "session": {
                "credits": session_data["credits"],
                "amount": session_data["amount"],
                "pack_name": session_data["pack_name"],
                "discount": session_data["discount_percentage"]
            }
        })

    except HTTPException as he:
        return error_response(str(he.detail), "Payment Verification Error", he.status_code)
    except Exception as error:
        logger.error(f"Payment verification error: {str(error)}")
        return error_response(
            "Failed to verify payment", 
            "Internal Server Error", 
            500
        )
