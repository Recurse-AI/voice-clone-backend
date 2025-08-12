from fastapi import APIRouter, Security, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from app.schemas.user import *
from app.config.database import db 
from app.config.settings import settings
from app.config.constants import MIN_PAYMENT_AMOUNT_USD, ERROR_USER_NOT_FOUND
from app.services.user_service import *
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
    price: int

def mongo_json(data):
    return jsonable_encoder(
        data,
        custom_encoder={
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }
    )

# Dummy auth dependency placeholder, replace with your actual auth
def auth_protect():
    pass

@stripe_route.get("/plans")
async def get_pricing_plans(request: Request):
    try:
        plans = await db["pricings"].find({"isActive": True}).sort("displayOrder").to_list(length=100)
        if not plans:
            return JSONResponse(
                status_code=404,
                content=mongo_json({
                    "error": "No pricing plans found",
                    "message": "No active pricing plans available"
                })
            )
        return JSONResponse(
            status_code=200,
            content=mongo_json({
                "success": True,
                "plans": plans,
                "count": len(plans)
            })
        )
    except Exception as error:
        logger.error(f"Get pricing plans error: {str(error)}")
        from os import getenv
        message = "An error occurred while retrieving pricing plans" if getenv("NODE_ENV") == "production" else str(error)
        return JSONResponse(
            status_code=500,
            content=mongo_json({
                "error": "Server error",
                "message": message
            })
        )

def calculate_credits(price, premium_plan):
    credits = 0
    discount_percentage = 0
    final_price = price
    packs = premium_plan.get("pricing", {}).get("creditPacks", [])
    matching_pack = next((pack for pack in packs if pack["price"] == price), None)
    if matching_pack:
        credits = matching_pack["credits"]
        discount_percentage = matching_pack.get("discountPercentage", 0)
        final_price = price * (1 - discount_percentage / 100)
    elif price > 180:
        base_credits = 5000
        base_price = 180
        base_discount = 10
        credits = int((price * base_credits) / base_price)
        discount_percentage = base_discount
        final_price = price * (1 - discount_percentage / 100)
    else:
        credits = int(price * 25)
        discount_percentage = 0
        final_price = price
    return credits, discount_percentage, final_price

async def handle_stripe_customer(user, user_id):
    customer_id = getattr(getattr(user, "subscription", None), "stripeCustomerId", None)

    if customer_id:
        try:
            stripe.Customer.retrieve(customer_id)
            return customer_id
        except Exception:
            customer_id = None

    # Create new customer
    stripe_customer = stripe.Customer.create(
        email=user.email,
        name=getattr(user, "name", ""),
        metadata={"userId": user_id}
    )
    customer_id = stripe_customer.id

    await db["users"].update_one(
        {"_id": user.id},
        {"$set": {"subscription.stripeCustomerId": customer_id}}
    )

    return customer_id

@stripe_route.post("/create-checkout-session", dependencies=[Depends(auth_protect)])
async def create_checkout_session(request: Checkout, current_user: TokenUser = Security(get_current_user)):
    
    try:
        logger.info("controller  reached here in checkout session line 121")
        price = request.price
        # price = data.get("price")
        user = await get_user_id(current_user.id)
        user_id = current_user.id
        logger.info(f"price: {price}")
        if user is None:
            return not_found_response(ERROR_USER_NOT_FOUND, "User no longer exists")
        # Input validation
        if not user_id:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Authentication required",
                    "message": "User authentication needed"
                }
            )
        if not price or price < MIN_PAYMENT_AMOUNT_USD:
            return error_response(
                f"Price is required and must be at least ${MIN_PAYMENT_AMOUNT_USD}",
                "Invalid price",
                status_code=400
            )

        # Find user (already fetched as 'user')
        if not user:
            return not_found_response(ERROR_USER_NOT_FOUND)

        # Get premium plan from database
        premium_plan = await db["pricings"].find_one({"name": "premium"})
        if not premium_plan:
            return not_found_response("Premium plan configuration not found", "Pricing plan not found")

        # Calculate credits and discount
        credits, discount_percentage, final_price = calculate_credits(price, premium_plan)


        logger.info(f"Credits calculation - Original: ${price}, Credits: {credits}, Discount: {discount_percentage}%, Final: ${final_price}")

        # Handle Stripe customer
        customer_id = await handle_stripe_customer(user, user_id)

        # Create checkout session
        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {   
                        "name": f"{credits} Credits",
                        "description": f"{credits} processing minutes for audio separation"
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
                "discountPercentage": discount_percentage
            }
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "sessionId": session.id,
                "url": session.url,
                "credits": credits,
                "price": final_price,
                "discount": discount_percentage
            }
        )

    except Exception as error:
        logger.error(f"Stripe checkout error: {str(error)}")
        from os import getenv
        message = "An error occurred while creating checkout session" if getenv("NODE_ENV") == "production" else str(error)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "message": message
            }
        )


@stripe_route.get("/customer-portal", dependencies=[Depends(auth_protect)])
def get_customer_portal():
    pass

@stripe_route.post("/webhook")  # raw body handled separately in actual implementation
async def webhook(request: Request):
    sig = request.headers.get('stripe-signature')
    logger.info(f"Received webhook with signature: {'present' if sig else 'missing'}")
    payload = await request.body()
    event = None
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
        logger.info(f"Webhook event verified successfully: {event['type']}")
    except Exception as err:
        logger.error(f"Webhook signature verification failed: {str(err)}")
        return JSONResponse(status_code=400, content={"error": f"Webhook Error: {str(err)}"})

    logger.info(f"Processing webhook event: {event['type']} with ID: {event['id']}")
    try:
        event_type = event['type']
        data_object = event['data']['object']
        if event_type == 'checkout.session.completed':
            await handle_checkout_completed(data_object)
        elif event_type == 'customer.created':
            logger.info(f"Stripe customer created: {data_object.get('id')}")
        else:
            logger.info(f"Unhandled Stripe event: {event_type}")
        logger.info(f"Webhook event processed successfully: {event_type}")
        return JSONResponse(status_code=200, content={"received": True})
    except Exception as error:
        logger.error(f"Error handling webhook event: {str(error)}")
        return JSONResponse(status_code=200, content={"received": True, "error": str(error)})

@stripe_route.get("/purchase-history", dependencies=[Depends(auth_protect)])
async def get_purchase_history(user: TokenUser = Security(get_current_user)):
    try:
        user_id = str(user.id)
        cursor = db["creditTransaction"].find(
            {"userId": user_id}
        ).sort("createdAt", -1)
        transactions = [txn async for txn in cursor]
        transactions = convert_objectids(transactions)
        return JSONResponse(
            status_code=200,
            content=jsonable_encoder({
                "message": "Purchase history retrieved successfully",
                "transactions": transactions
            })
        )
    except Exception as error:
        logger.error(f"Error retrieving purchase history: {str(error)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "details": str(error)
            }
        )
    
from pymongo.errors import DuplicateKeyError
@stripe_route.get("/verify-payment/{sessionId}", dependencies=[Depends(auth_protect)])
async def verify_payment(sessionId: str, current_user: TokenUser = Security(get_current_user)):
    try:
        _user = await get_user_id(current_user.id)
        user: dict = _user.model_dump()
        user_id = str(user.get("id"))

        if not sessionId:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Session ID required",
                    "message": "Stripe session ID is required"
                }
            )
        session = stripe.checkout.Session.retrieve(sessionId)
        if not session:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Session not found",
                    "message": "Payment session not found"
                }
            )
        if session.metadata.get("userId") != user_id:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Unauthorized",
                    "message": "This payment session does not belong to you"
                }
            )
        if session.payment_status != "paid":
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Payment not completed",
                    "message": "Payment has not been completed successfully"
                }
            )
        existing = await db["creditTransaction"].find_one({"stripeSessionId": sessionId, "userId": user_id})
        if existing:
            return JSONResponse(
                status_code=200,
                content=jsonable_encoder({
                    "success": True,
                    "message": "Payment already processed",
                    "transaction": convert_objectids(existing)
                })
            )
        credits_to_add = float(session.metadata.get("credit", 0))
        if not credits_to_add or credits_to_add <= 0:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid credits",
                    "message": "Invalid credits in payment session"
                }
            )
        # Update user credits and subscription type to premium
        try:
            txn = await db["creditTransaction"].insert_one({
                "userId": user_id,
                "type": "purchase",
                "credits": credits_to_add,
                "amount": session.amount_total / 100,
                "stripeSessionId": sessionId,
                "description": f"Purchase of {credits_to_add} credits",
                "status": "success",
                "createdAt": datetime.now()
            })
        except DuplicateKeyError:
            # Transaction already exists, so don't add credits again
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Payment already processed",
                    "transaction": await db["creditTransaction"].find_one({"stripeSessionId": sessionId, "userId": user_id})
                }
            )
        result = await db["users"].update_one(
            {"_id": ObjectId(user_id)},
            {"$inc": {"credits": credits_to_add}, "$set": {"subscription.type": "premium", "subscription.status": "active"}}
        )
        if result.modified_count == 0:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "User not found",
                    "message": "User not found"
                }
            )
        logger.info(f"Successfully processed payment verification for user {user_id}. Added {credits_to_add} credits")
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Payment verified and processed successfully",
                "transaction": str(txn.inserted_id),
                "user": {
                    "credits": user.get("credits", 0) + credits_to_add,
                    "subscription": {"type": "premium", "status": "active"}
                }
            }
        )
    except Exception as error:
        logger.error(f"Payment verification error: {str(error)}")
        from os import getenv
        message = "An error occurred while verifying payment" if getenv("NODE_ENV") == "production" else str(error)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "message": message
            }
        )

async def handle_checkout_completed(session):
    try:
        existing_txn = await db["creditTransaction"].find_one({
            "stripeSessionId": session["id"]
        })

        if existing_txn:
            print("Transaction already recorded.")
            return  # or handle however you prefer
        
        logger.info(f"Processing checkout.session.completed for session: {session['id']}")
        user_id = session['metadata'].get('userId')
        credits_to_add = float(session['metadata'].get('credit', 0))
        if not user_id:
            logger.error(f"Missing userId in session metadata for session: {session['id']}")
            return
        if not credits_to_add or credits_to_add <= 0:
            logger.error(f"Missing or invalid credits in session metadata for session: {session['id']}")
            return
        logger.info(f"Attempting to add {credits_to_add} credits to user: {user_id}")
        # Update user credits and subscription type to premium
        # Create transaction record
        # await db["creditTransaction"].insert_one({
        #     "userId": user_id,
        #     "type": "purchase",
        #     "credits": credits_to_add,
        #     "amount": session["amount_total"] / 100,
        #     "stripeSessionId": session["id"],
        #     "description": f"Purchase of {credits_to_add} credits",
        #     "status": "success",
        #     "createdAt": datetime.now(timezone.utc)
        # })
        try:
            txn = await db["creditTransaction"].insert_one({
                "userId": user_id,
                "type": "purchase",
                "credits": credits_to_add,
                "amount": session.amount_total / 100,
                "stripeSessionId": session["id"],
                "description": f"Purchase of {credits_to_add} credits",
                "status": "success",
                "createdAt": datetime.now()
            })
        except DuplicateKeyError:
            # Transaction already exists, so don't add credits again
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Payment already processed",
                    "transaction": await db["creditTransaction"].find_one({"stripeSessionId": session["id"], "userId": user_id})
                }
            )
        result = await db["users"].update_one(
            {"_id": ObjectId(user_id)},
            {"$inc": {"credits": credits_to_add}, "$set": {"subscription.type": "premium", "subscription.status": "active"}}
        )
        if result.modified_count == 0:
            logger.error(f"User not found: {user_id}")
            return
        logger.info(f"Successfully added {credits_to_add} credits to user {user_id}. Updated subscription type to premium.")
    except Exception as error:
        logger.error(f"Error handling checkout completed: {str(error)}")


