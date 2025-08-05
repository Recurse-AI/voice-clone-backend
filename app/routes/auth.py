from fastapi import APIRouter, Security, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from app.schemas.user import *
from app.config.database import db 
from app.services.user_service import *
from app.utils.user_helper import *
from app.dependencies.auth import get_current_user
from app.utils.logger import logger
from app.utils.email_helper import *
from app.utils.token_helper import *
from datetime import datetime, timezone, timedelta

auth = APIRouter()

def prepare_subscription_data(subscription):
    """Helper function to prepare subscription data for FullUser schema"""
    if subscription:
        if hasattr(subscription, 'model_dump'):
            return subscription.model_dump()
        elif hasattr(subscription, 'dict'):
            return subscription.dict()
        else:
            return subscription
    else:
        return {
            "type": "free",
            "status": "none",
            "stripeCustomerId": None,
            "stripeSubscriptionId": None,
            "currentPeriodEnd": None
        }

# Define the security scheme
security = HTTPBearer(
    bearerFormat="JWT",  
    description="Enter your Bearer token"
)

@auth.get("/")
async def auth_home():
    return {
        "message": "this is auth home"
    }

@auth.post("/register")
async def register(user: UserCreate, background_tasks: BackgroundTasks):
    try:
        existing = await db["users"].find_one({"email": user.email})
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        token = generate_url_safe_token()
        user_dict = user.model_dump()
        user_dict["emailVerificationToken"] = token 
        created_user = await create_user(user_dict) 
        logger.info(f"user : ${created_user}")

        background_tasks.add_task(send_verification_email_background_task, background_tasks, user.email, user.name, token)

        return JSONResponse(status_code=201, content={
            "message": "User created successfully. Please verify email",
        })
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Server error",
                "message": "An error occurred while creating account. Please try again."
            }
        )

@auth.get("/verify-email")
async def verify_email(token: str):
    try:
        user_doc = await db["users"].find_one({"emailVerificationToken": token})
        
        if user_doc is None:
            raise HTTPException(status_code=404, detail="Invalid verification token")
        
        if user_doc.get("isEmailVerified", False):
            return JSONResponse(status_code=200, content={
                "message": "Email already verified"
            })
        
        
        await db["users"].update_one(
            {"_id": user_doc["_id"]},
            {
                "$set": {
                    "isEmailVerified": True,
                    "updatedAt": datetime.now(timezone.utc)
                },
                "$unset": {
                    "emailVerificationToken": ""
                }
            }
        )

        return JSONResponse(status_code=200, content={
            "message": "Email verified successfully"
        })
    except Exception as e:
        logger.error(f"Email verification error: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail="Verification failed. Please contact support."
        )

@auth.post("/login")
async def login_user(req: LoginData):
    try:
        user: dict = await get_user(req)

        # Convert to FullUser schema to include subscription information
        subscription_data = prepare_subscription_data(user.subscription)
        
        full_user = FullUser(
            id=user.id,
            name=user.name,
            email=user.email,
            isEmailVerified=user.isEmailVerified,
            profilePicture=user.profilePicture,
            role=user.role,
            credits=user.credits,
            subscription=subscription_data
        )
        
        user_data = full_user.model_dump(mode='json')
        logger.info(f"get the user : {user_data}")
        remaining_credits = calculate_remaining_seconds(user_data)
        
        token = create_jwt_token(user_data)
        return JSONResponse(
            status_code=200,
            content={
                "message": "Login successfull",
                "user": user_data,
                "token": token,
                "remainingCredits": remaining_credits
            }
        )
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Server error", 
                "message": "Login failed. Please check your credentials and try again."
            }
        )
    

@auth.get("/profile")
async def profile(
    current_user: TokenUser = Security(get_current_user)
    ):
    try:
        user = await get_user_id(current_user.id)

        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "User not found",
                    "details": "User no longer exists"
                }
            )

        # Convert to FullUser schema to include subscription information
        subscription_data = prepare_subscription_data(user.subscription)
        
        full_user = FullUser(
            id=user.id,
            name=user.name,
            email=user.email,
            isEmailVerified=user.isEmailVerified,
            profilePicture=user.profilePicture,
            role=user.role,
            credits=user.credits,
            subscription=subscription_data
        )
        
        user_data = full_user.model_dump(mode='json')
        remaining_credits = calculate_remaining_seconds(user_data)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Profile retrieved successfully",
                "user": user_data,
                "remainingCredits": remaining_credits
            }
        )

    except Exception as e:
        logger.error(f"Get profile error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "details": "An error occurred while retrieving the profile"
            }
        )

@auth.put("/profile")
async def update_profile( data: UpdateProfileRequest, current_user: TokenUser = Security(get_current_user)) -> JSONResponse:
    try:
        if not current_user.id:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": "not found"
                }
            )
        
        user = await update_user_name(current_user.id, data)

        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "User not found",
                    "details": "User no longer exists"
                }
            )
        
        # Convert to FullUser schema to include subscription information
        subscription_data = prepare_subscription_data(user.subscription)
        
        full_user = FullUser(
            id=user.id,
            name=user.name,
            email=user.email,
            isEmailVerified=user.isEmailVerified,
            profilePicture=user.profilePicture,
            role=user.role,
            credits=user.credits,
            subscription=subscription_data
        )
        
        user_data = full_user.model_dump(mode='json')
        remaining_credits = calculate_remaining_seconds(user_data)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Profile updated successfully",
                "user": user_data,
                "remainingCredits": remaining_credits
            }
        )

    except Exception as e:
        logger.error(f"Error while updating profile: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "details": "An error occurred while updating profile"
            }
        )

@auth.post("/forgot-password")
async def request_password_reset(req: ResetPasswordRequest, background_tasks: BackgroundTasks) -> JSONResponse:
    
    try:
        user = await get_user_email(req.email)
        logger.info("controller reached here")
        if not user : 
            return JSONResponse(
                status_code=401,
                content="User not found"
            )

        token = generate_url_safe_token()
        expiry = datetime.now(timezone.utc) + timedelta(milliseconds=settings.RESET_PASSWORD_EXPIRES)

        updated_user = await update_reset_password(str(user.id), token, expiry)

        background_tasks.add_task(send_reset_email_background_task, background_tasks, updated_user.email, updated_user.name, token)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Password reset email send successfully",
                "details": "Please check your email for reset instructions",
                "email": req.email,
            }
        )
    except Exception as e:
        logger.error(f"Found error while sending reset email: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "details": "An error occurred while processing the forgot password request"
            }
        )

@auth.get("/reset-password")
async def reset_password_token_check(token: str) -> JSONResponse:
    try:
        user: User = await db["users"].find_one({"resetPasswordToken": token})
        if not user:
            return JSONResponse(
                status_code=404,
                content={"error": "Invalid token", "details": "Token not found"}
            )
        email: EmailStr = user["email"]
        expiry = user.get("resetPasswordExpiry")
        if not expiry or datetime.now(timezone.utc) > expiry:
            return JSONResponse(
                status_code=400,
                content={"error": "Token expired", "details": "Please request a new reset link"}
            )

        return JSONResponse(
            status_code=200,
            content={
                "message": "Token is valid",
                "email": email,
            }
        )

    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Server error", "details": "Something went wrong"}
        )

@auth.post("/reset-password")
async def reset_password(token: str, body: ResetPasswordBody) -> JSONResponse:
    try:
        user =         await db["users"].find_one({
            "resetPasswordToken": token,
            "resetPasswordExpiry": {"$gt": datetime.now(timezone.utc)}
        })

        if not user:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid or expired token",
                    "details": "Password reset token is invalid or expired"
                }
            )

        hashed_password = hash_password(body.password)

        await db["users"].update_one(
            {"_id": user["_id"]},
            {
                "$set": {"password": hashed_password},
                "$unset": {"resetPasswordToken": "", "resetPasswordExpiry": ""}
            }
        )

        return JSONResponse(
            status_code=200,
            content={
                "message": "Password reset successful",
            }
        )

    except HTTPException as http_err:
        raise http_err

    except Exception as error:
        logger.error(f"Reset password error: {error}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Server error",
                "details": "An error occurred while resetting password"
            }
        )
