from fastapi import APIRouter, Security, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
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
from app.services.google_auth import verify_google_token, handle_google_user
from app.config.settings import settings
from authlib.integrations.starlette_client import OAuth

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
            status_code=500, 
            detail="Verification failed. Please contact support."
        )

@auth.post("/login")
async def login_user(req: LoginData):
    try:
        user: dict = await get_user(req)
        # logger.info(f"user email : {user["email"]} is verfied: {user["isEmailVerified"]}")
        if user.isEmailVerified == False:
            return JSONResponse(
                status_code=403,
                content={
                    "details": "Email is not verified",
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
        logger.info(f"getting get user. {current_user.id}")
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
        logger.info(f"----> forgot-password {req.email}")
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


GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"
GOOGLE_REDIRECT_URI= f"{settings.BACKEND_URL}/api/auth/google/callback"

from urllib.parse import urlencode
import httpx

@auth.get("/google")
async def google_login(request: Request):
    """Initiates Google OAuth flow"""
    try:
        query_params = {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "prompt": "consent",
        }
        url = f"{GOOGLE_AUTH_ENDPOINT}?{urlencode(query_params)}"
        return RedirectResponse(url)
    
    except Exception as e:
        logger.error(f"Google login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initiate Google login")

@auth.get("/google/callback")
async def google_callback(request: Request):
    try:
        code = request.query_params.get("code")
        if not code:
            raise HTTPException(status_code=400, detail="Missing authorization code")
        
        data = {
            "code": code,
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        }
        
        async with httpx.AsyncClient() as client:
            token_response = await client.post(GOOGLE_TOKEN_ENDPOINT, data=data)
            token_data = token_response.json()
            access_token = token_data.get("access_token")

            if not access_token:
                raise HTTPException(status_code=400, detail="Failed to retrieve access token")

            headers = {"Authorization": f"Bearer {access_token}"}
            userinfo_response = await client.get(GOOGLE_USERINFO_ENDPOINT, headers=headers)
            user_info = userinfo_response.json()

        logger.info(f"userinfo: {user_info}")
        
        try:
            # Try to find user by email
            user = await get_user_email(user_info['email'])
            user = user.dict()
        except HTTPException:
            new_user = {
                "email": user_info['email'],
                "name": user_info.get('name'),
                "password": None,
                "profilePicture": user_info.get('picture'),
                "isEmailVerified": True,
            }

            user = await create_user(new_user)

        # Generate JWT token
        jwt_token = create_jwt_token({
            "id": user["id"],
            "email": user["email"]
        })

        # Redirect to frontend with token
        redirect_url = f"{settings.FRONTEND_URL}/auth/google/callback?token={jwt_token}"
        return RedirectResponse(url=redirect_url, status_code=302)
    
    except Exception as e:
        logger.error(f"Google callback error: {str(e)}")
        return RedirectResponse(url=f"{settings.FRONTEND_URL}/auth")

