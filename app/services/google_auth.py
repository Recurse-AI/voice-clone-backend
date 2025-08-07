from google.oauth2 import id_token
from google.auth.transport import requests
from fastapi import HTTPException
from app.config.settings import settings
from app.utils.logger import logger
from app.services.user_service import get_user_email, create_user
from datetime import datetime, timezone

async def verify_google_token(token: str, client_id: str):
    try:
        idinfo = id_token.verify_oauth2_token(
            token, 
            requests.Request(), 
            client_id
        )

        if idinfo['aud'] != client_id:
            raise ValueError('Wrong audience.')
            
        return idinfo
    except ValueError as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

async def handle_google_user(google_data: dict):
    try:
        # Try to find user by email
        email = google_data.get('email')
        try:
            user = await get_user_email(email)
            # User exists, update if needed
            return user
        except HTTPException:
            # User doesn't exist, create new one
            new_user = {
                "email": email,
                "name": google_data.get('name'),
                "password": None,  # Google users don't need password
                "profilePicture": google_data.get('picture'),
                "isEmailVerified": True,  # Google emails are verified
                "credits": 25,  # Default credits for new users
                "role": "user",
                "verificationAttempts": 0,
                "subscription": {
                    "type": "free",
                    "status": "none",
                    "stripeCustomerId": None,
                    "stripeSubscriptionId": None,
                    "currentPeriodEnd": None
                }
            }
            
            created_user = await create_user(new_user)
            # Convert dict to User object
            return await get_user_email(email)  # This will return a User object

    except Exception as e:
        logger.error(f"Google auth error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")
