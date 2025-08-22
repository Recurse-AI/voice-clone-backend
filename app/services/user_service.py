from app.config.database import db
from app.schemas.user import *
from app.models.user import *
from app.utils.password_helper import hash_password
from datetime import datetime, timezone
from fastapi import HTTPException
from app.utils.password_helper import verify_password
import logging
from bson import ObjectId

logger = logging.getLogger(__name__)
from datetime import datetime, timezone
from pymongo import ReturnDocument

async def create_user(user_dict: dict):
    if user_dict.get("password") is not None:  # Only hash if password exists
        user_dict["password"] = hash_password(user_dict["password"])
    
    # Rest of your existing code stays exactly the same
    user_dict["createdAt"] = datetime.now(timezone.utc)
    user_dict["updatedAt"] = datetime.now(timezone.utc)
    user_dict["isEmailVerified"] = False if "isEmailVerified" not in user_dict else user_dict["isEmailVerified"]
    user_dict["credits"] = 25  # New users get 25 free credits
    user_dict["role"] = "user"
    user_dict["verificationAttempts"] = 0
    
    # Initialize default subscription for new users
    user_dict["subscription"] = {
        "type": "free",
        "status": "none",
        "stripeCustomerId": None,
        "stripeSubscriptionId": None,
        "currentPeriodEnd": None
    }

    result = await db["users"].insert_one(user_dict)
    user_dict["id"] = str(result.inserted_id)
    user_dict.pop("password", None)
    return user_dict

async def get_user(data: LoginData) -> User:
    user_data = await db["users"].find_one({"email": data.email})
    logger.info(f"----> poniter reached here.")
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(data.password, user_data.get("password", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_data["id"] = str(user_data["_id"])
    user_data["_id"] = user_data["id"]
    user_data.pop("password", None)
    
    return User(**user_data)

async def get_user_id(id: str) -> User:
    try:
        # Validate ObjectId format first
        if not ObjectId.is_valid(id):
            raise HTTPException(status_code=401, detail="Invalid id format")
        
        user_data = await db["users"].find_one({"_id": ObjectId(id)})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error(f"Error finding user with id {id}: {e}")
        if "User not found" in str(e):
            raise HTTPException(status_code=404, detail="User not found")
        raise HTTPException(status_code=401, detail="Invalid id format")

    user_data["id"] = str(user_data["_id"])
    user_data["_id"] = user_data["id"]
    user_data.pop("password", None)
    
    return User(**user_data)

async def update_user_name(id: str, data: UpdateProfileRequest) -> User:
    user_data = await db["users"].find_one_and_update(
        {"_id": ObjectId(id)},  # filter
        {
            "$set": {
                "name": data.name,
                "updatedAt": datetime.now(timezone.utc)
            }
        },
        return_document=ReturnDocument.AFTER  # tells Motor to return updated doc
    )

    if not user_data:
        logger.error(f"User not found with id {str(id)}")
        raise HTTPException(
            status_code=404,
            detail="Invalid id"
        )

    user_data["id"] = str(user_data["_id"])
    user_data["_id"] = user_data["id"]
    user_data.pop("password", None)

    return User(**user_data)

async def get_user_email(email: EmailStr) -> User:
    try:
        user_data = await db["users"].find_one({"email": email})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found with this email")
    except Exception as e:
        logger.error(f"Error finding user with email {email}: {e}")
        raise HTTPException(status_code=404, detail="User not found with this email")

    logger.info(f"user find by email: {user_data}")
    user_data["id"] = str(user_data["_id"])
    user_data["_id"] = user_data["id"]
    user_data.pop("password", None)
    
    return User(**user_data)

async def update_reset_password(id: str, token: str, expiry: datetime) -> User:
    try:
        # Validate ObjectId format first
        if not ObjectId.is_valid(id):
            raise HTTPException(status_code=401, detail="Invalid id format")
        
        update_result = await db["users"].find_one_and_update(
            {"_id": ObjectId(id)},
            {
                "$set": {
                    "resetPasswordToken": token,
                    "resetPasswordExpiry": expiry,
                    "updatedAt": datetime.now(timezone.utc)
                }
            },
            return_document=ReturnDocument.AFTER  # tells 
        )
        if not update_result:
            raise HTTPException(
                status_code=500,
                detail="Failed to update reset token"
            )
        logger.info("Token and expiry date updated")
        return User(**update_result)
    except Exception as e:
        logger.error(f"Error updating reset password for id {id}: {e}")
        raise HTTPException(status_code=401, detail="Invalid id")