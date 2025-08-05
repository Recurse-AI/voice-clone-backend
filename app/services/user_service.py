from app.config.database import db
from app.schemas.user import *
from app.models.user import *
from app.utils.password_helper import hash_password
from datetime import datetime, timezone
from fastapi import HTTPException
from app.utils.password_helper import verify_password
from app.utils.logger import logger
from bson import ObjectId
from datetime import datetime, timezone
from pymongo import ReturnDocument

async def create_user(user_dict: dict):
    user_dict["password"] = hash_password(user_dict["password"])
    user_dict["createdAt"] = datetime.now(timezone.utc)
    user_dict["updatedAt"] = datetime.now(timezone.utc)
    user_dict["isEmailVerified"] = False
    user_dict["credits"] = 25  # New users get 25 free credits
    user_dict["role"] = "user"
    user_dict["verificationAttempts"] = 0

    result = await db["users"].insert_one(user_dict)
    user_dict["id"] = str(result.inserted_id)
    user_dict.pop("password", None)
    return user_dict

async def get_user(data: LoginData) -> User:
    user_data = await db["users"].find_one({"email": data.email})
    
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(data.password, user_data.get("password", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_data["id"] = str(user_data["_id"])
    user_data["_id"] = user_data["id"]
    user_data.pop("password", None)
    
    return User(**user_data)

async def get_user_id(id: str) -> User:
    user_data = await db["users"].find_one({"_id": ObjectId(id)})
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid id")

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
    user_data = await db["users"].find_one({"email": email})
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid id")

    logger.info(f"user find by email: {user_data}")
    user_data["id"] = str(user_data["_id"])
    user_data["_id"] = user_data["id"]
    user_data.pop("password", None)
    
    return User(**user_data)

async def update_reset_password(id: str, token: str, expiry: datetime) -> User:
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