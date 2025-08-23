from fastapi import HTTPException, Request, Query
from fastapi.security.http import HTTPAuthorizationCredentials
from typing import Optional
import logging
from datetime import datetime, timezone
from app.config.database import db, users_collection
from app.schemas.user import TokenUser

logger = logging.getLogger(__name__)

async def get_user_from_share_token(
    job_id: str,
    share_token: str = Query(None, description="Share token for accessing job")
) -> TokenUser:
    """Validates share token and returns the job owner as authenticated user."""
    if not share_token:
        raise HTTPException(
            status_code=401,
            detail="Share token required"
        )
    
    try:
        # Get share token from database
        share_tokens_collection = db["share_tokens"]
        token_doc = await share_tokens_collection.find_one({
            "token": share_token,
            "job_id": job_id
        })
        
        if not token_doc:
            raise HTTPException(
                status_code=401,
                detail="Invalid share token"
            )
        
        try:
            expires_at = token_doc["expires_at"]
            current_time = datetime.now(timezone.utc)
            
            if hasattr(expires_at, 'tzinfo'):
                if not expires_at.tzinfo:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
            else:
                if isinstance(expires_at, str):
                    from datetime import datetime as dt
                    expires_at = dt.fromisoformat(expires_at.replace('Z', '+00:00'))
            
            if current_time > expires_at:
                await share_tokens_collection.delete_one({"_id": token_doc["_id"]})
                raise HTTPException(
                    status_code=401,
                    detail="Share token has expired"
                )
        except Exception:
            pass
        
        user_id = token_doc["user_id"]
        
        user = await users_collection.find_one({"_id": user_id})
        if not user and isinstance(user_id, str):
            try:
                from bson import ObjectId
                user = await users_collection.find_one({"_id": ObjectId(user_id)})
            except Exception:
                pass
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="User not found"
            )
        
        return TokenUser(id=str(user["_id"]), email=user["email"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Share token validation error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Share token validation failed"
        )

def create_share_token_auth(job_id_param: str = "job_id"):
    """Creates dependency that supports both JWT and share token authentication."""
    async def auth_dependency(
        request: Request,
        share_token: Optional[str] = Query(None, description="Share token for accessing job")
    ) -> TokenUser:
        if share_token:
            job_id = request.path_params.get(job_id_param)
            if not job_id:
                raise HTTPException(status_code=400, detail="Job ID not found in path")
            return await get_user_from_share_token(job_id, share_token)
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Authentication required"
            )
        
        token = auth_header.split(" ")[1]
        
        try:
            import jwt
            from app.config.settings import settings
            
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=[settings.ALGORITHM]
            )
            
            user_id = payload.get("id") or payload.get("sub")
            email = payload.get("email")
            
            if not user_id or not email:
                raise ValueError("Invalid token payload")
                
            return TokenUser(id=user_id, email=email)
            
        except Exception as e:
            logger.error(f"JWT validation failed: {e}")
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
    
    return auth_dependency

get_video_dub_user = create_share_token_auth("job_id")
