from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import jwt
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from app.config.settings import settings
from app.config.database import users_collection, db

logger = logging.getLogger(__name__)

class AuthMiddleware:
    """Enhanced authentication middleware that handles both JWT tokens and share tokens"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Check if this endpoint requires authentication
        if self._requires_auth(request.url.path):
            try:
                # Get user (either from share token or JWT)
                user = await self._get_user_from_token(request)
                if not user:
                    response = JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Invalid or missing authentication token"}
                    )
                    await response(scope, receive, send)
                    return
                
                # Add user to request state
                scope["user"] = user
                
            except Exception as e:
                logger.error(f"Authentication error: {e}")
                response = JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Authentication failed"}
                )
                await response(scope, receive, send)
                return
        
        await self.app(scope, receive, send)
    
    def _requires_auth(self, path: str) -> bool:
        """Check if the path requires authentication"""
        protected_paths = [
            "/api/jobs/",           # User job APIs
            "/api/video-dub/",      # Video dubbing segments, actions
            "/api/audio-separation", # Audio separation
            "/upload-file",         # File uploads
            "/api/voice-clone-segment", # Voice cloning

            "/api/dubbing/",        # Dubbing APIs
            "/api/stripe/",         # Stripe APIs (except webhooks)
        ]
        
        # Skip auth for specific endpoints
        skip_auth = [
            "/",
            "/api/download-video",  # Public download
            "/api/process-video-complete",  # Public video processing
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/auth/",           # Auth endpoints themselves
        ]
        
        # Check if path should skip auth
        for skip_path in skip_auth:
            if path.startswith(skip_path):
                return False
        
        # Check if path requires auth
        for protected_path in protected_paths:
            if path.startswith(protected_path):
                return True
        
        return False
    

    

    
    async def _get_user_from_token(self, request: Request) -> Optional[Dict[str, Any]]:
        """Extract user from JWT token"""
        try:
            # JWT authentication only
            authorization = request.headers.get("Authorization")
            if not authorization or not authorization.startswith("Bearer "):
                return None
            
            token = authorization.split(" ")[1]
            
            # Decode JWT token
            try:
                payload = jwt.decode(
                    token, 
                    settings.SECRET_KEY, 
                    algorithms=[settings.ALGORITHM]
                )
                user_id = payload.get("sub")
                if not user_id:
                    return None
                
            except jwt.ExpiredSignatureError:
                logger.warning("Token has expired")
                return None
            except jwt.InvalidTokenError as e:
                logger.warning(f"Invalid token: {e}")
                return None
            
            # Get user from database
            return await self._get_user_by_id(user_id)
            
        except Exception as e:
            logger.error(f"Error extracting user from token: {e}")
            return None
    

    
    async def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Common user lookup logic - reusable"""
        try:
            user = await users_collection.find_one({"_id": user_id})
            if not user:
                return None
                
            # Convert ObjectId to string for JSON serialization
            user["id"] = str(user["_id"])
            del user["_id"]
            
            # Remove password field for security
            if "password" in user:
                del user["password"]
            
            return user
        except Exception:
            return None

# Helper function to get current user from request
def get_current_user_from_request(request: Request) -> Optional[Dict[str, Any]]:
    """Get current user from request scope (set by middleware)"""
    return getattr(request.state, "user", None) or request.scope.get("user")