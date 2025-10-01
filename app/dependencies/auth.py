from fastapi import HTTPException, Depends, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.utils.token_helper import decode_jwt_token
import logging 
from app.schemas.user import TokenUser

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> TokenUser:
    # Check if user was already set by middleware (share token or normal auth)
    user = request.scope.get("user")
    if user:
        return TokenUser(id=user["id"], email=user["email"])
    
    if not credentials:
        raise HTTPException(
            status_code=403,
            detail="Not authenticated"
        )
        
    try:
        token = credentials.credentials
        if not token:
            raise HTTPException(
                status_code=403,
                detail="Not authenticated"
            )
        payload = decode_jwt_token(token)
        return TokenUser(**payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )
    