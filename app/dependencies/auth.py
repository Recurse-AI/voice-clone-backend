from fastapi import HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.utils.token_helper import decode_jwt_token
from app.utils.logger import logger 
from app.schemas.user import TokenUser

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> TokenUser:
    try:
        token = credentials.credentials
        if not token:
            raise HTTPException(
                status_code=403,
                detail="Not authenticated"
            )
        payload = decode_jwt_token(token)
        return TokenUser(**payload)

    except Exception as e:
        logger.error(f"Token error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )