from app.models.user import User 
import base64
import os
from datetime import datetime, timezone, timedelta
from app.config.settings import settings
import jwt
import logging

logger = logging.getLogger(__name__)


def generate_url_safe_token(byte_length: int = 16) -> str:
    token = base64.b64encode(os.urandom(byte_length)).decode("utf-8")
    token = token.replace('+', '-').replace('/', '_').rstrip('=')
    return token

def create_jwt_token(user: dict) -> str:
    SECRET_KEY = settings.SECRET_KEY
    ALGORITHM = settings.ALGORITHM
    SINGIN_TOKEN_EXPIRES = settings.SINGIN_TOKEN_EXPIRES
    payload = {
        "id": user["id"],
        "email": user["email"],
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=SINGIN_TOKEN_EXPIRES)
    }
    token = jwt.encode(payload=payload, key=SECRET_KEY, algorithm=ALGORITHM)
    return token

def decode_jwt_token(token: str) -> dict:
    SECRET_KEY = settings.SECRET_KEY
    ALGORITHM = settings.ALGORITHM
    try:
        payload = jwt.decode(
            jwt=token,
            key=SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        
        return {
            "id": payload["id"],
            "email": payload["email"]
        }
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        raise 
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {str(e)}")
        raise 
        
    