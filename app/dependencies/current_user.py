from fastapi import Request, HTTPException, status
from typing import Dict, Any

def get_current_user(request: Request) -> Dict[str, Any]:
    """Get current user from request (set by auth middleware)"""
    user = getattr(request.state, "user", None) or request.scope.get("user")
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    return user