"""
Response Helper Utilities
Centralized error and success response formatting with common error handling
"""

from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, Callable
from app.config.constants import *
import logging


def success_response(
    message: str,
    data: Optional[Dict[str, Any]] = None,
    status_code: int = HTTP_STATUS_SUCCESS
) -> JSONResponse:
    """Create standardized success response"""
    content = {
        "success": True,
        "message": message
    }
    
    if data:
        content.update(data)
    
    return JSONResponse(status_code=status_code, content=content)


def error_response(
    message: str,
    error_type: str = "Server error",
    details: Optional[str] = None,
    status_code: int = HTTP_STATUS_INTERNAL_ERROR
) -> JSONResponse:
    """Create standardized error response"""
    content = {
        "success": False,
        "error": error_type,
        "message": message
    }
    
    if details:
        content["details"] = details
    
    return JSONResponse(status_code=status_code, content=content)


def validation_error_response(
    message: str = "Validation failed",
    details: Optional[str] = None
) -> JSONResponse:
    """Create validation error response"""
    return error_response(
        message=message,
        error_type="Validation error",
        details=details,
        status_code=HTTP_STATUS_BAD_REQUEST
    )


def auth_error_response(
    message: str = ERROR_AUTHENTICATION_REQUIRED,
    details: Optional[str] = None
) -> JSONResponse:
    """Create authentication error response"""
    return error_response(
        message=message,
        error_type="Authentication error",
        details=details,
        status_code=HTTP_STATUS_UNAUTHORIZED
    )


def not_found_response(
    message: str = "Resource not found",
    details: Optional[str] = None
) -> JSONResponse:
    """Create not found error response"""
    return error_response(
        message=message,
        error_type="Not found",
        details=details,
        status_code=HTTP_STATUS_NOT_FOUND
    )


def forbidden_response(
    message: str = "Access denied",
    details: Optional[str] = None
) -> JSONResponse:
    """Create forbidden error response"""
    return error_response(
        message=message,
        error_type="Access denied",
        details=details,
        status_code=HTTP_STATUS_FORBIDDEN
    )


def handle_errors(func_name: str, logger: Optional[logging.Logger] = None):
    """
    Decorator for common error handling pattern
    Reduces repetitive try/catch blocks
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.error(f"{func_name} failed: {str(e)}")
                return error_response(
                    message=f"{func_name} failed",
                    details=str(e),
                    status_code=HTTP_STATUS_INTERNAL_ERROR
                )
        return wrapper
    return decorator


def log_and_continue(func_name: str, logger: Optional[logging.Logger] = None):
    """
    Decorator that logs errors but doesn't change return behavior
    Useful for background tasks where we want to log but continue
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.error(f"{func_name} error: {str(e)}")
                raise  # Re-raise to maintain original behavior
        return wrapper
    return decorator


def validate_user_id(user_id: str) -> tuple:
    """
    Common user ID validation utility
    Returns (is_valid, error_message)
    """
    if not user_id or not isinstance(user_id, str) or len(user_id.strip()) == 0:
        return False, "Invalid user ID"
    return True, None


def validate_required_field(field_value: Any, field_name: str) -> tuple:
    """
    Common required field validation
    Returns (is_valid, error_message)
    """
    if field_value is None or (isinstance(field_value, str) and len(field_value.strip()) == 0):
        return False, f"{field_name} is required"
    return True, None