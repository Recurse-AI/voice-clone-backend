"""
Response Helper Utilities
Centralized error and success response formatting
"""

from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from app.config.constants import *


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