"""
Common decorators for error handling, validation, and logging
"""

import functools
import logging
from typing import Dict, Any, Callable
from fastapi import HTTPException
import stripe
from app.config.credit_constants import ErrorCodes

logger = logging.getLogger(__name__)

def handle_stripe_errors(func: Callable) -> Callable:
    """Decorator to handle Stripe-specific errors consistently"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except stripe.error.CardError as e:
            logger.error(f"Stripe card error in {func.__name__}: {e}")
            raise HTTPException(status_code=400, detail=f"Card error: {str(e)}")
        except stripe.error.RateLimitError as e:
            logger.error(f"Stripe rate limit in {func.__name__}: {e}")
            raise HTTPException(status_code=429, detail="Rate limit exceeded, please try again later")
        except stripe.error.InvalidRequestError as e:
            logger.error(f"Stripe invalid request in {func.__name__}: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
        except stripe.error.AuthenticationError as e:
            logger.error(f"Stripe auth error in {func.__name__}: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
        except stripe.error.APIConnectionError as e:
            logger.error(f"Stripe connection error in {func.__name__}: {e}")
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        except stripe.error.StripeError as e:
            logger.error(f"General Stripe error in {func.__name__}: {e}")
            raise HTTPException(status_code=400, detail=f"Payment processing error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    return wrapper

def handle_credit_operations(func: Callable) -> Callable:
    """Decorator to handle credit operation errors consistently"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValueError as e:
            error_msg = str(e).lower()
            if "insufficient" in error_msg:
                return {
                    "success": False,
                    "error": ErrorCodes.INSUFFICIENT_CREDITS,
                    "message": str(e)
                }
            raise
        except Exception as e:
            logger.error(f"Credit operation error in {func.__name__}: {e}")
            return {
                "success": False,
                "error": "CREDIT_OPERATION_FAILED",
                "message": f"Credit operation failed: {str(e)}"
            }
    
    return wrapper

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper

def validate_user_access(func: Callable) -> Callable:
    """Decorator to validate user access and existence"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract user_id from args or kwargs
        user_id = None
        
        # Check kwargs first
        if 'user_id' in kwargs:
            user_id = kwargs['user_id']
        else:
            # For methods, args[0] is self, so look for user_id in later args
            # Common patterns: method(self, user_id) or method(self, user, user_id)
            for arg in args[1:]:  # Skip self (args[0])
                if isinstance(arg, str) and len(arg) == 24:  # MongoDB ObjectId length
                    try:
                        from bson import ObjectId
                        ObjectId(arg)
                        user_id = arg
                        break
                    except:
                        continue
        
        if not user_id:
            # Silently skip validation if user_id not found - it might not be needed
            return await func(*args, **kwargs)
        
        # Validate ObjectId format
        from bson import ObjectId
        try:
            ObjectId(user_id)
        except:
            raise HTTPException(status_code=400, detail="Invalid user ID format")
        
        return await func(*args, **kwargs)
    
    return wrapper
