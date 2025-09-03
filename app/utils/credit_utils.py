"""
Simple credit calculation utilities for $10 threshold billing
"""

from typing import Dict, Any
from datetime import datetime, timezone
from app.config.credit_constants import CreditRates, JobType
import logging

logger = logging.getLogger(__name__)

class CreditCalculatorUtil:
    """Simple credit calculation for threshold billing"""
    
    @staticmethod
    def calculate_job_credits(job_type: JobType, duration_seconds: float) -> float:
        """Calculate credits required for a job"""
        if duration_seconds < 0:
            raise ValueError(f"Duration cannot be negative: {duration_seconds}")
        
        if job_type == JobType.DUB:
            credits = duration_seconds * CreditRates.DUB_RATE_PER_SECOND
        elif job_type == JobType.SEPARATION:
            duration_minutes = duration_seconds / 60.0
            credits = duration_minutes * CreditRates.SEPARATION_RATE_PER_MINUTE
        else:
            raise ValueError(f"Unknown job type: {job_type}")
        
        return round(credits, 2)
    
    @staticmethod
    def calculate_cost_estimate(credits: float) -> float:
        """Calculate estimated cost in USD for given credits"""
        return round(credits * CreditRates.COST_PER_CREDIT_USD, 2)

class UserValidationUtil:
    """Simple user validation utilities"""
    
    @staticmethod
    def is_payg_user(user_data: Dict[str, Any]) -> bool:
        """Check if user is pay-as-you-go with active subscription"""
        subscription = user_data.get("subscription", {})
        return (
            subscription.get("type") == "pay as you go" and
            subscription.get("status") == "active"
        )
    
    @staticmethod
    def has_sufficient_credits(user_data: Dict[str, Any], required_credits: float) -> tuple[bool, float]:
        """Check if user has sufficient credits and return availability status with credit amount"""
        available_credits = user_data.get("credits", 0.0)
        has_sufficient = available_credits >= required_credits
        return has_sufficient, available_credits

class ResponseUtil:
    """Simple API response formatting utilities"""
    
    @staticmethod
    def create_success_response(message: str, **extra_data) -> Dict[str, Any]:
        """Create standardized success response"""
        response = {
            "success": True,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        response.update(extra_data)
        return response
    
    @staticmethod
    def create_error_response(error_code: str, message: str, **extra_data) -> Dict[str, Any]:
        """Create standardized error response"""
        response = {
            "success": False,
            "error_code": error_code,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        response.update(extra_data)
        return response

# Simple utility functions
def calculate_threshold_cost(credits: float) -> float:
    """Calculate cost for threshold billing"""
    return credits * CreditRates.COST_PER_CREDIT_USD

def is_threshold_reached(total_credits: float, threshold_usd: float = 10.0) -> bool:
    """Check if usage has reached billing threshold"""
    cost = calculate_threshold_cost(total_credits)
    return cost >= threshold_usd
