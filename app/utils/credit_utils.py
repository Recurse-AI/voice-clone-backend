"""
Credit calculation and validation utilities
Centralized, reusable credit-related functions
"""

from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timezone, timedelta
from app.config.credit_constants import (
    CreditRates, CreditPackDiscount, SubscriptionType, 
    BillingType, JobCollection, SpendingPeriod, JobType, BillingPeriod, DefaultLimits
)
import logging

logger = logging.getLogger(__name__)

class CreditCalculatorUtil:
    """Improved credit calculation utility with validation"""
    
    @staticmethod
    def calculate_job_credits(job_type: JobType, duration_seconds: float) -> float:
        """
        Calculate credits required for a job
        
        Args:
            job_type: Type of job (DUB or SEPARATION)
            duration_seconds: Duration in seconds
            
        Returns:
            Required credits (rounded to 2 decimal places)
            
        Raises:
            ValueError: If duration is negative or job_type is invalid
        """
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
    
    @staticmethod
    def calculate_custom_credits(price: float) -> Tuple[float, int]:
        """
        Calculate credits for custom price with discount tiers
        
        Returns:
            Tuple of (total_credits, discount_percentage)
        """
        if price <= 0:
            raise ValueError("Price must be positive")
        
        # Find discount tier
        discount = 0
        for tier in CreditPackDiscount.TIERS:
            if price >= tier["min_price"]:
                if "max_price" not in tier or price <= tier["max_price"]:
                    discount = tier["discount"]
        
        # Calculate credits
        base_credits = price * CreditPackDiscount.CREDITS_PER_DOLLAR
        bonus_credits = base_credits * discount / 100
        total_credits = base_credits + bonus_credits
        
        return round(total_credits, 2), discount

class UserValidationUtil:
    """User validation utilities"""
    
    @staticmethod
    def validate_subscription_type(subscription_type: str) -> bool:
        """Validate subscription type"""
        return subscription_type in [SubscriptionType.CREDIT_PACK, SubscriptionType.PAY_AS_YOU_GO]
    
    @staticmethod
    def validate_spending_period(period: str) -> bool:
        """Validate spending limit period"""
        return period in SpendingPeriod.get_all()
    
    @staticmethod
    def is_payg_user(user_data: Dict[str, Any]) -> bool:
        """Check if user is pay-as-you-go with active subscription"""
        subscription = user_data.get("subscription", {})
        return (
            subscription.get("type") == SubscriptionType.PAY_AS_YOU_GO and
            subscription.get("status") == "active" and
            subscription.get("stripeSubscriptionId") is not None
        )
    
    @staticmethod
    def has_sufficient_credits(user_data: Dict[str, Any], required_credits: float) -> Tuple[bool, float]:
        """
        Check if user has sufficient credits
        
        Returns:
            Tuple of (has_sufficient, available_credits)
        """
        available_credits = user_data.get("credits", 0.0)
        return available_credits >= required_credits, available_credits

class DatabaseUtil:
    """Database operation utilities"""
    
    @staticmethod
    def get_collection_name(job_type: JobType) -> str:
        """Get collection name for job type"""
        return JobCollection.DUB_JOBS if job_type == JobType.DUB else JobCollection.SEPARATION_JOBS
    
    @staticmethod
    def create_job_data(
        job_id: str, user_id: str, credits_required: float,
        billing_type: str, **extra_fields
    ) -> Dict[str, Any]:
        """Create standardized job data structure"""
        from datetime import datetime, timezone
        
        job_data = {
            "job_id": job_id,
            "user_id": user_id,
            "credits_required": credits_required,
            "credits_reserved": True,
            "billing_type": billing_type,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        
        # Add billing-specific fields
        if billing_type == BillingType.PAY_AS_YOU_GO:
            job_data.update({
                "virtual_reservation": True,
                "credits_deducted": False
            })
        elif billing_type == BillingType.CREDIT_PACK:
            job_data.update({
                "credits_deducted": True,
                "virtual_reservation": False
            })
        
        # Add extra fields
        job_data.update(extra_fields)
        return job_data
    
    @staticmethod
    def create_credit_info(job_type: JobType, credits_amount: float, **extra_info) -> Dict[str, Any]:
        """Create standardized credit information structure"""
        from datetime import datetime, timezone
        
        credit_info = {
            "credits_confirmed": True,
            "confirmed_at": datetime.now(timezone.utc).isoformat(),
            "job_type": job_type.value,
            "credits_amount": credits_amount
        }
        credit_info.update(extra_info)
        return credit_info

class ResponseUtil:
    """API response formatting utilities"""
    
    @staticmethod
    def create_success_response(message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create standardized success response"""
        response = {
            "success": True,
            "message": message
        }
        if data:
            response["data"] = data
        return response
    
    @staticmethod
    def create_error_response(
        error_code: str, message: str, 
        details: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create standardized error response"""
        response = {
            "success": False,
            "error": error_code,
            "message": message
        }
        if details:
            response["details"] = details
        return response
    
    @staticmethod
    def create_credit_operation_response(
        success: bool, message: str, **extra_data
    ) -> Dict[str, Any]:
        """Create standardized credit operation response"""
        response = {
            "success": success,
            "message": message
        }
        response.update(extra_data)
        return response

class SpendingLimitUtil:
    """Spending limit utilities"""
    
    @staticmethod
    def calculate_period_start(period: str) -> datetime:
        """Calculate period start date based on period type"""
        
        now = datetime.now(timezone.utc)
        
        if period == SpendingPeriod.DAILY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == SpendingPeriod.WEEKLY:
            days_since_monday = now.weekday()
            return (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == SpendingPeriod.MONTHLY:
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError(f"Unknown period: {period}")
    
    @staticmethod
    def is_within_limit(current_spent: float, limit_amount: float, additional_amount: float) -> bool:
        """Check if additional spending would exceed limit"""
        return (current_spent + additional_amount) <= limit_amount
    
    @staticmethod
    def create_spending_limit_data(amount: float, period: str) -> Dict[str, Any]:
        """Create standardized spending limit data structure"""
        return {
            "amount": amount,
            "period": period,
            "currentSpent": 0.0,
            "periodStartDate": SpendingLimitUtil.calculate_period_start(period)
        }
