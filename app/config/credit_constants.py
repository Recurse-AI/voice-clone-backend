"""
Credit System Constants
Centralized configuration for all credit-related operations
"""

from enum import Enum
from typing import Dict, List

class JobType(Enum):
    """Job types with their credit rates"""
    DUB = "dub"
    SEPARATION = "separation"

class SubscriptionType:
    """User subscription types"""
    CREDIT_PACK = "credit_pack"
    PAY_AS_YOU_GO = "pay as you go"  # User-facing subscription type (with spaces)

class BillingType:
    """Job billing types"""
    CREDIT_PACK = "credit_pack"
    PAY_AS_YOU_GO = "pay_as_you_go"  # Internal billing type (with underscores)

class JobCollection:
    """Database collection names"""
    DUB_JOBS = "dub_jobs"
    SEPARATION_JOBS = "separation_jobs"

class CreditRates:
    """Credit calculation rates"""
    DUB_RATE_PER_SECOND = 0.135  # Increased by 2.7x (0.05 * 2.7)
    SEPARATION_RATE_PER_MINUTE = 2.5  # Increased by 2.5x (1.0 * 2.5)
    COST_PER_CREDIT_USD = 0.04

class SpendingPeriod:
    """Spending limit periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    
    @classmethod
    def get_all(cls) -> List[str]:
        return [cls.DAILY, cls.WEEKLY, cls.MONTHLY]

class BillingPeriod:
    """Pay-as-you-go billing periods"""
    WEEKLY = 7  # days
    PAYG_CYCLE_DAYS = 7  # Pay-as-you-go billing every 7 days

class DefaultLimits:
    """Default spending limits for user types"""
    PAYG_WEEKLY_LIMIT_USD = 50.0  # $50 weekly limit for new PAYG users
    PAYG_DAILY_LIMIT_USD = 10.0   # $10 daily limit option
    NEW_USER_SAFETY_LIMIT = 25.0  # $25 safety limit for first week

class CreditPackDiscount:
    """Credit pack discount tiers"""
    TIERS = [
        {"min_price": 0.00, "max_price": 4.98, "discount": 0},
        {"min_price": 4.99, "max_price": 9.98, "discount": 20},
        {"min_price": 9.99, "max_price": 19.98, "discount": 40},
        {"min_price": 19.99, "max_price": 39.98, "discount": 70},
        {"min_price": 39.99, "discount": 85}
    ]
    CREDITS_PER_DOLLAR = 25

class ErrorCodes:
    """Standard error codes"""
    INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"
    SPENDING_LIMIT_EXCEEDED = "SPENDING_LIMIT_EXCEEDED"
    INACTIVE_SUBSCRIPTION = "INACTIVE_SUBSCRIPTION"
    USER_NOT_FOUND = "USER_NOT_FOUND"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    STRIPE_ERROR = "STRIPE_ERROR"

class StripeConfig:
    """Stripe-related constants"""
    SETUP_INTENT_USAGE = "off_session"
    CHECKOUT_BASE_URL = "https://checkout.stripe.com"
    
class JobStatus:
    """Job status constants"""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
