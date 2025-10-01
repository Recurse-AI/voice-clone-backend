"""
Simple credit system constants for $10 threshold billing
"""

from enum import Enum

class JobType(Enum):
    """Job types with their credit rates"""
    DUB = "dub"
    SEPARATION = "separation"
    CLIP = "clip"

class CreditRates:
    """Credit calculation rates"""
    DUB_RATE_PER_SECOND = 0.135
    SEPARATION_RATE_PER_MINUTE = 2.5
    CLIP_RATE_PER_MINUTE = 3.75
    COST_PER_CREDIT_USD = 0.04

class ThresholdBilling:
    """Simple threshold billing constants"""
    DEFAULT_THRESHOLD_USD = 10.0
    BILLING_CURRENCY = "usd"

class ErrorCodes:
    """Standard error codes"""
    INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"
    USER_NOT_FOUND = "USER_NOT_FOUND"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    STRIPE_ERROR = "STRIPE_ERROR"
    THRESHOLD_NOT_REACHED = "THRESHOLD_NOT_REACHED"

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
