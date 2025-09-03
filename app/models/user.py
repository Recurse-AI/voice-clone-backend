from pydantic import BaseModel, EmailStr
from typing import Optional, Literal, List
from datetime import datetime

class PaymentMethod(BaseModel):
    id: str
    brand: str
    last4: str
    expiryMonth: int
    expiryYear: int
    isDefault: bool = False



class Subscription(BaseModel):
    type: Literal['free', 'pay as you go', 'credit pack'] = 'free'
    status: Literal['active', 'trialing', 'past_due', 'canceled', 'cancelled', 'none'] = 'none'
    stripeCustomerId: Optional[str] = None
    stripeSubscriptionId: Optional[str] = None
    currentPeriodEnd: Optional[datetime] = None
    cancelledAt: Optional[datetime] = None

class User(BaseModel):
    id: Optional[str] = None
    name: str
    email: EmailStr
    password: Optional[str] = None
    isEmailVerified: bool = False
    emailVerificationToken: Optional[str] = None
    emailVerificationExpiry: Optional[datetime] = None
    verificationAttempts: int = 0
    credits: float = 0.0
    googleId: Optional[str] = None
    profilePicture: Optional[str] = None
    role: Literal['user', 'admin'] = 'user'
    subscription: Optional[Subscription] = Subscription()
    total_usage: Optional[float] = None
    paymentMethods: List[PaymentMethod] = []
    hasPaymentMethod: bool = False
    paymentMethodAddedAt: Optional[datetime] = None
    resetPasswordToken: Optional[str] = None
    resetPasswordExpiry: Optional[datetime] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None