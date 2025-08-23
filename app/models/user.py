from pydantic import BaseModel, EmailStr
from typing import Optional, Literal
from datetime import datetime

class Subscription(BaseModel):
    type: str = 'pay as you go'  
    status: str = 'none'
    stripeCustomerId: Optional[str] = None
    stripeSubscriptionId: Optional[str] = None
    currentPeriodEnd: Optional[datetime] = None

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
    resetPasswordToken: Optional[str] = None
    resetPasswordExpiry: Optional[datetime] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None