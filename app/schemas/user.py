from pydantic import BaseModel, EmailStr
from typing import Optional, Literal, Union, Dict, Any
from datetime import datetime

class Subscription(BaseModel):
    type: Literal['free', 'pay as you go', 'credit pack', 'premium', 'pro'] = 'free'
    status: Literal['active', 'trialing', 'past_due', 'canceled', 'cancelled', 'none'] = 'none'
    stripeCustomerId: Optional[str] = None
    stripeSubscriptionId: Optional[str] = None
    currentPeriodEnd: Optional[datetime] = None
    cancelledAt: Optional[datetime] = None

class UserBase(BaseModel):
    name: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class LoginData(BaseModel):
    email: EmailStr
    password: str

class SpendingLimit(BaseModel):
    amount: float
    period: Literal['daily', 'weekly', 'monthly'] = 'monthly'
    currentSpent: float = 0.0
    periodStartDate: Optional[datetime] = None

class UserOut(UserBase):
    id: Optional[str]
    isEmailVerified: bool
    profilePicture: Optional[str] = None
    role: Literal['user', 'admin'] = 'user'
    credits: float = 0.0

    class Config:
        from_attributes = True

class FullUser(UserOut):
    subscription: Optional[Union[Subscription, Dict[str, Any]]] = None
    spendingLimit: Optional[SpendingLimit] = None 
    hasPaymentMethod: bool = False
    paymentMethodAddedAt: Optional[datetime] = None

class TokenUser(BaseModel):
    id: str
    email: EmailStr

class UpdateProfileRequest(BaseModel):
    name: str

class ResetPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordBody(BaseModel):
    password: str

