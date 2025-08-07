from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

class CreditPack(BaseModel):
    credits: float
    price: float
    discountPercentage: Optional[float] = None

class BasePackage(BaseModel):
    price: float
    credits: float

class CustomCredits(BaseModel):
    minPrice: float
    minCredits: float
    pricePerCredit: float

class PricingDetails(BaseModel):
    creditPacks: Optional[List[CreditPack]] = None
    basePackage: Optional[BasePackage] = None
    customCredits: Optional[CustomCredits] = None

class StripePriceIds(BaseModel):
    monthly: Optional[str] = None
    yearly: Optional[str] = None
    creditPack25: Optional[str] = None
    creditPack250: Optional[str] = None
    creditPack625: Optional[str] = None
    creditPack1250: Optional[str] = None
    creditPack2500: Optional[str] = None
    creditPack5000: Optional[str] = None
    customCredits: Optional[str] = None

class Limits(BaseModel):
    uploadsPerDay: Optional[int] = None
    maxFileSize: Optional[float] = None  # MB
    maxDuration: Optional[float] = None  # minutes
    maxConcurrentJobs: Optional[int] = None
    requiresCaptcha: bool = False
    showsAds: bool = False
    priority: Optional[Literal['low', 'medium', 'high']] = None
    hasLimitedApiAccess: bool = False
    batchProcessing: bool = False
    fullApiAccess: bool = False

class Pricing(BaseModel):
    id: Optional[str] = None
    name: Literal['free', 'premium', 'pro', 'registered']
    description: str
    features: List[str]
    isFree: bool = False
    freeCreditGift: Optional[float] = None
    creditBased: bool = False
    creditsPerMinute: Optional[float] = None
    pricing: Optional[PricingDetails] = None
    stripePriceIds: Optional[StripePriceIds] = None
    limits: Optional[Limits] = None
    outputFormats: Optional[List[str]] = None
    isActive: bool = True
    displayOrder: int
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

    