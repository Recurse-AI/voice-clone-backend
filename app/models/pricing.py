from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

class CreditPack(BaseModel):
    name: str
    credits: float
    originalPrice: float
    discountedPrice: float
    pricePerCredit: float
    savingsPercentage: int
    stripePriceId: Optional[str] = None

class Pricing(BaseModel):
    id: Optional[str] = None
    name: Literal['free', 'small', 'medium', 'special', 'limited']
    description: str
    features: List[str]
    creditPack: Optional[CreditPack] = None
    isActive: bool = True
    displayOrder: int
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None