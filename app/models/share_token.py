from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone

class ShareToken(BaseModel):
    """Share token model for job sharing"""
    token: str = Field(..., description="Share token string")
    job_id: str = Field(..., description="Associated job ID")
    user_id: str = Field(..., description="Original job owner")
    expires_at: datetime = Field(..., description="Token expiration time")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Config:
        collection_name = "share_tokens"
