from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

class Dubbing(BaseModel):
    id: Optional[str] = None  # Better to keep as str if MongoDB ObjectId
    user: str  # User id (ObjectId)
    job_id: str
    project_title: str = "Untitled project"
    source_language: str = "Auto Detect"
    target_language: str = "English"
    speakers: int = 1
    fileDuration: float = 0
    startTime: float = 0
    endTime: float = 0
    mediaType: Literal["audio", "video"] = "audio"
    isFullVideoCreditDeducted: bool = False 
    deductCredits: float = 0
    createdAt: Optional[datetime] = None


