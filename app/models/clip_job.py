from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List
from datetime import datetime, timezone

class ClipSegment(BaseModel):
    start: float
    end: float
    reason: Optional[str] = None
    ratings: Optional[Dict[str, Any]] = None
    clip_url: Optional[str] = None
    local_path: Optional[str] = None
    text: Optional[str] = None
    words: Optional[List[Dict[str, Any]]] = None
    sentences: Optional[List[Dict[str, Any]]] = None

class ClipJob(BaseModel):
    id: Optional[str] = None
    job_id: str = Field(..., description="Unique job ID")
    user_id: str = Field(..., description="User who submitted the job")
    
    status: Literal['pending', 'downloading', 'trimming', 'transcribing', 'segmenting', 'rendering', 'uploading', 'completed', 'failed'] = 'pending'
    progress: int = Field(0, ge=0, le=100)
    error_message: Optional[str] = None
    
    video_url: str = Field(..., description="R2 URL of source video")
    srt_url: Optional[str] = None
    start_time: float
    end_time: float
    expected_duration: float
    
    subtitle_style: Optional[str] = None
    subtitle_preset: Optional[str] = "reels"
    subtitle_font: Optional[str] = None
    subtitle_font_size: Optional[int] = None
    subtitle_wpl: Optional[int] = None
    
    transcript: Optional[str] = None
    segments: Optional[List[ClipSegment]] = Field(default_factory=list)
    overall_rating: Optional[Dict[str, Any]] = None
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
