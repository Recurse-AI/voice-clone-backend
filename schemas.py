from pydantic import BaseModel
from typing import Optional, Dict, Any

class ProcessingResponse(BaseModel):
    success: bool
    audio_id: str
    message: str
    processing_details: Optional[Dict[str, Any]] = None
    r2_storage: Optional[Dict[str, Any]] = None
    final_audio_url: Optional[str] = None
    subtitles_url: Optional[str] = None
    video_url: Optional[str] = None
    original_audio_details: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    status: str
    message: str
    audio_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class StartProcessingResponse(BaseModel):
    success: bool
    audio_id: str
    message: str
    status: str
    estimated_time: str
    status_check_url: str 