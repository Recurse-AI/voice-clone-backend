from pydantic import BaseModel
from typing import Optional, Dict, Any

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

class RegenerateSegmentRequest(BaseModel):
    text: str
    reference_audio_url: str
    duration: float
    seed: Optional[int] = None
    temperature: Optional[float] = 1.3
    cfg_scale: Optional[float] = 3.0
    top_p: Optional[float] = 0.95

class RegenerateSegmentResponse(BaseModel):
    success: bool
    message: str
    audio_url: Optional[str] = None
    audio_data: Optional[bytes] = None
    duration: Optional[float] = None
    generation_time: Optional[float] = None
    parameters_used: Optional[Dict[str, Any]] = None

class SegmentEdit(BaseModel):
    segment_url: str  # URL to the segment audio (original or regenerated)
    start_time: float  # New start time in seconds
    duration: float    # New duration in seconds
    speaker: str       # Speaker identifier (A, B, etc.)

class ReconstructVideoRequest(BaseModel):
    segments: list[SegmentEdit]  # List of edited segments
    video_url: Optional[str] = None  # Original video URL (optional)
    instruments_url: Optional[str] = None  # Instruments audio URL (optional)
    subtitles_url: Optional[str] = None  # Subtitles file URL (optional)
    include_subtitles: bool = True
    include_instruments: bool = True
    output_name: Optional[str] = None  # Custom output name

class ReconstructVideoResponse(BaseModel):
    success: bool
    message: str
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    subtitles_url: Optional[str] = None
    processing_time: Optional[float] = None
    reconstruction_id: Optional[str] = None 