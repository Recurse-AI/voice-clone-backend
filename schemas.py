from pydantic import BaseModel, Field, validator
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

# Export Video Schemas
class ExportVideoRequest(BaseModel):
    audioId: str = Field(..., min_length=1, description="Audio ID for the processed video")
    format: str = Field("mp4", pattern="^(mp4|avi|mov|mkv)$", description="Video output format")
    settings: Dict[str, Any] = Field(..., description="Export settings including quality, resolution, etc.")
    timeline: Dict[str, Any] = Field(..., description="Timeline data with items and configuration")
    editingChanges: Dict[str, Any] = Field(..., description="Changes applied during editing")
    voiceCloneData: Dict[str, Any] = Field(..., description="Voice cloning data and segments")
    exportMetadata: Dict[str, Any] = Field(..., description="Export metadata like title, description")
    instrumentsUrl: Optional[str] = Field(None, description="Optional instruments audio URL")
    subtitlesUrl: Optional[str] = Field(None, description="Optional SRT subtitle file URL")
    
    @validator('timeline')
    def validate_timeline(cls, v):
        required_fields = ['duration', 'fps', 'size', 'items']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Timeline missing required field: {field}")
        
        if not isinstance(v['items'], list):
            raise ValueError("Timeline items must be a list")
        
        return v
    
    @validator('settings')
    def validate_settings(cls, v):
        # Validate quality settings
        if 'quality' in v and v['quality'] not in ['low', 'medium', 'high', 'ultra']:
            raise ValueError("Quality must be one of: low, medium, high, ultra")
        
        return v

class ExportJobResponse(BaseModel):
    jobId: str
    status: str
    message: str
    estimatedDuration: Optional[int] = None

class ProcessingLogs(BaseModel):
    logs: list[str]

class ExportStatusResponse(BaseModel):
    jobId: str
    status: str
    progress: int
    downloadUrl: Optional[str] = None
    error: Optional[str] = None
    processingLogs: Optional[ProcessingLogs] = None

# Audio Separation API Schemas
class AudioSeparationRequest(BaseModel):
    audioUrl: str
    callerInfo: Optional[str] = None

class AudioSeparationResponse(BaseModel):
    success: bool
    requestId: str
    message: str
    estimatedTime: str
    statusCheckUrl: str
    queuePosition: Optional[int] = None

class SeparationStatusResponse(BaseModel):
    requestId: str
    status: str  # pending, processing, completed, failed, cancelled
    progress: int  # 0-100
    queuePosition: Optional[int] = None
    vocalUrl: Optional[str] = None
    instrumentUrl: Optional[str] = None
    error: Optional[str] = None
    createdAt: str
    startedAt: Optional[str] = None
    completedAt: Optional[str] = None
    callerInfo: Optional[str] = None

class QueueStatsResponse(BaseModel):
    totalRequests: int
    pending: int
    processing: int
    completed: int
    failed: int
    maxConcurrent: int
    queueLength: int 