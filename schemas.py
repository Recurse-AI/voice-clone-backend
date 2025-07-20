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

# Export Video Schemas
class ExportVideoRequest(BaseModel):
    audioId: str
    format: str = "mp4"
    settings: Dict[str, Any]
    timeline: Dict[str, Any] 
    editingChanges: Dict[str, Any]
    voiceCloneData: Dict[str, Any]
    exportMetadata: Dict[str, Any]
    instrumentsUrl: Optional[str] = None  # Optional instruments audio URL
    subtitlesUrl: Optional[str] = None    # Optional SRT subtitle file URL

class ExportJobResponse(BaseModel):
    jobId: str
    status: str
    message: str
    estimatedDuration: Optional[int] = None

class ExportStatusResponse(BaseModel):
    jobId: str
    status: str
    progress: int
    downloadUrl: Optional[str] = None
    error: Optional[str] = None
    processingLogs: Optional[Dict[str, Any]] = None

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