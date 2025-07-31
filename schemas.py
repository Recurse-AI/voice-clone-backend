from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List

class StatusResponse(BaseModel):
    status: str
    message: str
    audio_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Video-dub related schemas removed as requested
# (StartProcessingResponse, RegenerateSegmentRequest, RegenerateSegmentResponse)

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

# Video Download Schemas
class VideoDownloadRequest(BaseModel):
    url: str = Field(..., min_length=1, description="Video URL from supported platforms")
    quality: Optional[str] = Field(None, description="Video quality preference (e.g., 'best', 'worst', 'best[height<=720]')")
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v

class VideoDownloadResponse(BaseModel):
    success: bool
    message: str
    download_id: Optional[str] = None
    video_info: Optional[Dict[str, Any]] = None
    cloudflare: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# File Upload Schemas (for separate upload endpoint)
class FileUploadResponse(BaseModel):
    success: bool
    message: str
    file_id: str
    file_url: str
    original_filename: str
    file_size: int

# Upload Status Schema
class UploadStatusResponse(BaseModel):
    file_id: str
    status: str  # uploading, completed, failed
    progress: int  # 0-100
    message: str
    original_filename: Optional[str] = None
    file_url: Optional[str] = None

# Simple Dubbing API Schemas
class SimpleDubbingRequest(BaseModel):
    audioUrl: str = Field(..., min_length=1, description="Audio URL to process for dubbing")
    speakersCount: int = Field(2, ge=1, le=10, description="Expected number of speakers (1-10)")
    targetLanguage: str = Field("English", description="Target language for dubbing")
    
    @validator('audioUrl')
    def validate_audio_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Audio URL must start with http:// or https://")
        return v

class SimpleDubbingResponse(BaseModel):
    success: bool
    message: str
    transcriptId: Optional[str] = None
    segmentsCount: Optional[int] = None
    targetLanguage: Optional[str] = None
    finalAudioUrl: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None