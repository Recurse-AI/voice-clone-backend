from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import re

class StatusResponse(BaseModel):
    status: str
    message: str
    audio_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Video-dub related schemas
class VideoDubRequest(BaseModel):
    """
    ভিডিও ডাবিং প্রসেসের জন্য রিকোয়েস্ট।
    ভিডিও upload-file API দিয়ে upload করতে হবে, তারপর এখানে সেই job_id দিতে হবে।
    video_url ফিল্ড নেই, কারণ ভিডিও local-এ সংরক্ষিত থাকবে।
    """
    job_id: str = Field(..., description="Unique job ID for the dubbing process (from /upload-file API)")
    target_language: str = Field(..., description="Target language for dubbing")
    project_title: Optional[str] = Field("Untitled Project", description="Project title for the dubbing job")
    duration: Optional[float] = Field(None, gt=0, le=14400, description="Video duration in seconds (max 4 hours)")
    expected_speaker: Optional[str] = Field(None, description="Expected speaker name or ID")
    source_video_language: Optional[str] = Field(None, description="Source video language (default: None, auto-detect)")
    subtitle: bool = Field(False, description="Whether to add subtitles")
    instrument: bool = Field(False, description="Whether to add instrument track")

class VideoDubResponse(BaseModel):
    success: bool
    message: str
    job_id: str
    status_check_url: str

class VideoDubStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    result_url: Optional[str] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

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
    audioUrl: str = Field(..., min_length=1, description="Audio file URL")
    duration: float = Field(..., gt=0, le=7200, description="Audio duration in seconds (max 2 hours)")
    callerInfo: Optional[str] = Field(None, max_length=255, description="Caller information")
    
    @validator('audioUrl')
    def validate_audio_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Audio URL must start with http:// or https://")
        return v

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


# Video Download Schemas
class VideoDownloadRequest(BaseModel):
    url: str = Field(..., min_length=1, description="Video URL from supported platforms")
    quality: Optional[str] = Field(None, description="Video quality preference (e.g., 'best', 'worst', 'best[height<=720]')")
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        if not url_pattern.match(v):
            raise ValueError("Invalid URL format")
        return v

class VideoDownloadResponse(BaseModel):
    success: bool
    message: str
    job_id: Optional[str] = None       # New field for consistency with other APIs
    video_info: Optional[Dict[str, Any]] = None
    cloudflare: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Upload Status Schema
class UploadStatusResponse(BaseModel):
    job_id: str
    status: str  # uploading, completed, failed
    progress: int  # 0-100
    message: str
    original_filename: Optional[str] = None
    file_url: Optional[str] = None

# Voice Clone Segment Schemas
class VoiceCloneRequest(BaseModel):
    referenceAudioUrl: str = Field(..., min_length=1, description="URL to the reference audio segment")
    referenceText: str = Field(..., min_length=1, max_length=5000, description="Text spoken in the reference audio")
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize with cloned voice")
    speakerLabel: Optional[str] = Field(None, max_length=100, description="Optional speaker label")
    
    @validator('referenceAudioUrl')
    def validate_reference_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Reference audio URL must start with http:// or https://")
        return v

class VoiceCloneResponse(BaseModel):
    success: bool
    message: str
    jobId: str
    audioUrl: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None

# User Job List Schemas
class UserSeparationJob(BaseModel):
    job_id: str
    status: str
    progress: int
    audio_url: str
    vocal_url: Optional[str] = None
    instrument_url: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None

class UserSeparationListResponse(BaseModel):
    success: bool
    message: str
    jobs: List[UserSeparationJob]
    total: int

class UserDubJob(BaseModel):
    job_id: str
    status: str
    progress: int
    original_filename: Optional[str] = None
    target_language: str
    result_url: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None

class UserDubListResponse(BaseModel):
    success: bool
    message: str
    jobs: List[UserDubJob]
    total: int

# Individual Job Detail Schemas
class SeparationJobDetailResponse(BaseModel):
    success: bool
    job: Optional[UserSeparationJob] = None
    error: Optional[str] = None

class DubJobDetailResponse(BaseModel):
    success: bool
    job: Optional[UserDubJob] = None
    error: Optional[str] = None

# Credit Management Schemas
class CreditDeductRequest(BaseModel):
    job_id: str = Field(..., min_length=1, max_length=255, description="Dubbing job ID") 
    duration: float = Field(..., gt=0, le=14400, description="Duration in seconds (max 4 hours)")