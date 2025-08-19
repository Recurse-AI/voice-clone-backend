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
    Request for video dubbing process.
    Video must be uploaded using upload-file API first, then provide that job_id here.
    No video_url field needed as video will be stored locally.
    """
    job_id: str = Field(..., description="Unique job ID for the dubbing process (from /upload-file API)")
    target_language: str = Field(..., description="Target language for dubbing")
    project_title: Optional[str] = Field("Untitled Project", description="Project title for the dubbing job")
    duration: Optional[float] = Field(None, gt=0, le=14400, description="Video duration in seconds (max 4 hours)")
    expected_speaker: Optional[str] = Field(None, description="Expected speaker name or ID")
    source_video_language: Optional[str] = Field(None, description="Source video language (default: None, auto-detect)")
    humanReview: bool = Field(False, description="If true, pause after transcription+translation for human review")

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

# Review/Segments Schemas
class SegmentItem(BaseModel):
    id: str
    segment_index: int
    start: int
    end: int
    duration_ms: int
    original_text: str
    dubbed_text: str
    original_audio_file: Optional[str] = None
    speaker_label: Optional[str] = None

class SegmentsResponse(BaseModel):
    job_id: str
    segments: list[SegmentItem]
    manifestUrl: Optional[str] = None
    version: Optional[int] = None
    target_language: Optional[str] = None

class SegmentEdit(BaseModel):
    id: str
    dubbed_text: str
    start: Optional[int] = None
    end: Optional[int] = None

class SaveEditsRequest(BaseModel):
    segments: list[SegmentEdit]

class ApproveReviewRequest(BaseModel):
    pass

class RedubRequest(BaseModel):
    target_language: str = Field(..., description="New target language for re-dub")
    humanReview: Optional[bool] = False
    
    @validator('target_language')
    def validate_target_language(cls, v):
        if not v or not v.strip():
            raise ValueError("Target language cannot be empty")
        # Clean and normalize the language name
        return v.strip().title()

class RedubResponse(BaseModel):
    success: bool
    message: str
    job_id: str
    status: str
    details: Optional[Dict[str, Any]] = None

class RegenerateSegmentRequest(BaseModel):
    dubbed_text: Optional[str] = None
    tone: Optional[str] = Field(None, description="Optional tone marker like excited, sad, whispering")
    prompt: Optional[str] = Field(None, description="Optional custom prompt for voice generation")
    target_language: Optional[str] = None

class RegenerateSegmentResponse(BaseModel):
    success: bool
    message: str
    job_id: str
    segment_id: str
    manifestUrl: str
    version: int
    segment: SegmentItem

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
    job_id: str = Field(..., min_length=1, description="Unique job ID for audio separation (from /upload-file API)")
    duration: float = Field(..., gt=0, le=7200, description="Audio duration in seconds (max 2 hours)")
    callerInfo: Optional[str] = Field(None, max_length=255, description="Caller information")

class AudioSeparationResponse(BaseModel):
    success: bool
    job_id: str
    message: str
    estimatedTime: str
    statusCheckUrl: str
    queuePosition: Optional[int] = None

class SeparationStatusResponse(BaseModel):
    job_id: str
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
    queuePosition: Optional[int] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None

class UserSeparationListResponse(BaseModel):
    success: bool
    message: str
    jobs: List[UserSeparationJob]
    total: int
    page: Optional[int] = None
    limit: Optional[int] = None
    total_pages: Optional[int] = None
    total_completed: Optional[int] = None
    total_processing: Optional[int] = None

class FileInfo(BaseModel):
    filename: str
    url: str
    size: Optional[int] = None
    type: str  # video, audio, subtitle, summary, metadata, other

class UserDubJob(BaseModel):
    job_id: str
    status: str
    progress: int
    original_filename: Optional[str] = None
    target_language: str
    source_video_language: Optional[str] = None
    expected_speaker: Optional[str] = None
    result_url: Optional[str] = None
    files: Optional[List[FileInfo]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None

class UserDubListResponse(BaseModel):
    success: bool
    message: str
    jobs: List[UserDubJob]
    total: int
    page: Optional[int] = None
    limit: Optional[int] = None
    total_pages: Optional[int] = None
    total_completed: Optional[int] = None
    total_processing: Optional[int] = None

# Individual Job Detail Schemas
class SeparationJobDetailResponse(BaseModel):
    success: bool
    job: Optional[UserSeparationJob] = None
    error: Optional[str] = None

class DubJobDetailResponse(BaseModel):
    success: bool
    job: Optional[UserDubJob] = None
    error: Optional[str] = None

