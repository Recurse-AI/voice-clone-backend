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
    quality: Optional[str] = Field(
        "best", 
        description="Video quality preference. Supports yt-dlp format selectors like 'best', 'worst', 'best[height<=720]', etc."
    )
    resolution: Optional[str] = Field(
        None,
        description="Preferred resolution (e.g., '1080', '720', '480', '360'). Will try to get best quality at this resolution."
    )
    max_filesize: Optional[str] = Field(
        None,
        description="Maximum file size (e.g., '100M', '500M', '1G'). Downloads will be limited to this size."
    )
    format_preference: Optional[str] = Field(
        "mp4",
        description="Preferred video format (mp4, webm, mkv, etc.). Default is mp4 for better compatibility."
    )
    audio_quality: Optional[str] = Field(
        "best",
        description="Audio quality preference: 'best', 'worst', or specific codec like 'aac', 'opus', 'm4a'"
    )
    prefer_free_formats: Optional[bool] = Field(
        False,
        description="Prefer free/open formats (webm, ogg) over proprietary ones (mp4, m4a)"
    )
    include_subtitles: Optional[bool] = Field(
        False,
        description="Download available subtitles/captions if available"
    )
    
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
    
    @validator('quality')
    def validate_quality(cls, v):
        if v is None:
            return "best"
        # Allow any yt-dlp format selector for flexibility
        return v
    
    @validator('resolution')
    def validate_resolution(cls, v):
        if v is None:
            return v
        # Common resolution heights
        valid_resolutions = ['144', '240', '360', '480', '720', '1080', '1440', '2160', '4320']
        if v not in valid_resolutions:
            # Allow any numeric value for custom resolutions
            try:
                int(v)
            except ValueError:
                raise ValueError(f"Resolution must be numeric (height in pixels) or one of: {', '.join(valid_resolutions)}")
        return v
    
    @validator('max_filesize')
    def validate_max_filesize(cls, v):
        if v is None:
            return v
        # Validate filesize format (e.g., 100M, 1.5G, 500K)
        import re
        pattern = r'^\d+(\.\d+)?[KMG]$'
        if not re.match(pattern, v, re.IGNORECASE):
            raise ValueError("File size must be in format like '100M', '1.5G', '500K'")
        return v
    
    @validator('format_preference')
    def validate_format_preference(cls, v):
        if v is None:
            return "mp4"
        # Common video formats
        valid_formats = ['mp4', 'webm', 'mkv', 'avi', 'mov', 'flv', 'm4v']
        if v.lower() not in valid_formats:
            # Allow any format for flexibility
            pass
        return v.lower()
    
    @validator('audio_quality')
    def validate_audio_quality(cls, v):
        if v is None:
            return "best"
        # Allow any audio quality specification
        return v

class VideoDownloadResponse(BaseModel):
    success: bool
    message: str
    job_id: Optional[str] = None       # New field for consistency with other APIs
    video_info: Optional[Dict[str, Any]] = None
    download_info: Optional[Dict[str, Any]] = None  # Extended download details
    available_formats: Optional[List[Dict[str, Any]]] = None  # Available quality options
    cloudflare: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# File Delete Schemas  
class FileDeleteRequest(BaseModel):
    job_id: str = Field(..., min_length=1, description="Job ID of the downloaded file to delete")

class FileDeleteResponse(BaseModel):
    success: bool
    message: str
    deleted_files: Optional[List[str]] = None
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
    queuePosition: Optional[int] = None
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

