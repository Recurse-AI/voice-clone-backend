from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from fastapi import UploadFile
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
    Provide file_url (video/audio from R2 or external URL).
    Video files will be downloaded and audio extracted locally.
    """
    job_id: str = Field(..., description="Unique job ID for the dubbing process")
    file_url: str = Field(..., description="URL to video or audio file (from /upload-file or external)")
    target_language: str = Field(..., description="Target language for dubbing")
    project_title: Optional[str] = Field("Untitled Project", description="Project title for the dubbing job")
    duration: Optional[float] = Field(None, gt=0, le=14400, description="Video duration in seconds (max 4 hours)")
    source_video_language: Optional[str] = Field(None, description="Source video language (default: None, auto-detect)")
    humanReview: bool = Field(False, description="If true, pause after transcription+translation for human review")
    video_subtitle: bool = Field(False, description="If true, use provided SRT file instead of WhisperX transcription")
    voice_premium_model: bool = Field(False, description="If true, use premium Fish Audio API for voice cloning (costs double)")
    # Voice config
    voice_type: Optional[str] = Field(None, description="Voice mode: 'voice_clone' or 'ai_voice'")
    reference_ids: Optional[List[str]] = Field(default_factory=list, max_length=10, description="List of reference IDs for different speakers")
    
    @field_validator('file_url')
    @classmethod
    def validate_file_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("file_url must start with http:// or https://")
        return v

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
    video_url: Optional[str] = None
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
    speaker: Optional[str] = None
    reference_id: Optional[str] = None

class SegmentsResponse(BaseModel):
    job_id: str
    segments: list[SegmentItem]
    manifestUrl: Optional[str] = None
    version: Optional[int] = None
    target_language: Optional[str] = None
    reference_ids: Optional[List[str]] = None

class SegmentEdit(BaseModel):
    id: str
    dubbed_text: str
    start: Optional[int] = None
    end: Optional[int] = None
    reference_id: Optional[str] = None
    original_text: Optional[str] = None

class SaveEditsRequest(BaseModel):
    segments: list[SegmentEdit]

class ApproveReviewRequest(BaseModel):
    pass

class RedubRequest(BaseModel):
    target_language: str = Field(..., description="New target language for re-dub")
    humanReview: Optional[bool] = False
    voice_premium_model: bool = Field(False, description="If true, use premium Fish Audio API for voice cloning (costs double)")
    voice_type: Optional[str] = Field(None, description="Voice mode: 'voice_clone' or 'ai_voice'")
    reference_ids: Optional[List[str]] = Field(default_factory=list, max_length=10, description="List of reference IDs for different speakers")
    
    @field_validator('target_language')
    @classmethod
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
    target_language: Optional[str] = None
    prompt: Optional[str] = Field(None, description="Optional custom prompt for text regeneration using OpenAI")
    start: Optional[int] = None
    end: Optional[int] = None
    reference_id: Optional[str] = Field(None, description="Optional reference_id to change voice model")

    @field_validator('start')
    @classmethod
    def validate_start(cls, v):
        if v is None:
            return v
        if v < 0:
            raise ValueError("Start time must be >= 0")
        return v

    @field_validator('end')
    @classmethod
    def validate_end(cls, v, info):
        if v is None:
            return v
        if v < 0:
            raise ValueError("End time must be >= 0")
        if hasattr(info, 'data') and 'start' in info.data and info.data['start'] is not None:
            if v <= info.data['start']:
                raise ValueError("End time must be greater than start time")
        return v

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
    job_id: str = Field(..., min_length=1, description="Unique job ID for audio separation")
    file_url: str = Field(..., description="URL to audio/video file (from /upload-file or external)")
    duration: float = Field(..., gt=0, le=7200, description="Audio duration in seconds (max 2 hours)")
    callerInfo: Optional[str] = Field(None, max_length=255, description="Caller information")
    
    @field_validator('file_url')
    @classmethod
    def validate_file_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("file_url must start with http:// or https://")
        return v

class AudioSeparationResponse(BaseModel):
    success: bool
    job_id: str
    message: str
    estimatedTime: str
    statusCheckUrl: str
    queuePosition: Optional[int] = None

class SeparationStatusResponse(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
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
    format_id: Optional[str] = Field(
        None,
        description="Specific format ID from available-formats endpoint. If provided, overrides quality/resolution."
    )
    quality: Optional[str] = Field(
        "best", 
        description="Video quality preference. Use 'best' for highest quality, 'worst' for smallest file, or specify custom yt-dlp format selectors."
    )
    resolution: Optional[str] = Field(
        None,
        description="Preferred resolution (e.g., '1080', '720', '480', '360'). Max 1080p. Will try to get best quality at this resolution."
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
    
    @field_validator('url')
    @classmethod
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
    
    @field_validator('quality')
    @classmethod
    def validate_quality(cls, v):
        if v is None:
            return "best"
        # Allow any yt-dlp format selector for flexibility
        return v
    
    @field_validator('resolution')
    @classmethod
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
    
    @field_validator('max_filesize')
    @classmethod
    def validate_max_filesize(cls, v):
        if v is None:
            return v
        # Validate filesize format (e.g., 100M, 1.5G, 500K)
        import re
        pattern = r'^\d+(\.\d+)?[KMG]$'
        if not re.match(pattern, v, re.IGNORECASE):
            raise ValueError("File size must be in format like '100M', '1.5G', '500K'")
        return v
    
    @field_validator('format_preference')
    @classmethod
    def validate_format_preference(cls, v):
        if v is None:
            return "mp4"
        # Common video formats
        valid_formats = ['mp4', 'webm', 'mkv', 'avi', 'mov', 'flv', 'm4v']
        if v.lower() not in valid_formats:
            # Allow any format for flexibility
            pass
        return v.lower()
    
    @field_validator('audio_quality')
    @classmethod
    def validate_audio_quality(cls, v):
        if v is None:
            return "best"
        # Allow any audio quality specification
        return v

class VideoDownloadResponse(BaseModel):
    success: bool
    message: str
    job_id: Optional[str] = None
    r2_url: Optional[str] = None
    r2_key: Optional[str] = None
    video_info: Optional[Dict[str, Any]] = None
    download_info: Optional[Dict[str, Any]] = None
    available_formats: Optional[List[Dict[str, Any]]] = None
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



# Voice Clone Segment Schemas
class VoiceCloneRequest(BaseModel):
    referenceAudioUrl: str = Field(..., min_length=1, description="URL to the reference audio segment")
    referenceText: str = Field(..., min_length=1, max_length=5000, description="Text spoken in the reference audio")
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize with cloned voice")
    
    @field_validator('referenceAudioUrl')
    @classmethod
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
    audio_url: Optional[str] = None
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

# Workspace Status Schemas
class JobSummary(BaseModel):
    job_id: str
    status: str
    progress: int
    original_filename: Optional[str] = None
    target_language: Optional[str] = None
    source_video_language: Optional[str] = None
    result_url: Optional[str] = None
    files: Optional[List[FileInfo]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    # Separation specific fields
    audio_url: Optional[str] = None
    vocal_url: Optional[str] = None
    instrument_url: Optional[str] = None

class WorkspaceStats(BaseModel):
    """Workspace statistics summary"""
    total_dubs: int = Field(..., description="Total number of dub jobs")
    total_separations: int = Field(..., description="Total number of separation jobs")
    total_clips: int = Field(..., description="Total number of clip jobs")
    total_completed_dubs: int = Field(..., description="Number of completed dub jobs")
    total_completed_separations: int = Field(..., description="Number of completed separation jobs")
    total_completed_clips: int = Field(..., description="Number of completed clip jobs")
    total_processing_dubs: int = Field(..., description="Number of dubs currently processing")
    total_processing_separations: int = Field(..., description="Number of separations currently processing")
    total_processing_clips: int = Field(..., description="Number of clips currently processing")

class WorkspaceStatusResponse(BaseModel):
    """Comprehensive workspace status response"""
    success: bool = Field(..., description="Request success status")
    message: str = Field(..., description="Response message")
    stats: WorkspaceStats = Field(..., description="Workspace statistics")
    recent_dubs: List[JobSummary] = Field(..., description="Recent dub jobs")
    recent_separations: List[JobSummary] = Field(..., description="Recent separation jobs")
    recent_clips: List[JobSummary] = Field(..., description="Recent clip jobs")

# Individual Job Detail Schemas
class SeparationJobDetailResponse(BaseModel):
    success: bool
    job: Optional[UserSeparationJob] = None
    error: Optional[str] = None

class DubJobDetailResponse(BaseModel):
    success: bool
    job: Optional[UserDubJob] = None
    error: Optional[str] = None

# Clip Job Request Schema
class GenerateClipsRequest(BaseModel):
    video_url: str = Field(..., description="R2 bucket URL to video file")
    srt_url: Optional[str] = Field(None, description="R2 bucket URL to SRT subtitle file")
    start_time: float
    end_time: float
    expected_duration: float
    subtitle_style: Optional[str] = None
    subtitle_preset: Optional[str] = "reels"
    subtitle_font: Optional[str] = None
    subtitle_font_size: Optional[int] = None
    subtitle_wpl: Optional[int] = None

# Clip Job Response Schemas
class ClipJobListResponse(BaseModel):
    success: bool
    message: str
    jobs: List[Any]  # List of ClipJob from models
    total: int
    page: Optional[int] = None
    limit: Optional[int] = None
    total_pages: Optional[int] = None

class ClipJobDetailResponse(BaseModel):
    success: bool
    job: Optional[Any] = None  # ClipJob from models
    error: Optional[str] = None

# Process Video Complete API Schemas
class TimelineAudioSegment(BaseModel):
    start: int = Field(..., description="Start time in milliseconds")
    end: int = Field(..., description="End time in milliseconds") 
    audio_url: str = Field(..., description="URL to audio segment")
    
    @field_validator('start')
    @classmethod
    def validate_start(cls, v):
        if v < 0:
            raise ValueError("Start time must be >= 0")
        return v
    
    @field_validator('end')
    @classmethod
    def validate_end(cls, v, info):
        if v < 0:
            raise ValueError("End time must be >= 0")
        if hasattr(info, 'data') and 'start' in info.data and v <= info.data['start']:
            raise ValueError("End time must be greater than start time")
        return v

class VideoProcessingOptions(BaseModel):
    class Config:
        validate_by_name = True

    resolution: Optional[str] = Field("1080p", description="Output resolution: 720p, 1080p, 4k")
    format: Optional[str] = Field("mp4", description="Output format: mp4, webm, mov")
    quality: Optional[str] = Field("high", description="Video quality: medium, high")
    audio_quality: Optional[str] = Field("high", alias="audioQuality", description="Audio quality: medium, high")
    instrument_volume: Optional[float] = Field(0.3, alias="instrumentVolume", ge=0.0, le=1.0, description="Instrument audio volume (0.0-1.0)")
    include_subtitles: Optional[bool] = Field(True, alias="includeSubtitles", description="Include subtitles if subtitle_url provided")
    audio_only: Optional[bool] = Field(False, alias="audioOnly", description="Output audio only (no video)")
    audio_format: Optional[str] = Field("mp3", alias="audioFormat", description="Audio format for audio-only output: wav, mp3, aac")
    target_duration: Optional[int] = Field(None, description="Target duration in milliseconds")
    original_job_id: Optional[str] = Field(None, alias="originalJobId", description="Original job ID that created this processing request")
    use_background_sound: Optional[bool] = Field(False, alias="useBackgroundSound", description="Use background music if provided")

    @field_validator('resolution')
    @classmethod
    def validate_resolution(cls, v):
        if v and v not in ['720p', '1080p', '4k']:
            raise ValueError("Resolution must be: 720p, 1080p, or 4k")
        return v
    
    @field_validator('format')
    @classmethod
    def validate_format(cls, v):
        if v and v.lower() not in ['mp4', 'webm', 'mov']:
            raise ValueError("Format must be: mp4, webm, or mov")
        return v.lower() if v else v
    
    @field_validator('quality', 'audio_quality')
    @classmethod
    def validate_quality(cls, v):
        if v and v not in ['medium', 'high']:
            raise ValueError("Quality must be: medium or high")
        return v
    
    @field_validator('audio_format')
    @classmethod
    def validate_audio_format(cls, v):
        if v and v.lower() not in ['wav', 'mp3', 'aac']:
            raise ValueError("Audio format must be: wav, mp3, or aac")
        return v.lower() if v else v

class VideoProcessingResponse(BaseModel):
    success: bool
    message: str
    job_id: str
    download_url: Optional[str] = None
    output_filename: Optional[str] = None
    output_type: str = Field("video", description="Output type: video or audio")
    file_size_mb: Optional[float] = None
    duration_seconds: Optional[float] = None
    applied_options: Optional[VideoProcessingOptions] = None
    error: Optional[str] = None
    details: Optional[str] = None
    error_code: Optional[str] = None

