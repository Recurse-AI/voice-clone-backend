from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any
from datetime import datetime, timezone

class DubJob(BaseModel):
    id: Optional[str] = None  # MongoDB ObjectId
    job_id: str = Field(..., description="Unique job ID for dubbing")
    user_id: str = Field(..., description="User who submitted the job")
    
    # Job Status
    status: Literal['pending', 'downloading', 'separating', 'transcribing', 'processing', 'uploading', 'completed', 'failed'] = 'pending'
    progress: int = Field(0, ge=0, le=100, description="Progress percentage")
    
    # Input Details
    original_filename: Optional[str] = Field(None, description="Original uploaded file name")
    target_language: str = Field(..., description="Target language for dubbing")
    expected_speaker: Optional[str] = Field(None, description="Expected speaker name or ID")
    source_video_language: Optional[str] = Field(None, description="Source video language")
    subtitle: bool = Field(False, description="Whether to add subtitles")
    instrument: bool = Field(False, description="Whether to add instrument track")
    
    # File Processing
    local_video_path: Optional[str] = Field(None, description="Local path of uploaded video")
    audio_path: Optional[str] = Field(None, description="Extracted audio path")
    vocal_path: Optional[str] = Field(None, description="Separated vocal path")
    instrument_path: Optional[str] = Field(None, description="Separated instrument path")
    
    # Results (populated when completed)
    result_url: Optional[str] = Field(None, description="Final dubbed video URL")
    result_urls: Optional[Dict[str, str]] = Field(None, description="Multiple result URLs")
    
    # RunPod/External Service tracking
    separation_request_id: Optional[str] = Field(None, description="Audio separation job ID")
    
    # Processing Details
    details: Optional[Dict[str, Any]] = Field(None, description="Additional processing details")
    
    # Error handling
    error: Optional[str] = Field(None, description="Error message if failed")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now())
    updated_at: datetime = Field(default_factory=lambda: datetime.now())
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        # MongoDB collection name
        collection_name = "dub_jobs"
        
    def update_status(self, status: str, progress: int = None, **kwargs):
        """Update job status and progress"""
        self.status = status
        if progress is not None:
            self.progress = progress
        self.updated_at = datetime.now()
        
        # Set timestamps based on status
        if status in ['downloading', 'processing'] and not self.started_at:
            self.started_at = datetime.now()
        elif status in ['completed', 'failed']:
            self.completed_at = datetime.now()
            
        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def get_status_message(self) -> str:
        """Get human-readable status message"""
        messages = {
            'pending': "Processing queued",
            'downloading': "Downloading video...",
            'separating': "Separating audio tracks...",
            'transcribing': "Transcribing audio...",
            'processing': "Processing audio and video...",
            'uploading': "Uploading results...",
            'completed': "Processing completed successfully",
            'failed': "Processing failed"
        }
        return messages.get(self.status, "Unknown status")