from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, timezone

class SeparationJob(BaseModel):
    id: Optional[str] = None  # MongoDB ObjectId
    job_id: str = Field(..., description="Unique job ID for separation")
    user_id: str = Field(..., description="User who submitted the job")
    
    # Job Status
    status: Literal['pending', 'processing', 'completed', 'failed'] = 'pending'
    progress: int = Field(0, ge=0, le=100, description="Progress percentage")
    
    # Input
    audio_url: str = Field(..., description="Original audio URL for separation")
    caller_info: Optional[str] = Field(None, description="API caller information")
    
    # Results (populated when completed)
    vocal_url: Optional[str] = Field(None, description="Separated vocal audio URL")
    instrument_url: Optional[str] = Field(None, description="Separated instrument audio URL")
    
    # RunPod tracking
    runpod_request_id: Optional[str] = Field(None, description="RunPod job ID")
    
    # Error handling
    error: Optional[str] = Field(None, description="Error message if failed")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        # MongoDB collection name
        collection_name = "separation_jobs"
        
    def update_status(self, status: str, progress: int = None, **kwargs):
        """Update job status and progress"""
        self.status = status
        if progress is not None:
            self.progress = progress
        self.updated_at = datetime.now(timezone.utc)
        
        # Set timestamps based on status
        if status == 'processing' and not self.started_at:
            self.started_at = datetime.now(timezone.utc)
        elif status in ['completed', 'failed']:
            self.completed_at = datetime.now(timezone.utc)
            
        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)