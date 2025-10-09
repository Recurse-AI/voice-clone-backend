from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class DubbingContext:
    job_id: str
    target_language: str
    target_language_code: str
    source_video_language: Optional[str] = None
    source_language_code: Optional[str] = None
    model_type: str = "normal"
    voice_type: Optional[str] = None
    reference_ids: List[str] = field(default_factory=list)
    num_of_speakers: int = 1
    review_mode: bool = False
    add_subtitle_to_video: bool = False
    video_subtitle: bool = False
    
    process_temp_dir: str = None
    manifest: Optional[Dict[str, Any]] = None
    separation_urls: Optional[Dict[str, str]] = None
    
    segments: List[Dict[str, Any]] = field(default_factory=list)
    transcript_id: Optional[str] = None
    transcription_result: Optional[Dict[str, Any]] = None
    
    vocal_url: Optional[str] = None
    instrument_url: Optional[str] = None
    
    created_voice_ids: List[str] = field(default_factory=list)
    audio_already_split: bool = False
    
    def is_redub(self) -> bool:
        return self.manifest is not None and self.manifest.get("parent_job_id") is not None
    
    def is_resume(self) -> bool:
        return self.manifest is not None and not self.is_redub()
    
    def is_fresh_dub(self) -> bool:
        return self.manifest is None

