"""
Data models for video export processing
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime

@dataclass
class Position:
    x: float
    y: float
    width: Optional[float] = None
    height: Optional[float] = None

@dataclass
class Transform:
    scale: float = 1.0
    rotation: float = 0.0
    opacity: float = 1.0

@dataclass
class Effects:
    brightness: float = 1.0
    blur: float = 0.0
    flipX: bool = False
    flipY: bool = False

@dataclass
class AudioProperties:
    volume: float = 1.0
    playback_rate: float = 1.0

@dataclass
class VoiceCloneInfo:
    audio_type: Optional[str] = None  # "instruments", "cloned", "source"
    speaker: Optional[str] = None
    segment_index: Optional[int] = None
    original_text: Optional[str] = None
    english_text: Optional[str] = None

@dataclass
class VideoItem:
    id: str
    src: str
    start_time: float
    end_time: float
    duration: float
    trim_start: float = 0.0
    trim_end: Optional[float] = None
    position: Position = None
    transform: Transform = None
    effects: Effects = None
    audio: AudioProperties = None
    
    def __post_init__(self):
        if self.position is None:
            self.position = Position(0, 0)
        if self.transform is None:
            self.transform = Transform()
        if self.effects is None:
            self.effects = Effects()
        if self.audio is None:
            self.audio = AudioProperties()

@dataclass
class AudioItem:
    id: str
    src: str
    start_time: float
    end_time: float
    duration: float
    trim_start: float = 0.0
    trim_end: Optional[float] = None
    volume: float = 1.0
    playback_rate: float = 1.0
    voice_clone_info: VoiceCloneInfo = None
    
    def __post_init__(self):
        if self.voice_clone_info is None:
            self.voice_clone_info = VoiceCloneInfo()

@dataclass
class TextStyling:
    font_family: str = "Arial"
    font_size: int = 24
    color: str = "#ffffff"
    background_color: str = "transparent"
    text_align: str = "left"
    font_weight: str = "normal"
    opacity: float = 1.0
    transform: str = "none"

@dataclass
class TextItem:
    id: str
    text: str
    start_time: float
    end_time: float
    position: Position = None
    styling: TextStyling = None
    font_url: Optional[str] = None
    
    def __post_init__(self):
        if self.position is None:
            self.position = Position(0, 0)
        if self.styling is None:
            self.styling = TextStyling()

@dataclass
class ImageItem:
    id: str
    src: str
    start_time: float
    end_time: float
    position: Position = None
    transform: Transform = None
    effects: Effects = None
    
    def __post_init__(self):
        if self.position is None:
            self.position = Position(0, 0)
        if self.transform is None:
            self.transform = Transform()
        if self.effects is None:
            self.effects = Effects()

@dataclass
class Subtitle:
    start_time: float
    end_time: float
    text: str
    speaker: Optional[str] = None

@dataclass
class VideoConfig:
    width: int
    height: int
    fps: int
    duration: float
    format: str = "mp4"

@dataclass
class ProcessedItems:
    video_layers: List[VideoItem]
    audio_tracks: List[AudioItem]
    text_overlays: List[TextItem]
    image_overlays: List[ImageItem]

@dataclass
class ExportJob:
    job_id: str
    status: str
    progress: int
    created_at: datetime
    export_data: Dict[str, Any]
    processing_logs: List[str]
    download_url: Optional[str] = None
    error: Optional[str] = None
    
    @classmethod
    def create_new(cls, export_data: Dict[str, Any]) -> 'ExportJob':
        return cls(
            job_id=str(uuid.uuid4()),
            status="STARTED",
            progress=0,
            created_at=datetime.now(),
            export_data=export_data,
            processing_logs=["Export job created"]
        ) 