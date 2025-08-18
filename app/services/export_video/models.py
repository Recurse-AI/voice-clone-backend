"""
Data models for video export processing
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import uuid
from datetime import datetime, timezone

from .constants import DEFAULT_VIDEO_FORMAT

# New schema interfaces to match frontend ExportPayload

@dataclass
class TimelineSize:
    width: int
    height: int

@dataclass
class TimelineDisplay:
    from_time: float  # in milliseconds
    to_time: float    # in milliseconds

@dataclass
class TimelineTrim:
    from_time: float  # in milliseconds  
    to_time: float    # in milliseconds

@dataclass
class ItemPosition:
    x: float
    y: float
    width: float
    height: float

@dataclass
class ItemEffects:
    opacity: Optional[float] = 100
    brightness: Optional[float] = 100
    blur: Optional[float] = 0
    borderWidth: Optional[float] = None
    borderColor: Optional[str] = None
    borderRadius: Optional[float] = None

@dataclass
class TextStyle:
    fontSize: Optional[float] = None
    fontFamily: Optional[str] = None
    color: Optional[str] = None
    backgroundColor: Optional[str] = None
    textAlign: Optional[str] = None
    textDecoration: Optional[str] = None
    fontWeight: Optional[str] = None
    fontStyle: Optional[str] = None

@dataclass
class ItemMetadata:
    type: Optional[str] = None
    trackId: Optional[str] = None
    audioType: Optional[str] = None  # "instruments" | "vocal" | "cloned"
    speaker: Optional[str] = None
    segmentIndex: Optional[int] = None
    speakerTrackId: Optional[str] = None
    displayName: Optional[str] = None

@dataclass
class TimelineItemSchema:
    """
    Schema-compliant TimelineItem matching frontend interface
    """
    id: str
    type: str  # 'video' | 'audio' | 'image' | 'text' | 'caption'
    startTime: float  # in milliseconds
    duration: float   # in milliseconds
    url: Optional[str] = None
    volume: Optional[float] = None
    
    # Timing and display
    display: Optional[TimelineDisplay] = None
    trim: Optional[TimelineTrim] = None
    playbackRate: Optional[float] = 1.0
    
    # Positioning and styling
    position: Optional[ItemPosition] = None
    effects: Optional[ItemEffects] = None
    
    # Content properties
    text: Optional[str] = None
    style: Optional[TextStyle] = None
    
    # Metadata
    metadata: Optional[ItemMetadata] = None

@dataclass
class ExportSettings:
    quality: Optional[str] = 'medium'  # 'low' | 'medium' | 'high'
    bitrate: Optional[str] = None
    resolution: Optional[str] = None
    instrumentsEnabled: Optional[bool] = True
    subtitlesEnabled: Optional[bool] = True

@dataclass
class Timeline:
    duration: float  # in milliseconds
    fps: int
    size: TimelineSize
    items: List[TimelineItemSchema]

@dataclass
class PositionChange:
    itemId: str
    originalPosition: Dict[str, float]
    newPosition: Dict[str, float]

@dataclass
class TrimChange:
    itemId: str
    originalTrim: Optional[Dict[str, float]] = None
    newTrim: Optional[Dict[str, float]] = None

@dataclass
class SpeedChange:
    itemId: str
    originalSpeed: float
    newSpeed: float

@dataclass
class EditingChanges:
    volumeAdjustments: Optional[Dict[str, float]] = None
    filters: Optional[Dict[str, List[str]]] = None
    positionChanges: Optional[List[PositionChange]] = None
    trimChanges: Optional[List[TrimChange]] = None
    speedChanges: Optional[List[SpeedChange]] = None

@dataclass
class VoiceSegment:
    segmentIndex: int
    speaker: str
    originalText: str
    englishText: str
    startTime: float  # in seconds
    endTime: float    # in seconds
    duration: float   # in seconds
    confidence: float
    segmentUrl: str
    clonedFilename: str

@dataclass
class VoiceCloneData:
    audioId: Optional[str] = None
    originalVideoUrl: Optional[str] = None
    clonedAudioUrl: Optional[str] = None
    speakers: Optional[List[str]] = None
    processing_id: Optional[str] = None
    segments: Optional[List[VoiceSegment]] = None

@dataclass
class ExportPayload:
    """
    Complete export payload matching frontend schema
    """
    audioId: str
    format: str  # 'mp4' | 'webm' | 'mov' | 'avi'
    settings: ExportSettings
    timeline: Timeline
    editingChanges: Optional[EditingChanges] = None
    voiceCloneData: VoiceCloneData = None
    instrumentsUrl: Optional[str] = None
    subtitlesUrl: Optional[str] = None

# Legacy models for backward compatibility

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
    format: str = DEFAULT_VIDEO_FORMAT

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
            created_at=datetime.now(timezone.utc),
            export_data=export_data,
            processing_logs=["Export job created"]
        ) 