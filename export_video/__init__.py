"""
Export Video Module
Clean, modular video export functionality for voice cloning projects
"""

from .job_manager import ExportJobManager
from .timeline_processor import TimelineProcessor
from .audio_mixer import AudioMixer
from .canvas_processor import CanvasProcessor
from .video_renderer import VideoRenderer
from .background_processor import BackgroundProcessor

# Export new schema models
from .models import (
    ExportPayload, ExportSettings, Timeline, TimelineItemSchema,
    EditingChanges, VoiceCloneData, VoiceSegment,
    PositionChange, TrimChange, SpeedChange,
    TimelineSize, ItemPosition, ItemEffects, TextStyle, ItemMetadata
)

__all__ = [
    'ExportJobManager',
    'TimelineProcessor', 
    'AudioMixer',
    'CanvasProcessor',
    'VideoRenderer',
    'BackgroundProcessor',
    # Schema models
    'ExportPayload',
    'ExportSettings', 
    'Timeline',
    'TimelineItemSchema',
    'EditingChanges',
    'VoiceCloneData',
    'VoiceSegment',
    'PositionChange',
    'TrimChange', 
    'SpeedChange',
    'TimelineSize',
    'ItemPosition',
    'ItemEffects',
    'TextStyle',
    'ItemMetadata'
] 