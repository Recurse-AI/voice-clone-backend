"""
Timeline Processor
Converts timeline data into structured items for video processing
"""

import re
from typing import List, Dict, Any
import logging

from .models import (
    VideoItem, AudioItem, TextItem, ImageItem, ProcessedItems,
    Position, Transform, Effects, AudioProperties, VoiceCloneInfo, TextStyling
)
from .constants import (
    DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT, 
    DEFAULT_OVERLAY_WIDTH, DEFAULT_OVERLAY_HEIGHT, DEFAULT_TEXT_OVERLAY_HEIGHT
)

logger = logging.getLogger(__name__)

class TimelineProcessor:
    """
    Processes timeline items and converts them to structured data models
    """
    
    @staticmethod
    def parse_pixel_value(value) -> float:
        """Convert '100px' to 100"""
        if isinstance(value, str) and value.endswith('px'):
            return float(value[:-2])
        return float(value) if value else 0.0
    
    @staticmethod
    def extract_scale_from_transform(transform_str: str) -> float:
        """Extract scale value from 'scale(1.5)' format"""
        if not transform_str:
            return 1.0
        match = re.search(r'scale\(([\d.]+)\)', transform_str)
        return float(match.group(1)) if match else 1.0
    
    @staticmethod
    def parse_rotation(rotate_str: str) -> float:
        """Parse rotation from 'rotate(45deg)' format"""
        if not rotate_str:
            return 0.0
        match = re.search(r'rotate\(([\d.-]+)deg\)', rotate_str)
        return float(match.group(1)) if match else 0.0
    
    def process_timeline_items(self, timeline_items: List[Dict[str, Any]]) -> ProcessedItems:
        """
        Process each timeline item based on type
        """
        video_layers = []
        audio_tracks = []
        text_overlays = []
        image_overlays = []
        
        for item in timeline_items:
            item_type = item.get("type")
            
            try:
                if item_type == "video":
                    video_layers.append(self._process_video_item(item))
                elif item_type == "audio":
                    audio_tracks.append(self._process_audio_item(item))
                elif item_type == "text":
                    text_overlays.append(self._process_text_item(item))
                elif item_type == "image":
                    image_overlays.append(self._process_image_item(item))
                else:
                    logger.warning(f"Unknown item type: {item_type}")
            except Exception as e:
                logger.error(f"Error processing {item_type} item {item.get('id')}: {e}")
        
        return ProcessedItems(
            video_layers=video_layers,
            audio_tracks=audio_tracks,
            text_overlays=text_overlays,
            image_overlays=image_overlays
        )
    
    def _detect_audio_type(self, url: str, metadata: Dict[str, Any]) -> str:
        """
        Detect audio type from URL pattern and metadata
        """
        if not url:
            return "unknown"
        
        url_lower = url.lower()
        
        # Check metadata first
        if metadata.get("audioType"):
            return metadata["audioType"]
        
        # Detect from URL pattern
        if "instruments" in url_lower or "instrumental" in url_lower:
            return "instruments"
        elif "cloned" in url_lower or "/segments/" in url_lower:
            return "cloned"
        elif "original" in url_lower or "source" in url_lower:
            return "source"
        else:
            # Default based on URL structure for voice cloning
            if "/segments/" in url_lower and "/cloned/" in url_lower:
                return "cloned"
            elif "/instruments/" in url_lower:
                return "instruments"
            else:
                return "unknown"
    
    def _process_video_item(self, item: Dict[str, Any]) -> VideoItem:
        """Process video item with all transformations"""
        details = item.get("details", {})
        display = item.get("display", {})
        trim = item.get("trim", {})
        
        # Timeline positioning - use direct fields from frontend
        start_time = item.get("startTime", 0) / 1000  # Convert ms to seconds
        duration = item.get("duration", 0) / 1000      # Convert ms to seconds
        end_time = start_time + duration
        
        # Video trimming
        trim_start = trim.get("from", 0) / 1000 if trim else 0
        trim_end = trim.get("to") / 1000 if trim and trim.get("to") else None
        
        # Canvas positioning - use direct position data from frontend
        position_data = item.get("position", {})
        position = Position(
            x=self.parse_pixel_value(position_data.get("x", "0px")),
            y=self.parse_pixel_value(position_data.get("y", "0px")),
            width=position_data.get("width", DEFAULT_VIDEO_WIDTH),
            height=position_data.get("height", DEFAULT_VIDEO_HEIGHT)
        )
        
        # Transformations - use effects for opacity, default for others
        transform = Transform(
            scale=1.0,  # Default scale (no transform data in frontend schema)
            rotation=0.0,  # Default rotation
            opacity=effects_data.get("opacity", 100) / 100
        )
        
        # Visual effects - use direct effects data from frontend
        effects_data = item.get("effects", {})
        effects = Effects(
            brightness=effects_data.get("brightness", 100) / 100,
            blur=effects_data.get("blur", 0),
            flipX=effects_data.get("flipX", False),
            flipY=effects_data.get("flipY", False)
        )
        
        # Audio settings - use direct volume from item level
        audio = AudioProperties(
            volume=item.get("volume", 1.0),  # Frontend sends 0-1 scale directly
            playback_rate=item.get("playbackRate", 1)
        )
        
        return VideoItem(
            id=item.get("id"),
            src=item.get("url") or details.get("src"),  # Use direct URL from frontend
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            trim_start=trim_start,
            trim_end=trim_end,
            position=position,
            transform=transform,
            effects=effects,
            audio=audio
        )
    
    def _process_audio_item(self, item: Dict[str, Any]) -> AudioItem:
        """Process audio item with voice clone metadata"""
        details = item.get("details", {})
        display = item.get("display", {})
        trim = item.get("trim", {})
        metadata = item.get("metadata", {})
        
        # Timeline positioning - use direct fields from frontend
        start_time = item.get("startTime", 0) / 1000  # Convert ms to seconds
        duration = item.get("duration", 0) / 1000      # Convert ms to seconds
        end_time = start_time + duration
        
        # Audio trimming
        trim_start = trim.get("from", 0) / 1000 if trim else 0
        trim_end = trim.get("to") / 1000 if trim and trim.get("to") else None
        
        # Get URL from frontend data
        audio_url = item.get("url") or details.get("src")
        
        # Detect audio type from URL pattern and metadata
        audio_type = self._detect_audio_type(audio_url, metadata)
        
        # Voice clone metadata
        voice_clone_info = VoiceCloneInfo(
            audio_type=audio_type,
            speaker=metadata.get("speaker"),
            segment_index=metadata.get("segmentIndex"),
            original_text=metadata.get("originalText"),
            english_text=metadata.get("englishText")
        )
        
        return AudioItem(
            id=item.get("id"),
            src=audio_url,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            trim_start=trim_start,
            trim_end=trim_end,
            volume=item.get("volume", 1.0),  # Frontend sends 0-1 scale directly
            playback_rate=item.get("playbackRate", 1),
            voice_clone_info=voice_clone_info
        )
    
    def _process_text_item(self, item: Dict[str, Any]) -> TextItem:
        """Process text overlay with all styling"""
        details = item.get("details", {})
        display = item.get("display", {})
        
        # Timeline positioning - use direct fields from frontend like audio/video
        start_time = item.get("startTime", 0) / 1000  # Convert ms to seconds
        duration = item.get("duration", 0) / 1000      # Convert ms to seconds
        end_time = start_time + duration
        
        # Canvas positioning - use direct position data from frontend
        position_data = item.get("position", {})
        position = Position(
            x=position_data.get("x", 0),
            y=position_data.get("y", 0),
            width=position_data.get("width", DEFAULT_OVERLAY_WIDTH),
            height=position_data.get("height", DEFAULT_TEXT_OVERLAY_HEIGHT)
        )
        
        # Text styling - use direct style data from frontend
        style_data = item.get("style", {})
        styling = TextStyling(
            font_family=style_data.get("fontFamily", "Arial"),
            font_size=style_data.get("fontSize", 24),
            color=style_data.get("color", "#ffffff"),
            background_color=style_data.get("backgroundColor", "transparent"),
            text_align=style_data.get("textAlign", "left"),
            font_weight=style_data.get("fontWeight", "normal"),
            opacity=style_data.get("opacity", 100) / 100,
            transform=style_data.get("transform", "none")
        )
        
        return TextItem(
            id=item.get("id"),
            text=item.get("text", ""),
            start_time=start_time,
            end_time=end_time,
            position=position,
            styling=styling,
            font_url=style_data.get("fontUrl")
        )
    
    def _process_image_item(self, item: Dict[str, Any]) -> ImageItem:
        """Process image overlay with transformations"""
        details = item.get("details", {})
        display = item.get("display", {})
        
        # Timeline positioning - use direct fields from frontend
        start_time = item.get("startTime", 0) / 1000  # Convert ms to seconds
        duration = item.get("duration", 0) / 1000      # Convert ms to seconds
        end_time = start_time + duration
        
        # Canvas positioning - use direct position data from frontend
        position_data = item.get("position", {})
        position = Position(
            x=position_data.get("x", 0),
            y=position_data.get("y", 0),
            width=position_data.get("width", DEFAULT_OVERLAY_WIDTH),
            height=position_data.get("height", DEFAULT_OVERLAY_HEIGHT)
        )
        
        # Visual effects - use direct effects data from frontend
        effects_data = item.get("effects", {})
        transform = Transform(
            scale=1.0,  # Default scale
            rotation=0.0,  # Default rotation
            opacity=effects_data.get("opacity", 100) / 100
        )
        
        # Visual effects
        effects = Effects(
            brightness=effects_data.get("brightness", 100) / 100,
            blur=effects_data.get("blur", 0),
            flipX=effects_data.get("flipX", False),
            flipY=effects_data.get("flipY", False)
        )
        
        return ImageItem(
            id=item.get("id"),
            src=item.get("url") or details.get("src"),  # Use direct URL from frontend
            start_time=start_time,
            end_time=end_time,
            position=position,
            transform=transform,
            effects=effects
        ) 