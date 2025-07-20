"""
Canvas Processor
Applies visual transformations, positioning changes, and styling modifications
"""

from typing import List, Dict, Any, Union
import logging

from .models import VideoItem, TextItem, ImageItem, Position, Transform, Effects

logger = logging.getLogger(__name__)

class CanvasProcessor:
    """
    Apply all canvas positioning and styling changes
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
        import re
        match = re.search(r'scale\(([\d.]+)\)', transform_str)
        return float(match.group(1)) if match else 1.0
    
    def apply_canvas_changes(self, items: List[Union[VideoItem, TextItem, ImageItem]], 
                           editing_changes: Dict[str, Any]) -> List[Union[VideoItem, TextItem, ImageItem]]:
        """
        Apply all canvas positioning and styling changes
        """
        canvas_changes = editing_changes.get("canvasChanges", [])
        
        if not canvas_changes:
            logger.info("No canvas changes to apply")
            return items
        
        # Create mapping of item ID to changes
        changes_map = {change["itemId"]: change for change in canvas_changes}
        
        logger.info(f"Applying canvas changes to {len(changes_map)} items")
        
        for item in items:
            if item.id in changes_map:
                change = changes_map[item.id]
                self._apply_item_changes(item, change)
        
        return items
    
    def _apply_item_changes(self, item: Union[VideoItem, TextItem, ImageItem], 
                          change: Dict[str, Any]):
        """
        Apply changes to a specific item
        """
        item_type = type(item).__name__
        logger.debug(f"Applying changes to {item_type} item: {item.id}")
        
        # Apply positioning changes
        if "positioning" in change:
            self._apply_positioning_changes(item, change["positioning"])
        
        # Apply styling changes
        if "styling" in change:
            self._apply_styling_changes(item, change["styling"])
        
        # Apply transform changes
        if "transform" in change:
            self._apply_transform_changes(item, change["transform"])
    
    def _apply_positioning_changes(self, item: Union[VideoItem, TextItem, ImageItem], 
                                 positioning: Dict[str, Any]):
        """
        Apply positioning changes (left, top, width, height)
        """
        if "left" in positioning:
            item.position.x = self.parse_pixel_value(positioning["left"])
            logger.debug(f"Updated position.x for {item.id}: {item.position.x}")
        
        if "top" in positioning:
            item.position.y = self.parse_pixel_value(positioning["top"])
            logger.debug(f"Updated position.y for {item.id}: {item.position.y}")
        
        if "width" in positioning:
            item.position.width = positioning["width"]
            logger.debug(f"Updated width for {item.id}: {item.position.width}")
        
        if "height" in positioning:
            item.position.height = positioning["height"]
            logger.debug(f"Updated height for {item.id}: {item.position.height}")
        
        # Handle transform within positioning (for scale changes)
        if "transform" in positioning:
            if hasattr(item, 'transform') and item.transform:
                item.transform.scale = self.extract_scale_from_transform(positioning["transform"])
                logger.debug(f"Updated scale for {item.id}: {item.transform.scale}")
    
    def _apply_styling_changes(self, item: Union[VideoItem, TextItem, ImageItem], 
                             styling: Dict[str, Any]):
        """
        Apply styling changes (opacity, brightness, blur, etc.)
        """
        # Apply opacity changes
        if "opacity" in styling:
            if hasattr(item, 'transform') and item.transform:
                item.transform.opacity = styling["opacity"] / 100
                logger.debug(f"Updated opacity for {item.id}: {item.transform.opacity}")
        
        # Apply effects changes
        if hasattr(item, 'effects') and item.effects:
            if "brightness" in styling:
                item.effects.brightness = styling["brightness"] / 100
                logger.debug(f"Updated brightness for {item.id}: {item.effects.brightness}")
            
            if "blur" in styling:
                item.effects.blur = styling["blur"]
                logger.debug(f"Updated blur for {item.id}: {item.effects.blur}")
            
            if "flipX" in styling:
                item.effects.flipX = styling["flipX"]
                logger.debug(f"Updated flipX for {item.id}: {item.effects.flipX}")
            
            if "flipY" in styling:
                item.effects.flipY = styling["flipY"]
                logger.debug(f"Updated flipY for {item.id}: {item.effects.flipY}")
        
        # Apply text-specific styling for TextItem
        if isinstance(item, TextItem) and hasattr(item, 'styling'):
            if "color" in styling:
                item.styling.color = styling["color"]
                logger.debug(f"Updated text color for {item.id}: {item.styling.color}")
            
            if "fontSize" in styling:
                item.styling.font_size = styling["fontSize"]
                logger.debug(f"Updated font size for {item.id}: {item.styling.font_size}")
            
            if "fontFamily" in styling:
                item.styling.font_family = styling["fontFamily"]
                logger.debug(f"Updated font family for {item.id}: {item.styling.font_family}")
            
            if "backgroundColor" in styling:
                item.styling.background_color = styling["backgroundColor"]
                logger.debug(f"Updated background color for {item.id}: {item.styling.background_color}")
    
    def _apply_transform_changes(self, item: Union[VideoItem, TextItem, ImageItem], 
                               transform: Dict[str, Any]):
        """
        Apply transform changes (scale, rotation, etc.)
        """
        if not hasattr(item, 'transform') or not item.transform:
            return
        
        if "scale" in transform:
            item.transform.scale = transform["scale"]
            logger.debug(f"Updated scale for {item.id}: {item.transform.scale}")
        
        if "rotation" in transform:
            item.transform.rotation = transform["rotation"]
            logger.debug(f"Updated rotation for {item.id}: {item.transform.rotation}")
        
        if "opacity" in transform:
            item.transform.opacity = transform["opacity"] / 100
            logger.debug(f"Updated opacity for {item.id}: {item.transform.opacity}")
    
    def apply_position_timeline_changes(self, items: List[Union[VideoItem, TextItem, ImageItem]], 
                                      position_changes: List[Dict[str, Any]]) -> List[Union[VideoItem, TextItem, ImageItem]]:
        """
        Apply exact timeline positioning from editingChanges
        """
        if not position_changes:
            return items
        
        position_map = {change["itemId"]: change["newPosition"] 
                       for change in position_changes}
        
        logger.info(f"Applying timeline position changes to {len(position_map)} items")
        
        for item in items:
            if item.id in position_map:
                new_pos = position_map[item.id]
                original_start = item.start_time
                original_end = item.end_time
                
                item.start_time = new_pos["from"] / 1000  # Convert ms to seconds
                item.end_time = new_pos["to"] / 1000
                
                # Set duration for items that have it (VideoItem, AudioItem)
                if hasattr(item, 'duration'):
                    item.duration = (new_pos["to"] - new_pos["from"]) / 1000
                
                logger.debug(f"Timeline position change for {item.id}: "
                           f"{original_start:.2f}-{original_end:.2f}s -> "
                           f"{item.start_time:.2f}-{item.end_time:.2f}s")
        
        return items 