"""
Audio Mixer
Handles audio track filtering and mixing based on user settings
"""

from typing import List, Dict, Any
import logging

from .models import AudioItem

logger = logging.getLogger(__name__)

class AudioMixer:
    """
    Mix audio tracks based on user's audio mode settings
    """
    
    def mix_audio_tracks(self, audio_tracks: List[AudioItem], settings: Dict[str, Any], 
                        voice_clone_data: Dict[str, Any]) -> List[AudioItem]:
        """
        Mix audio based on user's audio mode settings
        IMPORTANT: Original video is ALWAYS muted, only segment audio is used
        """
        instruments_enabled = settings.get("instrumentsEnabled", True)
        
        logger.info(f"🎵 AUDIO MIXER: instruments={instruments_enabled}")
        logger.info(f"🎵 Input tracks: {len(audio_tracks)} total")
        
        # Log input track types
        for track in audio_tracks:
            audio_type = track.voice_clone_info.audio_type if hasattr(track, 'voice_clone_info') else "unknown"
            logger.info(f"  - Track {track.id}: type={audio_type}")
        
        # Filter tracks based on new logic: ALWAYS mute original video
        final_audio_tracks = self._filter_tracks_for_export(
            audio_tracks, instruments_enabled
        )
        
        logger.info(f"🎵 RESULT: {len(final_audio_tracks)} tracks selected for final mix")
        return final_audio_tracks
    
    def _filter_tracks_for_export(self, audio_tracks: List[AudioItem], 
                                 instruments_enabled: bool) -> List[AudioItem]:
        """
        Filter audio tracks for export: Always use segment audio, optionally add instruments
        CRITICAL: Original video audio is ALWAYS muted
        """
        final_tracks = []
        
        logger.info(f"🎛️ FILTERING: instruments_enabled={instruments_enabled}")
        
        for track in audio_tracks:
            audio_type = track.voice_clone_info.audio_type
            
            # NEW LOGIC: Original video is ALWAYS muted, only segment audio is used
            if audio_type == "cloned":
                # Always include cloned/segment audio - this is the main audio track
                final_tracks.append(track)
                logger.info(f"✅ INCLUDED: {track.id} (cloned voice - main audio track)")
                
            elif audio_type == "instruments" and instruments_enabled:
                # Only include instruments if enabled in settings
                final_tracks.append(track)
                logger.info(f"✅ INCLUDED: {track.id} (instruments - enabled in settings)")
                
            elif audio_type == "instruments" and not instruments_enabled:
                logger.info(f"❌ SKIPPED: {track.id} (instruments - disabled in settings)")
                
            elif audio_type in ["source", None]:
                # ALWAYS skip original video audio - it's muted
                logger.info(f"❌ MUTED: {track.id} (original video audio - always muted as per requirement)")
                
            else:
                logger.info(f"❌ SKIPPED: {track.id} (type={audio_type} - unknown type)")
        
        logger.info(f"🎯 FILTER RESULT: {len(final_tracks)} out of {len(audio_tracks)} tracks selected")
        return final_tracks
    
    def apply_volume_changes(self, audio_tracks: List[AudioItem], 
                           volume_changes: Dict[str, float]) -> List[AudioItem]:
        """
        Apply volume changes to audio tracks from editingChanges.volumeAdjustments
        """
        for track in audio_tracks:
            if track.id in volume_changes:
                original_volume = track.volume
                track.volume = volume_changes[track.id]  # Already 0-1 scale
                logger.debug(f"Volume change for {track.id}: {original_volume} -> {track.volume}")
        
        return audio_tracks
    
    def apply_speed_changes(self, audio_tracks: List[AudioItem], 
                          speed_changes: List[Dict[str, Any]]) -> List[AudioItem]:
        """
        Apply speed changes to audio tracks
        """
        speed_map = {change["itemId"]: change["newSpeed"] 
                    for change in speed_changes}
        
        for track in audio_tracks:
            if track.id in speed_map:
                original_speed = track.playback_rate
                track.playback_rate = speed_map[track.id]
                logger.debug(f"Speed change for {track.id}: {original_speed} -> {track.playback_rate}")
        
        return audio_tracks
    
    def apply_trim_changes(self, audio_tracks: List[AudioItem], 
                         trim_changes: List[Dict[str, Any]]) -> List[AudioItem]:
        """
        Apply trim changes to audio tracks
        """
        trim_map = {change["itemId"]: change.get("newTrim") 
                   for change in trim_changes if change.get("newTrim")}
        
        for track in audio_tracks:
            if track.id in trim_map:
                new_trim = trim_map[track.id]
                original_trim_start = track.trim_start
                original_trim_end = track.trim_end
                
                track.trim_start = new_trim.get("from", 0) / 1000  # Convert ms to seconds
                track.trim_end = new_trim.get("to") / 1000 if new_trim.get("to") else None
                
                logger.debug(f"Trim change for {track.id}: "
                           f"{original_trim_start:.2f}-{original_trim_end}s -> "
                           f"{track.trim_start:.2f}-{track.trim_end}s")
        
        return audio_tracks
    
    def apply_position_changes(self, audio_tracks: List[AudioItem], 
                             position_changes: List[Dict[str, Any]]) -> List[AudioItem]:
        """
        Apply timeline position changes to audio tracks
        """
        position_map = {change["itemId"]: change["newPosition"] 
                       for change in position_changes}
        
        for track in audio_tracks:
            if track.id in position_map:
                new_pos = position_map[track.id]
                original_start = track.start_time
                original_end = track.end_time
                
                track.start_time = new_pos["from"] / 1000  # Convert ms to seconds
                track.end_time = new_pos["to"] / 1000
                track.duration = (new_pos["to"] - new_pos["from"]) / 1000
                
                logger.debug(f"Position change for {track.id}: "
                           f"{original_start:.2f}-{original_end:.2f}s -> "
                           f"{track.start_time:.2f}-{track.end_time:.2f}s")
        
        return audio_tracks 