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
        """
        active_mode = settings.get("activeAudioMode", "english")  # "original" or "english"
        instruments_enabled = settings.get("instrumentsEnabled", True)
        
        logger.info(f"Mixing audio tracks: mode={active_mode}, instruments={instruments_enabled}")
        
        # Filter tracks based on audio mode
        final_audio_tracks = self._filter_tracks_by_mode(
            audio_tracks, active_mode, instruments_enabled
        )
        
        logger.info(f"Filtered {len(audio_tracks)} tracks to {len(final_audio_tracks)} tracks")
        return final_audio_tracks
    
    def _filter_tracks_by_mode(self, audio_tracks: List[AudioItem], active_mode: str, 
                              instruments_enabled: bool) -> List[AudioItem]:
        """
        Filter audio tracks based on active mode
        """
        final_tracks = []
        
        for track in audio_tracks:
            audio_type = track.voice_clone_info.audio_type
            
            # Audio mixing logic based on mode
            if active_mode == "english":
                # English mode: Play cloned audio + instruments (if enabled)
                if audio_type == "cloned":
                    final_tracks.append(track)
                    logger.debug(f"Including cloned audio track: {track.id}")
                elif audio_type == "instruments" and instruments_enabled:
                    final_tracks.append(track)
                    logger.debug(f"Including instruments track: {track.id}")
                # Skip original/source audio in English mode
                else:
                    logger.debug(f"Skipping {audio_type} track in English mode: {track.id}")
                    
            elif active_mode == "original":
                # Original mode: Play source video audio only
                if audio_type == "source" or audio_type is None:
                    final_tracks.append(track)
                    logger.debug(f"Including source audio track: {track.id}")
                # Skip cloned audio and instruments in Original mode
                else:
                    logger.debug(f"Skipping {audio_type} track in Original mode: {track.id}")
        
        return final_tracks
    
    def apply_volume_changes(self, audio_tracks: List[AudioItem], 
                           volume_changes: List[Dict[str, Any]]) -> List[AudioItem]:
        """
        Apply volume changes to audio tracks
        """
        volume_map = {change["itemId"]: change["newVolume"] / 100 
                     for change in volume_changes}
        
        for track in audio_tracks:
            if track.id in volume_map:
                original_volume = track.volume
                track.volume = volume_map[track.id]
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
    
    def get_active_audio_tracks(self, all_tracks: List[AudioItem], active_mode: str, 
                               instruments_enabled: bool) -> List[AudioItem]:
        """
        EXACT audio filtering based on user settings
        """
        active_tracks = []
        
        for track in all_tracks:
            audio_type = track.voice_clone_info.audio_type
            
            if active_mode == "english":
                # English mode: cloned voices + instruments (if enabled)
                if audio_type == "cloned":
                    active_tracks.append(track)
                elif audio_type == "instruments" and instruments_enabled:
                    active_tracks.append(track)
                    
            elif active_mode == "original": 
                # Original mode: source video audio only
                if audio_type in ["source", None]:
                    active_tracks.append(track)
        
        return active_tracks 