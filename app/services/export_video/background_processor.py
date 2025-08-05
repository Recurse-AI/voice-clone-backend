"""
Background Video Export Processor
Handles the background processing of video export jobs
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime
import requests
import soundfile as sf
import numpy as np
from scipy import signal

from .constants import AUDIO_SAMPLE_RATE, VOICE_SEGMENT_VOLUME, INSTRUMENT_VOLUME_RATIO

logger = logging.getLogger(__name__)

class BackgroundProcessor:
    """
    Handles background video export processing
    """
    
    def __init__(self, settings, r2_storage):
        self.settings = settings
        self.r2_storage = r2_storage
    
    def process_video_export_background(self, job_id: str, export_data: Dict[str, Any]):
        """
        Background video export processing function
        """
        try:
            from .job_manager import export_job_manager
            from .timeline_processor import TimelineProcessor
            from .audio_mixer import AudioMixer
            from .canvas_processor import CanvasProcessor
            from .video_renderer import VideoRenderer
            from .models import VideoConfig
            
            # Update status to processing
            export_job_manager.update_status(job_id, "PROCESSING", 5)
            export_job_manager.add_log(job_id, "Starting video processing")
            
            # Step 1: Process timeline items (15%)
            timeline_processor = TimelineProcessor()
            processed_items = timeline_processor.process_timeline_items(
                export_data["timeline"]["items"]
            )
            export_job_manager.update_status(job_id, "PROCESSING", 15)
            export_job_manager.add_log(job_id, "Timeline items processed")
            
            # Step 2: Apply editing changes (25%)
            canvas_processor = CanvasProcessor()
            audio_mixer = AudioMixer()
            
            # Apply canvas changes to visual items
            all_visual_items = (processed_items.video_layers + 
                               processed_items.text_overlays + 
                               processed_items.image_overlays)
            canvas_processor.apply_canvas_changes(all_visual_items, export_data["editingChanges"])
            
            # Apply position timeline changes to text and image overlays
            canvas_processor.apply_position_timeline_changes(
                processed_items.text_overlays + processed_items.image_overlays,
                export_data["editingChanges"].get("positionChanges", [])
            )
            
            # Apply audio changes
            editing_changes = export_data["editingChanges"]
            if "volumeAdjustments" in editing_changes:
                audio_mixer.apply_volume_changes(
                    processed_items.audio_tracks, editing_changes["volumeAdjustments"]
                )
            if "speedChanges" in editing_changes:
                audio_mixer.apply_speed_changes(
                    processed_items.audio_tracks, editing_changes["speedChanges"]
                )
            if "trimChanges" in editing_changes:
                audio_mixer.apply_trim_changes(
                    processed_items.audio_tracks, editing_changes["trimChanges"]
                )
            if "positionChanges" in editing_changes:
                audio_mixer.apply_position_changes(
                    processed_items.audio_tracks, editing_changes["positionChanges"]
                )
            
            export_job_manager.update_status(job_id, "PROCESSING", 25)
            export_job_manager.add_log(job_id, "Editing changes applied")
            
            # Step 3: Mix audio tracks (40%)
            final_audio_tracks = audio_mixer.mix_audio_tracks(
                processed_items.audio_tracks,
                export_data["settings"],
                export_data["voiceCloneData"]
            )
            export_job_manager.update_status(job_id, "PROCESSING", 40)
            export_job_manager.add_log(job_id, "Audio tracks mixed")
            
            # Step 4: Create final audio file (50%)
            timeline_duration = export_data["timeline"]["duration"]
            final_audio_path = self._create_final_audio_file(final_audio_tracks, job_id, timeline_duration)
            export_job_manager.update_status(job_id, "PROCESSING", 50)
            export_job_manager.add_log(job_id, "Final audio created")
            
            # Step 5: Render final video (80%)
            video_renderer = VideoRenderer(self.settings.TEMP_DIR)
            timeline_data = export_data["timeline"]
            
            video_config = VideoConfig(
                width=timeline_data["size"]["width"],
                height=timeline_data["size"]["height"],
                fps=timeline_data["fps"],
                duration=timeline_data["duration"] / 1000,  # Convert ms to seconds
                format=export_data["format"]
            )
            
            # Get optional URLs from export data - check settings first
            video_url = export_data["voiceCloneData"].get("originalVideoUrl")
            
            # Only include instruments if enabled in settings
            instruments_url = None
            if export_data["settings"].get("instrumentsEnabled", False):
                instruments_url = export_data.get("instrumentsUrl")
                logger.info(f"Instruments enabled: URL = {instruments_url}")
            else:
                logger.info("Instruments disabled in settings")
            
            # Only include subtitles if enabled in settings
            subtitles_url = None
            if export_data["settings"].get("subtitlesEnabled", False):
                subtitles_url = export_data.get("subtitlesUrl")
                logger.info(f"Subtitles enabled: URL = {subtitles_url}")
            else:
                logger.info("Subtitles disabled in settings")
            
            render_result = asyncio.run(video_renderer.render_video(
                video_url=video_url,
                final_audio_path=final_audio_path,
                instruments_url=instruments_url,
                subtitles_url=subtitles_url,
                config=video_config,
                job_id=job_id,
                processed_items=processed_items,
                export_settings=export_data["settings"]
            ))
            
            if not render_result["success"]:
                export_job_manager.fail_job(job_id, f"Video rendering failed: {render_result.get('error')}")
                return
            
            export_job_manager.update_status(job_id, "PROCESSING", 80)
            export_job_manager.add_log(job_id, "Video rendering completed")
            
            # Step 6: Upload to cloud storage (95%)
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"exported_video_{timestamp}_{job_id}.{export_data['format']}"
                r2_key = f"exported-videos/{timestamp}/{video_filename}"
                
                upload_result = self.r2_storage.upload_file(
                    render_result["video_path"], 
                    r2_key, 
                    f"video/{export_data['format']}"
                )
                
                if not upload_result.get("success"):
                    export_job_manager.fail_job(job_id, "Failed to upload video to storage")
                    return
                
                download_url = upload_result.get("url")
                export_job_manager.update_status(job_id, "PROCESSING", 95)
                export_job_manager.add_log(job_id, "Video uploaded to storage")
                
            except Exception as e:
                export_job_manager.fail_job(job_id, f"Upload failed: {str(e)}")
                return
            
            # Step 7: Complete job (100%)
            export_job_manager.complete_job(job_id, download_url)
            
            # Clean up temp files
            self._cleanup_temp_files(job_id)
            
        except Exception as e:
            logger.error(f"Video export failed for job {job_id}: {e}")
            from .job_manager import export_job_manager
            export_job_manager.fail_job(job_id, f"Unexpected error: {str(e)}")
    
    def _create_final_audio_file(self, audio_tracks: List, job_id: str, timeline_duration: float = None) -> str:
        """Create final audio file from mixed tracks"""
        try:
            # Use timeline duration or calculate from tracks
            if timeline_duration:
                total_duration = timeline_duration / 1000  # Convert ms to seconds
            else:
                total_duration = max([track.end_time for track in audio_tracks]) if audio_tracks else 10
            
            logger.info(f"Creating final audio with duration: {total_duration}s for {len(audio_tracks)} tracks")
            
            if not audio_tracks:
                # Create silent audio with proper duration
                temp_audio_path = os.path.join(self.settings.TEMP_DIR, f"silent_audio_{job_id}.wav")
                silent_audio = np.zeros(int(AUDIO_SAMPLE_RATE * total_duration), dtype=np.float32)
                sf.write(temp_audio_path, silent_audio, AUDIO_SAMPLE_RATE)
                return temp_audio_path
            
            # Mix multiple audio tracks properly
            final_audio = self._mix_multiple_audio_tracks(audio_tracks, total_duration, job_id)
            temp_audio_path = os.path.join(self.settings.TEMP_DIR, f"final_mixed_audio_{job_id}.wav")
            sf.write(temp_audio_path, final_audio, AUDIO_SAMPLE_RATE)
            return temp_audio_path
            
        except Exception as e:
            logger.error(f"Failed to create final audio file: {e}")
            # Create silent audio as fallback with proper duration
            duration = timeline_duration / 1000 if timeline_duration else 10
            temp_audio_path = os.path.join(self.settings.TEMP_DIR, f"error_audio_{job_id}.wav")
            silent_audio = np.zeros(int(AUDIO_SAMPLE_RATE * duration), dtype=np.float32)
            sf.write(temp_audio_path, silent_audio, AUDIO_SAMPLE_RATE)
            return temp_audio_path
    
    def _mix_multiple_audio_tracks(self, audio_tracks: List, total_duration: float, job_id: str) -> np.ndarray:
        """
        Mix audio tracks: voice segments + instruments (if enabled)
        """
        try:
            sample_rate = AUDIO_SAMPLE_RATE
            final_samples = int(sample_rate * total_duration)
            mixed_audio = np.zeros(final_samples, dtype=np.float32)
            
            logger.info(f"🎵 Mixing {len(audio_tracks)} tracks for {total_duration}s")
            
            # Separate tracks by type
            voice_segments = []
            instrument_tracks = []
            
            for track in audio_tracks:
                audio_type = track.voice_clone_info.audio_type if hasattr(track, 'voice_clone_info') else "unknown"
                if audio_type == "cloned":
                    voice_segments.append(track)
                elif audio_type == "instruments":
                    instrument_tracks.append(track)
            
            # 1. Process voice segments with proper trim handling
            if voice_segments:
                voice_audio = self._process_voice_segments(voice_segments, total_duration, job_id)
                mixed_audio += voice_audio
                logger.info(f"✅ Added {len(voice_segments)} voice segments")
            
            # 2. Process instruments (simple mixing)
            for track in instrument_tracks:
                instrument_audio = self._process_single_track(track, job_id, "instrument")
                if instrument_audio is not None:
                    # Apply track volume + soft background ratio
                    final_volume = track.volume * INSTRUMENT_VOLUME_RATIO
                    instrument_audio = instrument_audio * final_volume
                    mixed_audio[:min(len(instrument_audio), final_samples)] += instrument_audio[:final_samples]
                    logger.info(f"✅ Added instrument track: {track.id}")
            
            # Simple normalization
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 0.95:
                mixed_audio = mixed_audio * (0.95 / max_val)
            
            return mixed_audio
            
        except Exception as e:
            logger.error(f"Audio mixing failed: {e}")
            return np.zeros(final_samples, dtype=np.float32)
    
    def _process_voice_segments(self, voice_segments: List, total_duration: float, job_id: str) -> np.ndarray:
        """
        Process voice segments with proper trim handling and gap filling
        """
        sample_rate = AUDIO_SAMPLE_RATE
        final_samples = int(sample_rate * total_duration)
        voice_audio = np.zeros(final_samples, dtype=np.float32)
        
        # Sort by start time
        sorted_segments = sorted(voice_segments, key=lambda x: x.start_time)
        
        for segment in sorted_segments:
            segment_audio = self._process_single_track(segment, job_id, "voice")
            if segment_audio is None:
                continue
            
            # Apply trim if specified (IMPORTANT as user requested)
            if segment.trim_start > 0 or segment.trim_end is not None:
                trim_start_sample = int(segment.trim_start * sample_rate)
                if segment.trim_end is not None:
                    trim_end_sample = int(segment.trim_end * sample_rate)
                    segment_audio = segment_audio[trim_start_sample:trim_end_sample]
                else:
                    segment_audio = segment_audio[trim_start_sample:]
                logger.debug(f"Applied trim to segment {segment.id}")
            
            # Apply speed changes
            if segment.playback_rate != 1.0:
                new_length = int(len(segment_audio) / segment.playback_rate)
                segment_audio = signal.resample(segment_audio, new_length)
            
            # Place in timeline at correct position
            start_sample = int(segment.start_time * sample_rate)
            end_sample = min(start_sample + len(segment_audio), final_samples)
            
            if end_sample > start_sample:
                voice_audio[start_sample:end_sample] += segment_audio[:end_sample-start_sample] * VOICE_SEGMENT_VOLUME
        
        return voice_audio
    
    def _process_single_track(self, track, job_id: str, track_type: str) -> np.ndarray:
        """
        Download and process a single audio track
        """
        try:
            if not track.src:
                return None
            
            # Download audio
            response = requests.get(track.src, timeout=120)
            if response.status_code != 200:
                logger.warning(f"Failed to download {track_type} track: {track.id}")
                return None
            
            # Save temporarily
            temp_path = os.path.join(self.settings.TEMP_DIR, f"{track_type}_{track.id}_{job_id}.wav")
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            # Load and process audio
            audio_data, sr = sf.read(temp_path)
            
            # Resample if needed
            if sr != AUDIO_SAMPLE_RATE:
                audio_data = signal.resample(audio_data, int(len(audio_data) * AUDIO_SAMPLE_RATE / sr))
            
            # Convert stereo to mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Cleanup temp file
            os.unlink(temp_path)
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Failed to process {track_type} track {track.id}: {e}")
            return None


    
    def _cleanup_temp_files(self, job_id: str):
        """Clean up temp files for export job"""
        try:
            import shutil
            temp_job_dir = Path(self.settings.TEMP_DIR) / f"render_{job_id}"
            if temp_job_dir.exists():
                shutil.rmtree(temp_job_dir)
                
            # Clean up other temp files
            temp_dir = Path(self.settings.TEMP_DIR)
            for temp_file in temp_dir.glob(f"*{job_id}*"):
                if temp_file.is_file():
                    temp_file.unlink()
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files for job {job_id}: {e}") 