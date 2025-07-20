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
            
            # Apply audio changes
            editing_changes = export_data["editingChanges"]
            if "volumeChanges" in editing_changes:
                audio_mixer.apply_volume_changes(
                    processed_items.audio_tracks, editing_changes["volumeChanges"]
                )
            if "speedChanges" in editing_changes:
                audio_mixer.apply_speed_changes(
                    processed_items.audio_tracks, editing_changes["speedChanges"]
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
            final_audio_path = self._create_final_audio_file(final_audio_tracks, job_id)
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
            
            # Get optional URLs from export data
            video_url = export_data["voiceCloneData"].get("originalVideoUrl")
            instruments_url = export_data.get("instrumentsUrl")
            subtitles_url = export_data.get("subtitlesUrl")
            
            render_result = asyncio.run(video_renderer.render_video(
                video_url=video_url,
                final_audio_path=final_audio_path,
                instruments_url=instruments_url,
                subtitles_url=subtitles_url,
                config=video_config,
                job_id=job_id
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
    
    def _create_final_audio_file(self, audio_tracks: List, job_id: str) -> str:
        """Create final audio file from mixed tracks"""
        try:
            if not audio_tracks:
                # Create silent audio
                temp_audio_path = os.path.join(self.settings.TEMP_DIR, f"silent_audio_{job_id}.wav")
                silent_audio = np.zeros(int(44100 * 10), dtype=np.float32)  # 10 seconds silent
                sf.write(temp_audio_path, silent_audio, 44100)
                return temp_audio_path
            
            # For now, assume first track is the main audio
            # In a real implementation, you'd mix all tracks properly
            if len(audio_tracks) == 1:
                track = audio_tracks[0]
                if hasattr(track, 'src') and track.src:
                    # Download the audio track
                    response = requests.get(track.src, timeout=120)
                    if response.status_code == 200:
                        temp_audio_path = os.path.join(self.settings.TEMP_DIR, f"final_audio_{job_id}.wav")
                        with open(temp_audio_path, 'wb') as f:
                            f.write(response.content)
                        return temp_audio_path
            
            # Fallback: create silent audio
            temp_audio_path = os.path.join(self.settings.TEMP_DIR, f"fallback_audio_{job_id}.wav")
            silent_audio = np.zeros(int(44100 * 10), dtype=np.float32)
            sf.write(temp_audio_path, silent_audio, 44100)
            return temp_audio_path
            
        except Exception as e:
            logger.error(f"Failed to create final audio file: {e}")
            # Create silent audio as fallback
            temp_audio_path = os.path.join(self.settings.TEMP_DIR, f"error_audio_{job_id}.wav")
            silent_audio = np.zeros(int(44100 * 10), dtype=np.float32)
            sf.write(temp_audio_path, silent_audio, 44100)
            return temp_audio_path
    
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