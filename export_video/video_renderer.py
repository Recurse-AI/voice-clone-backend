"""
Video Renderer
Handles final video composition using existing video_processor functionality
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import requests
import soundfile as sf

from .models import VideoItem, AudioItem, TextItem, ImageItem, VideoConfig
from .constants import (
    HTTP_TIMEOUT_LONG, HTTP_TIMEOUT_MEDIUM, HTTP_TIMEOUT_SHORT,
    DEFAULT_CHUNK_SIZE, SUBTITLE_FONT_SIZE, SUBTITLE_MARGIN_BOTTOM
)

logger = logging.getLogger(__name__)

class VideoRenderer:
    """
    Render final video using existing video_processor functionality
    """
    
    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Use existing video processor
        from dub.video_processor import VideoProcessor
        self.video_processor = VideoProcessor(str(self.temp_dir))
    
    async def render_video(self, video_url: Optional[str], final_audio_path: str,
                          instruments_url: Optional[str], subtitles_url: Optional[str],
                          config: VideoConfig, job_id: str, processed_items=None, 
                          export_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main video rendering function using existing video processor
        """
        try:
            job_temp_dir = self.temp_dir / f"render_{job_id}"
            job_temp_dir.mkdir(exist_ok=True)
            
            logger.info(f"Starting video render for job {job_id}")
            
            # Download original video if provided
            video_path = None
            if video_url:
                video_path = await self._download_video(video_url, job_temp_dir)
            
            # Download instruments if provided
            instruments_path = None
            if instruments_url:
                logger.info(f"📁 Downloading instruments from: {instruments_url}")
                instruments_path = await self._download_instruments(instruments_url, job_temp_dir)
            else:
                logger.info("📁 No instruments URL provided - skipping instruments")
            
            # Download subtitle file if provided
            subtitle_file = None
            if subtitles_url:
                logger.info(f"📄 Downloading subtitles from: {subtitles_url}")
                subtitle_file = await self._download_subtitles(subtitles_url, job_temp_dir)
            else:
                logger.info("📄 No subtitles URL provided - skipping subtitles")
            
            # Use existing video processor functionality
            if video_path:
                # Create video with existing video
                if subtitle_file:
                    # Use subtitle file directly with video processor
                    result = self._create_video_with_overlays(
                        video_path, final_audio_path, subtitle_file, 
                        instruments_path, job_id, config, processed_items, export_settings
                    )
                else:
                    # Create video without subtitles but with overlays
                    result = self._create_video_with_overlays(
                        video_path, final_audio_path, None, 
                        instruments_path, job_id, config, processed_items, export_settings
                    )
            else:
                # Create blank video with audio
                result = self._create_blank_video_with_audio(
                    final_audio_path, instruments_path, config, job_id
                )
            
            if result and result.get("success"):
                return {
                    "success": True,
                    "video_path": result["video_path"],
                    "duration": config.duration,
                    "format": config.format,
                    "has_subtitles": bool(subtitle_file)
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Video creation failed")
                }
            
        except Exception as e:
            logger.error(f"Video rendering failed for job {job_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _download_video(self, video_url: str, temp_dir: Path) -> str:
        """Download original video file"""
        try:
            response = requests.get(video_url, stream=True, timeout=HTTP_TIMEOUT_LONG)
            response.raise_for_status()
            
            video_path = temp_dir / "original_video.mp4"
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=DEFAULT_CHUNK_SIZE):
                    f.write(chunk)
            
            logger.info(f"Downloaded video: {video_path}")
            return str(video_path)
            
        except Exception as e:
            logger.error(f"Failed to download video from {video_url}: {e}")
            raise
    
    async def _download_instruments(self, instruments_url: str, temp_dir: Path) -> str:
        """Download instruments audio file"""
        try:
            response = requests.get(instruments_url, stream=True, timeout=HTTP_TIMEOUT_LONG)
            response.raise_for_status()
            
            instruments_path = temp_dir / "instruments.wav"
            with open(instruments_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=DEFAULT_CHUNK_SIZE):
                    f.write(chunk)
            
            logger.info(f"Downloaded instruments: {instruments_path}")
            return str(instruments_path)
            
        except Exception as e:
            logger.error(f"Failed to download instruments from {instruments_url}: {e}")
            raise
    
    async def _download_subtitles(self, subtitles_url: str, temp_dir: Path) -> str:
        """Download SRT subtitle file"""
        try:
            response = requests.get(subtitles_url, stream=True, timeout=HTTP_TIMEOUT_MEDIUM)
            response.raise_for_status()
            
            subtitle_path = temp_dir / "subtitles.srt"
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logger.info(f"Downloaded subtitles: {subtitle_path}")
            return str(subtitle_path)
            
        except Exception as e:
            logger.error(f"Failed to download subtitles from {subtitles_url}: {e}")
            raise
    
    def _create_video_with_custom_subtitle(self, video_path: str, audio_path: str, 
                                         subtitle_path: str, instruments_path: Optional[str], 
                                         job_id: str, config: VideoConfig) -> Dict[str, Any]:
        """Create video with custom subtitle file using video processor"""
        try:
            import subprocess
            import tempfile
            
            # Create final audio with instruments if provided
            final_audio_path = audio_path
            if instruments_path and os.path.exists(instruments_path):
                final_audio_path = self._mix_audio_with_instruments(
                    audio_path, instruments_path, job_id
                )
            
            # Use FFmpeg to create video with custom subtitles
            output_path = self.temp_dir / f"video_with_custom_subtitles_{job_id}.mp4"
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', final_audio_path,
                '-vf', f"subtitles='{subtitle_path}':force_style='Fontname=Arial-Bold,Fontsize=18,Bold=1,PrimaryColour=&H00ffffff,OutlineColour=&H00000000,Outline=3,Alignment=2,MarginV=30'",
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-b:v', '2000k',  # Default - should be updated to use settings
                '-c:a', 'aac',
                '-b:a', '128k',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-t', str(config.duration),  # Use timeline duration
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "has_subtitles": True
                }
            else:
                return {
                    "success": False,
                    "error": f"FFmpeg error: {result.stderr}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Video creation with custom subtitles failed: {str(e)}"
            }
    
    def _create_video_without_subtitles(self, video_path: str, audio_path: str,
                                      instruments_path: Optional[str], job_id: str, 
                                      config: VideoConfig) -> Dict[str, Any]:
        """Create video without subtitles using timeline duration"""
        try:
            import subprocess
            
            # Create final audio with instruments if provided
            final_audio_path = audio_path
            if instruments_path and os.path.exists(instruments_path):
                final_audio_path = self._mix_audio_with_instruments(
                    audio_path, instruments_path, job_id
                )
            
            # Use FFmpeg to create video
            output_path = self.temp_dir / f"video_no_subtitles_{job_id}.mp4"
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', final_audio_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-b:v', '2000k',  # Optimal bitrate
                '-c:a', 'aac',
                '-b:a', '128k',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-t', str(config.duration),  # Use timeline duration
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "has_subtitles": False
                }
            else:
                return {
                    "success": False,
                    "error": f"FFmpeg error: {result.stderr}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Video creation without subtitles failed: {str(e)}"
            }
    
    def _create_video_with_overlays(self, video_path: str, audio_path: str,
                                   subtitle_path: Optional[str], instruments_path: Optional[str],
                                   job_id: str, config: VideoConfig, processed_items=None,
                                   export_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create video with text/image overlays and subtitles"""
        try:
            import subprocess
            
            # Create final audio with instruments if provided
            final_audio_path = audio_path
            if instruments_path and os.path.exists(instruments_path):
                final_audio_path = self._mix_audio_with_instruments(
                    audio_path, instruments_path, job_id
                )
            
            # Build FFmpeg command with overlays
            output_path = self.temp_dir / f"video_with_overlays_{job_id}.mp4"
            
            # Start with basic inputs
            cmd = ['ffmpeg', '-y', '-i', video_path, '-i', final_audio_path]
            input_count = 2
            
            # Download and add image overlays as inputs
            image_files = []
            text_overlays = []
            image_overlays = []
            
            if processed_items:
                # Process text overlays
                for text_item in processed_items.text_overlays:
                    if text_item.start_time < text_item.end_time:  # Valid duration
                        text_overlays.append(text_item)
                
                # Download and prepare image overlays
                for image_item in processed_items.image_overlays:
                    if image_item.start_time < image_item.end_time and image_item.src:
                        image_path = self._download_image_overlay(image_item.src, job_id)
                        if image_path:
                            cmd.extend(['-i', image_path])
                            image_files.append((input_count, image_item))
                            input_count += 1
            
            # Build filter complex
            filter_parts = []
            last_video_ref = "0:v"
            
            # Add image overlays
            for i, (input_idx, image_item) in enumerate(image_files):
                next_ref = f"v{i}"
                overlay_filter = (
                    f"[{last_video_ref}][{input_idx}:v]"
                    f"overlay={image_item.position.x}:{image_item.position.y}:"
                    f"enable='between(t,{image_item.start_time},{image_item.end_time})'"
                    f"[{next_ref}]"
                )
                filter_parts.append(overlay_filter)
                last_video_ref = next_ref
            
            # Add text overlays
            for i, text_item in enumerate(text_overlays):
                next_ref = f"t{i}"
                # Simple text overlay (escape quotes)
                text_safe = text_item.text.replace("'", "\\'").replace(":", "\\:")
                
                text_filter = (
                    f"[{last_video_ref}]drawtext="
                    f"text='{text_safe}':"
                    f"x={text_item.position.x}:y={text_item.position.y}:"
                    f"fontsize={text_item.styling.font_size}:"
                    f"fontcolor={text_item.styling.color}:"
                    f"enable='between(t,{text_item.start_time},{text_item.end_time})'"
                    f"[{next_ref}]"
                )
                filter_parts.append(text_filter)
                last_video_ref = next_ref
            
            # Add subtitle filter if provided
            if subtitle_path:
                next_ref = "final"
                subtitle_filter = (
                    f"[{last_video_ref}]subtitles='{subtitle_path}':force_style="
                    f"'Fontname=Arial-Bold,Fontsize=18,Bold=1,PrimaryColour=&H00ffffff,"
                    f"OutlineColour=&H00000000,Outline=3,Alignment=2,MarginV=30'[{next_ref}]"
                )
                filter_parts.append(subtitle_filter)
                last_video_ref = next_ref
            
            # Add filter complex if we have filters
            if filter_parts:
                cmd.extend(['-filter_complex', ';'.join(filter_parts)])
                cmd.extend(['-map', f'[{last_video_ref}]'])
            else:
                cmd.extend(['-map', '0:v:0'])
            
            # Get bitrate from frontend settings or use default
            bitrate = '2000k'  # Default fallback
            if export_settings and 'bitrate' in export_settings:
                bitrate = export_settings['bitrate']
                logger.info(f"Using frontend bitrate setting: {bitrate}")
            else:
                logger.info(f"Using default bitrate: {bitrate}")
            
            # Add audio mapping and codec settings
            cmd.extend([
                '-map', '1:a:0',
                '-c:v', 'libx264',
                '-preset', 'medium',  # Faster encoding
                '-b:v', bitrate,  # Use bitrate from frontend settings
                '-c:a', 'aac',
                '-b:a', '128k',
                '-t', str(config.duration),
                str(output_path)
            ])
            
            logger.info(f"Running FFmpeg with overlays: {len(text_overlays)} text, {len(image_files)} images")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "has_overlays": len(text_overlays) + len(image_files) > 0,
                    "has_subtitles": bool(subtitle_path)
                }
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return {
                    "success": False,
                    "error": f"FFmpeg error: {result.stderr}"
                }
        
        except Exception as e:
            logger.error(f"Video creation with overlays failed: {e}")
            return {
                "success": False,
                "error": f"Video creation with overlays failed: {str(e)}"
            }
    
    def _download_image_overlay(self, image_url: str, job_id: str) -> Optional[str]:
        """Download image for overlay"""
        try:
            response = requests.get(image_url, timeout=HTTP_TIMEOUT_SHORT)
            response.raise_for_status()
            
            # Determine file extension
            content_type = response.headers.get('content-type', '')
            if 'png' in content_type:
                ext = 'png'
            elif 'gif' in content_type:
                ext = 'gif'
            elif 'webp' in content_type:
                ext = 'webp'
            else:
                ext = 'jpg'
            
            image_path = self.temp_dir / f"overlay_image_{job_id}.{ext}"
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded image overlay: {image_path}")
            return str(image_path)
        
        except Exception as e:
            logger.warning(f"Failed to download image overlay {image_url}: {e}")
            return None
    
    def _mix_audio_with_instruments(self, audio_path: str, instruments_path: str, 
                                   job_id: str) -> str:
        """Mix cloned audio with instruments"""
        try:
            mixed_audio_path = self.temp_dir / f"mixed_audio_{job_id}.wav"
            
            # Load audio files
            cloned_audio, sr1 = sf.read(audio_path)
            instruments_audio, sr2 = sf.read(instruments_path)
            
            # Ensure same sample rate
            if sr1 != sr2:
                # Simple resampling by padding/truncating
                if sr2 > sr1:
                    instruments_audio = instruments_audio[::int(sr2/sr1)]
                
            # Match lengths
            min_length = min(len(cloned_audio), len(instruments_audio))
            cloned_audio = cloned_audio[:min_length]
            instruments_audio = instruments_audio[:min_length]
            
            # Mix: 100% vocals (full volume), 15% instruments (very soft background)
            mixed_audio = cloned_audio * 1.0 + instruments_audio * 0.15
            
            # Save mixed audio
            sf.write(mixed_audio_path, mixed_audio, sr1)
            
            return str(mixed_audio_path)
            
        except Exception as e:
            logger.error(f"Failed to mix audio with instruments: {e}")
            return audio_path
    
    def _create_blank_video_with_audio(self, audio_path: str, instruments_path: Optional[str], 
                                     config: VideoConfig, job_id: str) -> Dict[str, Any]:
        """Create blank video with audio for audio-only projects"""
        try:
            import subprocess
            
            # Mix audio with instruments if provided
            final_audio_path = audio_path
            if instruments_path and os.path.exists(instruments_path):
                final_audio_path = self._mix_audio_with_instruments(
                    audio_path, instruments_path, job_id
                )
            
            # Create blank video with audio
            output_path = self.temp_dir / f"blank_video_with_audio_{job_id}.mp4"
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', f'color=black:size={config.width}x{config.height}:rate={config.fps}',
                '-i', final_audio_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-b:v', '2000k',  # Optimal bitrate
                '-c:a', 'aac',
                '-b:a', '128k',
                '-t', str(config.duration),
                '-shortest',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "has_subtitles": False
                }
            else:
                return {
                    "success": False,
                    "error": f"FFmpeg error: {result.stderr}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Blank video creation failed: {str(e)}"
            } 