"""
Video Processing Module

Handles video processing with audio replacement.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from app.config.settings import settings




class VideoProcessor:
    
    def __init__(self, temp_dir: str = "./tmp/video_processing"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.subtitle_font_size = 18
        self.subtitle_margin_bottom = 30
        self.words_per_subtitle = 3


    def _load_subtitles_from_manifest(self, manifest_path: str) -> List[Dict[str, Any]]:
        try:
            from app.services.dub.manifest_manager import manifest_manager
            manifest = manifest_manager.load_manifest(manifest_path)
            return manifest_manager.get_segments_for_subtitles(manifest)
        except Exception:
            return []


    def _create_video_ffmpeg(self, video_path: str, audio_path: str, 
                           subtitle_path: Optional[str], output_path: Path) -> Dict[str, Any]:
        try:
            # Validate input files
            if not video_path or not os.path.exists(video_path):
                return {"success": False, "error": f"Invalid video file: {video_path}"}
            if not audio_path or not os.path.exists(audio_path):
                return {"success": False, "error": f"Invalid audio file: {audio_path}"}
            
            ffmpeg_cmd = self._get_ffmpeg_path()

            cmd = [
                ffmpeg_cmd, '-y',
                '-i', video_path,
                '-i', str(audio_path),
            ]
            
            if subtitle_path and Path(subtitle_path).exists():
                # Proper Windows path handling for FFmpeg
                import platform
                if platform.system() == 'Windows':
                    # For Windows, escape backslashes properly
                    subtitle_path_str = str(subtitle_path).replace('\\', '\\\\')
                else:
                    # For Unix-like systems, use as-is
                    subtitle_path_str = str(subtitle_path)
                
                # High-quality settings optimized for social media platforms
                video_codec = 'h264_nvenc' if settings.FFMPEG_USE_GPU else 'libx264'
                preset = 'fast' if settings.FFMPEG_USE_GPU else 'slow'  # Better quality preset
                cmd.extend([
                    '-vf', f"subtitles='{subtitle_path_str}':force_style='Fontname=Arial-Bold,Fontsize={self.subtitle_font_size},Bold=1,PrimaryColour=&H00ffffff,OutlineColour=&H00000000,Outline=3,Alignment=2,MarginV={self.subtitle_margin_bottom}'",
                    '-c:v', video_codec,
                    '-preset', preset,
                    '-crf', '20',  # High quality for platform upload
                    '-maxrate', '6000k',  # Max bitrate for platform compatibility
                    '-bufsize', '12000k',  # Buffer size for consistent quality
                    '-profile:v', 'high',  # High profile for better compression
                    '-level', '4.1',  # Compatibility with most platforms
                ])
            else:
                # High-quality re-encoding even without subtitles for platform optimization
                video_codec = 'h264_nvenc' if settings.FFMPEG_USE_GPU else 'libx264'
                preset = 'fast' if settings.FFMPEG_USE_GPU else 'slow'
                cmd.extend([
                    '-c:v', video_codec,
                    '-preset', preset,
                    '-crf', '20',  # Consistent high quality
                    '-maxrate', '6000k',
                    '-bufsize', '12000k',
                    '-profile:v', 'high',
                    '-level', '4.1',
                ])
            
            cmd.extend([
                '-c:a', 'aac',
                '-b:a', '128k',  # Optimized audio quality for platform uploads
                '-ar', '48000',  # Professional audio sample rate
                '-ac', '2',  # Stereo audio
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                '-movflags', '+faststart',  # Optimize for streaming
                str(output_path)
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                return {"success": True}
            else:
                return {"success": False, "error": f"FFmpeg error: {result.stderr}"}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "FFmpeg timed out after 5 minutes"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_final_audio(self, audio_path: str, instruments_path: str, audio_id: str) -> Path:
        """Mix original audio with instrument track and return mixed file path"""
        try:
            from .audio_utils import AudioUtils
            output_path = self.temp_dir / f"final_mix_{audio_id}.wav"
            AudioUtils.mix_audio_files(audio_path, instruments_path, str(output_path))
            return output_path
        except Exception:
            # If mixing fails, fallback to original audio
            return Path(audio_path)


    def create_srt_file(self, subtitle_data: List[Dict], output_path: Path) -> None:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, subtitle in enumerate(subtitle_data, 1):
                start_time = self._seconds_to_srt_time(subtitle['start'])
                end_time = self._seconds_to_srt_time(subtitle['end'])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{subtitle['text']}\n\n")
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    # Removed cleanup_temp_files - cleanup only happens after successful upload

    def create_video_with_subtitles(self, video_path: str, audio_path: str, manifest_path: str, audio_id: str, instruments_path: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not video_path or not os.path.exists(video_path):
                return {"success": False, "error": f"Invalid video file: {video_path}"}
            if not audio_path or not os.path.exists(audio_path):
                return {"success": False, "error": f"Invalid audio file: {audio_path}"}
            if instruments_path and os.path.exists(instruments_path):
                final_audio_path = self._create_final_audio(audio_path, instruments_path, audio_id)
            else:
                final_audio_path = Path(audio_path)
            
            subtitle_data = self._load_subtitles_from_manifest(manifest_path)
            if not subtitle_data:
                return self.create_video_with_audio(video_path, str(final_audio_path), audio_id, instruments_path)
            
            subtitle_path = self.temp_dir / f"subtitles_{audio_id}.srt"
            self.create_srt_file(subtitle_data, subtitle_path)
            output_path = self.temp_dir / f"video_with_subtitles_{audio_id}.mp4"
            result = self._create_video_ffmpeg(video_path, str(final_audio_path), subtitle_path, output_path)
            if result["success"]:
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "subtitle_path": str(subtitle_path),
                    "subtitle_count": len(subtitle_data) if subtitle_data is not None else None,
                    "has_subtitles": True,
                    "final_audio_path": str(final_audio_path)
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error in video creation")
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Video creation with subtitles failed: {str(e)}"
            }

    def create_video_with_audio(self, video_path: str, audio_path: str, audio_id: str, instruments_path: Optional[str] = None, segments_dir: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Validate input files early
            if not video_path or not os.path.exists(video_path):
                return {"success": False, "error": f"Invalid video file: {video_path}"}
            if not audio_path or not os.path.exists(audio_path):
                return {"success": False, "error": f"Invalid audio file: {audio_path}"}
            output_path = self.temp_dir / f"video_no_subtitles_{audio_id}.mp4"
            if instruments_path and os.path.exists(instruments_path):
                final_audio_path = self._create_final_audio(audio_path, instruments_path, audio_id)
            else:
                final_audio_path = Path(audio_path)
            result = self._create_video_ffmpeg(video_path, str(final_audio_path), None, output_path)
            if result["success"]:
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "has_subtitles": False,
                    "final_audio_path": str(final_audio_path)
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error in video creation")
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Video creation with audio failed: {str(e)}"
            }
    
    # ---------------------------------------------------------------------
    # FFmpeg path discovery
    # ---------------------------------------------------------------------
    def _get_ffmpeg_path(self):
        """Get FFmpeg executable path based on platform"""
        from app.utils.ffmpeg_helper import get_ffmpeg_path
        return get_ffmpeg_path()
