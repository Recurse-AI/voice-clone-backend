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

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _load_subtitles(self, segments_dir: str) -> List[Dict[str, Any]]:
        """Load subtitle information from segment_*_info.json files.

        Returns a list of dicts with keys: start, end, text (seconds).
        If no JSON files found, returns an empty list so caller can fallback.
        """
        try:
            dir_path = Path(segments_dir)
            if not dir_path.exists():
                return []
            subtitle_entries: List[Dict[str, Any]] = []
            for info_file in sorted(dir_path.glob("segment_*_info.json")):
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    start_sec = data.get("start", 0) / 1000.0
                    end_sec = data.get("end", 0) / 1000.0
                    text = data.get("dubbed_text") or data.get("original_text")
                    if text:
                        subtitle_entries.append({"start": start_sec, "end": end_sec, "text": text})
                except Exception:
                    continue
            return subtitle_entries
        except Exception:
            return []

    # ---------------------------------------------------------------------
    # FFmpeg wrapper
    # ---------------------------------------------------------------------
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
                # Convert to POSIX style path to avoid backslash escaping issues on Windows
                subtitle_path_str = str(subtitle_path).replace('\\', '/')
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
                '-b:a', '192k',  # Higher audio quality for platform uploads
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

    # ---------------------------------------------------------------------
    # Public methods (create video)
    # ---------------------------------------------------------------------
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

    def create_video_with_subtitles(self, video_path: str, audio_path: str, segments_dir: str, audio_id: str, instruments_path: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Validate input files early
            if not video_path or not os.path.exists(video_path):
                return {"success": False, "error": f"Invalid video file: {video_path}"}
            if not audio_path or not os.path.exists(audio_path):
                return {"success": False, "error": f"Invalid audio file: {audio_path}"}
            if instruments_path and os.path.exists(instruments_path):
                final_audio_path = self._create_final_audio(audio_path, instruments_path, audio_id)
            else:
                final_audio_path = Path(audio_path)
            # If an SRT file already exists in segments_dir (generated upstream), use it directly
            pre_existing_srt = Path(segments_dir) / f"subtitles_{audio_id}.srt"
            if pre_existing_srt.exists():
                subtitle_path = pre_existing_srt
                subtitle_data = None  # Unknown, we won't count lines
            else:
                subtitle_data = self._load_subtitles(segments_dir)
                if not subtitle_data:
                    # No subtitle info → fallback to simple audio replace
                    return self.create_video_with_audio(video_path, str(final_audio_path), audio_id, instruments_path, segments_dir)
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
        # Check if ffmpeg is in PATH
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return 'ffmpeg'
        except FileNotFoundError:
            pass

        # Check local FFmpeg installation in project directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_ffmpeg_paths = [
            os.path.join(project_root, 'ffmpeg-master-latest-win64-gpl', 'ffmpeg-master-latest-win64-gpl', 'bin', 'ffmpeg.exe'),
            os.path.join(project_root, 'ffmpeg-master-latest-win64-gpl', 'bin', 'ffmpeg.exe'),
            os.path.join(project_root, 'ffmpeg', 'bin', 'ffmpeg.exe'),
            os.path.join(project_root, 'ffmpeg.exe')
        ]
        
        for ffmpeg_path in local_ffmpeg_paths:
            if os.path.exists(ffmpeg_path):
                return ffmpeg_path

        # Check if ffmpeg is in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ffmpeg_path = os.path.join(script_dir, 'ffmpeg.exe')
        if os.path.exists(ffmpeg_path):
            return ffmpeg_path
        
        return None
