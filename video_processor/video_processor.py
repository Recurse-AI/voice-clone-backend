"""
Video Processing Module - Production Clean Version

Handles video processing with voice-cloned audio and optional subtitles.
Only processes subtitles from voice-cloned content, not original video.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import soundfile as sf

# Try to import MoviePy with fallback
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


class VideoProcessor:
    """Clean video processor for voice-cloned content with optional subtitles"""
    
    def __init__(self, temp_dir: str = "/tmp/video_processing"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def create_video_with_subtitles(self, video_path: str, audio_path: str, 
                                   segments_dir: str, audio_id: str,
                                   instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with voice-cloned audio and subtitles"""
        try:
            output_path = self.temp_dir / f"video_with_subtitles_{audio_id}.mp4"
            
            # Load subtitle data from voice-cloned segments only
            subtitle_data = self._load_voice_cloned_subtitles(segments_dir)
            
            if not subtitle_data:
                return {"success": False, "error": "No voice-cloned subtitle data found"}
            
            # Create video with subtitles
            if MOVIEPY_AVAILABLE:
                result = self._create_video_with_subtitles_moviepy(
                    video_path, audio_path, subtitle_data, output_path, instruments_path
                )
            else:
                result = self._create_video_with_subtitles_ffmpeg(
                    video_path, audio_path, subtitle_data, output_path, instruments_path
                )
            
            if result["success"]:
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "subtitle_count": len(subtitle_data),
                    "duration": result.get("duration", 0),
                    "file_size": os.path.getsize(output_path) / (1024*1024)
                }
            else:
                return result
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_video_with_audio(self, video_path: str, audio_path: str, 
                              audio_id: str, instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with voice-cloned audio only (no subtitles)"""
        try:
            output_path = self.temp_dir / f"video_no_subtitles_{audio_id}.mp4"
            
            # Create video without subtitles
            if MOVIEPY_AVAILABLE:
                result = self._create_video_audio_only_moviepy(
                    video_path, audio_path, output_path, instruments_path
                )
            else:
                result = self._create_video_audio_only_ffmpeg(
                    video_path, audio_path, output_path, instruments_path
                )
            
            if result["success"]:
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "duration": result.get("duration", 0),
                    "file_size": os.path.getsize(output_path) / (1024*1024),
                    "subtitles_included": False
                }
            else:
                return result
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _load_voice_cloned_subtitles(self, segments_dir: str) -> List[Dict]:
        """Load subtitles from voice-cloned segments only"""
        segments_path = Path(segments_dir)
        subtitle_data = []
        
        # Process voice-cloned segments
        for speaker_dir in segments_path.glob("speaker_*/segments"):
            for json_file in speaker_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Only use English text from voice-cloned content
                    english_text = data.get('english_text', data.get('text', ''))
                    
                    if english_text and data.get('start') is not None and data.get('end') is not None:
                        subtitle_data.append({
                            'start': data['start'],
                            'end': data['end'],
                            'text': english_text.strip(),
                            'duration': data['end'] - data['start']
                        })
                
                except Exception:
                    continue
        
        # Sort by start time
        subtitle_data.sort(key=lambda x: x['start'])
        return subtitle_data
    
    def _create_video_with_subtitles_moviepy(self, video_path: str, audio_path: str, 
                                           subtitle_data: List[Dict], output_path: Path,
                                           instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with subtitles using MoviePy"""
        try:
            from moviepy.editor import TextClip, CompositeVideoClip
            
            # Load video and prepare audio
            video = VideoFileClip(video_path)
            final_audio = self._prepare_audio_moviepy(audio_path, instruments_path)
            
            # Create subtitle clips
            subtitle_clips = []
            for subtitle in subtitle_data:
                try:
                    text_clip = TextClip(
                        subtitle['text'],
                        fontsize=32,
                        font='Arial-Bold',
                        color='white',
                        stroke_color='black',
                        stroke_width=3,
                        method='caption',
                        size=(video.w - 100, None),
                        align='center'
                    ).set_duration(subtitle['duration']).set_start(subtitle['start'])
                    
                    # Position at bottom
                    text_clip = text_clip.set_position(('center', video.h - 120))
                    subtitle_clips.append(text_clip)
                    
                except Exception:
                    continue
            
            # Compose final video
            final_video = video.set_audio(final_audio)
            if subtitle_clips:
                final_video = CompositeVideoClip([final_video] + subtitle_clips)
            
            # Render
            final_video.write_videofile(
                str(output_path),
                fps=24,
                codec='libx264',
                audio_codec='aac',
                bitrate='1500k',
                audio_bitrate='128k',
                verbose=False,
                logger=None,
                threads=2
            )
            
            # Cleanup
            self._cleanup_moviepy_objects(video, final_audio, final_video)
            
            return {
                "success": True,
                "duration": video.duration,
                "subtitle_count": len(subtitle_clips)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_video_with_subtitles_ffmpeg(self, video_path: str, audio_path: str,
                                          subtitle_data: List[Dict], output_path: Path,
                                          instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with subtitles using FFmpeg"""
        try:
            # Create temporary files
            srt_path = self.temp_dir / f"temp_subtitles_{output_path.stem}.srt"
            mixed_audio_path = self.temp_dir / f"temp_audio_{output_path.stem}.wav"
            
            # Create SRT file
            self._create_srt_file(subtitle_data, srt_path)
            
            # Create mixed audio
            self._create_mixed_audio(audio_path, instruments_path, mixed_audio_path)
            
            # Get FFmpeg executable path
            ffmpeg_path = self._get_ffmpeg_path()
            
            # FFmpeg command with proper path handling
            # Convert paths to forward slashes for ffmpeg compatibility
            srt_path_str = str(srt_path).replace('\\', '/')
            
            cmd = [
                ffmpeg_path, '-y',
                '-i', video_path,
                '-i', str(mixed_audio_path),
                '-vf', f"subtitles='{srt_path_str}':force_style='Fontname=Arial-Bold,Fontsize=32,Bold=1,PrimaryColour=&H00ffffff,OutlineColour=&H00000000,Outline=3,Shadow=1,Alignment=2,MarginV=120'",
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-b:v', '1500k',
                '-b:a', '128k',
                '-r', '24',
                '-shortest',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup temp files
            self._cleanup_temp_files([srt_path, mixed_audio_path])
            
            if result.returncode == 0:
                return {"success": True, "subtitle_count": len(subtitle_data)}
            else:
                return {"success": False, "error": f"FFmpeg error: {result.stderr}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_video_audio_only_moviepy(self, video_path: str, audio_path: str, 
                                       output_path: Path, instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with audio only using MoviePy"""
        try:
            # Load video and prepare audio
            video = VideoFileClip(video_path)
            final_audio = self._prepare_audio_moviepy(audio_path, instruments_path)
            
            # Set new audio to video
            final_video = video.set_audio(final_audio)
            
            # Render
            final_video.write_videofile(
                str(output_path),
                fps=24,
                codec='libx264',
                audio_codec='aac',
                bitrate='1500k',
                audio_bitrate='128k',
                verbose=False,
                logger=None,
                threads=2
            )
            
            # Cleanup
            self._cleanup_moviepy_objects(video, final_audio, final_video)
            
            return {"success": True, "duration": video.duration}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_video_audio_only_ffmpeg(self, video_path: str, audio_path: str,
                                      output_path: Path, instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with audio only using FFmpeg"""
        try:
            # Prepare audio
            if instruments_path and os.path.exists(instruments_path):
                mixed_audio_path = self.temp_dir / f"temp_audio_{output_path.stem}.wav"
                self._create_mixed_audio(audio_path, instruments_path, mixed_audio_path)
                final_audio_path = mixed_audio_path
            else:
                final_audio_path = audio_path
            
            # Get FFmpeg executable path
            ffmpeg_path = self._get_ffmpeg_path()
            
            # FFmpeg command
            cmd = [
                ffmpeg_path, '-y',
                '-i', video_path,
                '-i', str(final_audio_path),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup temp files
            if final_audio_path != audio_path:
                self._cleanup_temp_files([final_audio_path])
            
            if result.returncode == 0:
                return {"success": True}
            else:
                return {"success": False, "error": f"FFmpeg error: {result.stderr}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _prepare_audio_moviepy(self, audio_path: str, instruments_path: Optional[str] = None):
        """Prepare audio for MoviePy"""
        generated_audio = AudioFileClip(audio_path)
        
        if instruments_path and os.path.exists(instruments_path):
            instruments_audio = AudioFileClip(instruments_path)
            instruments_audio = instruments_audio.volumex(0.15)
            generated_audio = generated_audio.volumex(0.85)
            return CompositeAudioClip([generated_audio, instruments_audio])
        else:
            return generated_audio
    
    def _create_srt_file(self, subtitle_data: List[Dict], output_path: Path) -> None:
        """Create clean SRT subtitle file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, subtitle in enumerate(subtitle_data, 1):
                start_time = self._seconds_to_srt_time(subtitle['start'])
                end_time = self._seconds_to_srt_time(subtitle['end'])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{subtitle['text']}\n\n")
    
    def _create_mixed_audio(self, audio_path: str, instruments_path: Optional[str], 
                          output_path: Path) -> None:
        """Create mixed audio file"""
        audio, sr = sf.read(audio_path)
        
        if instruments_path and os.path.exists(instruments_path):
            instruments, _ = sf.read(instruments_path)
            
            # Match lengths
            min_length = min(len(audio), len(instruments))
            audio = audio[:min_length]
            instruments = instruments[:min_length]
            
            # Mix with proper levels
            mixed_audio = audio * 0.85 + instruments * 0.15
        else:
            mixed_audio = audio
        
        sf.write(output_path, mixed_audio, sr)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _cleanup_moviepy_objects(self, *objects) -> None:
        """Clean up MoviePy objects"""
        for obj in objects:
            if obj:
                try:
                    obj.close()
                except:
                    pass
    
    def _cleanup_temp_files(self, file_paths: List[Path]) -> None:
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if file_path.exists():
                    file_path.unlink()
            except:
                pass
    
    def cleanup_temp_files(self, audio_id: str) -> None:
        """Clean up all temporary files for an audio ID"""
        try:
            for file in self.temp_dir.glob(f"*{audio_id}*"):
                file.unlink()
        except:
            pass
    
    def _get_ffmpeg_path(self) -> str:
        """Get the correct FFmpeg executable path"""
        import platform
        import shutil
        
        # Check for bundled FFmpeg first (Windows)
        if platform.system() == 'Windows':
            bundled_ffmpeg = Path('ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe')
            if bundled_ffmpeg.exists():
                return str(bundled_ffmpeg)
        
        # Check if ffmpeg is in PATH
        ffmpeg_in_path = shutil.which('ffmpeg')
        if ffmpeg_in_path:
            return 'ffmpeg'
        
        # Default fallback
        return 'ffmpeg' 