"""
Video Processing Module

Handles video processing with high-quality subtitle generation.
"""

import os
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import soundfile as sf
import numpy as np

# Try to import MoviePy with fallback
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, TextClip, CompositeVideoClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


class VideoProcessor:
    """Handles video processing with high-quality subtitles"""
    
    def __init__(self, temp_dir: str = "/tmp/video_processing"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimal subtitle settings
        self.subtitle_config = {
            "fontsize": 32,
            "font": "Arial-Bold",
            "color": "white",
            "stroke_color": "black", 
            "stroke_width": 3,
            "max_words_per_line": 5,
            "position_from_bottom": 120,
            "margin_horizontal": 100,
            "line_spacing": 1.2
        }
    
    def create_video_with_subtitles(self, video_path: str, audio_path: str, 
                                   segments_dir: str, audio_id: str,
                                   instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with perfectly styled subtitles"""
        try:
            output_path = self.temp_dir / f"video_with_subtitles_{audio_id}.mp4"
            
            # Load subtitle data from segments
            subtitle_data = self._load_subtitle_data(segments_dir)
            
            if not subtitle_data:
                return {"success": False, "error": "No subtitle data found"}
            
            # Create video based on available tools
            if MOVIEPY_AVAILABLE:
                result = self._create_video_moviepy(video_path, audio_path, subtitle_data, 
                                                  output_path, instruments_path)
            else:
                result = self._create_video_ffmpeg(video_path, audio_path, subtitle_data, 
                                                 output_path, instruments_path)
            
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
    
    def _load_subtitle_data(self, segments_dir: str) -> List[Dict]:
        """Load and process subtitle data from segments"""
        segments_path = Path(segments_dir)
        subtitle_data = []
        
        # Find all segment JSON files
        for speaker_dir in segments_path.glob("speaker_*/segments"):
            for json_file in speaker_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Get English text
                    english_text = data.get('english_text', data.get('text', ''))
                    
                    if english_text:
                        # Split into optimal chunks
                        chunks = self._split_text_optimally(english_text)
                        segment_duration = data['end'] - data['start']
                        
                        # Create subtitle entries for each chunk
                        if chunks:
                            chunk_duration = segment_duration / len(chunks)
                            
                            for i, chunk in enumerate(chunks):
                                chunk_start = data['start'] + (i * chunk_duration)
                                chunk_end = chunk_start + chunk_duration
                                
                                subtitle_data.append({
                                    'start': chunk_start,
                                    'end': chunk_end,
                                    'text': chunk.strip(),
                                    'duration': chunk_duration
                                })
                
                except Exception as e:
                    continue
        
        # Sort by start time
        subtitle_data.sort(key=lambda x: x['start'])
        return subtitle_data
    
    def _split_text_optimally(self, text: str) -> List[str]:
        """Split text into optimal chunks for readability"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            
            # Check if we should end this chunk
            if len(current_chunk) >= self.subtitle_config["max_words_per_line"]:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add remaining words
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _create_video_moviepy(self, video_path: str, audio_path: str, 
                            subtitle_data: List[Dict], output_path: Path,
                            instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video using MoviePy with perfect subtitle styling"""
        try:
            # Load video
            video = VideoFileClip(video_path)
            video_size = (video.w, video.h)
            
            # Load and mix audio
            generated_audio = AudioFileClip(audio_path)
            
            if instruments_path and os.path.exists(instruments_path):
                instruments_audio = AudioFileClip(instruments_path)
                instruments_audio = instruments_audio.volumex(0.15)  # Very low background
                generated_audio = generated_audio.volumex(0.85)
                final_audio = CompositeAudioClip([generated_audio, instruments_audio])
            else:
                final_audio = generated_audio
            
            # Create subtitle clips with perfect styling
            subtitle_clips = []
            
            for subtitle in subtitle_data:
                try:
                    # Create text clip with optimal settings
                    text_clip = TextClip(
                        subtitle['text'],
                        fontsize=self.subtitle_config["fontsize"],
                        font=self.subtitle_config["font"],
                        color=self.subtitle_config["color"],
                        stroke_color=self.subtitle_config["stroke_color"],
                        stroke_width=self.subtitle_config["stroke_width"],
                        method='caption',
                        size=(video_size[0] - self.subtitle_config["margin_horizontal"], None),
                        align='center',
                        interline=self.subtitle_config["line_spacing"]
                    ).set_duration(subtitle['duration']).set_start(subtitle['start'])
                    
                    # Position at bottom with proper margin
                    text_clip = text_clip.set_position(
                        ('center', video_size[1] - self.subtitle_config["position_from_bottom"])
                    )
                    
                    subtitle_clips.append(text_clip)
                    
                except Exception as e:
                    continue
            
            # Combine video with audio and subtitles
            final_video = video.set_audio(final_audio)
            
            if subtitle_clips:
                final_video = CompositeVideoClip([final_video] + subtitle_clips)
            
            # Render with optimal settings
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
            video.close()
            generated_audio.close()
            if instruments_path:
                instruments_audio.close()
            final_audio.close()
            final_video.close()
            
            return {
                "success": True,
                "duration": video.duration,
                "subtitle_count": len(subtitle_clips)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_video_ffmpeg(self, video_path: str, audio_path: str,
                           subtitle_data: List[Dict], output_path: Path,
                           instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video using FFmpeg with perfect subtitle styling"""
        try:
            # Create SRT file
            srt_path = self.temp_dir / f"subtitles_{os.path.basename(output_path)}.srt"
            self._create_srt_file(subtitle_data, srt_path)
            
            # Create mixed audio
            mixed_audio_path = self.temp_dir / f"mixed_audio_{os.path.basename(output_path)}.wav"
            self._create_mixed_audio(audio_path, instruments_path, mixed_audio_path)
            
            # FFmpeg command with perfect subtitle styling
            subtitle_style = (
                f"Fontname={self.subtitle_config['font']},"
                f"Fontsize={self.subtitle_config['fontsize']},"
                f"Bold=1,"
                f"PrimaryColour=&H00ffffff,"  # White
                f"OutlineColour=&H00000000,"  # Black
                f"BackColour=&H80000000,"     # Semi-transparent black
                f"Outline={self.subtitle_config['stroke_width']},"
                f"Shadow=1,"
                f"Alignment=2,"  # Bottom center
                f"MarginV={self.subtitle_config['position_from_bottom']}"
            )
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', str(mixed_audio_path),
                '-vf', f"subtitles={srt_path}:force_style='{subtitle_style}'",
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
            if srt_path.exists():
                srt_path.unlink()
            if mixed_audio_path.exists():
                mixed_audio_path.unlink()
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "subtitle_count": len(subtitle_data)
                }
            else:
                return {"success": False, "error": f"FFmpeg error: {result.stderr}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_srt_file(self, subtitle_data: List[Dict], output_path: Path) -> None:
        """Create SRT subtitle file"""
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
        # Load main audio
        audio, sr = sf.read(audio_path)
        
        # Mix with instruments if provided
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
        
        # Save mixed audio
        sf.write(output_path, mixed_audio, sr)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def validate_video_file(self, video_path: str) -> bool:
        """Validate if video file is properly created"""
        try:
            if not os.path.exists(video_path):
                return False
            
            # Check file size
            file_size = os.path.getsize(video_path)
            if file_size < 100000:  # Less than 100KB
                return False
            
            # Try to get video info
            if MOVIEPY_AVAILABLE:
                try:
                    video = VideoFileClip(video_path)
                    duration = video.duration
                    video.close()
                    return duration > 0
                except:
                    pass
            
            return True
            
        except Exception:
            return False
    
    def cleanup_temp_files(self, audio_id: str) -> None:
        """Clean up temporary files"""
        try:
            for file in self.temp_dir.glob(f"*{audio_id}*"):
                file.unlink()
        except Exception:
            pass 