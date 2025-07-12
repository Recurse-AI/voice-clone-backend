"""
Video Processing Module

Handles video processing with voice-cloned audio and subtitle display.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import soundfile as sf
import logging

logger = logging.getLogger(__name__)

# Try to import MoviePy with fallback
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


class VideoProcessor:
    """Video processor with subtitle display"""
    
    def __init__(self, temp_dir: str = "/tmp/video_processing"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Subtitle configuration
        self.max_words_per_subtitle = 4  # Max 3-4 words at a time
        self.subtitle_font_size = 32
        self.subtitle_margin_bottom = 80
        
    def create_video_with_subtitles(self, video_path: str, audio_path: str, 
                                   segments_dir: str, audio_id: str,
                                   instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with voice-cloned audio and subtitles"""
        try:
            output_path = self.temp_dir / f"video_with_subtitles_{audio_id}.mp4"
            
            # Load subtitle data from voice-cloned segments
            subtitle_data = self._load_voice_cloned_subtitles(segments_dir)
            
            if not subtitle_data:
                logger.warning(f"No subtitle data found for {audio_id}")
                return {"success": False, "error": "No subtitle data found"}
            
            logger.info(f"Creating video with {len(subtitle_data)} subtitles for {audio_id}")
            
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
                    "file_size": os.path.getsize(output_path) / (1024*1024),
                    "audio_id": audio_id
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Video creation failed for {audio_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_video_with_audio(self, video_path: str, audio_path: str, 
                              audio_id: str, instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with voice-cloned audio only"""
        try:
            output_path = self.temp_dir / f"video_no_subtitles_{audio_id}.mp4"
            
            logger.info(f"Creating video without subtitles for {audio_id}")
            
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
                    "subtitles_included": False,
                    "audio_id": audio_id
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Video creation failed for {audio_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _load_voice_cloned_subtitles(self, segments_dir: str) -> List[Dict]:
        """Load subtitle data from voice-cloned segments"""
        subtitle_data = []
        segments_path = Path(segments_dir)
        
        # Process each speaker directory (including mixed)
        for speaker_dir in segments_path.glob("speaker_*"):
            if not speaker_dir.is_dir():
                continue
                
            segments_subdir = speaker_dir / "segments"
            if not segments_subdir.exists():
                continue
                
            # Load segment metadata
            for json_file in segments_subdir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check if corresponding cloned audio exists
                    json_stem = json_file.stem
                    cloned_audio_name = f"cloned_{json_stem}.wav"
                    cloned_audio_path = segments_subdir / cloned_audio_name
                    
                    if not cloned_audio_path.exists():
                        continue
                    
                    # Use English text from voice-cloned content
                    english_text = data.get('english_text', data.get('text', ''))
                    
                    if english_text and data.get('start') is not None and data.get('end') is not None:
                        # For mixed segments, clean the text by removing speaker tags for subtitles
                        if data.get('is_mixed', False):
                            # Remove [S1], [S2] tags for subtitle display
                            import re
                            clean_text = re.sub(r'\[S\d+\]\s*', '', english_text)
                            english_text = clean_text.strip()
                        
                        # Split text into word chunks (max 3-4 words per subtitle)
                        word_chunks = self._split_into_word_chunks(english_text, data['start'], data['end'])
                        subtitle_data.extend(word_chunks)
                
                except Exception as e:
                    logger.warning(f"Failed to process subtitle file {json_file}: {str(e)}")
                    continue
        
        # Sort by start time
        subtitle_data.sort(key=lambda x: x['start'])
        return subtitle_data
    
    def _split_into_word_chunks(self, text: str, start_time: float, end_time: float) -> List[Dict]:
        """Split text into chunks of max 3-4 words with proper timing"""
        words = text.strip().split()
        if not words:
            return []
        
        chunks = []
        total_duration = end_time - start_time
        
        # Split words into chunks
        for i in range(0, len(words), self.max_words_per_subtitle):
            chunk_words = words[i:i + self.max_words_per_subtitle]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate timing for this chunk
            chunk_start = start_time + (i / len(words)) * total_duration
            chunk_end = start_time + ((i + len(chunk_words)) / len(words)) * total_duration
            
            # Ensure minimum duration of 1 second
            if chunk_end - chunk_start < 1.0:
                chunk_end = chunk_start + 1.0
            
            chunks.append({
                'start': chunk_start,
                'end': chunk_end,
                'text': chunk_text,
                'duration': chunk_end - chunk_start
            })
        
        return chunks
    
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
            for i, subtitle in enumerate(subtitle_data):
                try:
                    # Create text clip
                    text_clip = TextClip(
                        subtitle['text'],
                        fontsize=self.subtitle_font_size,
                        font='Arial-Bold',
                        color='white',
                        stroke_color='black',
                        stroke_width=3,
                        align='center'
                    ).set_duration(subtitle['duration']).set_start(subtitle['start'])
                    
                    # Position at bottom center
                    text_clip = text_clip.set_position(('center', video.h - self.subtitle_margin_bottom))
                    subtitle_clips.append(text_clip)
                    
                except Exception as e:
                    logger.warning(f"Failed to create subtitle clip {i}: {str(e)}")
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
            logger.error(f"MoviePy video creation failed: {str(e)}")
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
            
            # FFmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', str(mixed_audio_path),
                '-vf', f"subtitles='{srt_path}':force_style='Fontname=Arial-Bold,Fontsize={self.subtitle_font_size},Bold=1,PrimaryColour=&H00ffffff,OutlineColour=&H00000000,Outline=3,Alignment=2,MarginV={self.subtitle_margin_bottom}'",
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
            logger.error(f"FFmpeg video creation failed: {str(e)}")
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
            logger.error(f"MoviePy audio-only video creation failed: {str(e)}")
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
            
            # FFmpeg command
            cmd = [
                'ffmpeg', '-y',
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
            logger.error(f"FFmpeg audio-only video creation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _prepare_audio_moviepy(self, audio_path: str, instruments_path: Optional[str] = None):
        """Prepare audio for MoviePy"""
        generated_audio = AudioFileClip(audio_path)
        
        if instruments_path and os.path.exists(instruments_path):
            instruments_audio = AudioFileClip(instruments_path)
            instruments_audio = instruments_audio.volumex(0.2)
            generated_audio = generated_audio.volumex(0.8)
            return CompositeAudioClip([generated_audio, instruments_audio])
        else:
            return generated_audio
    
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
        audio, sr = sf.read(audio_path)
        
        if instruments_path and os.path.exists(instruments_path):
            instruments, _ = sf.read(instruments_path)
            
            # Match lengths
            min_length = min(len(audio), len(instruments))
            audio = audio[:min_length]
            instruments = instruments[:min_length]
            
            # Mix with voice dominant
            mixed_audio = audio * 0.8 + instruments * 0.2
        else:
            mixed_audio = audio
        
        sf.write(output_path, mixed_audio, sr)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _cleanup_moviepy_objects(self, *objects):
        """Clean up MoviePy objects"""
        for obj in objects:
            try:
                if hasattr(obj, 'close'):
                    obj.close()
            except Exception:
                pass
    
    def _cleanup_temp_files(self, file_paths: List[Path]):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass
    
    def cleanup_temp_files(self, audio_id: str):
        """Clean up temporary files"""
        try:
            for pattern in [f"video_with_subtitles_{audio_id}.mp4", f"video_no_subtitles_{audio_id}.mp4"]:
                temp_file = self.temp_dir / pattern
                if temp_file.exists():
                    temp_file.unlink()
            
            for temp_file in self.temp_dir.glob(f"*{audio_id}*"):
                if temp_file.is_file():
                    temp_file.unlink()
                    
        except Exception:
            pass 