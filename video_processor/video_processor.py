"""
Video Processing Module

Handles video processing with voice-cloned audio replacement.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import soundfile as sf
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try to import MoviePy with fallback
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


class VideoProcessor:
    """Simple video processor that replaces original audio with cloned audio"""
    
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
        """Create video with cloned audio and subtitles"""
        try:
            output_path = self.temp_dir / f"video_with_subtitles_{audio_id}.mp4"
            
            # Load subtitle data
            subtitle_data = self._load_subtitles(segments_dir)
            
            if not subtitle_data:
                logger.warning(f"No subtitle data found for {audio_id}")
                return {"success": False, "error": "No subtitle data found"}
            
            logger.info(f"Creating video with {len(subtitle_data)} subtitles for {audio_id}")
            
            # Create subtitle file
            subtitle_path = self.temp_dir / f"subtitles_{audio_id}.srt"
            self._create_srt_file(subtitle_data, subtitle_path)
            
            # Create final audio (cloned + instruments if requested)
            final_audio_path = self._create_final_audio(audio_path, instruments_path, audio_id)
            
            # Create video with new audio and subtitles
            result = self._create_video_ffmpeg(
                video_path, final_audio_path, subtitle_path, output_path
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "subtitle_path": str(subtitle_path),
                    "subtitle_count": len(subtitle_data),
                    "duration": result.get("duration", 0),
                    "file_size": os.path.getsize(output_path) / (1024*1024),
                    "audio_id": audio_id,
                    "audio_used": str(final_audio_path),
                    "instruments_mixed": instruments_path is not None and os.path.exists(instruments_path)
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Video creation failed for {audio_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def create_video_with_audio(self, video_path: str, audio_path: str, 
                              audio_id: str, instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Create video with cloned audio only (no subtitles)"""
        try:
            output_path = self.temp_dir / f"video_no_subtitles_{audio_id}.mp4"
            
            logger.info(f"Creating video without subtitles for {audio_id}")
            
            # Create final audio (cloned + instruments if requested)
            final_audio_path = self._create_final_audio(audio_path, instruments_path, audio_id)
            
            # Create video with new audio only
            result = self._create_video_ffmpeg(
                video_path, final_audio_path, None, output_path
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "duration": result.get("duration", 0),
                    "file_size": os.path.getsize(output_path) / (1024*1024),
                    "subtitles_included": False,
                    "audio_id": audio_id,
                    "audio_used": str(final_audio_path),
                    "instruments_mixed": instruments_path is not None and os.path.exists(instruments_path)
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Video creation failed for {audio_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _load_subtitles(self, segments_dir: str) -> List[Dict]:
        """Load subtitle data from segments"""
        subtitle_data = []
        segments_path = Path(segments_dir)
        
        # Process each speaker directory
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
                    
                    # Use English text for subtitles
                    english_text = data.get('english_text', data.get('text', ''))
                    
                    if english_text and data.get('start') is not None and data.get('end') is not None:
                        # Clean mixed segment text (remove speaker tags for display)
                        if data.get('is_mixed', False):
                            import re
                            english_text = re.sub(r'\[S\d+\]\s*', '', english_text)
                        
                        # Split into word chunks
                        word_chunks = self._split_into_chunks(english_text, data['start'], data['end'])
                        subtitle_data.extend(word_chunks)
                
                except Exception as e:
                    logger.warning(f"Failed to process subtitle file {json_file}: {str(e)}")
                    continue
        
        # Sort by start time
        subtitle_data.sort(key=lambda x: x['start'])
        return subtitle_data
    
    def _split_into_chunks(self, text: str, start_time: float, end_time: float) -> List[Dict]:
        """Split text into small chunks for better subtitle display"""
        words = text.strip().split()
        if not words:
            return []
        
        chunks = []
        total_duration = end_time - start_time
        
        # Split words into chunks of max 3-4 words
        for i in range(0, len(words), self.max_words_per_subtitle):
            chunk_words = words[i:i + self.max_words_per_subtitle]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate timing
            chunk_start = start_time + (i / len(words)) * total_duration
            chunk_end = start_time + ((i + len(chunk_words)) / len(words)) * total_duration
            
            # Ensure minimum duration
            if chunk_end - chunk_start < 1.0:
                chunk_end = chunk_start + 1.0
            
            chunks.append({
                'start': chunk_start,
                'end': chunk_end,
                'text': chunk_text,
                'duration': chunk_end - chunk_start
            })
        
        return chunks
    
    def _create_final_audio(self, audio_path: str, instruments_path: Optional[str], audio_id: str) -> Path:
        """Create final audio by mixing cloned audio with instruments if provided"""
        final_audio_path = self.temp_dir / f"final_mixed_audio_{audio_id}.wav"
        
        # Load cloned audio
        cloned_audio, sr = sf.read(audio_path)
        
        if instruments_path and os.path.exists(instruments_path):
            logger.info(f"Mixing cloned audio with instruments")
            try:
                # Load instruments
                instruments_audio, _ = sf.read(instruments_path)
                
                # Match lengths
                min_length = min(len(cloned_audio), len(instruments_audio))
                cloned_audio = cloned_audio[:min_length]
                instruments_audio = instruments_audio[:min_length]
                
                # Mix: 80% cloned voice, 20% instruments
                mixed_audio = cloned_audio * 0.8 + instruments_audio * 0.2
                
                # Prevent clipping
                max_val = np.max(np.abs(mixed_audio))
                if max_val > 0.95:
                    mixed_audio = mixed_audio * (0.95 / max_val)
                
                sf.write(final_audio_path, mixed_audio, sr)
                logger.info(f"Mixed audio saved: voice 80%, instruments 20%")
                
            except Exception as e:
                logger.error(f"Failed to mix with instruments: {str(e)}, using cloned audio only")
                sf.write(final_audio_path, cloned_audio, sr)
        else:
            # Use cloned audio only
            sf.write(final_audio_path, cloned_audio, sr)
            logger.info(f"Using cloned audio only (no instruments)")
        
        return final_audio_path
    
    def _create_video_ffmpeg(self, video_path: str, audio_path: str, 
                           subtitle_path: Optional[str], output_path: Path) -> Dict[str, Any]:
        """Create video using FFmpeg - completely replace original audio"""
        try:
            # Build FFmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,        # Input video
                '-i', str(audio_path),   # Input audio (cloned)
            ]
            
            # Add subtitles if provided
            if subtitle_path and subtitle_path.exists():
                # Add video filter for subtitles
                cmd.extend([
                    '-vf', f"subtitles='{subtitle_path}':force_style='Fontname=Arial-Bold,Fontsize={self.subtitle_font_size},Bold=1,PrimaryColour=&H00ffffff,OutlineColour=&H00000000,Outline=3,Alignment=2,MarginV={self.subtitle_margin_bottom}'",
                    '-c:v', 'libx264',       # Re-encode video with subtitles
                ])
            else:
                # Copy video stream if no subtitles
                cmd.extend(['-c:v', 'copy'])
            
            # Add audio and mapping settings
            cmd.extend([
                '-c:a', 'aac',           # Encode audio as AAC
                '-b:a', '128k',          # Audio bitrate
                '-map', '0:v:0',         # Map video from first input
                '-map', '1:a:0',         # Map audio from second input (replaces original)
                '-shortest',             # End when shortest stream ends
                str(output_path)         # Output file
            ])
            
            logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
            
            # Execute FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Video created successfully: {output_path}")
                return {"success": True}
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return {"success": False, "error": f"FFmpeg error: {result.stderr}"}
                
        except Exception as e:
            logger.error(f"Video creation failed: {str(e)}")
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
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def cleanup_temp_files(self, audio_id: str):
        """Clean up temporary files"""
        try:
            patterns = [
                f"video_with_subtitles_{audio_id}.mp4",
                f"video_no_subtitles_{audio_id}.mp4",
                f"final_mixed_audio_{audio_id}.wav",
                f"subtitles_{audio_id}.srt"
            ]
            
            for pattern in patterns:
                temp_file = self.temp_dir / pattern
                if temp_file.exists():
                    temp_file.unlink()
            
            # Clean up any other temp files
            for temp_file in self.temp_dir.glob(f"*{audio_id}*"):
                if temp_file.is_file():
                    temp_file.unlink()
                    
        except Exception:
            pass 