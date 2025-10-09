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
        self.subtitle_font_size = 24
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

            cmd = [ffmpeg_cmd, '-y']
            if settings.FFMPEG_USE_GPU:
                cmd.extend(['-hwaccel', 'cuda'])
            cmd.extend([
                '-i', video_path,
                '-i', str(audio_path),
            ])
            
            if subtitle_path and Path(subtitle_path).exists():
                # Proper Windows path handling for FFmpeg
                import platform
                if platform.system() == 'Windows':
                    # For Windows, escape backslashes properly
                    subtitle_path_str = str(subtitle_path).replace('\\', '\\\\')
                else:
                    # For Unix-like systems, use as-is
                    subtitle_path_str = str(subtitle_path)
                
                video_codec = 'h264_nvenc' if settings.FFMPEG_USE_GPU else 'libx264'
                preset = 'fast' if settings.FFMPEG_USE_GPU else 'veryfast'
                
                # When subtitles needed, match original quality without forcing high bitrate
                base_cmd = [
                    '-vf', f"subtitles='{subtitle_path_str}':force_style='Fontname=Arial-Bold,Fontsize={self.subtitle_font_size},Bold=1,PrimaryColour=&H00ffffff,OutlineColour=&H00000000,Outline=3,Alignment=2,MarginV={self.subtitle_margin_bottom}'",
                    '-c:v', video_codec,
                    '-preset', preset,
                    '-crf', '23',  # Balanced quality matching typical source videos
                ]
                
                if not settings.FFMPEG_USE_GPU:
                    base_cmd.extend(['-threads', '0'])
                
                cmd.extend(base_cmd)
            else:
                # No subtitles, preserve original video quality
                cmd.extend(['-c:v', 'copy'])
            
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
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
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
            from app.utils.audio import AudioUtils
            output_path = self.temp_dir / f"final_mix_{audio_id}.wav"
            AudioUtils.mix_audio_files(audio_path, instruments_path, str(output_path))
            return output_path
        except Exception:
            # If mixing fails, fallback to original audio
            return Path(audio_path)


    def _detect_language_type(self, text: str) -> str:
        """Detect language type for optimal chunking"""
        if not text:
            return "latin"
        
        # Count different character types
        cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or  # Chinese
                                           '\u3040' <= c <= '\u309f' or  # Hiragana
                                           '\u30a0' <= c <= '\u30ff' or  # Katakana
                                           '\uac00' <= c <= '\ud7af')     # Korean
        
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06ff')
        
        total_chars = len(text)
        
        if cjk_chars / total_chars > 0.3:
            return "cjk"
        elif arabic_chars / total_chars > 0.3:
            return "arabic"
        else:
            return "latin"

    def _get_optimal_char_limit(self, text: str) -> int:
        """Get optimal character limit based on language type"""
        lang_type = self._detect_language_type(text)
        
        # Character limits optimized for single-line readability
        limits = {
            "cjk": 20,      # CJK characters are wider
            "arabic": 35,   # Arabic has different reading patterns
            "latin": 42     # Latin languages (English, Spanish, etc.)
        }
        
        return limits.get(lang_type, 42)

    def _chunk_subtitle_text(self, text: str) -> List[str]:
        """
        Intelligently chunk subtitle text for single-line display.
        Handles all languages with smart word/character boundary detection.
        """
        if not text:
            return []
        
        max_chars = self._get_optimal_char_limit(text)
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        lang_type = self._detect_language_type(text)
        
        if lang_type == "cjk":
            # For CJK languages, split by character groups since there are no spaces
            for i in range(0, len(text), max_chars):
                chunk = text[i:i + max_chars]
                if chunk.strip():
                    chunks.append(chunk.strip())
        else:
            # For languages with word boundaries (Latin, Arabic, etc.)
            words = text.split()
            current_chunk = ""
            
            for word in words:
                test_chunk = f"{current_chunk} {word}".strip()
                
                if len(test_chunk) <= max_chars:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = word
                    else:
                        # Single word is too long - force split
                        if len(word) > max_chars:
                            for i in range(0, len(word), max_chars):
                                chunk_part = word[i:i + max_chars]
                                if chunk_part:
                                    chunks.append(chunk_part)
                        else:
                            current_chunk = word
            
            if current_chunk:
                chunks.append(current_chunk)
        
        return [chunk for chunk in chunks if chunk.strip()]

    def demo_chunking(self, text: str) -> Dict[str, Any]:
        """
        Demonstrate subtitle chunking for different languages.
        Returns chunking analysis for testing purposes.
        """
        if not text:
            return {"error": "No text provided"}
        
        lang_type = self._detect_language_type(text)
        char_limit = self._get_optimal_char_limit(text)
        chunks = self._chunk_subtitle_text(text)
        
        return {
            "original_text": text,
            "language_type": lang_type,
            "character_limit": char_limit,
            "original_length": len(text),
            "chunks": chunks,
            "chunk_count": len(chunks),
            "chunk_lengths": [len(chunk) for chunk in chunks],
            "max_chunk_length": max(len(chunk) for chunk in chunks) if chunks else 0,
            "all_chunks_single_line": all(len(chunk) <= char_limit for chunk in chunks)
        }

    def create_ass_file(self, subtitle_data: List[Dict], output_path: Path) -> None:
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            f.write("[Script Info]\n")
            f.write("ScriptType: v4.00+\n")
            f.write("PlayResX: 1920\n")
            f.write("PlayResY: 1080\n\n")
            
            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
            f.write("Style: Default,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2.5,1.5,2,20,20,30,1\n\n")
            
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            
            for subtitle in subtitle_data:
                text = subtitle['text'].strip()
                if not text:
                    continue
                
                chunks = self._chunk_subtitle_text(text)
                if not chunks:
                    continue
                
                start_time = subtitle['start']
                end_time = subtitle['end']
                duration = end_time - start_time
                chunk_duration = duration / len(chunks)
                
                for i, chunk in enumerate(chunks):
                    chunk_start = start_time + (i * chunk_duration)
                    chunk_end = chunk_start + chunk_duration
                    
                    start = self._seconds_to_ass_time(chunk_start)
                    end = self._seconds_to_ass_time(chunk_end)
                    
                    f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{chunk}\n")
    
    def _seconds_to_ass_time(self, seconds: float) -> str:
        """Convert seconds to ASS time format (h:mm:ss.cc)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
    
    def create_srt_file(self, subtitle_data: List[Dict], output_path: Path) -> None:
        subtitle_index = 1
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for subtitle in subtitle_data:
                text = subtitle['text'].strip()
                if not text:
                    continue
                
                chunks = self._chunk_subtitle_text(text)
                if not chunks:
                    continue
                
                start_time = subtitle['start']
                end_time = subtitle['end']
                duration = end_time - start_time
                chunk_duration = duration / len(chunks)
                
                for i, chunk in enumerate(chunks):
                    chunk_start = start_time + (i * chunk_duration)
                    chunk_end = chunk_start + chunk_duration
                    
                    start_time_str = self._seconds_to_srt_time(chunk_start)
                    end_time_str = self._seconds_to_srt_time(chunk_end)
                    
                    f.write(f"{subtitle_index}\n")
                    f.write(f"{start_time_str} --> {end_time_str}\n")
                    f.write(f"{chunk}\n\n")
                    
                    subtitle_index += 1
    
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
