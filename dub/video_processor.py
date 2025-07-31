"""
Video Processing Module

Handles video processing with audio replacement.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import soundfile as sf


class VideoProcessor:
    
    def __init__(self, temp_dir: str = "./tmp/video_processing"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.subtitle_font_size = 18
        self.subtitle_margin_bottom = 30
        self.words_per_subtitle = 3
        subtitle_data = []
        segments_path = Path(segments_dir)
        
        # Use unified segments folder
        segments_folder = segments_path / "segments"
        
        if not segments_folder.exists():
            return subtitle_data
        
        for json_file in segments_folder.glob("*_metadata.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not data.get('segment_index') or not data.get('start') and data.get('start') != 0 or not data.get('end'):
                    continue
                
                cloned_audio_exists = False
                segment_index = data.get('segment_index', 1)
                
                # Check if cloned audio exists in unified cloned folder
                cloned_folder = segments_path / "cloned"
                cloned_filename = f"cloned_segment_{segment_index:03d}.wav"
                cloned_path = cloned_folder / cloned_filename
                
                if cloned_path.exists():
                    cloned_audio_exists = True
                
                if not cloned_audio_exists:
                    base_name = json_file.stem.replace('_metadata', '')
                    old_cloned_file = segments_folder / f"cloned_{base_name}.wav"
                    cloned_audio_exists = old_cloned_file.exists()
                
                if not cloned_audio_exists:
                    continue
                
                english_text = data.get('english_text', data.get('text', ''))
                if not english_text:
                    continue
                
                import re
                display_text = re.sub(r'\[S\d+\]\s*', '', english_text).strip()
                display_text = re.sub(r'\n', ' ', display_text).strip()
                
                if display_text:
                    word_chunks = self._create_word_chunks(
                        display_text, 
                        data['start'], 
                        data['end']
                    )
                    subtitle_data.extend(word_chunks)
                
            except Exception:
                continue
        
        subtitle_data.sort(key=lambda x: x['start'])
        return self._resolve_overlaps(subtitle_data)
    


    def _create_video_ffmpeg(self, video_path: str, audio_path: str, 
                           subtitle_path: Optional[str], output_path: Path) -> Dict[str, Any]:
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', str(audio_path),
            ]
            
            if subtitle_path and subtitle_path.exists():
                # High-quality settings when adding subtitles
                cmd.extend([
                    '-vf', f"subtitles='{subtitle_path}':force_style='Fontname=Arial-Bold,Fontsize={self.subtitle_font_size},Bold=1,PrimaryColour=&H00ffffff,OutlineColour=&H00000000,Outline=3,Alignment=2,MarginV={self.subtitle_margin_bottom}'",
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-b:v', '5000k',
                ])
            else:
                # High quality without subtitles
                cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-b:v', '5000k',
                ])
            
            cmd.extend([
                '-c:a', 'aac',
                '-b:a', '128k',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
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
    
    def _create_srt_file(self, subtitle_data: List[Dict], output_path: Path) -> None:
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
    
    def create_video_with_subtitles(self, video_path: str, audio_path: str, 
                                   segments_dir: str, audio_id: str,
                                   instruments_path: Optional[str] = None) -> Dict[str, Any]:
        try:
            if instruments_path and os.path.exists(instruments_path):
                final_audio_path = self._create_final_audio(audio_path, instruments_path, audio_id)
            else:
                final_audio_path = Path(audio_path)
            
            subtitle_data = self._load_subtitles(segments_dir)
            
            if not subtitle_data:
                return self.create_video_with_audio(video_path, str(final_audio_path), audio_id, instruments_path, segments_dir)
            
            subtitle_path = self.temp_dir / f"subtitles_{audio_id}.srt"
            self._create_srt_file(subtitle_data, subtitle_path)
            
            output_path = self.temp_dir / f"video_with_subtitles_{audio_id}.mp4"
            
            result = self._create_video_ffmpeg(video_path, str(final_audio_path), subtitle_path, output_path)
            
            if result["success"]:
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "subtitle_path": str(subtitle_path),
                    "subtitle_count": len(subtitle_data),
                    "has_subtitles": True,
                    "final_audio_path": str(final_audio_path)  # Return the actual audio path used
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
    
    def create_video_with_audio(self, video_path: str, audio_path: str, 
                               audio_id: str, instruments_path: Optional[str] = None,
                               segments_dir: Optional[str] = None) -> Dict[str, Any]:
        try:

            output_path = self.temp_dir / f"video_no_subtitles_{audio_id}.mp4"
            
            # Create video without attaching subtitles
            result = self._create_video_ffmpeg(video_path, str(final_audio_path), None, output_path)
            
            if result["success"]:
                return {
                    "success": True,
                    "video_path": str(output_path),
                    "subtitle_path": str(subtitle_path) if subtitle_path else None,
                    "subtitle_count": subtitle_count,
                    "has_subtitles": False,
                    "final_audio_path": str(final_audio_path)  # Return the actual audio path used
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