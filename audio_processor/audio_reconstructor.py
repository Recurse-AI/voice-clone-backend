"""
Audio Reconstructor Module

Handles final audio reconstruction and subtitle generation.
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Optional
import json


class AudioReconstructor:
    """Handles final audio reconstruction and subtitle generation"""
    
    def __init__(self, temp_dir: str = "/tmp/voice_cloning"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def reconstruct_final_audio(self, segments_dir: str, audio_id: str, 
                               include_instruments: bool = False,
                               instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Reconstruct final audio with precise timing"""
        try:
            segments_path = Path(segments_dir)
            
            metadata_file = segments_path / "metadata" / f"{audio_id}_metadata.json"
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            segments = metadata.get('segments_info', [])
            segments.sort(key=lambda x: x['start'])
            
            final_audio_parts = []
            sample_rate = 44100
            current_time = 0
            
            for segment in segments:
                # Add silence gap if needed
                gap_duration = segment['start'] - current_time
                if gap_duration > 0.1:  # 100ms threshold
                    gap_samples = int(gap_duration * sample_rate)
                    silence = np.zeros(gap_samples)
                    final_audio_parts.append(silence)
                
                # Add cloned audio
                speaker = segment['speaker']
                speaker_dir = segments_path / f"speaker_{speaker}" / "segments"
                
                cloned_file = self._find_cloned_audio_file(speaker_dir, segment)
                
                if cloned_file and cloned_file.exists():
                    audio_segment, _ = sf.read(cloned_file)
                    final_audio_parts.append(audio_segment)
                    current_time = segment['end']
            
            # Combine all parts
            if final_audio_parts:
                combined_audio = np.concatenate(final_audio_parts)
                
                # Mix with instruments if requested
                if include_instruments and instruments_path and os.path.exists(instruments_path):
                    instruments_audio, _ = sf.read(instruments_path)
                    min_length = min(len(combined_audio), len(instruments_audio))
                    combined_audio = combined_audio[:min_length]
                    instruments_audio = instruments_audio[:min_length]
                    combined_audio = combined_audio * 0.8 + instruments_audio * 0.2
                
                # Save final audio
                final_path = self.temp_dir / f"final_output_{audio_id}.wav"
                sf.write(final_path, combined_audio, sample_rate)
                
                return {
                    "success": True,
                    "final_audio_path": str(final_path),
                    "duration": len(combined_audio) / sample_rate,
                    "sample_rate": sample_rate,
                    "audio_id": audio_id
                }
            
            return {"success": False, "error": "No audio segments found", "audio_id": audio_id}
            
        except Exception as e:
            return {"success": False, "error": str(e), "audio_id": audio_id}
    
    def _find_cloned_audio_file(self, speaker_dir: Path, segment: Dict) -> Optional[Path]:
        """Find the cloned audio file for a segment"""
        # Try to find cloned file by pattern
        cloned_file = None
        for file in speaker_dir.glob(f"cloned_*.wav"):
            if any(str(segment['start']).replace('.', '_') in file.name for _ in [1]):
                cloned_file = file
                break
        
        if not cloned_file:
            # Find by segment number
            segment_files = list(speaker_dir.glob(f"cloned_segment_{segment['speaker']}_*.wav"))
            if segment_files:
                cloned_file = segment_files[0]
        
        return cloned_file
    
    def generate_subtitles(self, segments_dir: str, audio_id: str) -> Dict[str, Any]:
        """Generate subtitle file"""
        try:
            segments_path = Path(segments_dir)
            metadata_file = segments_path / "metadata" / f"{audio_id}_metadata.json"
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            subtitles = []
            counter = 1
            
            for segment in metadata.get('segments_info', []):
                start_time = self._format_srt_time(segment['start'])
                end_time = self._format_srt_time(segment['end'])
                
                # Get English text from segment data
                speaker = segment['speaker']
                speaker_dir = segments_path / f"speaker_{speaker}" / "segments"
                
                english_text = segment.get('text', '')
                for json_file in speaker_dir.glob("*.json"):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        seg_data = json.load(f)
                        if abs(seg_data.get('start', 0) - segment['start']) < 0.1:
                            english_text = seg_data.get('english_text', english_text)
                            break
                
                subtitles.extend([
                    f"{counter}",
                    f"{start_time} --> {end_time}",
                    english_text,
                    ""
                ])
                counter += 1
            
            subtitle_path = self.temp_dir / f"subtitles_{audio_id}.srt"
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(subtitles))
            
            return {"success": True, "subtitle_path": str(subtitle_path), "audio_id": audio_id}
            
        except Exception as e:
            return {"success": False, "error": str(e), "audio_id": audio_id}
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT subtitles"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
