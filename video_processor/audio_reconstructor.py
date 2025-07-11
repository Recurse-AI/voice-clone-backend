"""
Audio Reconstructor Module - Production Clean Version

Handles final audio reconstruction only.
Subtitles are handled separately in video processor.
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Optional
import json


class AudioReconstructor:
    """Handles final audio reconstruction from voice-cloned segments"""
    
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
        speaker = segment['speaker']
        
        # First try to find by exact segment matching
        for json_file in speaker_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                seg_data = json.load(f)
            
            # Check if this is the matching segment by comparing start times
            if abs(seg_data.get('start', 0) - segment['start']) < 0.1:
                # Get the segment ID from the JSON file
                segment_id = seg_data.get('segment_id', '')
                if segment_id:
                    cloned_file = speaker_dir / f"cloned_{segment_id}.wav"
                    if cloned_file.exists():
                        return cloned_file
        
        # Fallback: use any cloned file for this speaker
        cloned_files = list(speaker_dir.glob(f"cloned_{speaker}_*.wav"))
        if cloned_files:
            return cloned_files[0]
        
        # Final fallback: use original file if no cloned file exists
        for json_file in speaker_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                seg_data = json.load(f)
            
            if abs(seg_data.get('start', 0) - segment['start']) < 0.1:
                segment_id = seg_data.get('segment_id', '')
                if segment_id:
                    original_file = speaker_dir / f"{segment_id}.wav"
                    if original_file.exists():
                        return original_file
        
        return None
    
    def cleanup_temp_files(self, audio_id: str):
        """Clean up temporary files created during reconstruction"""
        try:
            # Clean up final audio files
            final_audio_path = self.temp_dir / f"final_output_{audio_id}.wav"
            if final_audio_path.exists():
                final_audio_path.unlink()
            
            # Clean up temporary mixed audio files
            temp_mixed_path = self.temp_dir / f"temp_mixed_{audio_id}.wav"
            if temp_mixed_path.exists():
                temp_mixed_path.unlink()
            
            # Clean up any other temp files related to audio reconstruction
            for temp_file in self.temp_dir.glob(f"*{audio_id}*"):
                if temp_file.is_file() and temp_file.name.endswith(('.wav', '.mp3', '.flac')):
                    temp_file.unlink()
                    
        except Exception:
            pass