"""
Audio Reconstructor Module

Handles final audio reconstruction with precise timing and order maintenance.
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import logging

logger = logging.getLogger(__name__)


class AudioReconstructor:
    """Audio reconstructor with precise timing and order maintenance"""
    
    def __init__(self, temp_dir: str = "/tmp/voice_cloning"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = 44100
    
    def reconstruct_final_audio(self, segments_dir: str, audio_id: str, 
                               include_instruments: bool = False,
                               instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Reconstruct final audio with precise timing and order maintenance"""
        try:
            segments_path = Path(segments_dir)
            metadata_file = segments_path / "metadata" / f"{audio_id}_metadata.json"
            
            if not metadata_file.exists():
                return {"success": False, "error": "Metadata file not found", "audio_id": audio_id}
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            segments = metadata.get('segments_info', [])
            if not segments:
                return {"success": False, "error": "No segments found", "audio_id": audio_id}
            
            total_duration = metadata.get('total_duration', max(seg['end'] for seg in segments))
            if total_duration <= 0:
                return {"success": False, "error": "Invalid duration", "audio_id": audio_id}
            
            # Reconstruct audio with precise timing
            reconstructed_audio = self._reconstruct_audio(segments, segments_path, total_duration)
            
            if reconstructed_audio is None:
                return {"success": False, "error": "Reconstruction failed", "audio_id": audio_id}
            
            # Mix with instruments if requested
            if include_instruments and instruments_path and os.path.exists(instruments_path):
                reconstructed_audio = self._mix_with_instruments(reconstructed_audio, instruments_path)
            
            # Save final audio
            final_path = self.temp_dir / f"final_output_{audio_id}.wav"
            sf.write(final_path, reconstructed_audio, self.sample_rate)
            
            logger.info(f"Final audio saved: {final_path} (size: {os.path.getsize(final_path) / (1024*1024):.2f} MB)")
            logger.info(f"Reconstruction completed: {len(segments)} segments processed, duration: {len(reconstructed_audio) / self.sample_rate:.2f}s")
            
            return {
                "success": True,
                "final_audio_path": str(final_path),
                "duration": len(reconstructed_audio) / self.sample_rate,
                "sample_rate": self.sample_rate,
                "audio_id": audio_id,
                "segments_processed": len(segments),
                "instruments_mixed": include_instruments and instruments_path is not None,
                "reconstruction_method": "precise_timing_with_cloned_audio"
            }
            
        except Exception as e:
            logger.error(f"Audio reconstruction failed for {audio_id}: {str(e)}")
            return {"success": False, "error": str(e), "audio_id": audio_id}
    
    def _reconstruct_audio(self, segments: List[Dict], segments_path: Path, 
                          total_duration: float) -> Optional[np.ndarray]:
        """Reconstruct audio with precise timing"""
        try:
            # Sort segments by start time
            segments_sorted = sorted(segments, key=lambda x: x['start'])
            
            # Create timeline
            total_samples = int(total_duration * self.sample_rate)
            final_audio = np.zeros(total_samples, dtype=np.float32)
            
            # Process each segment
            for segment in segments_sorted:
                start_sample = int(segment['start'] * self.sample_rate)
                end_sample = int(segment['end'] * self.sample_rate)
                
                # Bounds check
                start_sample = max(0, min(start_sample, total_samples - 1))
                end_sample = max(start_sample + 1, min(end_sample, total_samples))
                
                # Find and load cloned audio
                cloned_audio = self._load_cloned_audio(segments_path, segment)
                
                if cloned_audio is not None:
                    # Adjust length to fit exact time slot
                    expected_samples = end_sample - start_sample
                    if len(cloned_audio) != expected_samples:
                        cloned_audio = self._adjust_length(cloned_audio, expected_samples)
                    
                    # Place audio at exact position
                    final_audio[start_sample:end_sample] = cloned_audio
            
            return final_audio
            
        except Exception as e:
            logger.error(f"Audio reconstruction failed: {str(e)}")
            return None
    
    def _load_cloned_audio(self, segments_path: Path, segment: Dict) -> Optional[np.ndarray]:
        """Load cloned audio for segment"""
        speaker = segment['speaker']
        
        # Handle mixed segments
        if speaker == 'mixed':
            mixed_dir = segments_path / "speaker_mixed" / "segments"
            if not mixed_dir.exists():
                return None
            
            # Find matching mixed segment by timing
            for json_file in mixed_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        seg_data = json.load(f)
                    
                    # Check timing match
                    if (abs(seg_data.get('start', 0) - segment['start']) < 0.1 and 
                        abs(seg_data.get('end', 0) - segment['end']) < 0.1):
                        
                        # Load cloned audio
                        segment_id = seg_data.get('segment_file', json_file.stem)
                        if segment_id.endswith('.json'):
                            segment_id = segment_id[:-5]  # Remove .json extension
                        
                        cloned_file = mixed_dir / f"cloned_{segment_id}.wav"
                        
                        if cloned_file.exists():
                            audio, _ = sf.read(cloned_file)
                            return audio
                            
                except Exception:
                    continue
        else:
            # Handle individual speaker segments
            speaker_dir = segments_path / f"speaker_{speaker}" / "segments"
            
            if not speaker_dir.exists():
                return None
            
            # Find matching segment by timing
            for json_file in speaker_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        seg_data = json.load(f)
                    
                    # Check timing match
                    if (abs(seg_data.get('start', 0) - segment['start']) < 0.1 and 
                        abs(seg_data.get('end', 0) - segment['end']) < 0.1):
                        
                        # Load cloned audio
                        segment_id = seg_data.get('segment_file', json_file.stem)
                        if segment_id.endswith('.json'):
                            segment_id = segment_id[:-5]  # Remove .json extension
                        
                        cloned_file = speaker_dir / f"cloned_{segment_id}.wav"
                        
                        if cloned_file.exists():
                            audio, _ = sf.read(cloned_file)
                            return audio
                            
                except Exception:
                    continue
        
        return None
    
    def _adjust_length(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        """Adjust audio length to target samples"""
        current_samples = len(audio)
        
        if current_samples == target_samples:
            return audio
        elif current_samples > target_samples:
            # Trim with fade
            trimmed = audio[:target_samples]
            if target_samples > 100:
                fade_samples = min(50, target_samples // 10)
                fade_curve = np.linspace(1, 0, fade_samples)
                trimmed[-fade_samples:] *= fade_curve
            return trimmed
        else:
            # Extend with silence
            padding = np.zeros(target_samples - current_samples)
            return np.concatenate([audio, padding])
    
    def _mix_with_instruments(self, audio: np.ndarray, instruments_path: str) -> np.ndarray:
        """Mix audio with instruments"""
        try:
            instruments_audio, _ = sf.read(instruments_path)
            
            # Match lengths
            min_length = min(len(audio), len(instruments_audio))
            audio = audio[:min_length]
            instruments_audio = instruments_audio[:min_length]
            
            # Mix with voice dominant
            mixed_audio = audio * 0.8 + instruments_audio * 0.2
            
            # Prevent clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 0.95:
                mixed_audio = mixed_audio * (0.95 / max_val)
            
            return mixed_audio
            
        except Exception as e:
            logger.error(f"Failed to mix with instruments: {str(e)}")
            return audio
    
    def cleanup_temp_files(self, audio_id: str):
        """Clean up temporary files"""
        try:
            final_audio_path = self.temp_dir / f"final_output_{audio_id}.wav"
            if final_audio_path.exists():
                final_audio_path.unlink()
            
            for temp_file in self.temp_dir.glob(f"*{audio_id}*"):
                if temp_file.is_file() and temp_file.name.endswith(('.wav', '.mp3', '.flac')):
                    temp_file.unlink()
                    
        except Exception:
            pass