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
    
    def _calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate the Root Mean Square of the audio signal."""
        if audio.size == 0:
            return 0.0
        return np.sqrt(np.mean(audio**2))

    def _match_target_volume(self, audio: np.ndarray, target_rms: float, gain_db: float = 3.0) -> np.ndarray:
        """Adjusts the audio volume to a target RMS level with an optional dB boost."""
        if audio.size == 0 or target_rms <= 1e-7:
            return audio
        
        current_rms = self._calculate_rms(audio)
        if current_rms <= 1e-7:
            return audio
            
        gain = target_rms / current_rms
        gain *= 10**(gain_db / 20.0)
        
        return audio * gain

    def reconstruct_final_audio(self, segments_dir: str, audio_id: str, 
                               include_instruments: bool = False,
                               instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Reconstruct final audio using timeline with speech segments and silent parts"""
        try:
            segments_path = Path(segments_dir)
            timeline_file = segments_path / "metadata" / "timeline.json"
            
            if not timeline_file.exists():
                return {"success": False, "error": "Timeline file not found", "audio_id": audio_id}
            
            with open(timeline_file, 'r', encoding='utf-8') as f:
                timeline_data = json.load(f)
            
            timeline = timeline_data.get('timeline', [])
            if not timeline:
                return {"success": False, "error": "No timeline items found", "audio_id": audio_id}
            
            # Calculate total duration
            total_duration = max(item['end'] for item in timeline)
            if total_duration <= 0:
                return {"success": False, "error": "Invalid duration", "audio_id": audio_id}
            
            # Reconstruct audio from timeline
            reconstructed_audio = self._reconstruct_from_timeline(timeline, segments_path, total_duration)
            
            if reconstructed_audio is None:
                return {"success": False, "error": "Reconstruction failed", "audio_id": audio_id}
            
            # Mix with instruments if requested
            if include_instruments and instruments_path and os.path.exists(instruments_path):
                reconstructed_audio = self._mix_with_instruments(reconstructed_audio, instruments_path)
            
            # Save final audio
            final_path = self.temp_dir / f"final_output_{audio_id}.wav"
            sf.write(final_path, reconstructed_audio, self.sample_rate)
            
            speech_count = len([t for t in timeline if t['segment_type'] == 'speech'])
            silent_count = len([t for t in timeline if t['segment_type'] == 'silent'])
            
            logger.info(f"Final audio saved: {final_path} (size: {os.path.getsize(final_path) / (1024*1024):.2f} MB)")
            logger.info(f"Reconstruction completed: {speech_count} speech segments + {silent_count} silent parts, duration: {len(reconstructed_audio) / self.sample_rate:.2f}s")
            
            return {
                "success": True,
                "final_audio_path": str(final_path),
                "duration": len(reconstructed_audio) / self.sample_rate,
                "sample_rate": self.sample_rate,
                "audio_id": audio_id,
                "segments_processed": speech_count,
                "silent_parts_included": silent_count,
                "instruments_mixed": include_instruments and instruments_path is not None,
                "reconstruction_method": "timeline_based_with_silent_parts"
            }
            
        except Exception as e:
            logger.error(f"Audio reconstruction failed for {audio_id}: {str(e)}")
            return {"success": False, "error": str(e), "audio_id": audio_id}
    
    def _reconstruct_from_timeline(self, timeline: List[Dict], segments_path: Path, 
                                  total_duration: float) -> Optional[np.ndarray]:
        """Reconstruct audio from timeline with speech segments and silent parts"""
        try:
            # Sort timeline by start time
            timeline_sorted = sorted(timeline, key=lambda x: x['start'])
            
            # Create timeline
            total_samples = int(total_duration * self.sample_rate)
            final_audio = np.zeros(total_samples, dtype=np.float32)
            
            # Process each timeline item
            for item in timeline_sorted:
                start_sample = int(item['start'] * self.sample_rate)
                end_sample = int(item['end'] * self.sample_rate)
                
                # Bounds check
                start_sample = max(0, min(start_sample, total_samples - 1))
                end_sample = max(start_sample + 1, min(end_sample, total_samples))
                
                if item['segment_type'] == 'speech':
                    # Load cloned audio for speech segments
                    cloned_audio = self._load_cloned_speech_audio(segments_path, item)
                    
                    if cloned_audio is not None:
                        # Simple placement - no complex volume matching
                        expected_samples = end_sample - start_sample
                        if len(cloned_audio) != expected_samples:
                            cloned_audio = self._adjust_length(cloned_audio, expected_samples)
                        
                        # Place cloned audio directly
                        final_audio[start_sample:end_sample] = cloned_audio
                        print(f"Placed cloned audio: {item['start']:.2f}s-{item['end']:.2f}s, speaker: {item.get('speaker', 'unknown')}")
                    else:
                        print(f"WARNING: No cloned audio found for segment {item['start']:.2f}s-{item['end']:.2f}s")
                        
                elif item['segment_type'] == 'silent':
                    duration = item['end'] - item['start']
                    if duration > 0:
                        silent_samples = int(duration * self.sample_rate)
                        final_audio[start_sample:end_sample] = np.zeros(silent_samples, dtype=np.float32)

            return final_audio
            
        except Exception as e:
            logger.error(f"Audio reconstruction failed: {str(e)}")
            return None
    
    def _load_cloned_speech_audio(self, segments_path: Path, segment: Dict) -> Optional[np.ndarray]:
        """Load cloned audio for speech segment"""
        speaker = segment.get('speaker', 'A')
        speaker_dir = segments_path / f"speaker_{speaker}" / "segments"
        
        print(f"Looking for cloned audio in: {speaker_dir}")
        
        if not speaker_dir.exists():
            print(f"Speaker directory not found: {speaker_dir}")
            return None
        
        # Find matching segment by timing
        for json_file in speaker_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    seg_data = json.load(f)
                
                # Check timing match (with tolerance)
                if (abs(seg_data.get('start', 0) - segment['start']) < 0.1 and 
                    abs(seg_data.get('end', 0) - segment['end']) < 0.1):
                    
                    # Get audio file name
                    audio_file = seg_data.get('audio_file', '')
                    if audio_file:
                        # Look for cloned version first
                        cloned_file = f"cloned_{audio_file}"
                        cloned_path = speaker_dir / cloned_file
                        
                        print(f"Checking cloned file: {cloned_path}")
                        
                        if cloned_path.exists():
                            audio, _ = sf.read(cloned_path)
                            print(f"Found cloned audio: {cloned_path}, length: {len(audio)} samples")
                            return audio
                        else:
                            print(f"Cloned file not found: {cloned_path}")
                        
            except Exception as e:
                logger.warning(f"Error loading segment {json_file}: {str(e)}")
                continue
        
        print(f"No cloned audio found for segment {segment['start']:.2f}s-{segment['end']:.2f}s")
        return None
    
    def _load_original_speech_audio(self, segments_path: Path, segment: Dict) -> Optional[np.ndarray]:
        """Load original audio for a speech segment to be used for volume matching."""
        speaker = segment.get('speaker')
        if not speaker: 
            return None
    
        speaker_dir = segments_path / f"speaker_{speaker}" / "segments"
        if not speaker_dir.exists(): 
            return None

        for json_file in speaker_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    seg_data = json.load(f)
                
                if (abs(seg_data.get('start', 0) - segment['start']) < 0.1 and 
                    abs(seg_data.get('end', 0) - segment['end']) < 0.1):
                    
                    audio_file = seg_data.get('audio_file', '')
                    if audio_file:
                        original_path = speaker_dir / audio_file
                        if original_path.exists():
                            audio, _ = sf.read(original_path)
                            return audio
            except Exception as e:
                logger.warning(f"Could not load original audio for segment {segment.get('start')}: {e}")
                continue
        return None

    def _load_silent_audio(self, segments_path: Path, silent_part: Dict) -> Optional[np.ndarray]:
        """Load original silent audio"""
        silent_dir = segments_path / "silent_parts"
        
        if not silent_dir.exists():
            return None
        
        # Find matching silent part by timing
        for json_file in silent_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    silent_data = json.load(f)
                
                # Check timing match
                if (abs(silent_data.get('start', 0) - silent_part['start']) < 0.1 and 
                    abs(silent_data.get('end', 0) - silent_part['end']) < 0.1):
                    
                    # Load silent audio
                    audio_file = silent_data.get('audio_file', '')
                    if audio_file:
                        silent_path = silent_dir / audio_file
                        if silent_path.exists():
                            audio, _ = sf.read(silent_path)
                            return audio
                        
            except Exception:
                continue
        
        # If no silent audio found, return zeros (silence)
        duration = silent_part['end'] - silent_part['start']
        silent_samples = int(duration * self.sample_rate)
        return np.zeros(silent_samples, dtype=np.float32)
    
    def _adjust_length(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        current_samples = len(audio)
        
        if current_samples == target_samples:
            return audio
        elif current_samples > target_samples:
            return audio[:target_samples]
        else:
            padding = np.zeros(target_samples - current_samples)
            return np.concatenate([audio, padding])
    
    def _mix_with_instruments(self, audio: np.ndarray, instruments_path: str) -> np.ndarray:
        try:
            instruments_audio, _ = sf.read(instruments_path)
            
            min_length = min(len(audio), len(instruments_audio))
            audio = audio[:min_length]
            instruments_audio = instruments_audio[:min_length]
            
            mixed_audio = audio * 0.8 + instruments_audio * 0.2
            
            return mixed_audio
            
        except Exception as e:
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