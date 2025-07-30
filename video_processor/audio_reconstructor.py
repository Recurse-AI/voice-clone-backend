"""
Audio Reconstructor Module - Fish Speech Compatible
Complete audio reconstruction from cloned segments with timeline maintenance
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioReconstructor:
    """Complete audio reconstructor for dubbed content"""
    
    def __init__(self, temp_dir: str = "/tmp/voice_cloning"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = 44100
    
    def reconstruct_final_audio(self, segments_dir: str, audio_id: str, 
                               include_instruments: bool = False,
                               instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Reconstruct complete dubbed audio from cloned segments"""
        try:
            logger.info(f"🎵 Starting dubbed audio reconstruction for {audio_id}")
            
            segments_path = Path(segments_dir)
            
            # Get timeline and segments
            timeline_data = self._get_timeline_data(segments_path)
            if not timeline_data["success"]:
                return timeline_data
            
            # Reconstruct vocal track
            vocal_result = self._reconstruct_vocal_track(
                segments_path, timeline_data["segments"], audio_id
            )
            
            if not vocal_result["success"]:
                return vocal_result
            
            # Mix with instruments if requested
            if include_instruments and instruments_path:
                final_result = self._mix_with_instruments(
                    vocal_result["output_path"], instruments_path, audio_id
                )
            else:
                final_result = vocal_result
            
            logger.info(f"✅ Audio reconstruction completed: {final_result['output_path']}")
            
            return {
                "success": True,
                "output_path": final_result["output_path"],
                "duration": final_result["duration"],
                "sample_rate": self.sample_rate,
                "reconstruction_stats": {
                    "total_segments": timeline_data["total_segments"],
                    "speech_segments": timeline_data["speech_segments"],
                    "silence_segments": timeline_data["silence_segments"],
                    "cloned_segments": vocal_result.get("cloned_segments", 0),
                    "timeline_maintained": True,
                    "instruments_included": include_instruments
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Audio reconstruction failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _get_timeline_data(self, segments_path: Path) -> Dict[str, Any]:
        """Get complete timeline data from segments"""
        try:
            # Load segmentation summary
            summary_file = segments_path / "segmentation_summary.json"
            if not summary_file.exists():
                return {"success": False, "error": "Segmentation summary not found"}
            
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            # Get all segment metadata files
            segments_folder = segments_path / "segments"
            metadata_files = sorted(
                segments_folder.glob("*_metadata.json"),
                key=lambda x: self._extract_segment_index(x.name)
            )
            
            segments = []
            speech_count = 0
            silence_count = 0
            
            for metadata_file in metadata_files:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                segment_data = {
                    "index": metadata["segment_index"],
                    "start": metadata["start"],
                    "end": metadata["end"],
                    "duration": metadata["duration"],
                    "type": metadata["type"],
                    "speaker": metadata["speaker"],
                    "audio_file": segments_folder / metadata["audio_file"],
                    "metadata": metadata
                }
                
                # Check for cloned audio
                cloned_folder = segments_path / "cloned"
                cloned_file = cloned_folder / f"cloned_segment_{metadata['segment_index']:03d}.wav"
                
                if cloned_file.exists():
                    segment_data["cloned_audio"] = cloned_file
                    segment_data["has_cloned"] = True
                else:
                    segment_data["has_cloned"] = False
                
                segments.append(segment_data)
                
                if segment_data["type"] == "speech":
                    speech_count += 1
                else:
                    silence_count += 1
            
            return {
                "success": True,
                "segments": segments,
                "total_segments": len(segments),
                "speech_segments": speech_count,
                "silence_segments": silence_count,
                "total_duration": summary.get("total_duration", 0)
            }
                    
        except Exception as e:
            return {"success": False, "error": f"Timeline data extraction failed: {str(e)}"}
    
    def _reconstruct_vocal_track(self, segments_path: Path, segments: List[Dict[str, Any]], 
                               audio_id: str) -> Dict[str, Any]:
        """Reconstruct complete vocal track with proper timeline"""
        try:
            logger.info(f"🔄 Reconstructing vocal track with {len(segments)} segments")
            
            # Calculate total duration and samples
            if not segments:
                return {"success": False, "error": "No segments to reconstruct"}
            
            total_duration = max(seg["end"] for seg in segments)
            total_samples = int(total_duration * self.sample_rate)
            
            # Create empty audio buffer
            final_audio = np.zeros(total_samples, dtype=np.float32)
            
            cloned_count = 0
            
            # Process each segment in timeline order
            for segment in sorted(segments, key=lambda x: x["start"]):
                start_sample = int(segment["start"] * self.sample_rate)
                end_sample = int(segment["end"] * self.sample_rate)
                segment_samples = end_sample - start_sample
                
                # Determine audio source
                if segment["type"] == "speech" and segment["has_cloned"]:
                    # Use cloned audio
                    audio_source = segment["cloned_audio"]
                    segment_type = "cloned"
                    cloned_count += 1
                    
                elif segment["type"] == "speech":
                    # Use original audio if no cloned version
                    audio_source = segment["audio_file"]
                    segment_type = "original"
                    
                else:
                    # Create silence for non-speech segments
                    segment_audio = np.zeros(segment_samples, dtype=np.float32)
                    final_audio[start_sample:end_sample] = segment_audio
                    logger.debug(f"Added silence: segment {segment['index']} ({segment['duration']:.2f}s)")
                    continue
                
                # Load and process audio
                if audio_source.exists():
                    try:
                        audio_data, sr = sf.read(str(audio_source))
                        
                        # Convert to mono if stereo
                        if len(audio_data.shape) > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        
                        # Resample if needed
                        if sr != self.sample_rate:
                            audio_data = self._resample_audio(audio_data, sr, self.sample_rate)
                        
                        # Match exact length
                        audio_data = self._match_segment_length(audio_data, segment_samples)
                        
                        # Add to final audio
                        final_audio[start_sample:end_sample] = audio_data
                        
                        logger.debug(f"Added {segment_type}: segment {segment['index']} ({segment['duration']:.2f}s)")
                        
                    except Exception as e:
                        logger.warning(f"Failed to load segment {segment['index']}: {str(e)}")
                        # Fill with silence on error
                        segment_audio = np.zeros(segment_samples, dtype=np.float32)
                        final_audio[start_sample:end_sample] = segment_audio
                
                else:
                    logger.warning(f"Audio file not found for segment {segment['index']}")
                    # Fill with silence
                    segment_audio = np.zeros(segment_samples, dtype=np.float32)
                    final_audio[start_sample:end_sample] = segment_audio
            
            # Normalize audio
            if np.max(np.abs(final_audio)) > 0:
                final_audio = final_audio / np.max(np.abs(final_audio)) * 0.95
            
            # Save reconstructed vocal track
            output_filename = f"dubbed_vocal_{audio_id}.wav"
            output_path = self.temp_dir / output_filename
            
            sf.write(str(output_path), final_audio, self.sample_rate)
            
            logger.info(f"✅ Vocal track reconstructed: {cloned_count}/{len([s for s in segments if s['type'] == 'speech'])} segments cloned")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "duration": total_duration,
                "cloned_segments": cloned_count,
                "total_speech_segments": len([s for s in segments if s["type"] == "speech"])
            }
            
        except Exception as e:
            return {"success": False, "error": f"Vocal reconstruction failed: {str(e)}"}
    
    def _mix_with_instruments(self, vocal_path: str, instruments_path: str, 
                            audio_id: str) -> Dict[str, Any]:
        """Mix dubbed vocal with original instruments"""
        try:
            logger.info("🎶 Mixing dubbed vocal with instruments")
            
            # Load vocal and instrument tracks
            vocal_audio, vocal_sr = sf.read(vocal_path)
            instruments_audio, instruments_sr = sf.read(instruments_path)
            
            # Convert to mono if needed
            if len(vocal_audio.shape) > 1:
                vocal_audio = np.mean(vocal_audio, axis=1)
            if len(instruments_audio.shape) > 1:
                instruments_audio = np.mean(instruments_audio, axis=1)
            
            # Resample to common sample rate
            target_sr = self.sample_rate
            if vocal_sr != target_sr:
                vocal_audio = self._resample_audio(vocal_audio, vocal_sr, target_sr)
            if instruments_sr != target_sr:
                instruments_audio = self._resample_audio(instruments_audio, instruments_sr, target_sr)
            
            # Match lengths
            max_length = max(len(vocal_audio), len(instruments_audio))
            vocal_audio = self._match_segment_length(vocal_audio, max_length)
            instruments_audio = self._match_segment_length(instruments_audio, max_length)
            
            # Mix with appropriate levels - vocal prominent, instruments as background effect
            vocal_level = 0.8      # Vocal prominent (80%)
            instrument_level = 0.2  # Instruments background effect (20%)
            
            mixed_audio = (vocal_audio * vocal_level + instruments_audio * instrument_level)
            
            # Normalize to prevent clipping
            if np.max(np.abs(mixed_audio)) > 0:
                mixed_audio = mixed_audio / np.max(np.abs(mixed_audio)) * 0.95
            
            # Save mixed audio
            output_filename = f"dubbed_final_{audio_id}.wav"
            output_path = self.temp_dir / output_filename
            
            sf.write(str(output_path), mixed_audio, target_sr)
            
            logger.info("✅ Audio mixing completed")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "duration": len(mixed_audio) / target_sr
            }
            
        except Exception as e:
            return {"success": False, "error": f"Audio mixing failed: {str(e)}"}
    
    def _match_segment_length(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        """Match audio to exact sample length"""
        current_samples = len(audio)
        
        if current_samples == target_samples:
            return audio
        elif current_samples > target_samples:
            # Crop
            return audio[:target_samples]
        else:
            # Pad with silence
            padding = np.zeros(target_samples - current_samples, dtype=audio.dtype)
            return np.concatenate([audio, padding])
    
    def _resample_audio(self, audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling (basic implementation)"""
        if source_sr == target_sr:
            return audio
        
        # Basic resampling - in production, use librosa.resample
        ratio = target_sr / source_sr
        new_length = int(len(audio) * ratio)
        
        # Simple interpolation
        old_indices = np.linspace(0, len(audio) - 1, new_length)
        new_indices = np.arange(new_length)
        
        resampled = np.interp(old_indices, np.arange(len(audio)), audio)
        return resampled.astype(audio.dtype)
    
    def _extract_segment_index(self, filename: str) -> int:
        """Extract segment index from filename"""
        try:
            # Extract number from filename like "segment_001_metadata.json"
            import re
            match = re.search(r'segment_(\d+)', filename)
            return int(match.group(1)) if match else 0
        except:
            return 0
    
    def cleanup_temp_files(self, audio_id: str):
        """Clean up temporary reconstruction files"""
        try:
            patterns = [
                f"dubbed_vocal_{audio_id}.wav",
                f"dubbed_final_{audio_id}.wav",
                f"temp_*_{audio_id}.*"
            ]
            
            for pattern in patterns:
                for file_path in self.temp_dir.glob(pattern):
                    if file_path.exists():
                        file_path.unlink()
                        logger.debug(f"Cleaned up: {file_path}")
                        
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")


# Additional utility functions for audio processing
def validate_audio_file(file_path: str) -> Dict[str, Any]:
    """Validate audio file integrity"""
    try:
        audio_data, sample_rate = sf.read(file_path)
        duration = len(audio_data) / sample_rate
        
        return {
            "valid": True,
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": len(audio_data.shape),
            "samples": len(audio_data)
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }


def get_audio_info(segments_dir: str) -> Dict[str, Any]:
    """Get complete audio reconstruction info"""
    try:
        segments_path = Path(segments_dir)
        
        # Count segments
        cloned_folder = segments_path / "cloned"
        segments_folder = segments_path / "segments"
        
        cloned_files = list(cloned_folder.glob("*.wav")) if cloned_folder.exists() else []
        segment_files = list(segments_folder.glob("*.wav")) if segments_folder.exists() else []
        
        return {
            "segments_dir": str(segments_path),
            "total_segments": len(segment_files),
            "cloned_segments": len(cloned_files),
            "cloning_completion": len(cloned_files) / len(segment_files) * 100 if segment_files else 0,
            "ready_for_reconstruction": cloned_folder.exists() and len(cloned_files) > 0
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "ready_for_reconstruction": False
        }