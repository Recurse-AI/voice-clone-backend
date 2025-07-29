"""
Audio Reconstructor Module - Simplified
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AudioReconstructor:
    """Simplified audio reconstructor"""
    
    def __init__(self, temp_dir: str = "/tmp/voice_cloning"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = 44100
    
    def reconstruct_final_audio(self, segments_dir: str, audio_id: str, 
                               include_instruments: bool = False,
                               instruments_path: Optional[str] = None) -> Dict[str, Any]:
        """Reconstruct final audio from unified segments folder"""
        try:
            segments_path = Path(segments_dir)
            logger.info(f"Starting audio reconstruction for audio_id: {audio_id}")
            
            # Check unified structure
            segments_folder = segments_path / "segments"
            cloned_folder = segments_path / "cloned_segments"  # Fixed: match voice_cloning service directory name
            metadata_folder = segments_path / "metadata"
            
            if not segments_folder.exists():
                return {"success": False, "error": "Segments folder not found"}
            
            if not cloned_folder.exists():
                return {"success": False, "error": "Cloned folder not found"}
            
            # Try to get original audio duration from processing metadata
            original_duration = None
            if metadata_folder.exists():
                processing_metadata_file = metadata_folder / "processing_metadata.json"
                if processing_metadata_file.exists():
                    try:
                        with open(processing_metadata_file, 'r', encoding='utf-8') as f:
                            import json
                            processing_metadata = json.load(f)
                            original_duration = processing_metadata.get("original_audio", {}).get("duration")
                            if original_duration:
                                logger.info(f"Found original audio duration from metadata: {original_duration:.2f} seconds")
                    except Exception as e:
                        logger.warning(f"Could not read processing metadata: {e}")
            
            # Collect all segments from unified folders
            all_segments = []
            
            # Get all metadata files and sort by segment index
            metadata_files = sorted(list(segments_folder.glob("*_metadata.json")))
            logger.info(f"Found {len(metadata_files)} segment metadata files")
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        import json
                        metadata = json.load(f)
                    
                    segment_index = metadata.get('segment_index', 0)
                    
                    # Construct cloned audio path from segment index (voice cloning service saves with this pattern)
                    cloned_audio_filename = f"cloned_segment_{segment_index:03d}.wav"
                    cloned_audio_path = cloned_folder / cloned_audio_filename
                    
                    if not cloned_audio_path.exists():
                        logger.warning(f"Cloned audio not found for segment {segment_index}: {cloned_audio_path}")
                        continue
                    
                    # Load cloned audio
                    cloned_audio, sr = sf.read(cloned_audio_path)
                    
                    all_segments.append({
                        'segment_index': segment_index,
                        'start': metadata.get('start', 0),
                        'end': metadata.get('end', 0),
                        'duration': metadata.get('duration', 0),
                        'audio': cloned_audio,
                        'speaker': metadata.get('speaker', 'A'),
                        'sample_rate': sr,
                        'original_text': metadata.get('original_text', ''),
                        'english_text': metadata.get('english_text', ''),
                        'confidence': metadata.get('confidence', 0.0),
                        'word_count': metadata.get('word_count', 0)
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {metadata_file.name}: {e}")
                    continue
            
            if not all_segments:
                return {"success": False, "error": "No valid cloned segments found"}
            
            # Sort segments by segment index to maintain order
            all_segments.sort(key=lambda x: x['segment_index'])
            logger.info(f"Processing {len(all_segments)} segments for reconstruction")
            
            # Reconstruct audio with original duration
            reconstructed_audio = self._reconstruct_from_segments(all_segments, original_duration)
            
            if reconstructed_audio is None or len(reconstructed_audio) == 0:
                return {"success": False, "error": "Audio reconstruction failed"}
            
            # Calculate actual final duration
            final_duration = len(reconstructed_audio) / all_segments[0]['sample_rate']
            original_duration_str = f"{original_duration:.2f}" if original_duration is not None else "unknown"
            logger.info(f"Final reconstructed audio duration: {final_duration:.2f} seconds (original: {original_duration_str} seconds)")
            
            # Save final audio
            final_audio_filename = f"final_output_{audio_id}.wav"
            final_audio_path = segments_path / final_audio_filename
            sf.write(str(final_audio_path), reconstructed_audio, all_segments[0]['sample_rate'])
            
            # Mix with instruments if requested
            if include_instruments and instruments_path and os.path.exists(instruments_path):
                mixed_audio = self._mix_with_instruments(reconstructed_audio, instruments_path, all_segments[0]['sample_rate'])
                if mixed_audio is not None:
                    # Overwrite with mixed version
                    sf.write(str(final_audio_path), mixed_audio, all_segments[0]['sample_rate'])
                    logger.info("Mixed with instruments successfully")
            
            # Save reconstruction summary
            self._save_reconstruction_summary(segments_path, all_segments, audio_id, original_duration, final_duration)
            
            logger.info(f"Audio reconstruction completed: {final_audio_path}")
            
            return {
                "success": True,
                "final_audio_path": str(final_audio_path),
                "total_segments": len(all_segments),
                "total_duration": final_duration,
                "original_duration": original_duration,
                "sample_rate": all_segments[0]['sample_rate'],
                "instruments_mixed": include_instruments and instruments_path is not None
            }
            
        except Exception as e:
            logger.error(f"Audio reconstruction failed: {str(e)}")
            return {"success": False, "error": f"Audio reconstruction failed: {str(e)}"}
    
    def _reconstruct_from_segments(self, segments: List[Dict], original_duration: Optional[float]) -> Optional[np.ndarray]:
        """Simplified audio reconstruction with properly sized segments"""
        
        try:
            if not segments:
                return None
            
            # Use original duration if available, otherwise calculate from segments
            if original_duration and original_duration > 0:
                total_duration = original_duration
                logger.info(f"Using original audio duration: {total_duration:.2f} seconds for reconstruction")
            else:
                # Calculate total duration from segments
                total_duration = segments[-1]['end'] if segments else 0
                logger.warning(f"Original duration not available, using segment end time: {total_duration:.2f}s")
            
            total_samples = int(total_duration * self.sample_rate)
            
            # Create final audio array filled with silence
            final_audio = np.zeros(total_samples, dtype=np.float32)
            
            # Process each segment
            segments_placed = 0
            
            logger.info(f"Processing {len(segments)} segments to reconstruct {total_duration:.2f}s of audio")
            
            for segment in segments:
                try:
                    # Use pre-loaded audio data
                    cloned_audio = segment.get('audio')
                    segment_index = segment.get('segment_index', 'unknown')
                    segment_start = segment['start']
                    segment_end = segment['end']
                    segment_duration = segment_end - segment_start
                    
                    if cloned_audio is None or len(cloned_audio) == 0:
                        # Handle silent segments
                        logger.info(f"Segment {segment_index}: Keeping silence for {segment_duration:.2f}s from {segment_start:.2f}s to {segment_end:.2f}s")
                        segments_placed += 1
                        continue
                    
                    audio_duration = len(cloned_audio) / self.sample_rate
                    
                    # With our new duration enforcement, audio should already match segment duration
                    if abs(audio_duration - segment_duration) > 0.1:
                        logger.warning(f"Segment {segment_index} duration mismatch: timeline={segment_duration:.3f}s vs audio={audio_duration:.3f}s")
                    
                    # Calculate position in final audio
                    start_sample = int(segment_start * self.sample_rate)
                    end_sample = int(segment_end * self.sample_rate)
                    
                    # Bounds check
                    start_sample = max(0, min(start_sample, total_samples - 1))
                    end_sample = max(start_sample + 1, min(end_sample, total_samples))
                    
                    # Adjust cloned audio length if needed (should rarely happen now)
                    expected_samples = end_sample - start_sample
                    if len(cloned_audio) != expected_samples:
                        logger.debug(f"Segment {segment_index}: final adjustment from {len(cloned_audio)} to {expected_samples} samples")
                        cloned_audio = self._adjust_length(cloned_audio, expected_samples)
                    
                    # Place cloned voice in final audio
                    final_audio[start_sample:end_sample] = cloned_audio
                    segments_placed += 1
                    
                    logger.debug(f"Placed segment {segment_index} audio: {start_sample}-{end_sample} samples")
                    
                except Exception as e:
                    logger.error(f"Error processing segment {segment.get('segment_index', 'unknown')}: {e}")
                    segments_placed += 1
                    continue
            
            logger.info(f"Successfully placed {segments_placed}/{len(segments)} segments in final audio")
            logger.info(f"Final audio length: {len(final_audio)/self.sample_rate:.2f} seconds ({len(final_audio)} samples)")
            logger.info(f"Duration preservation: Original={total_duration:.2f}s, Final={len(final_audio)/self.sample_rate:.2f}s")
            
            return final_audio
            
        except Exception as e:
            logger.error(f"Audio reconstruction failed: {e}")
            return None
    
    def _adjust_length(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        """Simple audio length adjustment"""
        if len(audio) == target_samples:
            return audio
        elif len(audio) > target_samples:
            return audio[:target_samples]
        else:
            # Pad with silence
            padding = target_samples - len(audio)
            silence = np.zeros(padding, dtype=audio.dtype)
            return np.concatenate([audio, silence])
    
    def _mix_with_instruments(self, audio: np.ndarray, instruments_path: str, sample_rate: int) -> Optional[np.ndarray]:
        """Mix audio with instruments at original volume levels"""
        try:
            instruments_audio, _ = sf.read(instruments_path)
            
            min_length = min(len(audio), len(instruments_audio))
            audio = audio[:min_length]
            instruments_audio = instruments_audio[:min_length]
            
            # Keep both at original volume levels
            mixed_audio = audio + instruments_audio
            
            # Normalize to prevent clipping while preserving relative volumes
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 1.0:
                mixed_audio = mixed_audio / max_val
            
            return mixed_audio
            
        except Exception:
            return None
    
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

    def _save_reconstruction_summary(self, segments_path: Path, all_segments: List[Dict], audio_id: str, original_duration: Optional[float], final_duration: float):
        """Clean reconstruction summary without unnecessary speaker statistics"""
        summary_path = segments_path / "metadata" / "reconstruction_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        reconstruction_summary = {
            'audio_id': str(audio_id),
            'total_segments_used': int(len(all_segments)),
            'final_duration': float(final_duration),
            'original_duration': float(original_duration) if original_duration is not None else float('nan'),
            'sample_rate': int(all_segments[0]['sample_rate']) if all_segments else 44100,
            'instruments_included': False,
            'segments': [],
            'reconstruction_timestamp': str(datetime.now())
        }

        # Import R2Storage for URL generation
        from r2_storage import R2Storage
        r2_storage = R2Storage()

        # Prepare clean segment info
        for segment in all_segments:
            speaker = segment.get('speaker', 'A')
            segment_index = segment.get('segment_index', 1)
            cloned_filename = f"cloned_segment_{segment_index:03d}.wav"
            
            # Generate accurate R2 URL using R2Storage class
            r2_segment_url = r2_storage.generate_cloned_segment_url(audio_id, speaker, segment_index)
            
            # Create clean segment info
            segment_info = {
                "segment_url": r2_segment_url,
                "start_time": float(segment.get('start', 0.0)),
                "duration": float(segment.get('duration', segment.get('end', 0.0) - segment.get('start', 0.0))),
                "end_time": float(segment.get('end', 0.0)),
                "speaker": str(speaker),
                "segment_index": int(segment_index),
                "original_text": str(segment.get('original_text', '')),
                "english_text": str(segment.get('english_text', '')),
                "confidence": float(segment.get('confidence', 0.0)),
                "word_count": int(segment.get('word_count', 0)),
                "cloned_filename": str(cloned_filename),
                "processing_status": str(segment.get('processing_status', 'completed'))
            }
            
            reconstruction_summary['segments'].append(segment_info)
        
        # Save reconstruction summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(reconstruction_summary, f, ensure_ascii=False, indent=2)