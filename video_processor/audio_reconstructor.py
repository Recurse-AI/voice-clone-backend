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
        """Reconstruct final audio from cloned segments with proper metadata handling"""
        try:
            segments_path = Path(segments_dir)
            
            # Collect all segments with validation
            all_segments = []
            
            for speaker_dir in segments_path.glob("speaker_*"):
                if not speaker_dir.is_dir():
                    continue
                    
                segments_subdir = speaker_dir / "segments"
                if not segments_subdir.exists():
                    continue
                    
                for json_file in segments_subdir.glob("*_metadata.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            segment_data = json.load(f)
                        
                        # Validate essential fields
                        if not segment_data.get('segment_index'):
                            continue
                        
                        if not segment_data.get('start') and segment_data.get('start') != 0:
                            continue
                        
                        if not segment_data.get('end'):
                            continue
                        
                        # Try multiple cloned audio file patterns
                        segment_index = segment_data.get('segment_index', 1)
                        cloned_audio_path = None
                        
                        # Pattern 1: cloned_segment_XXX.wav (new format)
                        cloned_filename = f"cloned_segment_{segment_index:03d}.wav"
                        path1 = segments_subdir / cloned_filename
                        
                        # Pattern 2: segment_XXX_cloned.wav (old format)
                        cloned_filename2 = f"segment_{segment_index}_cloned.wav"
                        path2 = segments_subdir / cloned_filename2
                        
                        # Pattern 3: segment_XXX_cloned.wav with 3-digit padding
                        cloned_filename3 = f"segment_{segment_index:03d}_cloned.wav"
                        path3 = segments_subdir / cloned_filename3
                        
                        # Use the first existing file
                        if path1.exists():
                            cloned_audio_path = path1
                        elif path2.exists():
                            cloned_audio_path = path2
                        elif path3.exists():
                            cloned_audio_path = path3
                        
                        # Check if any cloned audio file exists
                        if cloned_audio_path and cloned_audio_path.exists():
                            segment_data['cloned_audio_path'] = str(cloned_audio_path)
                            segment_data['cloned_audio_exists'] = True
                            all_segments.append(segment_data)
                        else:
                            print(f"Warning: No cloned audio file found for segment {segment_index}")
                            print(f"  Tried: {path1}, {path2}, {path3}")
                            continue
                            
                    except Exception as e:
                        print(f"Error processing metadata file {json_file}: {e}")
                        continue
            
            if not all_segments:
                print("No valid cloned segments found. Checking available files...")
                # Debug: Print what files are actually available
                for speaker_dir in segments_path.glob("speaker_*"):
                    if speaker_dir.is_dir():
                        segments_subdir = speaker_dir / "segments"
                        if segments_subdir.exists():
                            print(f"Files in {segments_subdir}:")
                            for file in segments_subdir.iterdir():
                                print(f"  - {file.name}")
                return {"success": False, "error": "No valid cloned segments found"}
            
            print(f"Found {len(all_segments)} valid cloned segments")
            
            # Sort segments by start time
            all_segments.sort(key=lambda x: x.get('start', 0))
            
            # Reconstruct audio
            reconstructed_audio = self._reconstruct_from_segments(all_segments)
            
            if reconstructed_audio is None:
                return {"success": False, "error": "Audio reconstruction failed"}
            
            # Mix with instruments if requested
            if include_instruments and instruments_path and os.path.exists(instruments_path):
                reconstructed_audio = self._mix_with_instruments(reconstructed_audio, instruments_path)
            
            # Save final audio
            final_path = self.temp_dir / f"final_output_{audio_id}.wav"
            sf.write(final_path, reconstructed_audio, self.sample_rate)
            
            # Create reconstruction summary
            reconstruction_summary = {
                'audio_id': audio_id,
                'total_segments_used': len(all_segments),
                'final_duration': len(reconstructed_audio) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'instruments_included': include_instruments and instruments_path is not None,
                'segments_by_speaker': {},
                'segments': [],  # Added detailed segment information
                'reconstruction_timestamp': str(datetime.now())
            }
            
            # Import R2Storage for URL generation
            from r2_storage import R2Storage
            r2_storage = R2Storage()
            
            # Count segments by speaker and prepare detailed segment info
            for segment in all_segments:
                speaker = segment.get('speaker', 'Unknown')
                if speaker not in reconstruction_summary['segments_by_speaker']:
                    reconstruction_summary['segments_by_speaker'][speaker] = 0
                reconstruction_summary['segments_by_speaker'][speaker] += 1
                
                # Generate R2 URL for cloned segment
                segment_index = segment.get('segment_index', 1)
                cloned_filename = f"cloned_segment_{segment_index:03d}.wav"
                
                # Generate accurate R2 URL using R2Storage class
                r2_segment_url = r2_storage.generate_cloned_segment_url(audio_id, speaker, segment_index)
                
                # Create detailed segment info
                segment_info = {
                    "segment_url": r2_segment_url,
                    "start_time": segment.get('start', 0.0),
                    "duration": segment.get('duration', segment.get('end', 0.0) - segment.get('start', 0.0)),
                    "end_time": segment.get('end', 0.0),
                    "speaker": speaker,
                    "segment_index": segment_index,
                    "original_text": segment.get('original_text', ''),
                    "english_text": segment.get('english_text', ''),
                    "confidence": segment.get('confidence', 0.0),
                    "word_count": segment.get('word_count', 0),
                    "cloned_filename": cloned_filename,
                    "processing_status": segment.get('processing_status', 'completed')
                }
                
                reconstruction_summary['segments'].append(segment_info)
            
            # Save reconstruction summary
            summary_path = segments_path / "metadata" / "reconstruction_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(reconstruction_summary, f, ensure_ascii=False, indent=2)
            
            return {
                "success": True,
                "final_audio_path": str(final_path),
                "duration": len(reconstructed_audio) / self.sample_rate,
                "sample_rate": self.sample_rate,
                "audio_id": audio_id,
                "segments_processed": len(all_segments),
                "segments_by_speaker": reconstruction_summary['segments_by_speaker']
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "audio_id": audio_id}
    
    def _reconstruct_from_segments(self, segments: List[Dict]) -> Optional[np.ndarray]:
        """Reconstruct audio from segments"""
        try:
            if not segments:
                return None
            
            # Calculate total duration
            total_duration = segments[-1]['end']
            total_samples = int(total_duration * self.sample_rate)
            
            # Create final audio array
            final_audio = np.zeros(total_samples, dtype=np.float32)
            
            # Process each segment
            for segment in segments:
                try:
                    # Load cloned audio
                    cloned_audio, _ = sf.read(segment['cloned_audio_path'])
                    
                    # Calculate position in final audio
                    start_sample = int(segment['start'] * self.sample_rate)
                    end_sample = int(segment['end'] * self.sample_rate)
                    
                    # Bounds check
                    start_sample = max(0, min(start_sample, total_samples - 1))
                    end_sample = max(start_sample + 1, min(end_sample, total_samples))
                    
                    # Adjust cloned audio length
                    expected_samples = end_sample - start_sample
                    if len(cloned_audio) != expected_samples:
                        cloned_audio = self._adjust_length(cloned_audio, expected_samples)
                    
                    # Place in final audio
                    final_audio[start_sample:end_sample] = cloned_audio
                    
                except Exception:
                    continue
            
            return final_audio
            
        except Exception:
            return None
    
    def _adjust_length(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        """Adjust audio length to target samples"""
        current_samples = len(audio)
        
        if current_samples == target_samples:
            return audio
        elif current_samples > target_samples:
            return audio[:target_samples]
        else:
            padding = np.zeros(target_samples - current_samples, dtype=np.float32)
            return np.concatenate([audio, padding])
    
    def _mix_with_instruments(self, audio: np.ndarray, instruments_path: str) -> np.ndarray:
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