"""
File Manager Module

Handles file operations, metadata management, and cleanup.
"""

import json
import shutil
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import numpy as np
import soundfile as sf


class FileManager:
    """Manages file operations and metadata"""
    
    def __init__(self, temp_dir: str = "/tmp/voice_cloning"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def create_directory_structure(self, base_dir: Path, speakers: List[str]):
        """Create directory structure for processed audio"""
        # Create main directories
        (base_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (base_dir / "silent_parts").mkdir(parents=True, exist_ok=True)
        
        # Create speaker directories
        for speaker in speakers:
            speaker_dir = base_dir / f"speaker_{speaker}"
            (speaker_dir / "segments").mkdir(parents=True, exist_ok=True)
            (speaker_dir / "reference").mkdir(parents=True, exist_ok=True)
    
    def save_silent_parts(self, silent_parts: List[Tuple[float, float]], 
                         audio: np.ndarray, sr: int, base_dir: Path):
        """Save silent parts as separate audio files"""
        silent_dir = base_dir / "silent_parts"
        
        for i, (start, end) in enumerate(silent_parts):
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            if start_sample < len(audio) and end_sample <= len(audio):
                silent_audio = audio[start_sample:end_sample]
                silent_file = silent_dir / f"silent_{i+1:03d}.wav"
                sf.write(silent_file, silent_audio, sr)
    
    def save_metadata(self, transcript_data: Dict, segments: List[Dict], 
                     silent_parts: List[Tuple[float, float]], 
                     base_dir: Path, audio_id: str, audio_path: str, 
                     additional_metadata: Optional[Dict] = None):
        """Save comprehensive metadata"""
        metadata = {
            'audio_id': audio_id,
            'original_audio_path': audio_path,
            'transcription_source': 'AssemblyAI',
            'speakers': transcript_data['speakers'],
            'total_segments': len(segments),
            'total_duration': transcript_data['duration'],
            'segments_by_speaker': {},
            'silent_parts_count': len(silent_parts),
            'segments_info': segments,
            'processing_timestamp': datetime.now().isoformat(),
            'dia_guidelines_followed': True
        }
        
        # Add additional metadata if provided
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Count segments by speaker
        for segment in segments:
            speaker = segment['speaker']
            if speaker not in metadata['segments_by_speaker']:
                metadata['segments_by_speaker'][speaker] = 0
            metadata['segments_by_speaker'][speaker] += 1
        
        # Save metadata
        metadata_path = base_dir / "metadata" / f"{audio_id}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def save_reference_segments(self, reference_segments: Dict[str, Dict], 
                               audio: np.ndarray, sr: int, base_dir: Path, 
                               speakers: List[str]):
        """Save reference segments for each speaker"""
        for speaker in speakers:
            if speaker not in reference_segments:
                continue
            
            ref_segment = reference_segments[speaker]
            speaker_dir = base_dir / f"speaker_{speaker}" / "reference"
            
            # Extract reference audio
            start_sample = int(ref_segment['start'] * sr)
            end_sample = int(ref_segment['end'] * sr)
            reference_audio = audio[start_sample:end_sample]
            
            # Save reference audio
            ref_audio_name = f"{base_dir.name}_speaker_{speaker}_REFERENCE.wav"
            ref_audio_path = speaker_dir / ref_audio_name
            sf.write(ref_audio_path, reference_audio, sr)
            
            # Save reference metadata
            ref_metadata = {
                'speaker': speaker,
                'reference_audio': ref_audio_name,
                'start': ref_segment['start'],
                'end': ref_segment['end'],
                'duration': ref_segment['duration'],
                'text': ref_segment['text'],
                'confidence': ref_segment['confidence'],
                'selected_reason': 'highest_confidence_and_quality'
            }
            
            ref_metadata_path = speaker_dir / f"{base_dir.name}_speaker_{speaker}_REFERENCE_metadata.json"
            with open(ref_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(ref_metadata, f, ensure_ascii=False, indent=2)
    
    def save_multi_speaker_reference(self, audio: np.ndarray, sr: int, 
                                   base_dir: Path, speakers: List[str]):
        """Save multi-speaker reference audio if multiple speakers exist"""
        if len(speakers) > 1:
            # Create combined reference for multi-speaker scenarios
            multi_ref_name = f"{base_dir.name}_MULTI_SPEAKER_REFERENCE.wav"
            multi_ref_path = base_dir / "metadata" / multi_ref_name
            
            # Use a portion of the original audio as multi-speaker reference
            reference_duration = min(30.0, len(audio) / sr)  # Max 30 seconds
            reference_samples = int(reference_duration * sr)
            reference_audio = audio[:reference_samples]
            
            sf.write(multi_ref_path, reference_audio, sr)
            
            # Save metadata
            multi_ref_metadata = {
                'type': 'multi_speaker_reference',
                'speakers': speakers,
                'duration': reference_duration,
                'sample_rate': sr,
                'usage': 'For multi-speaker voice cloning scenarios'
            }
            
            multi_ref_metadata_path = base_dir / "metadata" / f"{base_dir.name}_MULTI_SPEAKER_REFERENCE_metadata.json"
            with open(multi_ref_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(multi_ref_metadata, f, ensure_ascii=False, indent=2)
    
    def cleanup_temp_files(self, audio_id: str):
        """Clean up temporary files"""
        try:
            for item in self.temp_dir.iterdir():
                if audio_id in item.name:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
        except Exception as e:
            # Silently handle cleanup errors
            pass
    
    def get_processing_stats(self, segments_dir: str) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            segments_path = Path(segments_dir)
            metadata_files = list((segments_path / "metadata").glob("*_metadata.json"))
            
            if not metadata_files:
                return {"error": "Metadata file not found"}
            
            with open(metadata_files[0], 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return {
                "total_duration": metadata.get('total_duration', 0),
                "total_segments": metadata.get('total_segments', 0),
                "speakers": metadata.get('speakers', []),
                "segments_by_speaker": metadata.get('segments_by_speaker', {}),
                "silent_parts_count": metadata.get('silent_parts_count', 0),
                "transcription_source": metadata.get('transcription_source', 'AssemblyAI'),
                "dia_guidelines_followed": metadata.get('dia_guidelines_followed', False)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def validate_file_structure(self, base_dir: Path) -> Dict[str, Any]:
        """Validate the processed file structure"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required directories
        required_dirs = ["metadata", "silent_parts"]
        for dir_name in required_dirs:
            if not (base_dir / dir_name).exists():
                validation_result["errors"].append(f"Missing directory: {dir_name}")
                validation_result["valid"] = False
        
        # Check speaker directories
        speaker_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("speaker_")]
        if not speaker_dirs:
            validation_result["errors"].append("No speaker directories found")
            validation_result["valid"] = False
        
        # Check each speaker directory
        for speaker_dir in speaker_dirs:
            required_subdirs = ["segments", "reference"]
            for subdir in required_subdirs:
                if not (speaker_dir / subdir).exists():
                    validation_result["errors"].append(f"Missing subdirectory: {speaker_dir.name}/{subdir}")
                    validation_result["valid"] = False
        
        return validation_result
