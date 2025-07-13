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
        """Create directory structure for segments and references"""
        # Create base directories
        (base_dir / "metadata").mkdir(parents=True, exist_ok=True)
        
        # Create speaker directories
        for speaker in speakers:
            speaker_dir = base_dir / f"speaker_{speaker}"
            (speaker_dir / "segments").mkdir(parents=True, exist_ok=True)
            (speaker_dir / "reference").mkdir(parents=True, exist_ok=True)
        
        # Create silent parts directory
        (base_dir / "silent_parts").mkdir(parents=True, exist_ok=True)
    
    def _create_timeline(self, segments: List[Dict], silent_parts: List[Tuple[float, float]], 
                        base_dir: Path, total_duration: float):
        """Create timeline for audio reconstruction"""
        timeline = []
        
        # Add speech segments with proper indexing
        for segment in segments:
            timeline.append({
                'segment_type': 'speech',
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'speaker': segment['speaker'],
                'segment_index': segment.get('segment_index'),  # Add segment index for mapping
                'audio_file': segment.get('audio_file'),  # Add audio file reference
                'is_continuous': segment.get('is_continuous', True),
                'group_id': segment.get('group_id')
            })
        
        # Add silent parts
        for start, end in silent_parts:
            timeline.append({
                'segment_type': 'silent',
                'start': start,
                'end': end,
                'duration': end - start
            })
        
        # Sort by start time
        timeline.sort(key=lambda x: x['start'])
        
        # Save timeline
        timeline_data = {
            'audio_id': base_dir.name.replace('segments_', ''),
            'total_duration': total_duration,
            'timeline': timeline,
            'speech_segments': len([t for t in timeline if t['segment_type'] == 'speech']),
            'silent_parts': len([t for t in timeline if t['segment_type'] == 'silent'])
        }
        
        timeline_path = base_dir / "metadata" / "timeline.json"
        with open(timeline_path, 'w', encoding='utf-8') as f:
            json.dump(timeline_data, f, ensure_ascii=False, indent=2)
    
    def save_silent_parts(self, silent_parts: List[Tuple[float, float]], 
                         audio: np.ndarray, sr: int, base_dir: Path):
        """Save silent parts as separate audio files"""
        silent_dir = base_dir / "silent_parts"
        
        for i, (start, end) in enumerate(silent_parts):
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            if start_sample < len(audio) and end_sample <= len(audio):
                silent_audio = audio[start_sample:end_sample]
                
                # Save audio file
                audio_filename = f"silent_{i+1:03d}.wav"
                silent_file = silent_dir / audio_filename
                sf.write(silent_file, silent_audio, sr)
                
                # Save metadata
                metadata = {
                    'silent_id': i + 1,
                    'start': start,
                    'end': end,
                    'duration': end - start,
                    'audio_file': audio_filename
                }
                
                metadata_path = silent_dir / f"silent_{i+1:03d}.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def save_reference_segments(self, reference_segments: Dict[str, Dict], 
                               audio: np.ndarray, sr: int, base_dir: Path, 
                               speakers: List[str]):
        """Save reference segments for each speaker with improved logic"""
        for speaker in speakers:
            speaker_dir = base_dir / f"speaker_{speaker}" / "reference"
            
            # Clear existing reference files to avoid duplicates
            if speaker_dir.exists():
                for existing_file in speaker_dir.glob("*"):
                    if existing_file.is_file():
                        existing_file.unlink()
            
            if speaker not in reference_segments:
                # Create a note that no reference was found
                no_ref_note = {
                    'speaker': speaker,
                    'reference_type': 'none',
                    'note': 'No suitable reference segment found for this speaker',
                    'reason': 'Insufficient audio duration or quality'
                }
                
                no_ref_path = speaker_dir / f"speaker_{speaker}_NO_REFERENCE.json"
                with open(no_ref_path, 'w', encoding='utf-8') as f:
                    json.dump(no_ref_note, f, ensure_ascii=False, indent=2)
                continue
            
            ref_segment = reference_segments[speaker]
            
            # Handle composite references differently
            if ref_segment and ref_segment.get('is_composite', False):
                self._save_composite_reference(ref_segment, speaker_dir, base_dir, speaker)
                continue
            
            # Save regular reference
            self._save_regular_reference(ref_segment, audio, sr, speaker_dir, base_dir, speaker)
    
    def _save_regular_reference(self, ref_segment: Dict, audio: np.ndarray, sr: int, 
                               speaker_dir: Path, base_dir: Path, speaker: str):
        """Save regular reference segment with audio"""
        start_time = ref_segment.get('start', 0)
        end_time = ref_segment.get('end', 5)
        duration = ref_segment.get('duration', 5)
        
        # Extract reference audio
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Ensure we don't exceed audio bounds
        if end_sample > len(audio):
            end_sample = len(audio)
        if start_sample >= end_sample:
            start_sample = max(0, end_sample - int(5 * sr))  # Fallback to 5 seconds minimum
        
        reference_audio = audio[start_sample:end_sample]
        
        # Save reference audio with consistent naming
        ref_audio_name = f"speaker_{speaker}_REFERENCE.wav"
        ref_audio_path = speaker_dir / ref_audio_name
        sf.write(ref_audio_path, reference_audio, sr)
        
        # Get clean English text
        english_text = ref_segment.get('english_text', ref_segment.get('text', f'Reference for speaker {speaker}'))
        original_text = ref_segment.get('original_text', ref_segment.get('text', english_text))
        
        # Save reference metadata
        ref_metadata = {
            'speaker': speaker,
            'reference_audio': ref_audio_name,
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'english_text': english_text,
            'original_text': original_text,
            'word_count': ref_segment.get('word_count', len(english_text.split())),
            'confidence': ref_segment.get('confidence', 0.5),
            'selected_reason': 'best_available_segment',
            'reference_quality': self._assess_reference_quality(ref_segment)
        }
        
        ref_metadata_path = speaker_dir / f"speaker_{speaker}_REFERENCE_metadata.json"
        with open(ref_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(ref_metadata, f, ensure_ascii=False, indent=2)
    
    def _assess_reference_quality(self, segment: Dict) -> str:
        """Assess reference quality for better feedback"""
        duration = segment.get('duration', 0)
        word_count = segment.get('word_count', 0)
        confidence = segment.get('confidence', 0)
        
        # Assess based on criteria
        if duration >= 8 and word_count >= 20 and confidence >= 0.7:
            return 'excellent'
        elif duration >= 5 and word_count >= 15 and confidence >= 0.5:
            return 'good'
        elif duration >= 3 and word_count >= 10 and confidence >= 0.3:
            return 'acceptable'
        else:
            return 'poor'
    
    def _save_composite_reference(self, ref_segment: Dict, speaker_dir: Path, 
                                 base_dir: Path, speaker: str):
        """Save composite reference created from mixed segments"""
        ref_metadata = {
            'speaker': speaker,
            'reference_type': 'composite',
            'source': ref_segment.get('source', 'mixed_segments'),
            'english_text': ref_segment.get('english_text', ref_segment.get('text', '')),
            'original_text': ref_segment.get('original_text', ''),
            'word_count': ref_segment.get('word_count', 0),
            'estimated_duration': ref_segment.get('duration', 10.0),
            'confidence': ref_segment.get('confidence', 0.5),
            'is_composite': True,
            'note': 'Composite reference created from mixed segments - audio will be generated during cloning'
        }
        
        ref_metadata_path = speaker_dir / f"speaker_{speaker}_COMPOSITE_REFERENCE_metadata.json"
        with open(ref_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(ref_metadata, f, ensure_ascii=False, indent=2)
    
    def save_metadata(self, transcript_data: Dict, segments: List[Dict], 
                     silent_parts: List[Tuple[float, float]], 
                     base_dir: Path, audio_id: str, audio_path: str, 
                     additional_metadata: Optional[Dict] = None):
        """Save comprehensive metadata with improved segment analysis"""
        # Calculate better statistics
        total_speech_duration = sum(seg['duration'] for seg in segments)
        total_silent_duration = sum(end - start for start, end in silent_parts)
        
        # Analyze segment quality
        segment_quality = {
            'high_quality': len([s for s in segments if s.get('confidence', 0) > 0.7]),
            'medium_quality': len([s for s in segments if 0.4 <= s.get('confidence', 0) <= 0.7]),
            'low_quality': len([s for s in segments if s.get('confidence', 0) < 0.4])
        }
        
        metadata = {
            'audio_id': audio_id,
            'original_audio_path': audio_path,
            'transcription_source': 'AssemblyAI',
            'speakers': transcript_data.get('speakers', ['A']),
            'total_segments': len(segments),
            'total_duration': transcript_data.get('duration', 0),
            'total_speech_duration': total_speech_duration,
            'total_silent_duration': total_silent_duration,
            'speech_ratio': total_speech_duration / max(transcript_data.get('duration', 1), 1),
            'segments_by_speaker': {},
            'segment_quality': segment_quality,
            'silent_parts_count': len(silent_parts),
            'segments_info': segments,
            'processing_timestamp': datetime.now().isoformat(),
            'improvements_applied': [
                'better_gap_detection',
                'improved_reference_selection',
                'enhanced_word_timing',
                'reduced_silent_parts'
            ],
            'raw_assemblyai_response': transcript_data.get('raw_assemblyai_response', {})
        }
        
        # Add additional metadata if provided
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Count segments by speaker
        for segment in segments:
            speaker = segment.get('speaker', 'A')
            if speaker not in metadata['segments_by_speaker']:
                metadata['segments_by_speaker'][speaker] = 0
            metadata['segments_by_speaker'][speaker] += 1
        
        # Save metadata
        metadata_path = base_dir / "metadata" / f"{audio_id}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Create and save timeline
        self._create_timeline(segments, silent_parts, base_dir, transcript_data.get('duration', 0))
    
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
                        # Clean up all subdirectories
                        if item.name.startswith("segments_"):
                            for speaker_dir in item.iterdir():
                                if speaker_dir.is_dir() and (speaker_dir.name.startswith("speaker_") or speaker_dir.name in ["silent_parts", "metadata"]):
                                    shutil.rmtree(speaker_dir)
                        shutil.rmtree(item)
                    else:
                        item.unlink()
        except Exception as e:
            # Silently handle cleanup errors
            pass
    
    def get_processing_stats(self, segments_dir: str) -> Dict[str, Any]:
        """Get processing statistics from timeline"""
        try:
            segments_path = Path(segments_dir)
            timeline_file = segments_path / "metadata" / "timeline.json"
            
            if not timeline_file.exists():
                return {"error": "Timeline file not found"}
            
            with open(timeline_file, 'r', encoding='utf-8') as f:
                timeline_data = json.load(f)
            
            timeline = timeline_data.get('timeline', [])
            speech_segments = [t for t in timeline if t.get('segment_type') == 'speech']
            silent_parts = [t for t in timeline if t.get('segment_type') == 'silent']
            
            # Calculate stats
            total_duration = max(t['end'] for t in timeline) if timeline else 0
            speakers = list(set(s['speaker'] for s in speech_segments))
            
            segments_by_speaker = {}
            for speaker in speakers:
                segments_by_speaker[speaker] = len([s for s in speech_segments if s['speaker'] == speaker])
            
            return {
                "total_duration": total_duration,
                "total_segments": len(speech_segments),
                "speakers": speakers,
                "segments_by_speaker": segments_by_speaker,
                "silent_parts_count": len(silent_parts),
                "transcription_source": "AssemblyAI",
                "processing_method": "word_based_segments_with_silent_parts"
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
