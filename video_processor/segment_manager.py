"""
Segment Manager Module

Segment creation optimized for overlapping voice cloning approach.
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SegmentManager:
    """Segment manager optimized for overlapping voice cloning"""
    
    def __init__(self, transcription_service):
        self.transcription_service = transcription_service
        self.words_per_chunk = 30  # Optimal for overlapping approach
        self.max_duration = 15.0   # Max duration per segment
        self.min_words = 5         # Minimum words per segment (reduced from 6)
        self.min_duration = 1.0    # Minimum duration per segment (reduced from 3.0)
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create segments optimized for overlapping voice cloning"""
        words = transcript_data.get('words', [])
        if not words:
            return []
        
        # Create initial segments with speaker awareness
        segments = []
        current_chunk = []
        current_speaker = None
        
        for word in words:
            speaker = word.get('speaker', 'A')
            
            # Check if we need to split due to speaker change or chunk size
            should_split = (
                (speaker != current_speaker and current_chunk) or 
                len(current_chunk) >= self.words_per_chunk
            )
            
            if should_split:
                if current_chunk:
                    segment = self._create_segment(current_chunk, current_speaker or 'A')
                    if segment:
                        segments.append(segment)
                current_chunk = [word]
                current_speaker = speaker
            else:
                current_chunk.append(word)
                current_speaker = speaker
        
        # Add final chunk
        if current_chunk:
            segment = self._create_segment(current_chunk, current_speaker or 'A')
            if segment:
                segments.append(segment)
        
        # Optimize segments for overlapping approach
        optimized_segments = self._optimize_for_overlapping(segments)
        
        return optimized_segments
    
    def _create_segment(self, words: List[Dict], speaker: str) -> Optional[Dict]:
        """Create a segment from words with validation"""
        if len(words) < self.min_words:
            return None
            
        start_time = words[0]['start'] / 1000.0
        end_time = words[-1]['end'] / 1000.0
        duration = end_time - start_time
        
        # Validate duration - more lenient
        if duration > self.max_duration:
            return None
            
        text = ' '.join(w['text'] for w in words)
        confidence = np.mean([w.get('confidence', 0.5) for w in words])
        
        return {
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'text': text,
            'speaker': speaker,
            'word_count': len(words),
            'confidence': confidence,
            'words': words
        }
    
    def _optimize_for_overlapping(self, segments: List[Dict]) -> List[Dict]:
        """Optimize segments for overlapping voice cloning approach"""
        if not segments:
            return segments
            
        optimized = []
        
        for segment in segments:
            # Keep segments that meet quality criteria - more lenient
            if (segment['confidence'] >= 0.3 and 
                segment['duration'] >= self.min_duration and 
                segment['word_count'] >= self.min_words):
                optimized.append(segment)
        
        return optimized
    
    def select_optimal_references(self, segments: List[Dict], speakers: List[str]) -> Dict[str, Dict]:
        """Select optimal reference segments for each speaker"""
        references = {}
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s['speaker'] == speaker]
            
            # Sort by quality metrics
            speaker_segments.sort(
                key=lambda x: (x['confidence'], x['word_count'], -abs(x['duration'] - 6.0)), 
                reverse=True
            )
            
            # Find best reference segment - more lenient criteria
            for segment in speaker_segments:
                if (segment['confidence'] >= 0.6 and 
                    segment['duration'] >= 4.0 and 
                    segment['duration'] <= 12.0 and
                    segment['word_count'] >= 10):
                    
                    references[speaker] = segment
                    break
        
        return references
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int,
                            output_dir: Path, speakers: List[str], 
                            target_language: str, detected_language: str):
        """Save segments optimized for overlapping voice cloning"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # First, select optimal references for each speaker
        references = self.select_optimal_references(segments, speakers)
        
        # Process segments for overlapping approach
        for i, segment in enumerate(segments):
            speaker = segment.get('speaker', 'A')
            speaker_dir = output_dir / f"speaker_{speaker}"
            segments_dir = speaker_dir / "segments"
            reference_dir = speaker_dir / "reference"
            
            # Create directories
            segments_dir.mkdir(parents=True, exist_ok=True)
            reference_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Extract audio segment
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # Save audio with consistent naming
                audio_filename = f"segment_{i+1:03d}.wav"
                audio_path = segments_dir / audio_filename
                sf.write(audio_path, segment_audio, sr)
                
                # Process text for overlapping approach
                english_text = self.transcription_service.translate_text_clean(segment['text'])
                
                # Create metadata optimized for overlapping approach
                metadata = {
                    'segment_index': i + 1,
                    'audio_file': audio_filename,
                    'original_text': segment['text'],
                    'english_text': english_text,
                    'speaker': speaker,
                    'start': segment['start'],
                    'end': segment['end'],
                    'duration': segment['duration'],
                    'word_count': segment['word_count'],
                    'confidence': segment['confidence'],
                    'detected_language': detected_language,
                    'target_language': target_language
                }
                
                # Save metadata
                metadata_path = segments_dir / f"segment_{i+1:03d}_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                # Save reference audio if this segment is selected as reference
                if speaker in references and references[speaker] == segment:
                    reference_audio_path = reference_dir / f"speaker_{speaker}_REFERENCE.wav"
                    sf.write(reference_audio_path, segment_audio, sr)
                    
                    # Save reference metadata
                    reference_metadata = {
                        'speaker': speaker,
                        'reference_text': english_text,
                        'original_segment_index': i + 1,
                        'duration': segment['duration'],
                        'confidence': segment['confidence']
                    }
                    reference_metadata_path = reference_dir / f"speaker_{speaker}_REFERENCE_metadata.json"
                    with open(reference_metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(reference_metadata, f, ensure_ascii=False, indent=2)
                
                # Update segment with paths
                segment.update({
                    'segment_index': i + 1,
                    'audio_path': str(audio_path),
                    'metadata_path': str(metadata_path),
                    'audio_file': audio_filename,
                    'english_text': english_text
                })
                
            except Exception as e:
                continue
    
    def identify_silent_parts(self, segments: List[Dict], total_duration: float) -> List[Tuple[float, float]]:
        """Identify silent parts between segments"""
        silent_parts = []
        
        if not segments:
            return silent_parts
        
        # Find gaps between segments
        for i in range(len(segments) - 1):
            current_end = segments[i]['end']
            next_start = segments[i + 1]['start']
            
            if next_start - current_end > 0.3:  # Gap > 0.3 seconds
                silent_parts.append((current_end, next_start))
        
        return silent_parts 