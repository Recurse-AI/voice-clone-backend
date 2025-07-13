"""
Segment Manager Module

Handles intelligent segment creation with continuous/non-continuous detection
and optimal chunking for voice cloning.
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SegmentManager:
    """Smart segment manager with continuous/non-continuous detection"""
    
    def __init__(self, transcription_service):
        self.transcription_service = transcription_service
        self.min_segment_duration = 2.0
        self.max_segment_duration = 20.0
        self.optimal_word_range = (25, 45)
        self.max_gap_for_merge = 2.0
        self.min_confidence = 0.8
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create optimal segments with smart chunking"""
        words = transcript_data.get('words', [])
        if not words:
            return []
        
        # Group words by speaker
        speaker_segments = self._group_by_speaker(words)
        
        # Process each speaker's segments
        optimal_segments = []
        for speaker, segments in speaker_segments.items():
            speaker_optimal = self._process_speaker_segments(speaker, segments)
            optimal_segments.extend(speaker_optimal)
        
        # Sort by start time
        optimal_segments.sort(key=lambda x: x['start'])
        return optimal_segments
    
    def _group_by_speaker(self, words: List[Dict]) -> Dict[str, List[List[Dict]]]:
        """Group words by speaker and create continuous segments"""
        speaker_segments = {}
        current_segment = []
        current_speaker = None
        
        for word in words:
            speaker = word.get('speaker', 'A')
            
            if speaker != current_speaker:
                if current_segment and current_speaker:
                    if current_speaker not in speaker_segments:
                        speaker_segments[current_speaker] = []
                    speaker_segments[current_speaker].append(current_segment)
                current_segment = [word]
                current_speaker = speaker
            else:
                current_segment.append(word)
        
        # Add last segment
        if current_segment and current_speaker:
            if current_speaker not in speaker_segments:
                speaker_segments[current_speaker] = []
            speaker_segments[current_speaker].append(current_segment)
        
        return speaker_segments
    
    def _process_speaker_segments(self, speaker: str, segments: List[List[Dict]]) -> List[Dict]:
        """Process segments for a speaker with continuous/non-continuous detection"""
        processed_segments = []
        
        for segment_words in segments:
            if not segment_words:
                continue
            
            # Check if segment is in optimal range
            word_count = len(segment_words)
            
            if self.optimal_word_range[0] <= word_count <= self.optimal_word_range[1]:
                # Single optimal segment - mark as continuous
                processed_segments.append(self._create_segment(
                    segment_words, speaker, is_continuous=True
                ))
            else:
                # Check for non-continuous opportunities
                non_continuous_groups = self._find_non_continuous_groups(segments)
                
                if non_continuous_groups:
                    for group in non_continuous_groups:
                        processed_segments.extend(self._create_non_continuous_segments(
                            group, speaker
                        ))
                else:
                    # Split large segments
                    if word_count > self.optimal_word_range[1]:
                        split_segments = self._split_large_segment(segment_words, speaker)
                        processed_segments.extend(split_segments)
                    else:
                        # Keep small segments as is
                        processed_segments.append(self._create_segment(
                            segment_words, speaker, is_continuous=True
                        ))
        
        return processed_segments
    
    def _find_non_continuous_groups(self, segments: List[List[Dict]]) -> List[List[List[Dict]]]:
        """Find groups of segments that can be merged for non-continuous processing"""
        groups = []
        current_group = []
        
        for i, segment in enumerate(segments):
            if not current_group:
                current_group.append(segment)
                continue
            
            # Check gap between segments
            last_word_prev = current_group[-1][-1]
            first_word_curr = segment[0]
            gap = (first_word_curr['start'] - last_word_prev['end']) / 1000.0
            
            if gap <= self.max_gap_for_merge:
                current_group.append(segment)
                
                # Check if group is in optimal range
                total_words = sum(len(s) for s in current_group)
                if self.optimal_word_range[0] <= total_words <= self.optimal_word_range[1]:
                    groups.append(current_group)
                    current_group = []
            else:
                if len(current_group) > 1:
                    groups.append(current_group)
                current_group = [segment]
        
        return groups
    
    def _create_non_continuous_segments(self, segment_group: List[List[Dict]], 
                                      speaker: str) -> List[Dict]:
        """Create non-continuous segments from a group"""
        segments = []
        
        for segment_words in segment_group:
            segment = self._create_segment(segment_words, speaker, is_continuous=False)
            segment['group_id'] = id(segment_group)  # Mark as part of same group
            segments.append(segment)
        
        return segments
    
    def _split_large_segment(self, words: List[Dict], speaker: str) -> List[Dict]:
        """Split large segment into optimal chunks"""
        segments = []
        chunk_size = self.optimal_word_range[1]
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                segments.append(self._create_segment(chunk_words, speaker, is_continuous=True))
        
        return segments
    
    def _create_segment(self, words: List[Dict], speaker: str, is_continuous: bool) -> Dict:
        """Create a segment from words"""
        start_time = words[0]['start'] / 1000.0
        end_time = words[-1]['end'] / 1000.0
        text = ' '.join(w['text'] for w in words)
        confidence = np.mean([w.get('confidence', 0.5) for w in words])
        
        return {
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time,
            'text': text,
            'speaker': speaker,
            'word_count': len(words),
            'confidence': confidence,
            'is_continuous': is_continuous,
            'words': words
        }
    
    def select_optimal_references(self, segments: List[Dict], speakers: List[str]) -> Dict[str, Dict]:
        """Select optimal reference audio for each speaker"""
        references = {}
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s['speaker'] == speaker]
            
            # Find best reference
            best_ref = self._find_best_reference(speaker_segments)
            
            if best_ref:
                references[speaker] = best_ref
            else:
                # Create composite reference if needed
                composite_ref = self._create_composite_reference(speaker_segments)
                if composite_ref:
                    references[speaker] = composite_ref
        
        return references
    
    def _find_best_reference(self, segments: List[Dict]) -> Optional[Dict]:
        """Find best single reference segment"""
        candidates = []
        
        for segment in segments:
            if (segment['confidence'] >= self.min_confidence and
                7.0 <= segment['duration'] <= 15.0 and
                segment['word_count'] >= 10):
                candidates.append(segment)
        
        if candidates:
            # Sort by confidence and optimal duration (11s is ideal)
            candidates.sort(key=lambda x: (x['confidence'], abs(11.0 - x['duration'])))
            best = candidates[0]
            
            # Ensure dia_text is included
            if 'dia_text' not in best:
                best['dia_text'] = f"[S1] {best.get('text', '')}"
            
            return best
        
        return None
    
    def _create_composite_reference(self, segments: List[Dict]) -> Optional[Dict]:
        """Create composite reference from multiple segments"""
        # Sort by confidence
        segments.sort(key=lambda x: x['confidence'], reverse=True)
        
        composite_segments = []
        total_duration = 0
        
        for segment in segments:
            if total_duration + segment['duration'] <= 15.0:
                composite_segments.append(segment)
                total_duration += segment['duration']
                if total_duration >= 7.0:
                    break
        
        if composite_segments and total_duration >= 7.0:
            return {
                'is_composite': True,
                'segments': composite_segments,
                'duration': total_duration,
                'text': ' '.join(s['text'] for s in composite_segments),
                'word_count': sum(s['word_count'] for s in composite_segments)
            }
        
        return None
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int,
                            output_dir: Path, speakers: List[str], 
                            target_language: str, detected_language: str):
        """Save segments with metadata"""
        for i, segment in enumerate(segments):
            speaker = segment['speaker']
            speaker_dir = output_dir / f"speaker_{speaker}" / "segments"
            
            # Extract audio
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Save audio
            audio_filename = f"segment_{i+1:03d}_{speaker}.wav"
            audio_path = speaker_dir / audio_filename
            sf.write(audio_path, segment_audio, sr)
            
            # Translate if needed
            if detected_language != 'en':
                english_text = self.transcription_service.translate_text_clean(segment['text'])
            else:
                english_text = segment['text']
            
            # Format for Dia
            dia_text = self.transcription_service.format_dia_text(
                english_text, speaker, speakers
            )
            
            # Save metadata
            metadata = {
                'segment_id': i + 1,
                'speaker': speaker,
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'text': segment['text'],
                'english_text': english_text,
                'dia_text': dia_text,
                'word_count': segment['word_count'],
                'confidence': segment['confidence'],
                'is_continuous': segment.get('is_continuous', True),
                'group_id': segment.get('group_id'),
                'audio_file': audio_filename
            }
            
            metadata_path = speaker_dir / f"segment_{i+1:03d}_{speaker}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def identify_silent_parts(self, segments: List[Dict], total_duration: float) -> List[Tuple[float, float]]:
        """Identify silent parts between segments"""
        silent_parts = []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        # Check gaps between segments
        for i in range(len(sorted_segments) - 1):
            gap_start = sorted_segments[i]['end']
            gap_end = sorted_segments[i + 1]['start']
            gap_duration = gap_end - gap_start
            
            if gap_duration >= 2.0:  # Only track significant silent parts
                silent_parts.append((gap_start, gap_end))
        
        # Check beginning
        if sorted_segments and sorted_segments[0]['start'] >= 2.0:
            silent_parts.append((0, sorted_segments[0]['start']))
        
        # Check end
        if sorted_segments and total_duration - sorted_segments[-1]['end'] >= 2.0:
            silent_parts.append((sorted_segments[-1]['end'], total_duration))
        
        return silent_parts 