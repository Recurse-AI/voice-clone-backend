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
        self.words_per_chunk = 80  # Increased for bigger segments
        self.max_duration = 20.0   # Maximum segment duration
        self.min_duration = 15.0   # Minimum segment duration
        self.min_words = 20        # Increased minimum words for bigger segments
        self.max_gap = 6.0         # Increased gap tolerance for bigger segments
        self.min_silent_duration = 3.0  # Minimum duration to consider as silent
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create larger segments with better vocal/silent separation"""
        words = transcript_data.get('words', [])
        if not words:
            return []
        
        # Create larger segments with aggressive combining
        segments = self._create_large_segments(words)
        
        # Further combine segments from same speaker
        combined_segments = self._aggressively_combine_segments(segments)
        
        return combined_segments
    
    def _create_large_segments(self, words: List[Dict]) -> List[Dict]:
        """Create larger segments by being more lenient with gaps"""
        segments = []
        current_chunk = []
        current_speaker = None
        
        for i, word in enumerate(words):
            speaker = word.get('speaker', 'A')
            word_start = word.get('start', 0) / 1000.0
            
            if current_chunk:
                prev_word = current_chunk[-1]
                prev_end = prev_word.get('end', 0) / 1000.0
                gap = word_start - prev_end
                
                # Much more lenient - only split on very large gaps or speaker change
                should_split = (
                    (speaker != current_speaker) or
                    (gap > self.max_gap) or  # Increased gap tolerance
                    (len(current_chunk) >= self.words_per_chunk)
                )
                
                if should_split:
                    if current_chunk:
                        segment = self._create_large_segment(current_chunk, current_speaker or 'A')
                        if segment:
                            segments.append(segment)
                    current_chunk = [word]
                    current_speaker = speaker
                else:
                    current_chunk.append(word)
            else:
                current_chunk.append(word)
                current_speaker = speaker
        
        # Add final chunk
        if current_chunk:
            segment = self._create_large_segment(current_chunk, current_speaker or 'A')
            if segment:
                segments.append(segment)
        
        return segments
    
    def _create_large_segment(self, words: List[Dict], speaker: str) -> Optional[Dict]:
        """Create larger segments with more lenient validation"""
        if len(words) < 5:  # Very lenient word count
            return None
            
        start_time = words[0]['start'] / 1000.0
        end_time = words[-1]['end'] / 1000.0
        duration = end_time - start_time
        
        # More lenient duration check
        if duration < 1.0:  # Very lenient minimum
            return None
        
        text = ' '.join(w['text'] for w in words)
        confidence = np.mean([w.get('confidence', 0.5) for w in words])
        
        # Basic quality check
        if len(text.strip()) < 3:
            return None
        
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
    
    def _aggressively_combine_segments(self, segments: List[Dict]) -> List[Dict]:
        """Aggressively combine segments from same speaker to make bigger segments"""
        if not segments:
            return segments
        
        combined = []
        i = 0
        
        while i < len(segments):
            current_segment = segments[i]
            
            # Look for consecutive segments from same speaker to combine
            segments_to_combine = [current_segment]
            j = i + 1
            
            while j < len(segments):
                next_segment = segments[j]
                gap = next_segment['start'] - current_segment['end']
                
                # Combine if same speaker and gap is reasonable
                if (next_segment['speaker'] == current_segment['speaker'] and 
                    gap < 8.0):  # Allow larger gaps for combining
                    
                    segments_to_combine.append(next_segment)
                    current_segment = next_segment  # Update end time
                    j += 1
                else:
                    break
            
            # Create combined segment if we have multiple segments
            if len(segments_to_combine) > 1:
                combined_segment = self._merge_multiple_segments(segments_to_combine)
                combined.append(combined_segment)
            else:
                # Check if single segment meets minimum requirements
                if (current_segment['duration'] >= self.min_duration and 
                    current_segment['word_count'] >= self.min_words):
                    combined.append(current_segment)
                elif len(combined) > 0 and combined[-1]['speaker'] == current_segment['speaker']:
                    # Merge with previous segment if same speaker
                    logger.info(f"Merging short segment ({current_segment['duration']:.1f}s) with previous segment")
                    combined[-1] = self._merge_segments(combined[-1], current_segment)
                else:
                    if current_segment['duration'] < self.min_duration:
                        logger.warning(f"Keeping short segment ({current_segment['duration']:.1f}s) - no adjacent segments to merge with")
                    combined.append(current_segment)  # Keep even if small
            
            i = j
        
        # Apply balanced splitting for segments that are too long
        final_segments = []
        for segment in combined:
            if segment['duration'] > self.max_duration:
                logger.info(f"Segment too long ({segment['duration']:.1f}s), splitting into balanced parts")
                # Split segment in a balanced way
                split_segments = self._split_segment_balanced(segment)
                final_segments.extend(split_segments)
            else:
                final_segments.append(segment)
        
        # Log segment statistics
        if final_segments:
            durations = [s['duration'] for s in final_segments]
            duration_strs = [f"{d:.1f}s" for d in durations]
            logger.info(f"Created {len(final_segments)} final segments with durations: {duration_strs}")
            logger.info(f"Duration range: {min(durations):.1f}s - {max(durations):.1f}s (target: {self.min_duration}-{self.max_duration}s)")
        
        return final_segments
    
    def _split_segment_balanced(self, segment: Dict) -> List[Dict]:
        """Split a long segment into balanced parts"""
        duration = segment['duration']
        words = segment.get('words', [])
        
        if duration <= self.max_duration:
            return [segment]
        
        # Calculate number of parts needed
        num_parts = int(np.ceil(duration / self.max_duration))
        
        # Try to make parts as equal as possible
        target_duration = duration / num_parts
        
        # Split by words for better balance
        if not words:
            # If no words, split by time
            return self._split_by_time(segment, num_parts)
        
        # Split by words
        words_per_part = len(words) // num_parts
        parts = []
        
        for i in range(num_parts):
            start_idx = i * words_per_part
            if i == num_parts - 1:
                # Last part gets remaining words
                end_idx = len(words)
            else:
                end_idx = (i + 1) * words_per_part
            
            part_words = words[start_idx:end_idx]
            
            if part_words:
                part_start = part_words[0]['start'] / 1000.0
                part_end = part_words[-1]['end'] / 1000.0
                part_duration = part_end - part_start
                
                part_text = ' '.join(w['text'] for w in part_words)
                part_confidence = np.mean([w.get('confidence', 0.5) for w in part_words])
                
                part = {
                    'start': part_start,
                    'end': part_end,
                    'duration': part_duration,
                    'text': part_text,
                    'speaker': segment['speaker'],
                    'word_count': len(part_words),
                    'confidence': part_confidence,
                    'words': part_words,
                    'is_split': True,
                    'original_segment': segment
                }
                parts.append(part)
        
        part_durations = [f"{p['duration']:.1f}s" for p in parts]
        logger.info(f"Split {duration:.1f}s segment into {len(parts)} balanced parts: {part_durations}")
        return parts
    
    def _split_by_time(self, segment: Dict, num_parts: int) -> List[Dict]:
        """Split segment by time when words are not available"""
        duration = segment['duration']
        start_time = segment['start']
        part_duration = duration / num_parts
        
        parts = []
        for i in range(num_parts):
            part_start = start_time + (i * part_duration)
            part_end = start_time + ((i + 1) * part_duration)
            
            part = {
                'start': part_start,
                'end': part_end,
                'duration': part_duration,
                'text': segment['text'],  # Same text for all parts
                'speaker': segment['speaker'],
                'word_count': segment['word_count'] // num_parts,
                'confidence': segment['confidence'],
                'words': [],
                'is_split': True,
                'original_segment': segment
            }
            parts.append(part)
        
        return parts
    
    def _merge_multiple_segments(self, segments: List[Dict]) -> Dict:
        """Merge multiple segments into one large segment"""
        if not segments:
            return None
        
        # Combine all words
        all_words = []
        for seg in segments:
            all_words.extend(seg.get('words', []))
        
        # Combine text
        combined_text = ' '.join(seg['text'] for seg in segments)
        
        # Calculate combined metrics
        start_time = segments[0]['start']
        end_time = segments[-1]['end']
        duration = end_time - start_time
        word_count = sum(seg['word_count'] for seg in segments)
        confidence = sum(seg['confidence'] * seg['word_count'] for seg in segments) / word_count if word_count > 0 else 0.5
        
        return {
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'text': combined_text,
            'speaker': segments[0]['speaker'],
            'word_count': word_count,
            'confidence': confidence,
            'words': all_words
        }
    
    def identify_silent_parts(self, segments: List[Dict], total_duration: float) -> List[Tuple[float, float]]:
        """Identify truly silent parts - only real gaps without any vocal"""
        silent_parts = []
        
        if not segments:
            return silent_parts
        
        # Only consider large gaps as silent (not small pauses)
        for i in range(len(segments) - 1):
            current_end = segments[i]['end']
            next_start = segments[i + 1]['start']
            gap_duration = next_start - current_end
            
            # Only mark as silent if gap is significant AND speakers are different
            # This avoids marking speech pauses as silent
            if (gap_duration >= self.min_silent_duration and
                segments[i]['speaker'] != segments[i + 1]['speaker']):
                
                # Further filter: only very large gaps
                if gap_duration >= 5.0:  # Only gaps of 5+ seconds
                    silent_parts.append((current_end, next_start))
        
        return silent_parts
    
    def select_optimal_references(self, segments: List[Dict], speakers: List[str]) -> Dict[str, Dict]:
        """Select references favoring larger segments"""
        references = {}
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s['speaker'] == speaker]
            
            if not speaker_segments:
                continue
            
            # Sort by duration and word count - favor larger segments
            speaker_segments.sort(
                key=lambda x: (
                    x['duration'],  # Prioritize duration first
                    x['word_count'],
                    x['confidence']
                ), 
                reverse=True
            )
            
            # Select the largest segment as reference
            best_segment = speaker_segments[0]
            
            # Very lenient criteria since we want larger segments
            if (best_segment['confidence'] >= 0.2 and 
                best_segment['duration'] >= 2.0 and
                best_segment['word_count'] >= 5):
                
                references[speaker] = best_segment
            else:
                # If no segment meets criteria, use the longest anyway
                references[speaker] = best_segment
        
        return references
    
    def _merge_segments(self, seg1: Dict, seg2: Dict) -> Dict:
        """Merge two segments into one longer segment"""
        combined_words = seg1.get('words', []) + seg2.get('words', [])
        combined_text = f"{seg1['text']} {seg2['text']}"
        combined_duration = seg2['end'] - seg1['start']
        combined_confidence = (seg1['confidence'] + seg2['confidence']) / 2
        
        return {
            'start': seg1['start'],
            'end': seg2['end'],
            'duration': combined_duration,
            'text': combined_text,
            'speaker': seg1['speaker'],
            'word_count': len(combined_words),
            'confidence': combined_confidence,
            'words': combined_words
        }
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int,
                            output_dir: Path, speakers: List[str], 
                            target_language: str, detected_language: str):
        """Save larger segments optimized for voice cloning"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # First, select optimal references for each speaker
        references = self.select_optimal_references(segments, speakers)
        
        # Process segments with larger segment approach
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
                
                # Process text for voice cloning
                english_text = self.transcription_service.translate_text_clean(segment['text'])
                
                # Create metadata for larger segments
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
                    'target_language': target_language,
                    'segment_type': 'large_combined'
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
                        'confidence': segment['confidence'],
                        'reference_type': 'large_segment'
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
                logger.warning(f"Failed to save segment {i+1}: {str(e)}")
                continue 