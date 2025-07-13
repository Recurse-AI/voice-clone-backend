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
import re

logger = logging.getLogger(__name__)


class SegmentManager:
    """Segment manager optimized for overlapping voice cloning"""
    
    def __init__(self, transcription_service):
        self.transcription_service = transcription_service
        self.words_per_chunk = 60  # Increased for bigger segments
        self.max_duration = 30.0   # Maximum duration before splitting
        self.min_duration = 15.0   # Minimum 15 seconds as requested
        self.min_words = 20        # Minimum words for a valid segment
        self.max_gap = 8.0         # Maximum gap tolerance
        self.min_silent_duration = 4.0  # Minimum duration to consider as silent
        self.target_split_duration = 15.0  # Target duration when splitting
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create segments with minimum 15 seconds duration"""
        words = transcript_data.get('words', [])
        if not words:
            return []
        
        # Create initial segments
        segments = self._create_initial_segments(words)
        
        # Merge short segments and split long ones
        final_segments = self._process_segments_for_duration(segments)
        
        return final_segments
    
    def _create_initial_segments(self, words: List[Dict]) -> List[Dict]:
        """Create initial segments based on speaker changes and gaps"""
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
                
                # Split on speaker change or large gap
                should_split = (
                    (speaker != current_speaker) or
                    (gap > self.max_gap) or
                    (len(current_chunk) >= self.words_per_chunk)
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
            else:
                current_chunk.append(word)
                current_speaker = speaker
        
        # Add final chunk
        if current_chunk:
            segment = self._create_segment(current_chunk, current_speaker or 'A')
            if segment:
                segments.append(segment)
        
        return segments
    
    def _create_segment(self, words: List[Dict], speaker: str) -> Optional[Dict]:
        """Create a segment from words"""
        if not words:
            return None
            
        start_time = words[0]['start'] / 1000.0
        end_time = words[-1]['end'] / 1000.0
        duration = end_time - start_time
        
        if duration < 1.0:  # Too short to be meaningful
            return None
        
        text = ' '.join(w['text'] for w in words).strip()
        confidence = np.mean([w.get('confidence', 0.5) for w in words])
        
        if len(text) < 3:
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
    
    def _process_segments_for_duration(self, segments: List[Dict]) -> List[Dict]:
        """Process segments to ensure minimum 15s duration and split long ones"""
        if not segments:
            return []
        
        processed = []
        i = 0
        
        while i < len(segments):
            current_segment = segments[i]
            
            # If segment is too short, try to merge with adjacent segments
            if current_segment['duration'] < self.min_duration:
                merged_segment = self._merge_with_adjacent(segments, i)
                if merged_segment:
                    processed.append(merged_segment)
                    # Skip merged segments
                    i = self._find_next_unmerged_index(segments, i, merged_segment)
                else:
                    # Can't merge, keep as is if it's the only option
                    logger.warning(f"Keeping short segment ({current_segment['duration']:.1f}s) - no merge possible")
                    processed.append(current_segment)
                    i += 1
            
            # If segment is too long, split it
            elif current_segment['duration'] > self.max_duration:
                split_segments = self._split_segment_intelligently(current_segment)
                processed.extend(split_segments)
                i += 1
            
            # Perfect duration, keep as is
            else:
                processed.append(current_segment)
                i += 1
        
        # Log final statistics
        if processed:
            durations = [s['duration'] for s in processed]
            duration_strs = [f"{d:.1f}s" for d in durations]
            logger.info(f"Final segments: {len(processed)} segments with durations: {duration_strs}")
            logger.info(f"Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
        
        return processed
    
    def _merge_with_adjacent(self, segments: List[Dict], index: int) -> Optional[Dict]:
        """Try to merge segment with adjacent segments to reach minimum duration"""
        current = segments[index]
        speaker = current['speaker']
        
        # Try to merge with next segments of same speaker
        segments_to_merge = [current]
        total_duration = current['duration']
        
        # Look forward for same speaker segments
        j = index + 1
        while j < len(segments) and total_duration < self.min_duration:
            next_seg = segments[j]
            gap = next_seg['start'] - segments_to_merge[-1]['end']
            
            # Merge if same speaker and reasonable gap
            if (next_seg['speaker'] == speaker and gap < 10.0):
                segments_to_merge.append(next_seg)
                total_duration = segments_to_merge[-1]['end'] - segments_to_merge[0]['start']
                j += 1
            else:
                break
        
        # If still too short, try to merge with previous segments
        if total_duration < self.min_duration and index > 0:
            k = index - 1
            while k >= 0 and total_duration < self.min_duration:
                prev_seg = segments[k]
                gap = segments_to_merge[0]['start'] - prev_seg['end']
                
                if (prev_seg['speaker'] == speaker and gap < 10.0):
                    segments_to_merge.insert(0, prev_seg)
                    total_duration = segments_to_merge[-1]['end'] - segments_to_merge[0]['start']
                    k -= 1
                else:
                    break
        
        # Merge if we have multiple segments
        if len(segments_to_merge) > 1:
            return self._merge_multiple_segments(segments_to_merge)
        
        return None
    
    def _find_next_unmerged_index(self, segments: List[Dict], start_index: int, merged_segment: Dict) -> int:
        """Find the next index after merged segments"""
        merged_end = merged_segment['end']
        
        for i in range(start_index, len(segments)):
            if segments[i]['end'] > merged_end:
                return i
        
        return len(segments)
    
    def _split_segment_intelligently(self, segment: Dict) -> List[Dict]:
        """Split long segment intelligently without breaking sentences"""
        duration = segment['duration']
        words = segment.get('words', [])
        
        if duration <= self.max_duration:
            return [segment]
        
        # Calculate target number of parts
        num_parts = int(np.ceil(duration / self.target_split_duration))
        
        if not words:
            return self._split_by_time(segment, num_parts)
        
        # Find good split points (sentence boundaries)
        split_points = self._find_sentence_boundaries(words)
        
        # Split based on sentence boundaries
        if split_points:
            return self._split_by_sentences(segment, split_points, num_parts)
        else:
            # Fallback to word-based splitting
            return self._split_by_words(segment, num_parts)
    
    def _find_sentence_boundaries(self, words: List[Dict]) -> List[int]:
        """Find sentence boundaries in words"""
        boundaries = []
        sentence_enders = ['.', '!', '?', '।', '|']  # Including Bengali sentence ender
        
        for i, word in enumerate(words):
            word_text = word.get('text', '').strip()
            if any(word_text.endswith(ender) for ender in sentence_enders):
                boundaries.append(i)
        
        return boundaries
    
    def _split_by_sentences(self, segment: Dict, sentence_boundaries: List[int], target_parts: int) -> List[Dict]:
        """Split segment by sentence boundaries"""
        words = segment['words']
        total_words = len(words)
        
        # Find optimal split points
        words_per_part = total_words // target_parts
        split_indices = []
        
        for i in range(1, target_parts):
            target_word_index = i * words_per_part
            
            # Find nearest sentence boundary
            best_boundary = min(sentence_boundaries, 
                              key=lambda x: abs(x - target_word_index),
                              default=target_word_index)
            
            if best_boundary not in split_indices:
                split_indices.append(best_boundary)
        
        # Create segments
        parts = []
        start_idx = 0
        
        for split_idx in sorted(split_indices):
            if start_idx < split_idx:
                part_words = words[start_idx:split_idx + 1]
                part = self._create_segment(part_words, segment['speaker'])
                if part:
                    parts.append(part)
                start_idx = split_idx + 1
        
        # Add final part
        if start_idx < len(words):
            part_words = words[start_idx:]
            part = self._create_segment(part_words, segment['speaker'])
            if part:
                parts.append(part)
        
        # Ensure parts meet minimum duration
        return self._ensure_minimum_duration(parts)
    
    def _split_by_words(self, segment: Dict, num_parts: int) -> List[Dict]:
        """Split segment by words when no sentence boundaries found"""
        words = segment['words']
        words_per_part = len(words) // num_parts
        
        parts = []
        for i in range(num_parts):
            start_idx = i * words_per_part
            if i == num_parts - 1:
                end_idx = len(words)
            else:
                end_idx = (i + 1) * words_per_part
            
            part_words = words[start_idx:end_idx]
            part = self._create_segment(part_words, segment['speaker'])
            if part:
                parts.append(part)
        
        return self._ensure_minimum_duration(parts)
    
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
                'text': segment['text'],
                'speaker': segment['speaker'],
                'word_count': segment['word_count'] // num_parts,
                'confidence': segment['confidence'],
                'words': [],
                'is_split': True
            }
            parts.append(part)
        
        return parts
    
    def _ensure_minimum_duration(self, segments: List[Dict]) -> List[Dict]:
        """Ensure all segments meet minimum duration by merging if necessary"""
        if not segments:
            return segments
        
        result = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            # If current segment is too short, try to merge with next
            if (current['duration'] < self.min_duration and 
                i + 1 < len(segments) and 
                segments[i + 1]['speaker'] == current['speaker']):
                
                merged = self._merge_segments(current, segments[i + 1])
                result.append(merged)
                i += 2  # Skip both segments
            else:
                result.append(current)
                i += 1
        
        return result
    
    def _merge_multiple_segments(self, segments: List[Dict]) -> Dict:
        """Merge multiple segments into one"""
        if not segments:
            return None
        
        if len(segments) == 1:
            return segments[0]
        
        # Combine all words
        all_words = []
        for seg in segments:
            all_words.extend(seg.get('words', []))
        
        # Combine text
        combined_text = ' '.join(seg['text'] for seg in segments)
        
        # Calculate metrics
        start_time = segments[0]['start']
        end_time = segments[-1]['end']
        duration = end_time - start_time
        word_count = sum(seg['word_count'] for seg in segments)
        
        # Weighted average confidence
        if word_count > 0:
            confidence = sum(seg['confidence'] * seg['word_count'] for seg in segments) / word_count
        else:
            confidence = 0.5
        
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
    
    def _merge_segments(self, seg1: Dict, seg2: Dict) -> Dict:
        """Merge two segments"""
        combined_words = seg1.get('words', []) + seg2.get('words', [])
        combined_text = f"{seg1['text']} {seg2['text']}"
        combined_duration = seg2['end'] - seg1['start']
        
        # Weighted average confidence
        total_words = seg1['word_count'] + seg2['word_count']
        if total_words > 0:
            combined_confidence = (seg1['confidence'] * seg1['word_count'] + 
                                 seg2['confidence'] * seg2['word_count']) / total_words
        else:
            combined_confidence = (seg1['confidence'] + seg2['confidence']) / 2
        
        return {
            'start': seg1['start'],
            'end': seg2['end'],
            'duration': combined_duration,
            'text': combined_text,
            'speaker': seg1['speaker'],
            'word_count': total_words,
            'confidence': combined_confidence,
            'words': combined_words
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
                
                # Preserve original text and format it for reference
                original_text = segment['text']  # This is the actual original text
                
                # Format for Dia if needed for reference
                if not original_text.startswith('[S'):
                    original_text_formatted = f"[S1] {original_text.strip()}"
                else:
                    original_text_formatted = original_text
                
                # Create metadata with proper original text preservation
                metadata = {
                    'segment_index': i + 1,
                    'audio_file': audio_filename,
                    'original_text': original_text,  # Keep actual original text
                    'original_text_formatted': original_text_formatted,  # Dia formatted version
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
                    
                    # Save reference metadata with proper original text
                    reference_metadata = {
                        'speaker': speaker,
                        'reference_audio': f"speaker_{speaker}_REFERENCE.wav",
                        'start': segment['start'],
                        'end': segment['end'],
                        'duration': segment['duration'],
                        'original_text': original_text,  # Actual original text
                        'original_text_formatted': original_text_formatted,  # Dia formatted
                        'english_text': english_text,
                        'word_count': segment['word_count'],
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