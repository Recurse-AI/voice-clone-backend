"""
Segment Manager Module

Handles intelligent audio segmentation with ordering, silent segments, and optimal segment duration.
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SegmentManager:
    """Manages audio segmentation with optimal duration and ordering"""
    
    def __init__(self, transcription_service):
        self.transcription_service = transcription_service
        self.min_silent_duration = 2.0  # Minimum 2 seconds for silent segments
        self.optimal_word_min = 35  # Optimal minimum words per segment
        self.optimal_word_max = 45  # Optimal maximum words per segment
        self.max_word_count = 60  # Hard maximum words
        self.max_segment_duration = 20.0  # Maximum 20 seconds for any segment
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create optimal segments from transcript data with proper ordering"""
        words = transcript_data.get('words', [])
        speakers = transcript_data.get('speakers', ['A'])
        
        if not words:
            logger.warning("No words found in transcript")
            return []
        
        # Store raw AssemblyAI response for later use
        raw_response = transcript_data.get('raw_assemblyai_response', {})
        
        # Create speaker segments based on word count
        speaker_segments = self._create_speaker_segments_by_words(words, speakers)
        
        # Optimize segment word counts (35-45 words preference)
        optimized_segments = self._optimize_segment_word_counts(speaker_segments)
        
        # Add raw response to each segment
        for segment in optimized_segments:
            segment['assemblyai_transcript_id'] = raw_response.get('id', '')
            segment['detected_language'] = raw_response.get('language_code', 'en')
        
        return optimized_segments
    
    def _create_speaker_segments_by_words(self, words: List[Dict], speakers: List[str]) -> List[Dict[str, Any]]:
        """Create segments based on speaker changes and word count"""
        if not words:
            return []
        
        segments = []
        current_segment = {
            'speaker': words[0].get('speaker', 'A'),
            'start': words[0]['start'] / 1000.0,  # Convert to seconds
            'words': [words[0]],
            'text': words[0]['text'],
            'word_count': 1
        }
        
        for word in words[1:]:
            word_speaker = word.get('speaker', 'A')
            word_start = word['start'] / 1000.0
            word_end = word['end'] / 1000.0
            
            # Check if we need to create a new segment
            should_split = False
            
            # Calculate current segment duration if we add this word
            potential_duration = word_end - current_segment['start']
            potential_word_count = current_segment['word_count'] + 1
            
            # Split on speaker change ONLY if current segment has at least 15 words
            if word_speaker != current_segment['speaker']:
                if current_segment['word_count'] >= 15:
                    should_split = True
            
            # Split if current segment is getting too long (word count or duration)
            if potential_word_count >= self.max_word_count or potential_duration >= self.max_segment_duration:
                should_split = True
            
            if should_split:
                # Finalize current segment
                current_segment['end'] = current_segment['words'][-1]['end'] / 1000.0
                current_segment['duration'] = current_segment['end'] - current_segment['start']
                segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    'speaker': word_speaker,
                    'start': word_start,
                    'words': [word],
                    'text': word['text'],
                    'word_count': 1
                }
            else:
                # Add word to current segment
                current_segment['words'].append(word)
                current_segment['text'] += ' ' + word['text']
                current_segment['word_count'] += 1
        
        # Add final segment
        if current_segment['words']:
            current_segment['end'] = current_segment['words'][-1]['end'] / 1000.0
            current_segment['duration'] = current_segment['end'] - current_segment['start']
            segments.append(current_segment)
        
        return segments
    
    def _optimize_segment_word_counts(self, segments: List[Dict]) -> List[Dict]:
        """Optimize segments to have 35-45 words each"""
        optimized = []
        i = 0
        
        while i < len(segments):
            current = segments[i].copy()
            
            # If segment has optimal word count, keep it
            if self.optimal_word_min <= current['word_count'] <= self.optimal_word_max:
                optimized.append(current)
                i += 1
            
            # If segment is too short, try to merge with next segments
            elif current['word_count'] < self.optimal_word_min:
                # Try to merge with following segments
                merged = current.copy()
                j = i + 1
                
                while j < len(segments) and merged['word_count'] < self.optimal_word_max:
                    next_seg = segments[j]
                    
                    # Check potential word count and duration after merge
                    potential_word_count = merged['word_count'] + next_seg['word_count']
                    potential_duration = next_seg['end'] - merged['start']
                    
                    # If same speaker and won't exceed limits, merge
                    if (next_seg['speaker'] == merged['speaker'] and 
                        potential_word_count <= self.max_word_count and 
                        potential_duration <= self.max_segment_duration):
                        
                        # Merge segments
                        merged['words'].extend(next_seg['words'])
                        merged['text'] += ' ' + next_seg['text']
                        merged['end'] = next_seg['end']
                        merged['duration'] = merged['end'] - merged['start']
                        merged['word_count'] = potential_word_count
                        j += 1
                        
                        # If we're in optimal range now, stop merging
                        if merged['word_count'] >= self.optimal_word_min:
                            break
                    else:
                        # Can't merge this segment, but check if we can create mixed segment
                        if next_seg['speaker'] != merged['speaker'] and len(segments[i:j+1]) > 1:
                            # Check if creating a mixed segment would help reach optimal word count
                            mixed_word_count = sum(s['word_count'] for s in segments[i:j+1])
                            mixed_duration = segments[j]['end'] - merged['start']
                            
                            if (self.optimal_word_min <= mixed_word_count <= self.optimal_word_max and
                                mixed_duration <= self.max_segment_duration):
                                # Create mixed segment
                                mixed_segment = self._create_mixed_segment(segments[i:j+1])
                                if mixed_segment:
                                    optimized.append(mixed_segment)
                                    i = j + 1
                                    break
                        break
                
                # If we couldn't reach optimal word count but have something, keep it
                if j > i + 1 or merged['word_count'] >= 15:  # At least 15 words
                    optimized.append(merged)
                    i = j
                else:
                    # Too short and can't merge, keep as is
                    optimized.append(current)
                    i += 1
            
            # If segment is too long, split it into optimal chunks
            elif current['word_count'] > self.optimal_word_max:
                split_segments = self._split_long_segment_by_words(current)
                optimized.extend(split_segments)
                i += 1
            
            else:
                optimized.append(current)
                i += 1
        
        return optimized
    
    def _split_long_segment_by_words(self, segment: Dict) -> List[Dict]:
        """Split a long segment into optimal chunks (35-45 words preferred)"""
        words = segment['words']
        if not words:
            return [segment]
        
        # Calculate how many splits we need
        total_words = len(words)
        num_splits = max(1, total_words // self.optimal_word_max)
        
        # Calculate target words per split
        target_words = total_words // num_splits
        # Ensure target is within optimal range
        target_words = max(self.optimal_word_min, min(target_words, self.optimal_word_max))
        
        splits = []
        current_split = {
            'speaker': segment['speaker'],
            'start': words[0]['start'] / 1000.0,
            'words': [],
            'text': '',
            'word_count': 0
        }
        
        for word in words:
            word_end = word['end'] / 1000.0
            current_duration = word_end - current_split['start']
            
            # Check if we should create a new split
            should_split = False
            
            # Split if we've reached target word count
            if current_split['word_count'] >= target_words and current_split['words']:
                # Look ahead to see if we have enough words left
                remaining_words = len([w for w in words if w['start'] > word['start']])
                if remaining_words >= 15:  # Still have words left for another segment
                    should_split = True
            
            # Force split if we're at max word count or duration
            if (current_split['word_count'] >= self.optimal_word_max or 
                current_duration >= self.max_segment_duration) and current_split['words']:
                should_split = True
            
            if should_split:
                # Finalize current split
                current_split['end'] = current_split['words'][-1]['end'] / 1000.0
                current_split['duration'] = current_split['end'] - current_split['start']
                splits.append(current_split)
                
                # Start new split
                current_split = {
                    'speaker': segment['speaker'],
                    'start': word['start'] / 1000.0,
                    'words': [word],
                    'text': word['text'],
                    'word_count': 1
                }
            else:
                # Add word to current split
                current_split['words'].append(word)
                if current_split['text']:
                    current_split['text'] += ' ' + word['text']
                else:
                    current_split['text'] = word['text']
                current_split['word_count'] += 1
        
        # Add final split
        if current_split['words']:
            current_split['end'] = current_split['words'][-1]['end'] / 1000.0
            current_split['duration'] = current_split['end'] - current_split['start']
            splits.append(current_split)
        
        return splits if splits else [segment]
    
    def _create_mixed_segment(self, segments: List[Dict]) -> Optional[Dict]:
        """Create a mixed segment from multiple speakers for optimal word count"""
        if not segments or len(segments) < 2:
            return None
        
        # Get all unique speakers
        speakers = list(set(seg['speaker'] for seg in segments))
        if len(speakers) < 2:
            return None  # Not actually mixed
        
        # Create mixed segment
        mixed = {
            'speaker': 'MIXED',  # Special marker for mixed segments
            'speakers': speakers,
            'start': segments[0]['start'],
            'end': segments[-1]['end'],
            'duration': segments[-1]['end'] - segments[0]['start'],
            'words': [],
            'text': '',
            'word_count': 0,
            'is_mixed': True
        }
        
        # Combine all words maintaining speaker info
        for seg in segments:
            speaker_tag = f"[S{speakers.index(seg['speaker']) + 1}]"
            if mixed['text']:
                mixed['text'] += f" {speaker_tag} {seg['text']}"
            else:
                mixed['text'] = f"{speaker_tag} {seg['text']}"
            mixed['words'].extend(seg['words'])
            mixed['word_count'] += seg['word_count']
        
        return mixed
    
    def identify_silent_parts(self, segments: List[Dict], total_duration: float) -> List[Tuple[float, float]]:
        """Identify silent parts between segments (minimum 2 seconds)"""
        silent_parts = []
        
        # Check silence at the beginning
        if segments and segments[0]['start'] >= self.min_silent_duration:
            silent_parts.append((0, segments[0]['start']))
        
        # Check silence between segments
        for i in range(len(segments) - 1):
            gap_start = segments[i]['end']
            gap_end = segments[i + 1]['start']
            gap_duration = gap_end - gap_start
            
            if gap_duration >= self.min_silent_duration:
                silent_parts.append((gap_start, gap_end))
        
        # Check silence at the end
        if segments and total_duration - segments[-1]['end'] >= self.min_silent_duration:
            silent_parts.append((segments[-1]['end'], total_duration))
        
        return silent_parts
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int,
                            base_dir: Path, speakers: List[str], 
                            target_language: str, detected_language: str):
        """Save segments with metadata and translations"""
        # Identify and save silent parts
        total_duration = len(audio) / sr
        silent_parts = self.identify_silent_parts(segments, total_duration)
        self._save_silent_parts(silent_parts, audio, sr, base_dir)
        
        # Create timeline for reconstruction
        timeline = []
        
        # Add silent parts to timeline
        for start, end in silent_parts:
            timeline.append({
                'segment_type': 'silent',
                'start': start,
                'end': end,
                'duration': end - start,
                'speaker': None
            })
        
        # Process and save speech segments
        segment_counter = {}  # Track segment count per speaker
        
        for segment in segments:
            # Handle mixed segments differently
            if segment.get('is_mixed', False) or segment.get('speaker') == 'MIXED':
                # Save mixed segment in first speaker's directory
                mixed_speakers = segment.get('speakers', speakers)
                speaker = mixed_speakers[0] if mixed_speakers else 'A'
                speaker_dir = base_dir / f"speaker_{speaker}" / "segments"
            else:
                speaker = segment['speaker']
                speaker_dir = base_dir / f"speaker_{speaker}" / "segments"
            
            speaker_dir.mkdir(parents=True, exist_ok=True)
            
            # Track segment count
            if speaker not in segment_counter:
                segment_counter[speaker] = 0
            segment_counter[speaker] += 1
            idx = segment_counter[speaker]
            
            # Extract segment audio
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Generate filenames
            segment_name = f"segment_{speaker}_{idx:03d}"
            audio_file = f"{segment_name}.wav"
            json_file = f"{segment_name}.json"
            
            # Save audio
            audio_path = speaker_dir / audio_file
            sf.write(audio_path, segment_audio, sr)
            
            # Translate text to English if needed
            original_text = segment['text']
            if detected_language != 'en' and target_language.lower() == 'english':
                english_text = self.transcription_service.translate_text_clean(original_text)
            else:
                english_text = original_text
            
            # Format text for Dia model - handle mixed segments
            if segment.get('is_mixed', False):
                # Mixed segment already has speaker tags
                dia_formatted_text = english_text
            else:
                dia_formatted_text = self.transcription_service.format_dia_text(
                    english_text, speaker, speakers
                )
            
            # Prepare metadata
            metadata = {
                'segment_id': segment_name,
                'speaker': speaker,
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'original_text': original_text,
                'english_text': english_text,
                'dia_formatted_text': dia_formatted_text,
                'detected_language': segment.get('detected_language', detected_language),
                'target_language': target_language,
                'audio_file': audio_file,
                'sample_rate': sr,
                'word_count': len(segment['words']),
                'assemblyai_transcript_id': segment.get('assemblyai_transcript_id', ''),
                'is_mixed': segment.get('is_mixed', False)
            }
            
            # Add mixed speaker info if applicable
            if segment.get('is_mixed', False):
                metadata['mixed_speakers'] = segment.get('speakers', [])
            
            # Save metadata
            json_path = speaker_dir / json_file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Add to timeline
            timeline.append({
                'segment_type': 'speech',
                'segment_id': segment_name,
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'speaker': speaker,
                'audio_file': audio_file,
                'has_translation': detected_language != 'en',
                'is_mixed': segment.get('is_mixed', False)
            })
        
        # Sort timeline by start time
        timeline.sort(key=lambda x: x['start'])
        
        # Save timeline
        timeline_path = base_dir / "metadata" / "timeline.json"
        with open(timeline_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timeline': timeline,
                'total_duration': total_duration,
                'speech_segments': len([t for t in timeline if t['segment_type'] == 'speech']),
                'silent_segments': len([t for t in timeline if t['segment_type'] == 'silent']),
                'speakers': speakers
            }, f, ensure_ascii=False, indent=2)
    
    def _save_silent_parts(self, silent_parts: List[Tuple[float, float]], 
                          audio: np.ndarray, sr: int, base_dir: Path):
        """Save silent parts with metadata"""
        silent_dir = base_dir / "silent_parts"
        silent_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (start, end) in enumerate(silent_parts):
            # Extract silent audio
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            if start_sample < len(audio) and end_sample <= len(audio):
                silent_audio = audio[start_sample:end_sample]
                
                # Save audio
                audio_file = f"silent_{idx+1:03d}.wav"
                audio_path = silent_dir / audio_file
                sf.write(audio_path, silent_audio, sr)
                
                # Save metadata
                metadata = {
                    'segment_id': f"silent_{idx+1:03d}",
                    'segment_type': 'silent',
                    'start': start,
                    'end': end,
                    'duration': end - start,
                    'audio_file': audio_file,
                    'sample_rate': sr
                }
                
                json_file = f"silent_{idx+1:03d}.json"
                json_path = silent_dir / json_file
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def select_optimal_references(self, segments: List[Dict], speakers: List[str]) -> Dict[str, Dict]:
        """Select best reference segment for each speaker (35-45 words preference)"""
        reference_segments = {}
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s['speaker'] == speaker and not s.get('is_mixed', False)]
            if not speaker_segments:
                # If no pure segments, create reference from mixed segments
                mixed_segments = [s for s in segments if s.get('is_mixed', False) and speaker in s.get('speakers', [])]
                if mixed_segments:
                    # Create composite reference from mixed segments
                    reference_segments[speaker] = self._create_composite_reference(mixed_segments, speaker)
                continue
            
            # Sort by how close they are to optimal word count (40 words center)
            optimal_center = (self.optimal_word_min + self.optimal_word_max) / 2
            speaker_segments.sort(key=lambda s: abs(s.get('word_count', 0) - optimal_center))
            
            # Select the segment closest to optimal word count
            best_segment = speaker_segments[0]
            
            # Ensure it's not too short
            if best_segment.get('word_count', 0) < 20 and len(speaker_segments) > 1:
                # Try to find a longer segment
                for seg in speaker_segments[1:]:
                    if seg.get('word_count', 0) >= 20:
                        best_segment = seg
                        break
            
            reference_segments[speaker] = best_segment
            logger.info(f"Selected reference for speaker {speaker}: {best_segment.get('word_count', 0)} words, {best_segment['duration']:.1f}s")
        
        return reference_segments
    
    def _create_composite_reference(self, mixed_segments: List[Dict], target_speaker: str) -> Dict:
        """Create composite reference from mixed segments for a specific speaker"""
        # Collect all words from mixed segments for the target speaker
        composite_words = []
        composite_text = ""
        
        for segment in mixed_segments:
            if target_speaker not in segment.get('speakers', []):
                continue
            
            # Extract speaker-specific text from mixed segment
            speaker_idx = segment['speakers'].index(target_speaker) + 1
            speaker_tag = f"[S{speaker_idx}]"
            
            # Parse mixed text to extract only target speaker's parts
            import re
            parts = re.split(r'(\[S\d+\])', segment['text'])
            current_speaker = None
            
            for i, part in enumerate(parts):
                if part == speaker_tag:
                    current_speaker = speaker_tag
                elif current_speaker == speaker_tag and part.strip():
                    if composite_text:
                        composite_text += " " + part.strip()
                    else:
                        composite_text = part.strip()
                    # Add approximate word count
                    composite_words.extend(part.strip().split())
        
        # Create composite reference with max 20s duration
        if composite_words:
            # Limit to approximately 40 words for optimal reference
            if len(composite_words) > 40:
                composite_text = ' '.join(composite_text.split()[:40])
                composite_words = composite_words[:40]
            
            # Create reference segment structure
            return {
                'speaker': target_speaker,
                'text': composite_text,
                'word_count': len(composite_words),
                'duration': min(20.0, len(composite_words) * 0.4),  # Approximate duration
                'is_composite': True,
                'source': 'mixed_segments'
            }
        
        return None