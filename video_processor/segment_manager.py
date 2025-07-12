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
        self.optimal_segment_min = 11.0  # Optimal minimum segment duration
        self.optimal_segment_max = 17.0  # Optimal maximum segment duration
        self.max_segment_duration = 20.0  # Hard maximum
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create optimal segments from transcript data with proper ordering"""
        words = transcript_data.get('words', [])
        speakers = transcript_data.get('speakers', ['A'])
        
        if not words:
            logger.warning("No words found in transcript")
            return []
        
        # Store raw AssemblyAI response for later use
        raw_response = transcript_data.get('raw_assemblyai_response', {})
        
        # Create speaker segments
        speaker_segments = self._create_speaker_segments(words, speakers)
        
        # Optimize segment durations (11-17s preference)
        optimized_segments = self._optimize_segment_durations(speaker_segments)
        
        # Add raw response to each segment
        for segment in optimized_segments:
            segment['assemblyai_transcript_id'] = raw_response.get('id', '')
            segment['detected_language'] = raw_response.get('language_code', 'en')
        
        return optimized_segments
    
    def _create_speaker_segments(self, words: List[Dict], speakers: List[str]) -> List[Dict[str, Any]]:
        """Create segments based on speaker changes and natural breaks"""
        if not words:
            return []
        
        segments = []
        current_segment = {
            'speaker': words[0].get('speaker', 'A'),
            'start': words[0]['start'] / 1000.0,  # Convert to seconds
            'words': [words[0]],
            'text': words[0]['text']
        }
        
        for word in words[1:]:
            word_speaker = word.get('speaker', 'A')
            word_start = word['start'] / 1000.0
            word_end = word['end'] / 1000.0
            
            # Check if we need to create a new segment
            should_split = False
            
            # Calculate current segment duration if we add this word
            potential_duration = word_end - current_segment['start']
            
            # Split on speaker change ONLY if current segment is at least 5 seconds
            if word_speaker != current_segment['speaker']:
                current_duration = current_segment['words'][-1]['end'] / 1000.0 - current_segment['start']
                if current_duration >= 5.0:  # Only split on speaker change if we have at least 5 seconds
                    should_split = True
            
            # Split if current segment is getting too long (beyond max)
            if potential_duration >= self.max_segment_duration:
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
                    'text': word['text']
                }
            else:
                # Add word to current segment
                current_segment['words'].append(word)
                current_segment['text'] += ' ' + word['text']
        
        # Add final segment
        if current_segment['words']:
            current_segment['end'] = current_segment['words'][-1]['end'] / 1000.0
            current_segment['duration'] = current_segment['end'] - current_segment['start']
            segments.append(current_segment)
        
        return segments
    
    def _optimize_segment_durations(self, segments: List[Dict]) -> List[Dict]:
        """Optimize segments to be 11-17 seconds when possible"""
        optimized = []
        i = 0
        
        while i < len(segments):
            current = segments[i].copy()
            
            # If segment is already in optimal range, keep it
            if self.optimal_segment_min <= current['duration'] <= self.optimal_segment_max:
                optimized.append(current)
                i += 1
                continue
            
            # If segment is too short, aggressively merge with next segments
            if current['duration'] < self.optimal_segment_min:
                merged = current.copy()
                j = i + 1
                
                # Keep merging until we reach optimal duration or can't merge anymore
                while j < len(segments):
                    next_seg = segments[j]
                    
                    # Check potential duration after merge
                    potential_duration = next_seg['end'] - merged['start']
                    
                    # If same speaker and won't exceed max, merge
                    if next_seg['speaker'] == merged['speaker'] and potential_duration <= self.max_segment_duration:
                        # Merge segments
                        merged['words'].extend(next_seg['words'])
                        merged['text'] += ' ' + next_seg['text']
                        merged['end'] = next_seg['end']
                        merged['duration'] = merged['end'] - merged['start']
                        j += 1
                        
                        # If we're in optimal range now, stop merging
                        if merged['duration'] >= self.optimal_segment_min:
                            break
                    else:
                        # Can't merge this segment, but check if we can create mixed segment
                        if next_seg['speaker'] != merged['speaker'] and len(segments[i:j+1]) > 1:
                            # Check if creating a mixed segment would help reach optimal duration
                            mixed_duration = next_seg['end'] - merged['start']
                            if mixed_duration >= self.optimal_segment_min and mixed_duration <= self.optimal_segment_max:
                                # Create mixed segment
                                mixed_segment = self._create_mixed_segment(segments[i:j+1])
                                if mixed_segment:
                                    optimized.append(mixed_segment)
                                    i = j + 1
                                    break
                        break
                
                # If we couldn't reach optimal duration but have something, keep it
                if j > i + 1 or merged['duration'] >= 5.0:  # At least 5 seconds
                    optimized.append(merged)
                    i = j
                else:
                    # Too short and can't merge, keep as is
                    optimized.append(current)
                    i += 1
            
            # If segment is too long, split it into optimal chunks
            elif current['duration'] > self.optimal_segment_max:
                split_segments = self._split_long_segment(current)
                optimized.extend(split_segments)
                i += 1
            
            else:
                optimized.append(current)
                i += 1
        
        return optimized
    
    def _split_long_segment(self, segment: Dict) -> List[Dict]:
        """Split a long segment into optimal chunks (11-17s preferred)"""
        words = segment['words']
        if not words:
            return [segment]
        
        # Calculate optimal split points
        total_duration = segment['duration']
        num_splits = int(total_duration / ((self.optimal_segment_min + self.optimal_segment_max) / 2))
        if num_splits < 1:
            num_splits = 1
        
        # Calculate target duration per split
        target_duration = total_duration / num_splits
        # Ensure target is within optimal range
        target_duration = max(self.optimal_segment_min, min(target_duration, self.optimal_segment_max))
        
        splits = []
        current_split = {
            'speaker': segment['speaker'],
            'start': words[0]['start'] / 1000.0,
            'words': [],
            'text': ''
        }
        
        for word in words:
            word_end = word['end'] / 1000.0
            current_duration = word_end - current_split['start']
            
            # Check if we should create a new split
            should_split = False
            
            # Split if we've reached target duration and have at least some words
            if current_duration >= target_duration and current_split['words']:
                # Look ahead to see if next few words would keep us under max
                remaining_words = len([w for w in words if w['start'] > word['start']])
                if remaining_words > 5:  # Still have words left for another segment
                    should_split = True
            
            # Force split if we're approaching max duration
            if current_duration >= self.optimal_segment_max - 1.0 and current_split['words']:
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
                    'text': word['text']
                }
            else:
                # Add word to current split
                current_split['words'].append(word)
                if current_split['text']:
                    current_split['text'] += ' ' + word['text']
                else:
                    current_split['text'] = word['text']
        
        # Add final split
        if current_split['words']:
            current_split['end'] = current_split['words'][-1]['end'] / 1000.0
            current_split['duration'] = current_split['end'] - current_split['start']
            splits.append(current_split)
        
        return splits if splits else [segment]
    
    def _create_mixed_segment(self, segments: List[Dict]) -> Optional[Dict]:
        """Create a mixed segment from multiple speakers for optimal duration"""
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
        """Select best reference segment for each speaker (11-17s preference)"""
        reference_segments = {}
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s['speaker'] == speaker]
            if not speaker_segments:
                continue
            
            # Sort by how close they are to optimal duration (14s center)
            optimal_center = (self.optimal_segment_min + self.optimal_segment_max) / 2
            speaker_segments.sort(key=lambda s: abs(s['duration'] - optimal_center))
            
            # Select the segment closest to optimal duration
            best_segment = speaker_segments[0]
            
            # Ensure it's not too short
            if best_segment['duration'] < 5.0 and len(speaker_segments) > 1:
                # Try to find a longer segment
                for seg in speaker_segments[1:]:
                    if seg['duration'] >= 5.0:
                        best_segment = seg
                        break
            
            reference_segments[speaker] = best_segment
            logger.info(f"Selected reference for speaker {speaker}: {best_segment['duration']:.1f}s")
        
        return reference_segments