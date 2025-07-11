"""
Segment Manager Module - Optimized for Dia Voice Cloning

Handles smart segmentation with Dia model constraints:
- Optimal: 35-50 words, 10-15 seconds
- Mixed speaker support with [S1], [S2] tags
- Intelligent segment combination
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from .audio_utils import AudioUtils
from .transcription import TranscriptionService


class SegmentManager:
    """Smart segment manager optimized for Dia voice cloning"""
    
    def __init__(self, transcription_service: TranscriptionService):
        self.transcription_service = transcription_service
        self.audio_utils = AudioUtils()
        
        # Dia model optimal constraints
        self.optimal_duration = (10.0, 15.0)  # 10-15 seconds
        self.optimal_words = (35, 50)         # 35-50 words
        self.min_segment_duration = 8.0       # Minimum viable segment
        self.max_segment_duration = 20.0      # Maximum segment length
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create optimal segments for Dia voice cloning"""
        words = transcript_data.get('words', [])
        speakers = transcript_data.get('speakers', [])
        
        if len(speakers) == 1:
            return self._create_single_speaker_segments(words, speakers[0])
        else:
            return self._create_multi_speaker_segments(words, speakers)
    
    def _create_single_speaker_segments(self, words: List[Dict], speaker: str) -> List[Dict[str, Any]]:
        """Create optimal segments for single speaker"""
        segments = []
        current_words = []
        
        for word in words:
            current_words.append(word)
            
            # Check if we have enough content for optimal segment
            if self._is_segment_optimal(current_words):
                segment = self._create_pure_segment(current_words, speaker, [speaker])
                if segment:
                    segments.append(segment)
                    current_words = []
            
            # Force segment if too long
            elif self._is_segment_too_long(current_words):
                segment = self._create_pure_segment(current_words, speaker, [speaker])
                if segment:
                    segments.append(segment)
                    current_words = []
        
        # Handle remaining words
        if current_words and self._is_segment_viable(current_words):
            segment = self._create_pure_segment(current_words, speaker, [speaker])
            if segment:
                segments.append(segment)
        
        return segments
    
    def _create_multi_speaker_segments(self, words: List[Dict], speakers: List[str]) -> List[Dict[str, Any]]:
        """Create pure speaker segments first, then mixed only when necessary"""
        # Step 1: Create pure speaker segments first
        pure_segments = []
        
        # Group words by speaker with timing
        for speaker in speakers:
            speaker_words = [w for w in words if w.get('speaker') == speaker]
            if not speaker_words:
                continue
            
            # Create segments for this speaker with gap detection
            current_segment = []
            last_end_time = 0
            
            for word in speaker_words:
                start_time = word['start'] / 1000
                end_time = word['end'] / 1000
                
                # Check for gap > 2 seconds to split segments
                if current_segment and (start_time - last_end_time > 2.0):
                    # Finish current segment
                    segment = self._create_pure_segment(current_segment, speaker, speakers)
                    if segment:
                        pure_segments.append(segment)
                    current_segment = []
                
                current_segment.append(word)
                last_end_time = end_time
                
                # Force segment if optimal length reached
                if self._is_segment_optimal(current_segment):
                    segment = self._create_pure_segment(current_segment, speaker, speakers)
                    if segment:
                        pure_segments.append(segment)
                        current_segment = []
            
            # Add final segment for this speaker
            if current_segment:
                segment = self._create_pure_segment(current_segment, speaker, speakers)
                if segment:
                    pure_segments.append(segment)
        
        # Step 2: Sort segments by start time
        pure_segments.sort(key=lambda x: x['start'])
        
        # Step 3: Check for short segments and create mixed segments
        final_segments = []
        i = 0
        
        while i < len(pure_segments):
            current_segment = pure_segments[i]
            
            # If segment is too short, try to combine with next segment
            if current_segment['duration'] < self.min_segment_duration:
                next_segment = None
                if i + 1 < len(pure_segments):
                    next_segment = pure_segments[i + 1]
                    
                    # Check if segments are close enough (< 2s gap)
                    gap = next_segment['start'] - current_segment['end']
                    if gap < 2.0:
                        # Create mixed segment
                        mixed_segment = self._create_mixed_segment_from_pure(
                            current_segment, next_segment, speakers
                        )
                        if mixed_segment and self._is_segment_viable(mixed_segment.get('words', [])):
                            final_segments.append(mixed_segment)
                            i += 2  # Skip both segments
                            continue
                
                # If can't combine, add as is if viable
                if self._is_segment_viable(current_segment.get('words', [])):
                    final_segments.append(current_segment)
            else:
                # Segment is good as is
                final_segments.append(current_segment)
            
            i += 1
        
        return final_segments
    
    def _create_pure_segment(self, words: List[Dict], speaker: str, all_speakers: List[str]) -> Optional[Dict]:
        """Create pure speaker segment"""
        if not words:
            return None
        
        start_time = words[0]['start'] / 1000
        end_time = words[-1]['end'] / 1000
        duration = end_time - start_time
        text = ' '.join(word['text'] for word in words)
        word_count = len(words)
        
        # Calculate confidence
        confidences = [w.get('confidence', 0.5) for w in words]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Quality score
        quality_score = self._calculate_quality_score(duration, word_count, avg_confidence)
        
        return {
            'speaker': speaker,
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'text': text,
            'words': words,
            'word_count': word_count,
            'confidence': avg_confidence,
            'quality_score': quality_score,
            'is_mixed_speaker': False,
            'segment_speakers': [speaker],
            'all_speakers': all_speakers
        }
    
    def _create_mixed_segment_from_pure(self, segment1: Dict, segment2: Dict, all_speakers: List[str]) -> Optional[Dict]:
        """Create mixed segment from two pure segments"""
        # Combine words from both segments
        all_words = segment1['words'] + segment2['words']
        all_words.sort(key=lambda x: x['start'])  # Sort by timeline
        
        start_time = all_words[0]['start'] / 1000
        end_time = all_words[-1]['end'] / 1000
        duration = end_time - start_time
        
        # Don't create if too long
        if duration > self.max_segment_duration:
            return None
        
        # Combine text
        text = ' '.join(word['text'] for word in all_words)
        word_count = len(all_words)
        
        # Calculate combined confidence
        confidences = [w.get('confidence', 0.5) for w in all_words]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Use primary speaker (first in time)
        primary_speaker = segment1['speaker']
        segment_speakers = list(set([segment1['speaker'], segment2['speaker']]))
        
        # Quality score
        quality_score = self._calculate_quality_score(duration, word_count, avg_confidence)
        
        return {
            'speaker': primary_speaker,
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'text': text,
            'words': all_words,
            'word_count': word_count,
            'confidence': avg_confidence,
            'quality_score': quality_score,
            'is_mixed_speaker': True,
            'segment_speakers': segment_speakers,
            'all_speakers': all_speakers
        }
    
    def _calculate_quality_score(self, duration: float, word_count: int, confidence: float) -> float:
        """Calculate quality score for reference selection"""
        score = 0.0
        
        # Duration score (40% weight)
        if self.optimal_duration[0] <= duration <= self.optimal_duration[1]:
            duration_score = 1.0
        elif self.min_segment_duration <= duration < self.optimal_duration[0]:
            duration_score = 0.7
        elif self.optimal_duration[1] < duration <= self.max_segment_duration:
            duration_score = 0.6
        else:
            duration_score = 0.3
        score += duration_score * 0.4
        
        # Word count score (35% weight)
        if self.optimal_words[0] <= word_count <= self.optimal_words[1]:
            word_score = 1.0
        elif 25 <= word_count < self.optimal_words[0]:
            word_score = 0.7
        elif self.optimal_words[1] < word_count <= 70:
            word_score = 0.6
        else:
            word_score = 0.3
        score += word_score * 0.35
        
        # Confidence score (25% weight)
        score += confidence * 0.25
        
        return score
    
    def _is_segment_optimal(self, words: List[Dict]) -> bool:
        """Check if segment meets optimal criteria"""
        if not words:
            return False
        
        duration = (words[-1]['end'] - words[0]['start']) / 1000
        word_count = len(words)
        
        return (self.optimal_duration[0] <= duration <= self.optimal_duration[1] and 
                self.optimal_words[0] <= word_count <= self.optimal_words[1])
    
    def _is_segment_viable(self, words: List[Dict]) -> bool:
        """Check if segment is viable for cloning"""
        if not words:
            return False
        
        duration = (words[-1]['end'] - words[0]['start']) / 1000
        word_count = len(words)
        
        return (duration >= self.min_segment_duration and 
                word_count >= 20)  # Minimum 20 words
    
    def _is_segment_too_long(self, words: List[Dict]) -> bool:
        """Check if segment exceeds maximum length"""
        if not words:
            return False
        
        duration = (words[-1]['end'] - words[0]['start']) / 1000
        return duration > self.max_segment_duration
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int, 
                            base_dir: Path, speakers: List[str], target_language: str = "English",
                            transcript_metadata: Optional[Dict] = None):
        """Save optimized segments for Dia voice cloning"""
        # Create directory structure
        for speaker in speakers:
            speaker_dir = base_dir / f"speaker_{speaker}"
            (speaker_dir / "segments").mkdir(parents=True, exist_ok=True)
            (speaker_dir / "reference").mkdir(parents=True, exist_ok=True)
        
        # Save segments
        for i, segment in enumerate(segments):
            primary_speaker = segment['speaker']
            speaker_dir = base_dir / f"speaker_{primary_speaker}"
            
            # Extract audio with exact timing
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Audio processing
            segment_audio = self.audio_utils.normalize_audio(segment_audio)
            
            # Generate filenames
            seg_id = f"{primary_speaker}_seg_{i+1:03d}"
            audio_file = speaker_dir / "segments" / f"{seg_id}.wav"
            json_file = speaker_dir / "segments" / f"{seg_id}.json"
            
            # Save audio with exact length preservation
            sf.write(audio_file, segment_audio, sr)
            
            # Translate text for dubbing
            english_text = self.transcription_service.translate_text_clean(segment['text'])
            
            # Format for Dia model with mixed speaker support
            dia_text = self._format_dia_text_optimal(
                english_text, segment, speakers, i
            )
            
            # Save segment metadata
            segment_data = {
                'segment_id': seg_id,
                'speaker': primary_speaker,
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'original_text': segment['text'],
                'english_text': english_text,
                'dia_text': dia_text,
                'word_count': segment['word_count'],
                'confidence': segment['confidence'],
                'quality_score': segment['quality_score'],
                'is_mixed_speaker': segment['is_mixed_speaker'],
                'segment_speakers': segment['segment_speakers'],
                'audio_file': audio_file.name,
                'sample_rate': sr,
                'translation_used': True
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(segment_data, f, ensure_ascii=False, indent=2)
    
    def _format_dia_text_optimal(self, english_text: str, segment: Dict, 
                               all_speakers: List[str], segment_index: int) -> str:
        """Format text for Dia model with optimal mixed speaker support (up to 10 speakers)"""
        if segment['is_mixed_speaker']:
            # Mixed speaker scenario - use specific speaker tags based on actual speakers
            segment_speakers = segment['segment_speakers']
            
            # Create speaker tags based on actual speaker indices
            speaker_tags = []
            for speaker in segment_speakers:
                try:
                    speaker_idx = all_speakers.index(speaker) + 1
                    speaker_tags.append(f"[S{speaker_idx}]")
                except ValueError:
                    # Fallback if speaker not found
                    speaker_tags.append("[S1]")
            
            # Format text with multiple speaker tags
            if len(speaker_tags) == 2:
                return f"{speaker_tags[0]} {english_text} {speaker_tags[1]}"
            else:
                # For more than 2 speakers in mixed segment
                primary_tag = speaker_tags[0]
                secondary_tag = speaker_tags[1] if len(speaker_tags) > 1 else "[S2]"
                return f"{primary_tag} {english_text} {secondary_tag}"
        else:
            # Pure speaker - use appropriate tag (S1 to S10)
            try:
                speaker_idx = all_speakers.index(segment['speaker']) + 1
                return f"[S{speaker_idx}] {english_text}"
            except ValueError:
                # Fallback if speaker not found
                return f"[S1] {english_text}"
    
    def select_optimal_references(self, segments: List[Dict], speakers: List[str]) -> Dict[str, Dict]:
        """Select optimal reference segments for each speaker"""
        references = {}
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s['speaker'] == speaker and not s['is_mixed_speaker']]
            
            if not speaker_segments:
                continue
            
            # Select highest quality segment
            best_segment = max(speaker_segments, key=lambda s: s['quality_score'])
            
            # Add English text
            english_text = self.transcription_service.translate_text_clean(best_segment['text'])
            reference_segment = best_segment.copy()
            reference_segment['english_text'] = english_text
            reference_segment['original_text'] = best_segment['text']
            
            references[speaker] = reference_segment
        
        return references
