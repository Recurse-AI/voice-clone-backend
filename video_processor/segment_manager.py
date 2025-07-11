"""
Segment Manager Module - Optimized for Dia Voice Cloning

Handles smart segmentation with Dia model constraints:
- Strict: 5-19 seconds duration
- Mixed speaker support with [S1], [S2] tags
- Clean segment creation without extra logs
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
        
        # Dia model constraints - 4-19 seconds (try to keep above 5s)
        self.absolute_min_duration = 4.0      # Absolute minimum 4 seconds
        self.preferred_min_duration = 5.0     # Preferred minimum 5 seconds
        self.max_duration = 19.0              # Maximum 19 seconds
        self.optimal_duration = (8.0, 15.0)   # Optimal range
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create optimal segments for Dia voice cloning"""
        words = transcript_data.get('words', [])
        speakers = transcript_data.get('speakers', [])
        
        if not words:
            return []
        
        if len(speakers) == 1:
            return self._create_single_speaker_segments(words, speakers[0])
        else:
            return self._create_multi_speaker_segments(words, speakers)
    
    def _create_single_speaker_segments(self, words: List[Dict], speaker: str) -> List[Dict[str, Any]]:
        """Create segments for single speaker with 4-19 seconds constraint (prefer 5s+)"""
        segments = []
        current_words = []
        
        for word in words:
            current_words.append(word)
            
            # Check current segment duration
            duration = self._calculate_duration(current_words)
            
            # If duration is optimal, create segment
            if self.optimal_duration[0] <= duration <= self.optimal_duration[1]:
                # Look ahead to see if we should include more words
                if not self._would_exceed_max_duration(current_words, words[words.index(word) + 1:]):
                    continue
                
                segment = self._create_segment(current_words, speaker, [speaker])
                if segment:
                    segments.append(segment)
                    current_words = []
            
            # Try to reach preferred minimum (5s), but don't wait too long
            elif duration >= self.preferred_min_duration:
                # Look ahead to see if we should create segment now
                if not self._would_exceed_max_duration(current_words, words[words.index(word) + 1:]):
                    continue
                
                segment = self._create_segment(current_words, speaker, [speaker])
                if segment:
                    segments.append(segment)
                    current_words = []
            
            # Force segment if max duration reached
            elif duration >= self.max_duration:
                segment = self._create_segment(current_words, speaker, [speaker])
                if segment:
                    segments.append(segment)
                    current_words = []
        
        # Handle remaining words - accept 4s+ but prefer 5s+
        if current_words:
            duration = self._calculate_duration(current_words)
            if duration >= self.absolute_min_duration:
                segment = self._create_segment(current_words, speaker, [speaker])
                if segment:
                    segments.append(segment)
        
        return segments
    
    def _create_multi_speaker_segments(self, words: List[Dict], speakers: List[str]) -> List[Dict[str, Any]]:
        """Create segments for multiple speakers with 4-19 seconds constraint (prefer 5s+)"""
        segments = []
        current_words = []
        current_speaker = None
        
        for word in words:
            word_speaker = word.get('speaker', 'A')
            
            # Start new segment or continue current
            if current_speaker is None:
                current_speaker = word_speaker
                current_words = [word]
            elif current_speaker == word_speaker:
                current_words.append(word)
            else:
                # Speaker change - finalize current segment if viable
                duration = self._calculate_duration(current_words)
                if duration >= self.absolute_min_duration:
                    segment = self._create_segment(current_words, current_speaker, speakers)
                    if segment:
                        segments.append(segment)
                
                # Start new segment
                current_speaker = word_speaker
                current_words = [word]
            
            # Check if current segment should be finalized
            duration = self._calculate_duration(current_words)
            if duration >= self.max_duration:
                segment = self._create_segment(current_words, current_speaker, speakers)
                if segment:
                    segments.append(segment)
                    current_words = []
                    current_speaker = None
        
        # Handle remaining words - accept 4s+ but prefer 5s+
        if current_words:
            duration = self._calculate_duration(current_words)
            if duration >= self.absolute_min_duration:
                segment = self._create_segment(current_words, current_speaker, speakers)
                if segment:
                    segments.append(segment)
        
        return segments
    
    def _create_segment(self, words: List[Dict], speaker: str, all_speakers: List[str]) -> Optional[Dict]:
        """Create segment with duration constraints (4-19s, prefer 5s+)"""
        if not words:
            return None
        
        # Safe access to start and end with fallbacks
        start_time = words[0].get('start', 0) / 1000
        end_time = words[-1].get('end', 0) / 1000
        duration = end_time - start_time
        
        # Duration check - accept 4s+ but reject if over 19s
        if duration < self.absolute_min_duration or duration > self.max_duration:
            return None
        
        text = ' '.join(word.get('text', '') for word in words)
        word_count = len(words)
        
        # Calculate confidence
        confidences = [w.get('confidence', 0.5) for w in words]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return {
            'speaker': speaker,
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'text': text,
            'words': words,
            'word_count': word_count,
            'confidence': avg_confidence,
            'all_speakers': all_speakers
        }
    
    def _calculate_duration(self, words: List[Dict]) -> float:
        """Calculate duration of word sequence"""
        if not words:
            return 0.0
        
        # Safe access to start and end with fallbacks
        start_time = words[0].get('start', 0)
        end_time = words[-1].get('end', 0)
        return (end_time - start_time) / 1000
    
    def _would_exceed_max_duration(self, current_words: List[Dict], remaining_words: List[Dict]) -> bool:
        """Check if adding next word would exceed max duration"""
        if not remaining_words:
            return False
        
        test_words = current_words + [remaining_words[0]]
        test_duration = self._calculate_duration(test_words)
        return test_duration > self.max_duration
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int, 
                            base_dir: Path, speakers: List[str], target_language: str = "English"):
        """Save segments with duration compliance (4-19s, prefer 5s+)"""
        reference_files = {}
        
        for i, segment in enumerate(segments):
            speaker = segment['speaker']
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            
            # Double-check duration constraint
            duration = end_time - start_time
            if duration < self.absolute_min_duration or duration > self.max_duration:
                continue  # Skip segments outside 4-19 seconds
            
            # Extract and save audio
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Save segment files
            segment_dir = base_dir / f"segment_{i:03d}"
            segment_dir.mkdir(exist_ok=True)
            
            # Save original audio
            original_path = segment_dir / "original.wav"
            sf.write(str(original_path), segment_audio, sr)
            
            # Process text for voice cloning
            if target_language.lower() != "english":
                english_text = self.transcription_service.translate_text_clean(text)
            else:
                english_text = text
            
            # Format for Dia model
            dia_text = self.transcription_service.format_dia_text(english_text, speaker, speakers)
            
            # Save metadata
            metadata = {
                "speaker": speaker,
                "original_text": text,
                "english_text": english_text,
                "dia_text": dia_text,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "word_count": segment['word_count'],
                "confidence": segment['confidence']
            }
            
            import json
            with open(segment_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Track for reference selection
            if speaker not in reference_files:
                reference_files[speaker] = []
            reference_files[speaker].append({
                "path": str(original_path),
                "duration": duration,
                "confidence": segment['confidence'],
                "segment_index": i
            })
        
        # Select best reference for each speaker
        selected_references = {}
        for speaker, files in reference_files.items():
            # Sort by duration (prefer 8-15 seconds) then by confidence
            files.sort(key=lambda x: (
                abs(x['duration'] - 11.5),  # Distance from 11.5 seconds (middle of optimal range)
                -x['confidence']  # Higher confidence first
            ))
            selected_references[speaker] = files[0] if files else None
        
        return selected_references

    def select_optimal_references(self, segments: List[Dict], speakers: List[str]) -> Dict[str, Dict]:
        """Select optimal reference segments for each speaker"""
        reference_segments = {}
        
        # Group segments by speaker
        speaker_segments = {}
        for segment in segments:
            speaker = segment.get('speaker', 'A')
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(segment)
        
        # Select best reference for each speaker
        for speaker in speakers:
            if speaker not in speaker_segments:
                continue
                
            speaker_segs = speaker_segments[speaker]
            
            # Sort by optimal criteria with safe access:
            # 1. Duration preference (8-15 seconds is optimal)
            # 2. Higher confidence
            # 3. Word count (longer is better for reference)
            best_segment = max(speaker_segs, key=lambda seg: (
                -abs(seg.get('duration', 5) - 11.5),  # Closer to 11.5 seconds (middle of optimal range)
                seg.get('confidence', 0.5),           # Higher confidence
                seg.get('word_count', 0)              # More words
            ))
            
            reference_segments[speaker] = {
                'segment': best_segment,
                'start': best_segment.get('start', 0),
                'end': best_segment.get('end', 5),
                'duration': best_segment.get('duration', 5),
                'confidence': best_segment.get('confidence', 0.5),
                'word_count': best_segment.get('word_count', 0),
                'text': best_segment.get('text', f'Reference for speaker {speaker}')
            }
        
        return reference_segments
