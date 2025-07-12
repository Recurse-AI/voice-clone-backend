"""
Segment Manager Module - Optimized for Dia Voice Cloning

Handles smart segmentation with Dia model constraints:
- Strict: 3-19 seconds duration (prefer 15-19s)
- Mixed speaker support with [S1], [S2] tags
- Clean segment creation
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Optional
from .audio_utils import AudioUtils
from .transcription import TranscriptionService


class SegmentManager:
    """Smart segment manager optimized for Dia voice cloning"""
    
    def __init__(self, transcription_service: TranscriptionService):
        self.transcription_service = transcription_service
        self.audio_utils = AudioUtils()
        
        # Dia model constraints
        self.min_duration = 3.0
        self.preferred_min_duration = 15.0
        self.max_duration = 19.0
        self.combine_threshold = 14.0
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create optimal segments for Dia voice cloning"""
        words = transcript_data.get('words', [])
        speakers = transcript_data.get('speakers', [])
        
        if not words:
            return []
        
        if len(speakers) == 1:
            segments = self._create_single_speaker_segments(words, speakers[0])
        else:
            segments = self._create_multi_speaker_segments(words, speakers)
        
        # Post-process to combine short segments
        return self._combine_short_segments(segments)
    
    def _create_single_speaker_segments(self, words: List[Dict], speaker: str) -> List[Dict[str, Any]]:
        """Create segments for single speaker"""
        segments = []
        current_words = []
        
        for word in words:
            current_words.append(word)
            duration = self._calculate_duration(current_words)
            
            # Create segment if we reach preferred minimum with good break point
            if duration >= self.preferred_min_duration and self._is_good_break_point(word):
                segment = self._create_segment(current_words, speaker, [speaker])
                if segment:
                    segments.append(segment)
                    current_words = []
                    continue
            
            # Force segment if max duration reached
            if duration >= self.max_duration:
                segment = self._create_segment(current_words, speaker, [speaker])
                if segment:
                    segments.append(segment)
                    current_words = []
        
        # Handle remaining words
        if current_words:
            segment = self._create_segment(current_words, speaker, [speaker])
            if segment:
                segments.append(segment)
        
        return segments
    
    def _create_multi_speaker_segments(self, words: List[Dict], speakers: List[str]) -> List[Dict[str, Any]]:
        """Create segments for multiple speakers"""
        segments = []
        current_words = []
        current_speaker = None
        
        for word in words:
            word_speaker = word.get('speaker', 'A')
            
            # Start new segment or continue current
            if current_speaker is None or current_speaker == word_speaker:
                if current_speaker is None:
                    current_speaker = word_speaker
                current_words.append(word)
                
                duration = self._calculate_duration(current_words)
                
                # Create segment if we reach preferred minimum with good break point
                if duration >= self.preferred_min_duration and self._is_good_break_point(word):
                    segment = self._create_segment(current_words, current_speaker, speakers)
                    if segment:
                        segments.append(segment)
                        current_words = []
                        current_speaker = None
                        continue
                
                # Force segment if max duration reached
                if duration >= self.max_duration:
                    segment = self._create_segment(current_words, current_speaker, speakers)
                    if segment:
                        segments.append(segment)
                        current_words = []
                        current_speaker = None
            else:
                # Speaker change - finalize current segment
                if current_words:
                    segment = self._create_segment(current_words, current_speaker, speakers)
                    if segment:
                        segments.append(segment)
                
                # Start new segment
                current_speaker = word_speaker
                current_words = [word]
        
        # Handle remaining words
        if current_words:
            segment = self._create_segment(current_words, current_speaker, speakers)
            if segment:
                segments.append(segment)
        
        return segments
    
    def _combine_short_segments(self, segments: List[Dict]) -> List[Dict[str, Any]]:
        """Combine short segments to reach optimal duration"""
        if len(segments) <= 1:
            return segments
        
        combined_segments = []
        i = 0
        
        while i < len(segments):
            current_segment = segments[i]
            
            # Try to combine with next segment if current is short
            if (current_segment['duration'] < self.combine_threshold and 
                i + 1 < len(segments) and 
                segments[i + 1]['speaker'] == current_segment['speaker']):
                
                next_segment = segments[i + 1]
                combined_duration = current_segment['duration'] + next_segment['duration']
                
                # Combine if within max duration
                if combined_duration <= self.max_duration:
                    combined_words = current_segment['words'] + next_segment['words']
                    combined_segment = self._create_segment(
                        combined_words, current_segment['speaker'], current_segment['all_speakers']
                    )
                    if combined_segment:
                        combined_segments.append(combined_segment)
                        i += 2
                        continue
            
            combined_segments.append(current_segment)
            i += 1
        
        return combined_segments
    
    def _is_good_break_point(self, word: Dict) -> bool:
        """Check if a word is a good break point for segmentation"""
        text = word.get('text', '').strip()
        if not text:
            return False
        
        # Sentence endings and punctuation
        return text.endswith(('.', '!', '?', '।', ';', ':', ',', '-'))
    
    def _create_segment(self, words: List[Dict], speaker: str, all_speakers: List[str]) -> Optional[Dict]:
        """Create segment with duration constraints"""
        if not words:
            return None
        
        start_time = words[0].get('start', 0) / 1000
        end_time = words[-1].get('end', 0) / 1000
        duration = end_time - start_time
        
        # Duration check
        if duration < self.min_duration or duration > self.max_duration:
            return None
        
        text = ' '.join(word.get('text', '') for word in words)
        confidences = [w.get('confidence', 0.5) for w in words]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return {
            'speaker': speaker,
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'text': text,
            'words': words,
            'word_count': len(words),
            'confidence': avg_confidence,
            'all_speakers': all_speakers
        }
    
    def _calculate_duration(self, words: List[Dict]) -> float:
        """Calculate duration of word sequence"""
        if not words:
            return 0.0
        
        start_time = words[0].get('start', 0)
        end_time = words[-1].get('end', 0)
        return (end_time - start_time) / 1000
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int, 
                            base_dir: Path, speakers: List[str], target_language: str = "English", 
                            detected_language: str = "en"):
        """Save segments with duration compliance"""
        reference_files = {}
        
        for i, segment in enumerate(segments):
            speaker = segment['speaker']
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            duration = end_time - start_time
            
            # Extract and save audio
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Save in speaker-wise directory structure
            speaker_dir = base_dir / f"speaker_{speaker}"
            segments_dir = speaker_dir / "segments"
            segments_dir.mkdir(parents=True, exist_ok=True)
            
            # Save segment files
            segment_filename = f"segment_{i:03d}_{speaker}.wav"
            segment_path = segments_dir / segment_filename
            sf.write(str(segment_path), segment_audio, sr)
            
            # Process text for voice cloning
            if detected_language.lower() not in ["en", "english"]:
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
                "start": start_time,
                "end": end_time,
                "duration": duration,
                "word_count": segment['word_count'],
                "confidence": segment['confidence'],
                "segment_file": segment_filename,
                "audio_path": str(segment_path)
            }
            
            import json
            metadata_filename = f"segment_{i:03d}_{speaker}.json"
            metadata_path = segments_dir / metadata_filename
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Track for reference selection
            if speaker not in reference_files:
                reference_files[speaker] = []
            reference_files[speaker].append({
                "path": str(segment_path),
                "duration": duration,
                "confidence": segment['confidence'],
                "segment_index": i,
                "metadata": metadata
            })
        
        # Select best reference for each speaker
        selected_references = {}
        for speaker, files in reference_files.items():
            # Sort by duration (prefer 15-19 seconds) then by confidence
            files.sort(key=lambda x: (
                abs(x['duration'] - 17.0),  # Distance from 17 seconds
                -x['confidence']            # Higher confidence first
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
            
            # Sort by optimal criteria
            best_segment = max(speaker_segs, key=lambda seg: (
                -abs(seg.get('duration', 5) - 17.0),  # Closer to 17 seconds
                seg.get('confidence', 0.5),           # Higher confidence
                seg.get('word_count', 0)              # More words
            ))
            
            # Format reference text
            original_text = best_segment.get('text', f'Reference for speaker {speaker}')
            english_text = original_text
            
            if hasattr(self.transcription_service, 'translate_text_clean'):
                try:
                    english_text = self.transcription_service.translate_text_clean(original_text)
                except:
                    pass
            
            formatted_reference_text = self.transcription_service.format_dia_text(english_text, speaker, speakers)
            
            reference_segments[speaker] = {
                'segment': best_segment,
                'start': best_segment.get('start', 0),
                'end': best_segment.get('end', 5),
                'duration': best_segment.get('duration', 5),
                'confidence': best_segment.get('confidence', 0.5),
                'word_count': best_segment.get('word_count', 0),
                'text': formatted_reference_text,
                'english_text': english_text,
                'original_text': original_text
            }
        
        return reference_segments
