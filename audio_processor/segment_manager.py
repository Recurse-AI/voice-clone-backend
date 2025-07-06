"""
Segment Manager Module

Handles creation and management of audio segments for voice cloning.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from .transcription import TranscriptionService
from .audio_utils import AudioUtils


class SegmentManager:
    """Manages audio segments for voice cloning processing"""
    
    def __init__(self, transcription_service: TranscriptionService):
        self.transcription_service = transcription_service
        self.audio_utils = AudioUtils()
    
    def create_speaker_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create speaker segments from transcript data"""
        words = transcript_data.get('words', [])
        speakers = transcript_data.get('speakers', [])
        
        segments = []
        
        for speaker in speakers:
            # Group words by speaker
            speaker_words = [w for w in words if w.get('speaker') == speaker]
            
            if not speaker_words:
                continue
            
            # Create segments for this speaker
            current_segment = []
            last_end_time = 0
            
            for word in speaker_words:
                start_time = word['start'] / 1000  # Convert to seconds
                end_time = word['end'] / 1000
                
                # Check if this word should start a new segment
                if current_segment and (start_time - last_end_time > 2.0):  # 2 second gap
                    # Finish current segment
                    segment = self.create_segment_from_words(current_segment, speaker)
                    if segment:
                        segments.append(segment)
                    current_segment = []
                
                current_segment.append(word)
                last_end_time = end_time
            
            # Add final segment
            if current_segment:
                segment = self.create_segment_from_words(current_segment, speaker)
                if segment:
                    segments.append(segment)
        
        return segments
    
    def create_segment_from_words(self, words: List[Dict], speaker: str) -> Optional[Dict]:
        """Create a segment from a list of words"""
        if not words:
            return None
        
        start_time = words[0]['start'] / 1000
        end_time = words[-1]['end'] / 1000
        text = ' '.join(word['text'] for word in words)
        
        # Calculate average confidence
        confidences = [w.get('confidence', 0.5) for w in words]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        segment = {
            'speaker': speaker,
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time,
            'text': text,
            'words': words,
            'confidence': avg_confidence,
            'word_count': len(words)
        }
        
        return segment if self.validate_segment_quality(segment) else None
    
    def validate_segment_quality(self, segment: Dict) -> bool:
        """Validate segment quality for voice cloning"""
        # Duration check: 5-17 seconds
        if segment['duration'] < 5.0:
            return False
        
        if segment['duration'] > 17.0:
            return False
        
        # Word count check: 35-50 words
        if segment['word_count'] < 35:
            return False
        
        if segment['word_count'] > 50:
            return False
        
        # Confidence threshold
        if segment['confidence'] < 0.6:
            return False
        
        return True
    
    def save_segments(self, segments: List[Dict], audio: np.ndarray, sr: int, 
                     base_dir: Path, speakers: List[str], target_language: str = "English"):
        """Save audio segments to files"""
        total_segments = len(segments)
        
        # Create directory structure
        for speaker in speakers:
            speaker_dir = base_dir / f"speaker_{speaker}"
            (speaker_dir / "segments").mkdir(parents=True, exist_ok=True)
            (speaker_dir / "reference").mkdir(parents=True, exist_ok=True)
        
        # Save segments
        for i, segment in enumerate(segments):
            speaker = segment['speaker']
            speaker_dir = base_dir / f"speaker_{speaker}"
            
            # Extract audio segment
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Apply fade in/out
            segment_audio = self.audio_utils.fade_in_out(segment_audio, 0.05, sr)
            
            # Normalize audio
            segment_audio = self.audio_utils.normalize_audio(segment_audio)
            
            # Generate filenames
            seg_id = f"{speaker}_seg_{i+1:03d}"
            audio_file = speaker_dir / "segments" / f"{seg_id}.wav"
            json_file = speaker_dir / "segments" / f"{seg_id}.json"
            
            # Save audio file
            sf.write(audio_file, segment_audio, sr)
            
            # Translate text if needed
            english_text = segment['text']
            if target_language.lower() != "english":
                english_text = self.transcription_service.translate_text_clean(segment['text'])
            
            # Format for Dia model
            is_last_segment = (i == total_segments - 1)
            dia_text = self.transcription_service.format_dia_text(
                english_text, speaker, speakers, i, is_last_segment
            )
            
            # Save segment metadata
            segment_data = {
                'segment_id': seg_id,
                'speaker': speaker,
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'original_text': segment['text'],
                'english_text': english_text,
                'dia_text': dia_text,
                'confidence': segment['confidence'],
                'word_count': segment['word_count'],
                'audio_file': audio_file.name,
                'sample_rate': sr,
                'is_last_segment': is_last_segment
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(segment_data, f, ensure_ascii=False, indent=2)
    
    def select_reference_segments(self, segments: List[Dict], speakers: List[str]) -> Dict[str, Dict]:
        """Select reference segments for each speaker"""
        reference_segments = {}
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s['speaker'] == speaker]
            
            if not speaker_segments:
                continue
            
            # Select best segment (highest confidence, good duration)
            best_segment = max(speaker_segments, key=lambda s: (
                s['confidence'],
                min(s['duration'], 17.0),  # Prefer segments up to 17 seconds
                s['word_count']
            ))
            
            reference_segments[speaker] = best_segment
        
        return reference_segments
