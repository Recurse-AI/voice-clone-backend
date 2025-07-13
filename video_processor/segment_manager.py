"""
Segment Manager Module

Simple segment creation with word-based chunking similar to Dia model approach.
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SegmentManager:
    """Simple segment manager with word-based chunking"""
    
    def __init__(self, transcription_service):
        self.transcription_service = transcription_service
        # Simple word-based chunking - similar to Dia model approach
        self.words_per_chunk = 35  # Optimal for voice cloning (similar to dia approach)
        self.max_duration = 18.0   # Max duration in seconds
        self.min_words = 8         # Minimum words per segment
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create simple word-based segments"""
        words = transcript_data.get('words', [])
        if not words:
            return []
        
        # Simple chunking by word count - no complex logic
        segments = []
        current_chunk = []
        current_speaker = None
        
        for word in words:
            speaker = word.get('speaker', 'A')
            
            # If speaker changes or chunk is full, finalize current chunk
            if (speaker != current_speaker and current_chunk) or len(current_chunk) >= self.words_per_chunk:
                if current_chunk:
                    segment = self._create_simple_segment(current_chunk, current_speaker or 'A')
                    if segment:
                        segments.append(segment)
                current_chunk = [word]
                current_speaker = speaker
            else:
                current_chunk.append(word)
                current_speaker = speaker
        
        # Add final chunk
        if current_chunk:
            segment = self._create_simple_segment(current_chunk, current_speaker or 'A')
            if segment:
                segments.append(segment)
        
        return segments
    
    def _create_simple_segment(self, words: List[Dict], speaker: str) -> Optional[Dict]:
        """Create a simple segment from words"""
        if len(words) < self.min_words:
            return None
            
        start_time = words[0]['start'] / 1000.0
        end_time = words[-1]['end'] / 1000.0
        duration = end_time - start_time
        
        # Skip if too long
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
    
    def select_optimal_references(self, segments: List[Dict], speakers: List[str]) -> Dict[str, Dict]:
        """Simple reference selection - just pick first good segment per speaker"""
        references = {}
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s['speaker'] == speaker]
            
            # Sort by confidence and pick first good one
            speaker_segments.sort(key=lambda x: x['confidence'], reverse=True)
            
            for segment in speaker_segments:
                if (segment['confidence'] >= 0.5 and 
                    segment['duration'] >= 2.0 and 
                    segment['duration'] <= 15.0 and
                    segment['word_count'] >= 8):
                    
                    # Format for Dia model
                    segment['dia_text'] = f"[S1] {segment['text']}"
                    references[speaker] = segment
                    break
        
        return references
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int,
                            output_dir: Path, speakers: List[str], 
                            target_language: str, detected_language: str):
        """Save segments with simple file naming into speaker subdirs"""
        # Ensure base directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each segment into its speaker-specific folder
        for i, segment in enumerate(segments):
            speaker = segment.get('speaker', 'A')
            speaker_dir = output_dir / f"speaker_{speaker}" / "segments"
            speaker_dir.mkdir(parents=True, exist_ok=True)
            try:
                # Extract audio segment
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # Simple file naming
                audio_filename = f"segment_{i+1:03d}.wav"
                audio_path = speaker_dir / audio_filename
                sf.write(audio_path, segment_audio, sr)
                
                # Clean or translate text
                english_text = self.transcription_service.translate_text_clean(segment['text'])
                dia_text = f"[S1] {english_text}"
                
                # Save metadata
                metadata = {
                    'original_text': segment['text'],
                    'english_text': english_text,
                    'dia_text': dia_text,
                    'speaker': speaker,
                    'start': segment['start'],
                    'end': segment['end'],
                    'duration': segment['duration'],
                    'word_count': segment['word_count'],
                    'confidence': segment['confidence'],
                    'detected_language': detected_language,
                    'target_language': target_language
                }
                metadata_path = speaker_dir / f"segment_{i+1:03d}_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                # Update segment with paths if needed
                segment.update({
                    'audio_path': str(audio_path),
                    'metadata_path': str(metadata_path)
                })
            except Exception as e:
                logger.error(f"Error saving segment {i+1} for speaker {speaker}: {str(e)}")
                continue
        logger.info(f"Saved {len(segments)} segments under speaker subdirectories in {output_dir}")
    
    def identify_silent_parts(self, segments: List[Dict], total_duration: float) -> List[Tuple[float, float]]:
        """Simple silent part identification"""
        silent_parts = []
        
        if not segments:
            return silent_parts
        
        # Check gaps between segments
        for i in range(len(segments) - 1):
            current_end = segments[i]['end']
            next_start = segments[i + 1]['start']
            
            if next_start - current_end > 0.5:  # Gap > 0.5 seconds
                silent_parts.append((current_end, next_start))
        
        return silent_parts 