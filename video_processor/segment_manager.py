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
        
        # Apply combination logic as separate step
        optimized_segments = self._apply_combination_logic(segments)
        
        return optimized_segments
    
    def _apply_combination_logic(self, segments: List[Dict]) -> List[Dict]:
        """Apply combination logic for short segments - separate and clean"""
        if not segments:
            return segments
            
        optimized = []
        min_duration = 5.0  # Minimum duration for standalone segment
        
        for segment in segments:
            # Check if this is a short segment that can be combined
            if (segment['duration'] < min_duration and 
                optimized and 
                optimized[-1]['speaker'] == segment['speaker']):
                
                # Check if combining won't exceed max duration
                prev_segment = optimized[-1]
                combined_duration = segment['end'] - prev_segment['start']
                
                if combined_duration <= self.max_duration:
                    # Combine with previous segment
                    self._combine_segments(optimized[-1], segment)
                    logger.info(f"Combined short segment ({segment['duration']:.2f}s) with previous segment. New duration: {combined_duration:.2f}s")
                    continue
            
            # Add segment as standalone
            optimized.append(segment)
        
        return optimized
    
    def _combine_segments(self, target_segment: Dict, source_segment: Dict) -> None:
        """Combine two segments cleanly"""
        combined_words = target_segment['words'] + source_segment['words']
        combined_text = ' '.join(w['text'] for w in combined_words)
        combined_confidence = np.mean([w.get('confidence', 0.5) for w in combined_words])
        combined_duration = source_segment['end'] - target_segment['start']
        
        # Update target segment with combined data
        target_segment.update({
            'end': source_segment['end'],
            'duration': combined_duration,
            'text': combined_text,
            'word_count': len(combined_words),
            'confidence': combined_confidence,
            'words': combined_words
        })
    
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
            
            # Sort by word count (prefer larger segments) then by confidence
            speaker_segments.sort(key=lambda x: (x['word_count'], x['confidence']), reverse=True)
            
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
                
                # Simple file naming with consistent format
                audio_filename = f"segment_{i+1:03d}_metadata.wav"
                audio_path = speaker_dir / audio_filename
                sf.write(audio_path, segment_audio, sr)
                
                # Clean or translate text
                english_text = self.transcription_service.translate_text_clean(segment['text'])
                dia_text = f"[S1] {english_text}"
                
                # Save metadata with segment index for proper mapping
                metadata = {
                    'segment_index': i + 1,  # Add segment index for mapping
                    'audio_file': audio_filename,
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
                
                # Update segment with paths and index for timeline
                segment.update({
                    'segment_index': i + 1,
                    'audio_path': str(audio_path),
                    'metadata_path': str(metadata_path),
                    'audio_file': audio_filename
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