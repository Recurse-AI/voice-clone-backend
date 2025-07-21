"""
Segment Manager Module - Simplified for Voice Cloning
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class SegmentManager:
    """Simplified segment manager for voice cloning"""
    
    def __init__(self, transcription_service):
        self.transcription_service = transcription_service
        self.target_words_per_segment = 30  # Target words per segment
        self.min_duration = 3.0             # Minimum segment duration
        self.max_duration = 17.0            # Maximum segment duration
        # Removed max_gap_seconds as requested
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create segments with simple word-based chunking"""
        if not transcript_data or not transcript_data.get('words'):
            logger.error("No words found in transcript_data")
            return []
            
        words = transcript_data.get('words', [])
        # Filter out invalid words but maintain them exactly as they are
        valid_words = [w for w in words if w and isinstance(w, dict) and 
                      w.get('text') and w.get('start') is not None and w.get('end') is not None]
        
        if not valid_words:
            logger.error("No valid words found after filtering")
            return []
        
        logger.info(f"Processing {len(valid_words)} valid words")
        
        segments = self._create_word_based_segments(valid_words)
        logger.info(f"Created {len(segments)} segments")
        
        # Log segment details
        for i, seg in enumerate(segments):
            if seg:
                duration = seg.get('duration', 0)
                word_count = seg.get('word_count', 0)
                text_preview = seg.get('text', '')[:50] + "..." if len(seg.get('text', '')) > 50 else seg.get('text', '')
                speakers_in_segment = seg.get('speakers_in_segment', [])
                logger.info(f"Segment {i+1}: {duration:.2f}s, {word_count} words, speakers: {speakers_in_segment} - '{text_preview}'")
        
        return segments
    
    def _create_word_based_segments(self, words: List[Dict]) -> List[Dict]:
        """Create segments based on word count with duration validation - allow multi-speaker"""
        segments = []
        current_chunk = []
        
        i = 0
        while i < len(words):
            current_chunk.append(words[i])
            
            # Check if we should create a segment
            if self._should_create_segment(current_chunk, i, words):
                segment = self._create_segment_from_words(current_chunk)
                if segment:
                    segments.append(segment)
                current_chunk = []
            
            i += 1
        
        # Handle remaining words
        if current_chunk:
            segment = self._create_segment_from_words(current_chunk)
            if segment:
                segments.append(segment)
        
        return segments
    
    def _should_create_segment(self, current_chunk: List[Dict], current_index: int, all_words: List[Dict]) -> bool:
        """Determine if we should create a segment now - reduced speaker change impact"""
        if not current_chunk:
            return False
        
        word_count = len(current_chunk)
        
        # Calculate current duration
        start_time = current_chunk[0].get('start', 0) / 1000.0
        end_time = current_chunk[-1].get('end', 0) / 1000.0
        duration = end_time - start_time
        
        # Removed speaker change check to allow multi-speaker segments
        # Removed max_gap_seconds check as requested
        
        # Check if reached target word count and minimum duration
        if word_count >= self.target_words_per_segment and duration >= self.min_duration:
            return True
        
        # Force split if exceeding max duration
        if duration > self.max_duration:
            logger.debug(f"Max duration exceeded: {duration:.2f}s")
            return True
        
        # If we're at the end of words
        if current_index + 1 >= len(all_words):
            return True
        
        return False
    
    def _create_segment_from_words(self, words: List[Dict]) -> Optional[Dict]:
        """Create a segment from a list of words - support multi-speaker segments"""
        if not words:
            return None
        
        # Calculate timing
        start_time = words[0].get('start', 0) / 1000.0
        end_time = words[-1].get('end', 0) / 1000.0
        duration = end_time - start_time
        
        # Build text exactly as the words are - maintain original words
        text = ' '.join(w.get('text', '') for w in words if w.get('text')).strip()
        if not text:
            return None
        
        # Get all speakers in the segment and their word counts
        speakers_in_segment = []
        speaker_word_counts = {}
        for w in words:
            speaker = w.get('speaker', 'A')
            if speaker not in speaker_word_counts:
                speaker_word_counts[speaker] = 0
                speakers_in_segment.append(speaker)
            speaker_word_counts[speaker] += 1
        
        # Always use the first speaker as the primary speaker for folder assignment
        primary_speaker = speakers_in_segment[0] if speakers_in_segment else 'A'
        
        # Calculate confidence
        confidences = [w.get('confidence', 0.5) for w in words if w.get('confidence') is not None]
        confidence = np.mean(confidences) if confidences else 0.5
        
        # Validate duration (be more lenient for edge cases)
        if duration < 0.5:  # Very short segments are usually noise
            logger.debug(f"Segment too short: {duration:.2f}s, text: '{text[:30]}...'")
            return None
        
        logger.debug(f"Created segment: {duration:.2f}s, {len(words)} words, speakers: {speakers_in_segment}, primary: {primary_speaker}")
        
        return {
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'text': text,
            'speaker': primary_speaker,  # Always assign to first speaker
            'speakers_in_segment': speakers_in_segment,  # Track all speakers
            'speaker_word_counts': speaker_word_counts,  # Track word distribution
            'word_count': len(words),
            'confidence': confidence,
            'words': words,  # Keep exact words as they are
            'is_multi_speaker': len(speakers_in_segment) > 1
        }
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int,
                            output_dir: Path, speakers: List[str], 
                            target_language: str, detected_language: str):
        """Save segments for voice cloning with parallel translation - support multi-speaker"""
        if not segments or audio is None or sr <= 0:
            logger.error("Invalid input parameters for save_optimal_segments")
            return
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        overall_metadata = {
            "total_segments": len(segments),
            "speakers": speakers,
            "target_language": target_language,
            "detected_language": detected_language,
            "processing_timestamp": str(datetime.now())
        }
        
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_dir / "processing_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(overall_metadata, f, ensure_ascii=False, indent=2)
        
        # Prepare all texts for parallel translation
        print("Starting parallel translation...")
        translation_start_time = time.time()
        
        segment_texts = []
        segment_speakers_data = []
        for segment in segments:
            original_text = segment.get('text', '').strip()
            if not original_text:
                raise ValueError(f"Segment has no text: {segment}")
            segment_texts.append(original_text)
            
            # Prepare speaker data for multi-speaker formatting
            speakers_in_segment = segment.get('speakers_in_segment', [segment.get('speaker', 'A')])
            segment_speakers_data.append({
                'speakers': speakers_in_segment,
                'is_multi_speaker': segment.get('is_multi_speaker', False),
                'primary_speaker': segment.get('speaker', 'A')
            })
        
        # Process all translations in parallel with multi-speaker support
        english_texts = self.transcription_service.format_dialogue_batch(
            segment_texts, segment_speakers_data
        )
        
        translation_time = time.time() - translation_start_time
        print(f"Parallel translation completed in {translation_time:.2f} seconds")
        
        for i, segment in enumerate(segments):
            if not segment or not isinstance(segment, dict):
                continue
                
            # Always use primary speaker (first speaker) for folder assignment
            speaker = segment.get('speaker', 'A')
            speaker_dir = output_dir / f"speaker_{speaker}"
            segments_dir = speaker_dir / "segments"
            
            segments_dir.mkdir(parents=True, exist_ok=True)
            
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            if start_sample >= len(audio) or end_sample > len(audio) or start_sample >= end_sample:
                continue
                
            segment_audio = audio[start_sample:end_sample]
            
            audio_filename = f"segment_{i+1:03d}.wav"
            audio_path = segments_dir / audio_filename
            sf.write(audio_path, segment_audio, sr)
            
            original_text = segment_texts[i] if i < len(segment_texts) else ""
            english_text = english_texts[i] if i < len(english_texts) else ""
            
            if not original_text:
                print(f"Warning: Segment {i+1} has no original text, skipping")
                continue
            
            if not english_text:
                print(f"Warning: Segment {i+1} translation failed, using original text")
                english_text = original_text
            
            metadata = {
                'segment_index': i + 1,
                'audio_file': audio_filename,
                'audio_path': str(audio_path),
                'original_text': original_text,
                'english_text': english_text,
                'speaker': speaker,  # Primary speaker for folder assignment
                'speakers_in_segment': segment.get('speakers_in_segment', [speaker]),
                'is_multi_speaker': segment.get('is_multi_speaker', False),
                'speaker_word_counts': segment.get('speaker_word_counts', {}),
                'speaker_index': ord(speaker) - ord('A') + 1,
                'start': start_time,
                'end': end_time,
                'duration': segment.get('duration', 0),
                'word_count': segment.get('word_count', 0),
                'confidence': segment.get('confidence', 0.5),
                'cloned_audio_file': f"cloned_segment_{i+1:03d}.wav",
                'cloned_audio_path': str(segments_dir / f"cloned_segment_{i+1:03d}.wav"),
                'metadata_complete': True,
                'processing_status': 'ready_for_cloning'
            }
            
            metadata_path = segments_dir / f"segment_{i+1:03d}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            segment.update({
                'segment_index': i + 1,
                'audio_path': str(audio_path),
                'metadata_path': str(metadata_path),
                'audio_file': audio_filename,
                'english_text': english_text,
                'metadata_complete': True
            }) 