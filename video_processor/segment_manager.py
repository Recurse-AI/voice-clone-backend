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
        self.max_gap_seconds = 3.0          # Maximum gap between words
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create segments with simple word-based chunking"""
        if not transcript_data or not transcript_data.get('words'):
            logger.error("No words found in transcript_data")
            return []
            
        words = transcript_data.get('words', [])
        # Filter out invalid words
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
                logger.info(f"Segment {i+1}: {duration:.2f}s, {word_count} words - '{text_preview}'")
        
        return segments
    
    def _create_word_based_segments(self, words: List[Dict]) -> List[Dict]:
        """Create segments based on word count with duration validation"""
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
        """Determine if we should create a segment now"""
        if not current_chunk:
            return False
        
        word_count = len(current_chunk)
        
        # Calculate current duration
        start_time = current_chunk[0].get('start', 0) / 1000.0
        end_time = current_chunk[-1].get('end', 0) / 1000.0
        duration = end_time - start_time
        
        # Check for speaker change (if next word exists)
        if current_index + 1 < len(all_words):
            current_speaker = current_chunk[-1].get('speaker', 'A')
            next_speaker = all_words[current_index + 1].get('speaker', 'A')
            if current_speaker != next_speaker:
                logger.debug(f"Speaker change detected: {current_speaker} -> {next_speaker}")
                return True
        
        # Check for long gap
        if current_index + 1 < len(all_words):
            current_end = current_chunk[-1].get('end', 0) / 1000.0
            next_start = all_words[current_index + 1].get('start', 0) / 1000.0
            gap = next_start - current_end
            if gap > self.max_gap_seconds:
                logger.debug(f"Long gap detected: {gap:.2f}s")
                return True
        
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
        """Create a segment from a list of words"""
        if not words:
            return None
        
        # Calculate timing
        start_time = words[0].get('start', 0) / 1000.0
        end_time = words[-1].get('end', 0) / 1000.0
        duration = end_time - start_time
        
        # Build text
        text = ' '.join(w.get('text', '') for w in words if w.get('text')).strip()
        if not text:
            return None
        
        # Get speaker (use most common speaker in the segment)
        speakers = [w.get('speaker', 'A') for w in words if w.get('speaker')]
        speaker = max(set(speakers), key=speakers.count) if speakers else 'A'
        
        # Calculate confidence
        confidences = [w.get('confidence', 0.5) for w in words if w.get('confidence') is not None]
        confidence = np.mean(confidences) if confidences else 0.5
        
        # Validate duration (be more lenient for edge cases)
        if duration < 0.5:  # Very short segments are usually noise
            logger.debug(f"Segment too short: {duration:.2f}s, text: '{text[:30]}...'")
            return None
        
        logger.debug(f"Created segment: {duration:.2f}s, {len(words)} words, speaker: {speaker}")
        
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
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int,
                            output_dir: Path, speakers: List[str], 
                            target_language: str, detected_language: str):
        """Save segments for voice cloning with parallel translation"""
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
        segment_speakers = []
        for segment in segments:
            original_text = segment.get('text', '').strip()
            if not original_text:
                raise ValueError(f"Segment has no text: {segment}")
            segment_texts.append(original_text)
            segment_speakers.append(segment.get('speaker', 'A'))
        
        # Process all translations in parallel
        english_texts = self.transcription_service.format_dialogue_batch(segment_texts, segment_speakers)
        
        translation_time = time.time() - translation_start_time
        print(f"Parallel translation completed in {translation_time:.2f} seconds")
        
        for i, segment in enumerate(segments):
            if not segment or not isinstance(segment, dict):
                continue
                
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
                'speaker': speaker,
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