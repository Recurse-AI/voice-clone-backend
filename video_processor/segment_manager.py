"""
Segment Manager Module - Enhanced for Complete Audio Coverage
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
    """Enhanced segment manager with complete audio coverage"""
    
    def __init__(self, transcription_service):
        self.transcription_service = transcription_service
        self.target_words_per_segment = 30  # Target words per segment
        self.max_duration = 17.0            # Maximum segment duration
        # Removed min_duration to allow any segment length
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create segments with complete audio coverage optimized for Dia performance"""
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
        
        # Get total audio duration
        total_duration = transcript_data.get('duration', 0)
        if total_duration <= 0 and valid_words:
            total_duration = valid_words[-1]['end'] / 1000.0
        
        # Create continuous segments covering entire audio with Dia optimization
        segments = self._create_continuous_segments_optimized(valid_words, total_duration)
        logger.info(f"Created {len(segments)} Dia-optimized segments covering {total_duration:.2f}s")
        
        # Log segment details with Dia performance analysis
        total_covered = 0
        for i, seg in enumerate(segments):
            if seg:
                duration = seg.get('duration', 0)
                word_count = seg.get('word_count', 0)
                text_preview = seg.get('text', '')[:50] + "..." if len(seg.get('text', '')) > 50 else seg.get('text', '')
                speakers_in_segment = seg.get('speakers_in_segment', [])
                dia_score = seg.get('dia_optimization_score', 'unknown')
                logger.info(f"Segment {i+1}: {seg.get('start', 0):.2f}-{seg.get('end', 0):.2f}s ({duration:.2f}s), {word_count} words, Dia score: {dia_score}, speakers: {speakers_in_segment} - '{text_preview}'")
                total_covered += duration
        
        logger.info(f"Total coverage: {total_covered:.2f}s of {total_duration:.2f}s audio")
        
        return segments
    
    def _create_continuous_segments_optimized(self, words: List[Dict], total_duration: float) -> List[Dict]:
        """Create continuous segments with Dia model optimization"""
        if not words or total_duration <= 0:
            return []
        
        segments = []
        current_start = 0.0  # Start from beginning of audio
        
        # Sort words by start time to ensure proper order
        sorted_words = sorted(words, key=lambda w: w.get('start', 0))
        
        # Group words into segments while ensuring continuity and Dia optimization
        current_chunk = []
        word_index = 0
        
        while current_start < total_duration and word_index < len(sorted_words):
            # Collect words for current segment with Dia-aware targeting
            segment_end_target = min(current_start + self.max_duration, total_duration)
            current_chunk = []
            
            # Adaptive word collection based on expected Dia performance
            target_words = self._get_optimal_word_count_for_dia(
                current_start, segment_end_target, sorted_words, word_index
            )
            
            # Add words that fit within current segment timeframe
            words_collected = 0
            while word_index < len(sorted_words) and words_collected < target_words:
                word = sorted_words[word_index]
                word_start = word.get('start', 0) / 1000.0
                word_end = word.get('end', 0) / 1000.0
                
                # If word fits in current segment
                if word_start < segment_end_target:
                    current_chunk.append(word)
                    word_index += 1
                    words_collected += 1
                else:
                    break
            
            # Create segment from collected words
            if current_chunk:
                # Calculate actual segment end
                last_word_end = current_chunk[-1].get('end', 0) / 1000.0
                actual_end = min(max(last_word_end, current_start + 1.0), total_duration)
                
                segment = self._create_segment_from_words_continuous(
                    current_chunk, current_start, actual_end
                )
                
                if segment:
                    # Add Dia optimization score
                    segment['dia_optimization_score'] = self._calculate_dia_score(segment)
                    segments.append(segment)
                    current_start = actual_end  # Move to next position
            else:
                # No words in this range, move forward
                current_start = min(current_start + self.max_duration, total_duration)
        
        return segments
    
    def _get_optimal_word_count_for_dia(self, start_time: float, end_time: float, 
                                       words: List[Dict], word_index: int) -> int:
        """Calculate optimal word count for Dia performance"""
        # Count available words in timeframe
        available_words = 0
        temp_index = word_index
        
        while temp_index < len(words):
            word = words[temp_index]
            word_start = word.get('start', 0) / 1000.0
            
            if word_start < end_time:
                available_words += 1
                temp_index += 1
            else:
                break
        
        # Dia optimization targets
        if available_words <= 8:
            return min(available_words, 12)  # Single line (8-12 words)
        elif available_words <= 25:
            return min(available_words, 24)  # 2-3 lines (16-24 words)
        elif available_words <= 45:
            return min(available_words, 36)  # Max 4 lines (36 words)
        else:
            return min(available_words, 42)  # Max 5 lines (42 words)
    
    def _calculate_dia_score(self, segment: Dict) -> str:
        """Calculate expected Dia performance score"""
        word_count = segment.get('word_count', 0)
        
        # Estimate formatting
        if word_count <= 10:
            expected_lines = 1
            expected_words_per_line = word_count
        elif word_count <= 25:
            expected_lines = 2.5
            expected_words_per_line = word_count / 2.5
        else:
            expected_lines = min(5, word_count / 8)
            expected_words_per_line = word_count / expected_lines
        
        # Score based on Dia observations
        if expected_lines == 1 and 8 <= word_count <= 12:
            return "excellent"
        elif expected_lines <= 3 and 6 <= expected_words_per_line <= 10:
            return "good"
        elif expected_lines > 5:
            return "poor"
        elif expected_words_per_line <= 2:  # Relaxed from <= 3
            return "poor"
        else:
            return "fair"
    
    def _create_segment_from_words_continuous(self, words: List[Dict], 
                                           segment_start: float, segment_end: float) -> Optional[Dict]:
        """Create a segment with guaranteed start/end times for continuity"""
        if not words:
            return None
        
        # Use provided start/end times for continuity
        start_time = segment_start
        end_time = segment_end
        duration = end_time - start_time
        
        # Build text from words
        text = ' '.join(w.get('text', '') for w in words if w.get('text')).strip()
        if not text:
            return None
        
        # Get all speakers in the segment
        speakers_in_segment = []
        speaker_word_counts = {}
        for w in words:
            speaker = w.get('speaker', 'A')
            if speaker not in speaker_word_counts:
                speaker_word_counts[speaker] = 0
                speakers_in_segment.append(speaker)
            speaker_word_counts[speaker] += 1
        
        # Calculate confidence
        confidences = [w.get('confidence', 0.5) for w in words if w.get('confidence') is not None]
        confidence = np.mean(confidences) if confidences else 0.5
        
        primary_speaker = speakers_in_segment[0] if speakers_in_segment else 'A'
        word_count = len(words)
        
        logger.debug(f"Created segment: {duration:.2f}s, {word_count} words, speakers: {speakers_in_segment}")
        
        return {
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'text': text,
            'speaker': primary_speaker,
            'speakers_in_segment': speakers_in_segment,
            'speaker_word_counts': speaker_word_counts,
            'word_count': word_count,
            'confidence': confidence,
            'words': words,
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