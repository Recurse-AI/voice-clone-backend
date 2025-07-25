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
    """Simplified segment manager for optimal Dia performance"""
    
    def __init__(self, transcription_service):
        self.transcription_service = transcription_service
        self.optimal_duration = 10.0        # Optimal 10 seconds
        self.min_duration = 9.0             # Minimum 9 seconds  
        self.max_duration = 11.0            # Maximum 11 seconds
        self.target_words_per_segment = 25  # Target words for good [S1]/[S2] formatting
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create simple 9-11 second segments with proper speaker tags"""
        if not transcript_data or not transcript_data.get('words'):
            logger.error("No words found in transcript_data")
            return []
            
        words = transcript_data.get('words', [])
        valid_words = [w for w in words if w and isinstance(w, dict) and 
                      w.get('text') and w.get('start') is not None and w.get('end') is not None]
        
        if not valid_words:
            logger.error("No valid words found after filtering")
            return []
        
        logger.info(f"Creating simple segments from {len(valid_words)} words")
        
        # Get total duration
        total_duration = transcript_data.get('duration', 0)
        if total_duration <= 0 and valid_words:
            total_duration = valid_words[-1]['end'] / 1000.0
        
        # Create simple continuous segments
        segments = self._create_simple_segments(valid_words, total_duration)
        
        logger.info(f"Created {len(segments)} segments with average duration: {sum(s['duration'] for s in segments) / len(segments):.1f}s")
        return segments
    
    def _create_simple_segments(self, words: List[Dict], total_duration: float) -> List[Dict]:
        """Create simple continuous segments targeting 9-11 seconds"""
        if not words or total_duration <= 0:
            return []
        
        segments = []
        current_start = 0.0
        sorted_words = sorted(words, key=lambda w: w.get('start', 0))
        word_index = 0
        loop_counter = 0
        max_loops = len(sorted_words) * 2  # Safety limit
        
        while current_start < total_duration and word_index < len(sorted_words) and loop_counter < max_loops:
            loop_counter += 1
            # Check if remaining duration is too short for a new segment
            remaining_duration = total_duration - current_start
            if remaining_duration < self.min_duration and segments:
                # Extend the last segment to include remaining words
                remaining_words = sorted_words[word_index:]
                if remaining_words:
                    last_segment = segments[-1]
                    # Extend last segment to include remaining words
                    last_segment['end'] = total_duration
                    last_segment['duration'] = total_duration - last_segment['start']
                    # Add remaining words to the last segment
                    last_segment['words'].extend(remaining_words)
                    
                    # Update text and speaker info
                    for word in remaining_words:
                        text = word.get('text', '').strip()
                        if text:
                            speaker = word.get('speaker', 'A')
                            if speaker not in last_segment['speaker_word_counts']:
                                last_segment['speaker_word_counts'][speaker] = 0
                                last_segment['speakers_in_segment'].append(speaker)
                            last_segment['speaker_word_counts'][speaker] += 1
                    
                    # Update text
                    all_texts = [w.get('text', '').strip() for w in last_segment['words'] if w.get('text', '').strip()]
                    last_segment['text'] = ' '.join(all_texts)
                    last_segment['word_count'] = len(all_texts)
                    last_segment['is_multi_speaker'] = len(last_segment['speakers_in_segment']) > 1
                break
            
            # Target segment end
            target_end = min(current_start + self.optimal_duration, total_duration)
            
            # Collect words for this segment
            segment_words = []
            while word_index < len(sorted_words):
                word = sorted_words[word_index]
                word_start = word.get('start', 0) / 1000.0
                word_end = word.get('end', 0) / 1000.0
                
                # If word starts before target end, include it
                if word_start < target_end:
                    segment_words.append(word)
                    word_index += 1
                else:
                    break
            
            # Create segment if we have words
            if segment_words:
                last_word_end = segment_words[-1].get('end', 0) / 1000.0
                actual_end = min(max(last_word_end, current_start + self.min_duration), total_duration)
                
                # Ensure minimum duration by adjusting end time if needed
                if actual_end - current_start < self.min_duration and actual_end < total_duration:
                    actual_end = min(current_start + self.min_duration, total_duration)
                
                # Ensure maximum duration
                if actual_end - current_start > self.max_duration:
                    actual_end = current_start + self.max_duration
                
                segment = self._create_simple_segment(segment_words, current_start, actual_end)
                if segment:
                    segments.append(segment)
                current_start = actual_end
            else:
                # No words collected - force increment to avoid infinite loop
                current_start += self.optimal_duration
                # Also increment word_index if we're stuck on the same word
                if word_index < len(sorted_words):
                    word_index += 1
        
        if loop_counter >= max_loops:
            logger.warning(f"Segment creation loop limit reached ({max_loops} iterations) - safety break")
        
        return segments
    
    def _create_simple_segment(self, words: List[Dict], 
                              segment_start: float, segment_end: float) -> Optional[Dict]:
        """Create a simple segment with clean speaker tags"""
        if not words:
            return None
        
        duration = segment_end - segment_start
        
        # Build text and collect speakers
        text_parts = []
        speakers_in_segment = []
        speaker_word_counts = {}
        
        for word in words:
            text = word.get('text', '').strip()
            if text:
                text_parts.append(text)
                
                speaker = word.get('speaker', 'A')
                if speaker not in speaker_word_counts:
                    speaker_word_counts[speaker] = 0
                    speakers_in_segment.append(speaker)
                speaker_word_counts[speaker] += 1
        
        if not text_parts:
            return None
        
        # Create clean text
        full_text = ' '.join(text_parts)
        
        # Calculate confidence
        confidences = [w.get('confidence', 0.5) for w in words if w.get('confidence') is not None]
        confidence = float(np.mean(confidences)) if confidences else 0.5
        
        # Primary speaker (most words)
        primary_speaker = max(speaker_word_counts.keys(), key=lambda k: speaker_word_counts[k]) if speaker_word_counts else 'A'
        
        logger.debug(f"Created segment: {duration:.1f}s, {len(text_parts)} words, speakers: {speakers_in_segment}")
        
        return {
            'start': segment_start,
            'end': segment_end, 
            'duration': duration,
            'text': full_text,
            'speaker': primary_speaker,
            'speakers_in_segment': speakers_in_segment,
            'speaker_word_counts': speaker_word_counts,
            'word_count': len(text_parts),
            'confidence': confidence,
            'words': words,
            'is_multi_speaker': len(speakers_in_segment) > 1
        }
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int,
                            output_dir: Path, speakers: List[str], 
                            target_language: str, detected_language: str, 
                            original_audio_details: Optional[Dict] = None):
        """Save segments for voice cloning with parallel translation - support multi-speaker"""
        if not segments or audio is None or sr <= 0:
            logger.error("Invalid input parameters for save_optimal_segments")
            return
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate original audio duration from the audio array
        original_duration = len(audio) / sr
        
        overall_metadata = {
            "total_segments": len(segments),
            "speakers": speakers,
            "target_language": target_language,
            "detected_language": detected_language,
            "processing_timestamp": str(datetime.now()),
            "original_audio": {
                "duration": original_duration,
                "sample_rate": sr,
                "total_samples": len(audio)
            }
        }
        
        # Add original audio details if provided
        if original_audio_details:
            overall_metadata["original_audio"].update(original_audio_details)
        
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_dir / "processing_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(overall_metadata, f, ensure_ascii=False, indent=2)
        
        # Prepare all texts for parallel translation
        print("Starting parallel translation...")
        translation_start_time = time.time()
        
        segment_texts = []
        segment_speakers_data = []
        segment_words_data = []
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
            
            # Add words data for multi-speaker processing
            segment_words_data.append(segment.get('words', []))
        
        # Process all translations in parallel with multi-speaker support
        english_texts = self.transcription_service.format_dialogue_batch(
            segment_texts, segment_speakers_data, segment_words_data
        )
        
        translation_time = time.time() - translation_start_time
        logger.info(f"Parallel translation completed in {translation_time:.2f} seconds")
        logger.info(f"Translation results: {len(english_texts)} texts generated from {len(segment_texts)} input texts")
        
        segments_saved = 0
        for i, segment in enumerate(segments):
            logger.info(f"Processing segment {i+1}/{len(segments)}")
            if not segment or not isinstance(segment, dict):
                logger.warning(f"Skipping segment {i+1}: Invalid segment data")
                continue
                
            # Store all segments in single segments folder
            segments_dir = output_dir / "segments"
            segments_dir.mkdir(parents=True, exist_ok=True)
            
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Clamp to audio bounds instead of skipping
            start_sample = max(0, min(start_sample, len(audio) - 1))
            end_sample = max(start_sample + 1, min(end_sample, len(audio)))
            
            logger.info(f"Segment {i+1} timing: start={start_time:.2f}s, end={end_time:.2f}s, start_sample={start_sample}, end_sample={end_sample}, audio_length={len(audio)}")
            
            if start_sample >= end_sample:
                logger.warning(f"Skipping segment {i+1}: Invalid timing - start_sample={start_sample}, end_sample={end_sample}")
                continue
                
            segment_audio = audio[start_sample:end_sample]
            
            audio_filename = f"segment_{i+1:03d}.wav"
            audio_path = segments_dir / audio_filename
            sf.write(audio_path, segment_audio, sr)
            
            original_text = segment_texts[i] if i < len(segment_texts) else ""
            english_text = english_texts[i] if i < len(english_texts) else ""
            
            if not original_text:
                logger.error(f"Segment {i+1} has no original text, skipping")
                continue
            
            if not english_text:
                logger.error(f"Segment {i+1} translation failed, skipping")
                continue
            
            metadata = {
                'segment_index': i + 1,
                'audio_file': audio_filename,
                'audio_path': str(audio_path),
                'original_text': original_text,
                'english_text': english_text,
                'speaker': segment.get('speaker', 'A'),  # Primary speaker for folder assignment
                'speakers_in_segment': segment.get('speakers_in_segment', [segment.get('speaker', 'A')]),
                'is_multi_speaker': segment.get('is_multi_speaker', False),
                'speaker_word_counts': segment.get('speaker_word_counts', {}),
                'speaker_index': ord(segment.get('speaker', 'A')) - ord('A') + 1,
                'start': start_time,
                'end': end_time,
                'duration': segment.get('duration', 0),
                'word_count': segment.get('word_count', 0),
                'confidence': segment.get('confidence', 0.5),
                'cloned_audio_file': f"cloned_segment_{i+1:03d}.wav",
                'cloned_audio_path': str(output_dir / "cloned" / f"cloned_segment_{i+1:03d}.wav"),
                'metadata_complete': True,
                'processing_status': 'ready_for_cloning'
            }
            
            metadata_path = segments_dir / f"segment_{i+1:03d}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved segment {i+1}: audio={audio_filename}, metadata={metadata_path.name}")
            segments_saved += 1
            
            segment.update({
                'segment_index': i + 1,
                'audio_path': str(audio_path),
                'metadata_path': str(metadata_path),
                'audio_file': audio_filename,
                'english_text': english_text,
                'metadata_complete': True
            })
        
        logger.info(f"Successfully saved {segments_saved} segments to {output_dir}")
        logger.info(f"Segments directory: {output_dir / 'segments'}")
        logger.info(f"Metadata directory: {output_dir / 'metadata'}") 