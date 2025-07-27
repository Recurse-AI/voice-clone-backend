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
import librosa
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


class SegmentManager:
    """Enhanced segment manager with audio length adjustment and complete coverage"""
    
    def __init__(self, transcription_service):
        self.transcription_service = transcription_service
        self.optimal_duration = 8.0  # Dia works best with 5-10s segments
        self.min_duration = 5.0  # Dia recommendation: at least 5s for natural sound
        self.max_duration = 10.0  # Dia recommendation: max 10s for best quality
        self.silence_threshold = -40  # dB
        self.min_gap_duration = 2.5  # Only consider gaps >= 2.5s as actual breaks
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create segments ensuring complete audio coverage with simplified logic"""
        if not transcript_data or not transcript_data.get('words'):
            return []
            
        words = transcript_data.get('words', [])
        valid_words = [w for w in words if w and isinstance(w, dict) and 
                      w.get('text') and w.get('start') is not None and w.get('end') is not None]
        
        if not valid_words:
            return []
        
        # Calculate actual audio duration
        audio_duration = transcript_data.get('audio_duration', 0)
        first_word_start = min(w['start'] for w in valid_words) / 1000.0
        last_word_end = max(w['end'] for w in valid_words) / 1000.0
        
        # Use full audio duration to avoid losing any portion
        total_duration = max(audio_duration, last_word_end)
        
        logger.info(f"Audio duration info: original={audio_duration:.2f}s, transcribed_end={last_word_end:.2f}s, using_total={total_duration:.2f}s")
        
        # Handle short audio clips (Dia works better with 5-10s but can handle shorter)
        if total_duration < 3.0:
            logger.warning(f"Audio duration ({total_duration:.2f}s) is too short for quality voice cloning - skipping segmentation")
            return []
        elif total_duration < self.min_duration:
            logger.info(f"Audio duration ({total_duration:.2f}s) is shorter than optimal ({self.min_duration}s) but proceeding - quality may be reduced")
            # Continue processing with a warning
        
        segments = self._create_simplified_segments(valid_words, total_duration, first_word_start)
        
        # Filter out segments shorter than minimum duration (with some flexibility)
        min_filter_duration = max(3.0, self.min_duration * 0.8)  # Allow 80% of min_duration but not below 3s
        valid_segments = [s for s in segments if s.get('duration', 0) >= min_filter_duration]
        
        if len(valid_segments) != len(segments):
            logger.info(f"Filtered {len(segments) - len(valid_segments)} segments shorter than {self.min_duration}s")
        
        return valid_segments
    
    def _create_simplified_segments(self, words: List[Dict], total_duration: float, 
                                   first_word_start: float) -> List[Dict]:
        """Create segments with simplified logic - only break on significant gaps (>=2.5s)"""
        segments = []
        current_segment = {
            'words': [],
            'speakers_in_segment': set(),
            'is_multi_speaker': False,
            'start': 0.0,  # Always start from 0 to ensure complete coverage
            'end': 0.0,
            'text': '',
            'speaker': 'A',
            'duration': 0.0
        }
        
        current_time = 0.0
        
        for word in words:
            word_start = word.get('start', 0) / 1000.0
            word_end = word.get('end', 0) / 1000.0
            word_text = word.get('text', '').strip()
            word_speaker = word.get('speaker', 'A')
            
            if not word_text:
                continue
            
            # Only consider significant gaps (>= min_gap_duration) as breaks
            significant_gap = word_start > current_time + self.min_gap_duration
            
            # Check if we need to finish current segment
            should_split = False
            
            # Duration-based splitting
            potential_end = word_end
            potential_duration = potential_end - current_segment['start']
            
            if potential_duration > self.max_duration:
                should_split = True
            elif significant_gap and len(current_segment['words']) > 0:
                # Only split on significant gaps
                gap_duration = word_start - current_time
                logger.info(f"Significant gap detected: {gap_duration:.2f}s at {current_time:.2f}s - creating new segment")
                should_split = True
            
            if should_split and current_segment['words']:
                # Finalize current segment
                current_segment['end'] = current_segment['words'][-1].get('end', 0) / 1000.0
                current_segment['duration'] = current_segment['end'] - current_segment['start']
                current_segment['text'] = ' '.join([w.get('text', '') for w in current_segment['words']])
                current_segment['is_multi_speaker'] = len(current_segment['speakers_in_segment']) > 1
                
                # Only add segment if it meets minimum duration
                if current_segment['duration'] >= self.min_duration:
                    segments.append(current_segment.copy())
                else:
                    logger.info(f"Skipping short segment: {current_segment['duration']:.2f}s < {self.min_duration}s")
                
                # Start new segment
                current_segment = {
                    'words': [],
                    'speakers_in_segment': set(),
                    'is_multi_speaker': False,
                    'start': current_segment['end'] if significant_gap else word_start,
                    'end': 0.0,
                    'text': '',
                    'speaker': word_speaker,
                    'duration': 0.0
                }
            
            # Add word to current segment
            current_segment['words'].append(word)
            current_segment['speakers_in_segment'].add(word_speaker)
            if not current_segment['speaker'] or current_segment['speaker'] == 'A':
                current_segment['speaker'] = word_speaker
            
            current_time = word_end
        
        # Finalize last segment
        if current_segment['words']:
            current_segment['end'] = total_duration  # Ensure coverage to end
            current_segment['duration'] = current_segment['end'] - current_segment['start']
            current_segment['text'] = ' '.join([w.get('text', '') for w in current_segment['words']])
            current_segment['is_multi_speaker'] = len(current_segment['speakers_in_segment']) > 1
            
            # Only add segment if it meets minimum duration
            if current_segment['duration'] >= self.min_duration:
                segments.append(current_segment)
            else:
                logger.info(f"Skipping final short segment: {current_segment['duration']:.2f}s < {self.min_duration}s")
                
                # If we have previous segments, extend the last one to cover the remaining duration
                if segments:
                    segments[-1]['end'] = total_duration
                    segments[-1]['duration'] = segments[-1]['end'] - segments[-1]['start']
                    segments[-1]['text'] += ' ' + current_segment['text']
                    segments[-1]['words'].extend(current_segment['words'])
                    logger.info(f"Extended last segment to cover remaining duration")
        
        logger.info(f"Created {len(segments)} segments covering 0.0s to {total_duration:.2f}s with minimum {self.min_duration}s duration")
        return segments
    
    def _create_segment(self, words: List[Dict], start: float, end: float) -> Optional[Dict]:
        """Create segment with speaker information"""
        duration = end - start
        
        # Extract text and speaker info
        text_parts = []
        speaker_counts = {}
        
        for word in words:
            text = word.get('text', '').strip()
            if text:
                text_parts.append(text)
                speaker = word.get('speaker', 'A')
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        # Handle segments without words (silence/music)
        if not text_parts:
            text_parts = ['[SILENCE]']
            speaker_counts = {'A': 1}
        
        full_text = ' '.join(text_parts)
        primary_speaker = max(speaker_counts.keys(), key=lambda k: speaker_counts[k])
        
        # Calculate confidence
        confidences = [w.get('confidence', 0.5) for w in words if w.get('confidence') is not None]
        confidence = float(np.mean(confidences)) if confidences else 0.5
        
        return {
            'start': start,
            'end': end,
            'duration': duration,
            'text': full_text,
            'speaker': primary_speaker,
            'speakers_in_segment': list(speaker_counts.keys()),
            'speaker_word_counts': speaker_counts,
            'word_count': len(text_parts),
            'confidence': confidence,
            'words': words,
            'is_multi_speaker': len(speaker_counts) > 1
        }
    
    def _adjust_length(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        """Simple audio length adjustment"""
        if len(audio) == target_samples:
            return audio
        elif len(audio) > target_samples:
            return audio[:target_samples]
        else:
            # Pad with silence
            padding = target_samples - len(audio)
            silence = np.zeros(padding, dtype=audio.dtype)
            return np.concatenate([audio, silence])
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int,
                            output_dir: Path, speakers: List[str], 
                            target_language: str, detected_language: str, 
                            original_audio_details: Optional[Dict] = None):
        """Save segments with complete audio coverage and handle silent segments"""
        
        # Prepare segment texts for translation (exclude silent segments)
        segment_texts = []
        segment_speakers_data = []
        segment_words_data = []
        
        for segment in segments:
            text = segment.get('text', '').strip()
            if text:  # Only translate non-silent segments
                segment_texts.append(text)
                segment_speakers_data.append({
                    'speakers': segment.get('speakers_in_segment', [segment.get('speaker', 'A')]),
                    'is_multi_speaker': segment.get('is_multi_speaker', False),
                    'primary_speaker': segment.get('speaker', 'A')
                })
                segment_words_data.append(segment.get('words', []))
            else:
                segment_texts.append("")
                segment_speakers_data.append({
                    'speakers': [segment.get('speaker', 'A')],
                    'is_multi_speaker': False,
                    'primary_speaker': segment.get('speaker', 'A')
                })
                segment_words_data.append([])
        
        # Translate only non-empty texts
        english_texts = self.transcription_service.format_dialogue_batch(
            segment_texts, segment_speakers_data, segment_words_data
        )
        
        # Save segments
        segments_dir = output_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        for i, segment in enumerate(segments):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Ensure valid sample range
            start_sample = max(0, min(start_sample, len(audio) - 1))
            end_sample = max(start_sample + 1, min(end_sample, len(audio)))
            
            segment_audio = audio[start_sample:end_sample]
            
            # Save audio
            audio_filename = f"segment_{i+1:03d}.wav"
            audio_path = segments_dir / audio_filename
            sf.write(audio_path, segment_audio, sr)
            
            # Prepare metadata
            original_text = segment_texts[i] if i < len(segment_texts) else ""
            english_text = english_texts[i] if i < len(english_texts) else ""
            
            # Calculate confidence for silent segments
            if segment.get('words'):
                confidence = sum(w.get('confidence', 0) for w in segment['words']) / len(segment['words'])
            else:
                confidence = 1.0  # Silent segments have perfect "confidence"
            
            metadata = {
                'segment_index': i + 1,
                'audio_file': audio_filename,
                'audio_path': str(audio_path),
                'original_text': original_text,
                'english_text': english_text,
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'speaker': segment.get('speaker', 'A'),
                'confidence': confidence,
                'word_count': len(segment.get('words', [])),
                'is_silent_segment': not bool(original_text.strip()),
                'segment_type': 'silent' if not original_text.strip() else 'speech'
            }
            
            # Save metadata
            metadata_filename = f"segment_{i+1:03d}_metadata.json"
            metadata_path = segments_dir / metadata_filename
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Save processing summary
        self._save_processing_summary(output_dir, segments, speakers, target_language, 
                                    detected_language, original_audio_details)
        
        logger.info(f"Saved {len(segments)} segments with complete coverage")
    
    def _save_processing_summary(self, output_dir: Path, segments: List[Dict], 
                               speakers: List[str], target_language: str, 
                               detected_language: str, original_audio_details: Optional[Dict]):
        """Save clean processing summary"""
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Count speech vs silent segments
        speech_segments = sum(1 for s in segments if s.get('text', '').strip())
        silent_segments = len(segments) - speech_segments
        total_duration = segments[-1]['end'] if segments else 0.0
        
        summary = {
            'processing_metadata': {
                'total_segments': len(segments),
                'speech_segments': speech_segments,
                'silent_segments': silent_segments,
                'total_duration': total_duration,
                'speakers': speakers,
                'target_language': target_language,
                'detected_language': detected_language,
                'processing_timestamp': str(datetime.now())
            }
        }
        
        if original_audio_details:
            summary['original_audio'] = original_audio_details
        
        summary_file = metadata_dir / "processing_metadata.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def adjust_generated_audio_length(self, generated_audio: np.ndarray, sr: int, 
                                     reference_segment: Dict[str, Any]) -> np.ndarray:
        """
        Simplified audio length adjustment - just ensure exact duration match
        """
        target_duration = reference_segment.get('duration', 10.0)
        current_duration = len(generated_audio) / sr
        tolerance = 0.05  # 50ms tolerance
        
        # If already within tolerance, return as is
        if abs(current_duration - target_duration) <= tolerance:
            return generated_audio
        
        target_samples = int(target_duration * sr)
        
        if current_duration > target_duration:
            # Audio is too long - simple trim
            logger.info(f"Trimming audio from {current_duration:.3f}s to {target_duration:.3f}s")
            return generated_audio[:target_samples]
        else:
            # Audio is too short - add silence padding
            padding_needed = target_samples - len(generated_audio)
            logger.info(f"Padding audio with {padding_needed/sr:.3f}s of silence to reach {target_duration:.3f}s")
            silence_padding = np.zeros(padding_needed, dtype=generated_audio.dtype)
            return np.concatenate([generated_audio, silence_padding])
    
    def _intelligent_shorten(self, audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
        """Simplified shortening - just trim to target duration"""
        target_samples = int(target_duration * sr)
        return audio[:target_samples]
    
    def _intelligent_lengthen(self, audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
        """Simplified lengthening - just add silence padding"""
        target_samples = int(target_duration * sr)
        padding_needed = target_samples - len(audio)
        
        if padding_needed <= 0:
            return audio[:target_samples]
        
        # Add silence to the end
        silence = np.zeros(padding_needed, dtype=audio.dtype)
        return np.concatenate([audio, silence])

