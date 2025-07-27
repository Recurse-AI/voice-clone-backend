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
        self.optimal_duration = 10.0
        self.min_duration = 9.0
        self.max_duration = 11.0
        self.silence_threshold = -40  # dB
        self.min_silence_duration = 0.3  # seconds
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create segments ensuring complete audio coverage"""
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
        
        segments = self._create_complete_segments(valid_words, total_duration, first_word_start)
        
        return segments
    
    def _create_complete_segments(self, words: List[Dict], total_duration: float, 
                                 first_word_start: float) -> List[Dict]:
        """Create segments with complete audio coverage"""
        if not words or total_duration <= 0:
            return []
        
        segments = []
        sorted_words = sorted(words, key=lambda w: w.get('start', 0))
        
        # Start from beginning of audio, not first word
        current_start = 0.0
        word_index = 0
        
        while current_start < total_duration:
            # Calculate segment end
            target_end = min(current_start + self.optimal_duration, total_duration)
            
            # Collect words for this segment
            segment_words = []
            segment_end = target_end
            
            # Find words within this time range
            while word_index < len(sorted_words):
                word = sorted_words[word_index]
                word_start = word.get('start', 0) / 1000.0
                word_end = word.get('end', 0) / 1000.0
                
                # Include word if it overlaps with segment
                if word_start < target_end and word_end > current_start:
                    segment_words.append(word)
                    segment_end = max(segment_end, word_end)
                    word_index += 1
                elif word_start >= target_end:
                    break
                else:
                    word_index += 1
            
            # Adjust segment end to maintain optimal length
            if segment_end - current_start > self.max_duration:
                segment_end = current_start + self.max_duration
            elif segment_end - current_start < self.min_duration and current_start + self.min_duration <= total_duration:
                segment_end = min(current_start + self.min_duration, total_duration)
            
            # Create segment
            segment = self._create_segment(segment_words, current_start, segment_end)
            if segment:
                segments.append(segment)
            
            current_start = segment_end
        
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
    
    def adjust_audio_length(self, audio: np.ndarray, sr: int, target_duration: float, 
                          reference_duration: float) -> np.ndarray:
        """Adjust generated audio length to match reference segment"""
        current_duration = len(audio) / sr
        
        if abs(current_duration - target_duration) < 0.1:  # Within 100ms tolerance
            return audio
        
        if current_duration > target_duration:
            return self._shorten_audio(audio, sr, target_duration, current_duration)
        else:
            return self._lengthen_audio(audio, sr, target_duration, current_duration)
    
    def _shorten_audio(self, audio: np.ndarray, sr: int, target_duration: float, 
                      current_duration: float) -> np.ndarray:
        """Shorten audio using intelligent methods"""
        target_samples = int(target_duration * sr)
        
        # Method 1: Remove silence from tail
        silence_removed = self._remove_tail_silence(audio, sr)
        if len(silence_removed) / sr <= target_duration * 1.05:  # Within 5% tolerance
            return silence_removed[:target_samples]
        
        # Method 2: Speed up slightly (0.9-1.1x)
        speed_factor = current_duration / target_duration
        if 0.9 <= speed_factor <= 1.1:
            speed_adjusted = librosa.effects.time_stretch(audio, rate=speed_factor)
            return speed_adjusted[:target_samples]
        
        # Method 3: Simple crop from tail
        return audio[:target_samples]
    
    def _lengthen_audio(self, audio: np.ndarray, sr: int, target_duration: float, 
                       current_duration: float) -> np.ndarray:
        """Lengthen audio using intelligent methods"""
        target_samples = int(target_duration * sr)
        
        # Method 1: Slow down slightly (0.9-1.1x)
        speed_factor = current_duration / target_duration
        if 0.9 <= speed_factor <= 1.1:
            speed_adjusted = librosa.effects.time_stretch(audio, rate=speed_factor)
            return speed_adjusted[:target_samples]
        
        # Method 2: Add silence to tail
        silence_needed = target_samples - len(audio)
        if silence_needed > 0:
            silence = np.zeros(silence_needed)
            return np.concatenate([audio, silence])
        
        return audio[:target_samples]
    
    def _remove_tail_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Remove silence from the end of audio"""
        # Convert to dB
        audio_db = librosa.amplitude_to_db(np.abs(audio))
        
        # Find last non-silent sample
        non_silent_indices = np.where(audio_db > self.silence_threshold)[0]
        
        if len(non_silent_indices) == 0:
            return audio  # All silence, return as is
        
        last_sound = non_silent_indices[-1]
        
        # Add small buffer (100ms) after last sound
        buffer_samples = int(0.1 * sr)
        end_sample = min(len(audio), last_sound + buffer_samples)
        
        return audio[:end_sample]
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int,
                            output_dir: Path, speakers: List[str], 
                            target_language: str, detected_language: str, 
                            original_audio_details: Optional[Dict] = None):
        """Save segments with complete audio coverage"""
        if not segments or audio is None or sr <= 0:
            return
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate original audio duration
        original_duration = len(audio) / sr
        
        # Ensure all segments cover the full audio
        total_segment_duration = sum(s['duration'] for s in segments)
        coverage_percentage = (total_segment_duration / original_duration) * 100
        
        overall_metadata = {
            "total_segments": len(segments),
            "speakers": speakers,
            "target_language": target_language,
            "detected_language": detected_language,
            "processing_timestamp": str(datetime.now()),
            "original_audio": {
                "duration": original_duration,
                "sample_rate": sr,
                "total_samples": len(audio),
                "coverage_percentage": coverage_percentage
            }
        }
        
        if original_audio_details:
            overall_metadata["original_audio"].update(original_audio_details)
        
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_dir / "processing_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(overall_metadata, f, ensure_ascii=False, indent=2)
        
        # Parallel translation
        segment_texts = [s.get('text', '').strip() for s in segments]
        segment_speakers_data = [{
            'speakers': s.get('speakers_in_segment', [s.get('speaker', 'A')]),
            'is_multi_speaker': s.get('is_multi_speaker', False),
            'primary_speaker': s.get('speaker', 'A')
        } for s in segments]
        segment_words_data = [s.get('words', []) for s in segments]
        
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
            
            metadata = {
                'segment_index': i + 1,
                'audio_file': audio_filename,
                'audio_path': str(audio_path),
                'original_text': original_text,
                'english_text': english_text,
                'speaker': segment.get('speaker', 'A'),
                'speakers_in_segment': segment.get('speakers_in_segment', []),
                'is_multi_speaker': segment.get('is_multi_speaker', False),
                'speaker_word_counts': segment.get('speaker_word_counts', {}),
                'start': start_time,
                'end': end_time,
                'duration': segment.get('duration', 0),
                'word_count': segment.get('word_count', 0),
                'confidence': segment.get('confidence', 0.5),
                'cloned_audio_file': f"cloned_segment_{i+1:03d}.wav",
                'processing_status': 'ready_for_cloning'
            }
            
            # Save metadata
            metadata_path = segments_dir / f"segment_{i+1:03d}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Update segment info
            segment.update({
                'segment_index': i + 1,
                'audio_path': str(audio_path),
                'metadata_path': str(metadata_path),
                'audio_file': audio_filename,
                'english_text': english_text,
                'metadata_complete': True
            }) 

    def adjust_generated_audio_length(self, generated_audio: np.ndarray, sr: int, 
                                     reference_segment: Dict[str, Any]) -> np.ndarray:
        """
        Advanced audio length adjustment for TTS output to match reference segment.
        Implements user's specific requirements for audio adjustment.
        """
        target_duration = reference_segment.get('duration', 10.0)
        current_duration = len(generated_audio) / sr
        tolerance = 0.15  # 150ms tolerance
        
        if abs(current_duration - target_duration) <= tolerance:
            return generated_audio
        
        if current_duration > target_duration:
            # Audio is too long - shorten it
            return self._intelligent_shorten(generated_audio, sr, target_duration)
        else:
            # Audio is too short - lengthen it
            return self._intelligent_lengthen(generated_audio, sr, target_duration)
    
    def _intelligent_shorten(self, audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
        """
        Intelligent shortening following user's priority:
        1. Crop silent parts from tail
        2. Use speed adjustment (0.9-1.10x)
        3. Normal crop from tail
        """
        target_samples = int(target_duration * sr)
        current_duration = len(audio) / sr
        
        # Step 1: Remove silence from tail
        trimmed_audio = self._remove_tail_silence(audio, sr)
        trimmed_duration = len(trimmed_audio) / sr
        
        if trimmed_duration <= target_duration:
            return trimmed_audio[:target_samples]
        
        # Step 2: Speed adjustment if within acceptable range
        speed_factor = trimmed_duration / target_duration
        if 0.9 <= speed_factor <= 1.10:
            try:
                speed_adjusted = librosa.effects.time_stretch(trimmed_audio, rate=speed_factor)
                return speed_adjusted[:target_samples]
            except:
                pass  # Fallback to crop if speed adjustment fails
        
        # Step 3: Normal crop from tail
        return trimmed_audio[:target_samples]
    
    def _intelligent_lengthen(self, audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
        """
        Intelligent lengthening (reverse of shortening):
        1. Speed adjustment (0.9-1.10x)
        2. Add silence padding
        """
        target_samples = int(target_duration * sr)
        current_duration = len(audio) / sr
        
        # Step 1: Speed adjustment if within acceptable range
        speed_factor = current_duration / target_duration
        if 0.9 <= speed_factor <= 1.10:
            try:
                speed_adjusted = librosa.effects.time_stretch(audio, rate=speed_factor)
                if len(speed_adjusted) >= target_samples:
                    return speed_adjusted[:target_samples]
                audio = speed_adjusted
            except:
                pass  # Fallback to padding if speed adjustment fails
        
        # Step 2: Add silence padding
        padding_needed = target_samples - len(audio)
        if padding_needed > 0:
            # Add silence to tail
            silence = np.zeros(padding_needed, dtype=audio.dtype)
            return np.concatenate([audio, silence])
        
        return audio[:target_samples]

    def process_hindi_transcript(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process Hindi transcript ensuring complete coverage and optimal segment lengths.
        Specifically designed for the provided Hindi audio data.
        """
        if not transcript_data:
            return []
        
        # Extract key information
        words = transcript_data.get('words', [])
        audio_duration = transcript_data.get('audio_duration', 0)
        
        if not words:
            return []
        
        # Calculate actual content duration
        content_start = min(w['start'] for w in words) / 1000.0  # Convert ms to seconds
        content_end = max(w['end'] for w in words) / 1000.0
        content_duration = content_end - content_start
        
        # Use the full audio duration to avoid losing any content
        total_duration = max(audio_duration, content_end)
        
        segments = self._create_complete_segments(words, total_duration, content_start)
        
        # Validate coverage
        total_segment_duration = sum(s['duration'] for s in segments)
        coverage_ratio = total_segment_duration / total_duration
        
        # Ensure 9-11 second segments as requested
        optimized_segments = []
        for segment in segments:
            duration = segment['duration']
            if duration < self.min_duration or duration > self.max_duration:
                # Split or merge segments if needed
                if duration > self.max_duration:
                    split_segments = self._split_long_segment(segment)
                    optimized_segments.extend(split_segments)
                elif duration < self.min_duration and optimized_segments:
                    # Try to merge with previous segment
                    merged = self._try_merge_segments(optimized_segments[-1], segment)
                    if merged:
                        optimized_segments[-1] = merged
                    else:
                        optimized_segments.append(segment)
                else:
                    optimized_segments.append(segment)
            else:
                optimized_segments.append(segment)
        
        return optimized_segments
    
    def _split_long_segment(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a segment that's longer than max_duration into smaller segments"""
        words = segment.get('words', [])
        if not words:
            return [segment]
        
        start_time = segment['start']
        end_time = segment['end']
        duration = segment['duration']
        
        # Calculate number of sub-segments needed
        num_segments = max(2, int(duration / self.optimal_duration))
        segment_duration = duration / num_segments
        
        sub_segments = []
        current_start = start_time
        word_index = 0
        
        for i in range(num_segments):
            sub_end = min(current_start + segment_duration, end_time)
            if i == num_segments - 1:  # Last segment gets remaining time
                sub_end = end_time
            
            # Collect words for this sub-segment
            sub_words = []
            while word_index < len(words):
                word = words[word_index]
                word_start = word.get('start', 0) / 1000.0
                if word_start < sub_end:
                    sub_words.append(word)
                    word_index += 1
                else:
                    break
            
            sub_segment = self._create_segment(sub_words, current_start, sub_end)
            if sub_segment:
                sub_segments.append(sub_segment)
            
            current_start = sub_end
        
        return sub_segments
    
    def _try_merge_segments(self, segment1: Dict[str, Any], segment2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try to merge two adjacent segments if the result is within acceptable duration"""
        combined_duration = segment1['duration'] + segment2['duration']
        
        if combined_duration > self.max_duration:
            return None
        
        # Merge the segments
        combined_words = segment1.get('words', []) + segment2.get('words', [])
        merged_segment = self._create_segment(combined_words, segment1['start'], segment2['end'])
        
        return merged_segment 