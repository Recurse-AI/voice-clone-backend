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
import re
import time

logger = logging.getLogger(__name__)


class SegmentManager:
    """Simplified segment manager for voice cloning"""
    
    def __init__(self, transcription_service):
        self.transcription_service = transcription_service
        self.min_duration = 3.0      # Minimum segment duration (aligned with reference)
        self.max_duration = 15.0     # Maximum segment duration
        self.optimal_duration = 8.0  # Optimal segment duration (middle of reference range)
        self.max_gap = 2.0           # Maximum gap between words to keep in same segment
        self.words_per_chunk = 30    # Approximate words per chunk (reduced for shorter segments)
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create segments with optimal duration for voice cloning"""
        if not transcript_data:
            logger.error("transcript_data is None or empty")
            return []
            
        words = transcript_data.get('words', [])
        if not words or not isinstance(words, list):
            logger.error("No words found in transcript_data or words is not a list")
            return []
        
        # Filter out None words
        words = [word for word in words if word is not None and isinstance(word, dict)]
        if not words:
            logger.error("No valid words found after filtering")
            return []
        
        segments = self._create_initial_segments(words)
        final_segments = self._process_segments_for_duration(segments)
        
        return final_segments
    
    def _create_initial_segments(self, words: List[Dict]) -> List[Dict]:
        """Create initial segments based on speaker changes and gaps"""
        if not words:
            return []
            
        segments = []
        current_chunk = []
        current_speaker = None
        
        for word in words:
            if not word or not isinstance(word, dict):
                continue
                
            speaker = word.get('speaker', 'A')
            word_start = word.get('start')
            
            if word_start is None:
                continue
                
            word_start = word_start / 1000.0
            
            if current_chunk:
                prev_word = current_chunk[-1]
                if not prev_word or not isinstance(prev_word, dict):
                    current_chunk = [word]
                    current_speaker = speaker
                    continue
                    
                prev_end = prev_word.get('end')
                if prev_end is None:
                    current_chunk.append(word)
                    continue
                    
                prev_end = prev_end / 1000.0
                gap = word_start - prev_end
                
                should_split = (
                    (speaker != current_speaker) or
                    (gap > self.max_gap) or
                    (len(current_chunk) >= self.words_per_chunk)
                )
                
                if should_split:
                    segment = self._create_segment(current_chunk, current_speaker or 'A')
                    if segment:
                        segments.append(segment)
                    current_chunk = [word]
                    current_speaker = speaker
                else:
                    current_chunk.append(word)
            else:
                current_chunk.append(word)
                current_speaker = speaker
        
        if current_chunk:
            segment = self._create_segment(current_chunk, current_speaker or 'A')
            if segment:
                segments.append(segment)
        
        return segments
    
    def _create_segment(self, words: List[Dict], speaker: str) -> Optional[Dict]:
        """Create a segment from words"""
        if not words or not isinstance(words, list):
            return None
            
        # Filter out None words and validate structure
        valid_words = []
        for word in words:
            if (word and isinstance(word, dict) and 
                'start' in word and 'end' in word and 
                'text' in word and 
                word['start'] is not None and 
                word['end'] is not None and 
                word['text'] is not None):
                valid_words.append(word)
        
        if not valid_words:
            return None
            
        start_time = valid_words[0]['start'] / 1000.0
        end_time = valid_words[-1]['end'] / 1000.0
        duration = end_time - start_time
        
        if duration < 1.0:
            return None
        
        text = ' '.join(w['text'] for w in valid_words if w.get('text')).strip()
        if not text:
            return None
            
        confidence = np.mean([w.get('confidence', 0.5) for w in valid_words if w.get('confidence') is not None])
        
        return {
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'text': text,
            'speaker': speaker,
            'word_count': len(valid_words),
            'confidence': confidence,
            'words': valid_words
        }
    
    def _process_segments_for_duration(self, segments: List[Dict]) -> List[Dict]:
        """Process segments to optimize duration and merge short segments aggressively"""
        if not segments:
            return []
        
        processed = []
        i = 0
        
        while i < len(segments):
            current_segment = segments[i]
            
            if not current_segment or not isinstance(current_segment, dict):
                i += 1
                continue
                
            if current_segment.get('duration', 0) < self.min_duration and processed:
                last_segment = processed[-1]
                if (last_segment and 
                    last_segment.get('speaker') == current_segment.get('speaker') and 
                    current_segment.get('start', 0) - last_segment.get('end', 0) <= self.max_gap):
                    
                    merged_segment = self._merge_segments(last_segment, current_segment)
                    if merged_segment:
                        processed[-1] = merged_segment
                    i += 1
                    continue
            
            if current_segment.get('duration', 0) < self.min_duration:
                merged_segment = self._merge_with_next(segments, i)
                if merged_segment and merged_segment.get('duration', 0) <= self.max_duration:
                    processed.append(merged_segment)
                    i = self._find_next_index(segments, i, merged_segment)
                    continue
            
            if current_segment.get('duration', 0) > self.max_duration:
                split_segments = self._split_segment(current_segment)
                if split_segments:
                    processed.extend(split_segments)
                i += 1
                continue
            
            processed.append(current_segment)
            i += 1
        
        return processed
    
    def _merge_with_next(self, segments: List[Dict], index: int) -> Optional[Dict]:
        """Merge segment with next segment if same speaker"""
        if index + 1 >= len(segments):
            return None
            
        current = segments[index]
        next_seg = segments[index + 1]
        
        if (not current or not next_seg or 
            not isinstance(current, dict) or not isinstance(next_seg, dict)):
            return None
        
        if current.get('speaker') != next_seg.get('speaker'):
            return None
        
        current_end = current.get('end', 0)
        next_start = next_seg.get('start', 0)
        
        gap = next_start - current_end
        if gap > self.max_gap:
            return None
            
        return self._merge_segments(current, next_seg)
    
    def _split_segment(self, segment: Dict) -> List[Dict]:
        """Split long segment into optimal-sized parts"""
        if not segment or not isinstance(segment, dict):
            return []
            
        duration = segment.get('duration', 0)
        words = segment.get('words', [])
        
        if not words or not isinstance(words, list):
            return [segment]
        
        # Filter valid words
        valid_words = [w for w in words if w and isinstance(w, dict)]
        if not valid_words:
            return [segment]
        
        num_parts = max(1, int(duration / self.optimal_duration))
        words_per_part = len(valid_words) // num_parts
        
        parts = []
        for i in range(num_parts):
            start_idx = i * words_per_part
            if i == num_parts - 1:
                end_idx = len(valid_words)
            else:
                end_idx = (i + 1) * words_per_part
            
            part_words = valid_words[start_idx:end_idx]
            if part_words:
                part = self._create_segment(part_words, segment.get('speaker', 'A'))
                if part:
                    parts.append(part)
        
        return parts if parts else [segment]
    
    def _find_next_index(self, segments: List[Dict], start_index: int, merged_segment: Dict) -> int:
        """Find next index after merged segments"""
        if not merged_segment or not isinstance(merged_segment, dict):
            return start_index + 1
            
        merged_end = merged_segment.get('end', 0)
        
        for i in range(start_index, len(segments)):
            if segments[i] and segments[i].get('end', 0) > merged_end:
                return i
        
        return len(segments)
    
    def _merge_segments(self, seg1: Dict, seg2: Dict) -> Optional[Dict]:
        """Merge two segments"""
        if (not seg1 or not seg2 or 
            not isinstance(seg1, dict) or not isinstance(seg2, dict)):
            return None
            
        seg1_words = seg1.get('words', [])
        seg2_words = seg2.get('words', [])
        
        if not isinstance(seg1_words, list):
            seg1_words = []
        if not isinstance(seg2_words, list):
            seg2_words = []
            
        combined_words = seg1_words + seg2_words
        seg1_text = seg1.get('text', '')
        seg2_text = seg2.get('text', '')
        
        combined_text = f"{seg1_text} {seg2_text}".strip()
        
        return {
            'start': seg1.get('start', 0),
            'end': seg2.get('end', 0),
            'duration': seg2.get('end', 0) - seg1.get('start', 0),
            'text': combined_text,
            'speaker': seg1.get('speaker', 'A'),
            'word_count': seg1.get('word_count', 0) + seg2.get('word_count', 0),
            'confidence': (seg1.get('confidence', 0.5) + seg2.get('confidence', 0.5)) / 2,
            'words': combined_words
        }
    

    
    def select_optimal_references(self, segments: List[Dict], speakers: List[str]) -> Dict[str, Dict]:
        """Select best reference for each speaker - simple selection"""
        if not segments or not speakers:
            return {}
            
        references = {}
        
        for speaker in speakers:
            if not speaker:
                continue
                
            # Find all segments for this speaker
            speaker_segments = [s for s in segments if s and s.get('speaker') == speaker]
            
            if not speaker_segments:
                continue
            
            # Simple selection: pick segment closest to 7 seconds with good confidence
            best_segment = max(speaker_segments, 
                             key=lambda s: s.get('confidence', 0.5) - abs(s.get('duration', 0) - 7.0) * 0.1)
            
            # Create reference from the best segment
            references[speaker] = {
                'start': best_segment.get('start', 0),
                'end': best_segment.get('end', 0),
                'duration': best_segment.get('duration', 0),
                'text': best_segment.get('text', '').strip(),
                'speaker': speaker,
                'word_count': best_segment.get('word_count', 0),
                'confidence': best_segment.get('confidence', 0.5),
                'words': best_segment.get('words', []),
                'is_reference': True,
                'reference_type': 'best_segment'
            }
        
        return references


    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int,
                            output_dir: Path, speakers: List[str], 
                            target_language: str, detected_language: str):
        """Save segments for voice cloning with sentence-based references and parallel translation"""
        if not segments or audio is None or sr <= 0:
            logger.error("Invalid input parameters for save_optimal_segments")
            return
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        references = self.select_optimal_references(segments, speakers)
        
        overall_metadata = {
            "total_segments": len(segments),
            "speakers": speakers,
            "target_language": target_language,
            "detected_language": detected_language,
            "processing_timestamp": str(datetime.now()),
            "reference_selections": {k: f"{k}_sentence_ref" for k in references.keys()}
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
            reference_dir = speaker_dir / "reference"
            
            segments_dir.mkdir(parents=True, exist_ok=True)
            reference_dir.mkdir(parents=True, exist_ok=True)
            
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
                'is_reference': False,
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
        
        # Process references with parallel translation
        reference_texts = []
        reference_speakers = []
        for speaker, reference in references.items():
            if reference and isinstance(reference, dict):
                reference_texts.append(reference.get('text', ''))
                reference_speakers.append(speaker)
        
        if reference_texts:
            print("Starting parallel reference translation...")
            ref_translation_start = time.time()
            reference_english_texts = self.transcription_service.format_dialogue_batch(reference_texts, reference_speakers)
            ref_translation_time = time.time() - ref_translation_start
            print(f"Reference translation completed in {ref_translation_time:.2f} seconds")
        else:
            reference_english_texts = []
        
        ref_index = 0
        for speaker, reference in references.items():
            if not reference or not isinstance(reference, dict):
                continue
                
            speaker_dir = output_dir / f"speaker_{speaker}"
            reference_dir = speaker_dir / "reference"
            
            ref_start = reference.get('start', 0)
            ref_end = reference.get('end', 0)
            
            ref_start_sample = int(ref_start * sr)
            ref_end_sample = int(ref_end * sr)
            
            if ref_start_sample >= len(audio) or ref_end_sample > len(audio) or ref_start_sample >= ref_end_sample:
                continue
                
            ref_audio = audio[ref_start_sample:ref_end_sample]
            
            reference_audio_path = reference_dir / f"speaker_{speaker}_REFERENCE.wav"
            sf.write(reference_audio_path, ref_audio, sr)
            
            ref_english_text = reference_english_texts[ref_index] if ref_index < len(reference_english_texts) else ""
            ref_index += 1
            
            if not ref_english_text:
                print(f"Warning: Reference translation failed for speaker {speaker}, using original text")
                ref_english_text = reference.get('text', '')
            
            reference_metadata = {
                'speaker': speaker,
                'speaker_index': ord(speaker) - ord('A') + 1,
                'reference_audio': f"speaker_{speaker}_REFERENCE.wav",
                'reference_audio_path': str(reference_audio_path),
                'start': ref_start,
                'end': ref_end,
                'duration': reference.get('duration', 0),
                'original_text': reference.get('text', ''),
                'english_text': ref_english_text,
                'word_count': reference.get('word_count', 0),
                'confidence': reference.get('confidence', 0.5),
                'is_reference': True,
                'reference_type': 'best_segment_based'
            }
            
            reference_metadata_path = reference_dir / f"speaker_{speaker}_REFERENCE_metadata.json"
            with open(reference_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(reference_metadata, f, ensure_ascii=False, indent=2) 