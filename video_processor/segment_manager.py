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

logger = logging.getLogger(__name__)


class SegmentManager:
    """Simplified segment manager for voice cloning"""
    
    def __init__(self, transcription_service):
        self.transcription_service = transcription_service
        self.min_duration = 2.0      
        self.max_duration = 20.0     
        self.optimal_duration = 12.0 
        self.max_gap = 2.0           
        self.words_per_chunk = 45    
        self.ref_min_duration = 2.0
        self.ref_max_duration = 5.0
        self.ref_min_words = 3
        self.ref_max_words = 10
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create segments with 7-17 seconds duration"""
        words = transcript_data.get('words', [])
        if not words:
            return []
        
        segments = self._create_initial_segments(words)
        final_segments = self._process_segments_for_duration(segments)
        
        return final_segments
    
    def _create_initial_segments(self, words: List[Dict]) -> List[Dict]:
        """Create initial segments based on speaker changes and gaps"""
        segments = []
        current_chunk = []
        current_speaker = None
        
        for word in words:
            speaker = word.get('speaker', 'A')
            word_start = word.get('start', 0) / 1000.0
            
            if current_chunk:
                prev_word = current_chunk[-1]
                prev_end = prev_word.get('end', 0) / 1000.0
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
        if not words:
            return None
            
        start_time = words[0]['start'] / 1000.0
        end_time = words[-1]['end'] / 1000.0
        duration = end_time - start_time
        
        if duration < 1.0:
            return None
        
        text = ' '.join(w['text'] for w in words).strip()
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
    
    def _process_segments_for_duration(self, segments: List[Dict]) -> List[Dict]:
        """Process segments to optimize duration and merge short segments aggressively"""
        if not segments:
            return []
        
        processed = []
        i = 0
        
        while i < len(segments):
            current_segment = segments[i]
            
            if current_segment['duration'] < self.min_duration and processed:
                last_segment = processed[-1]
                if (last_segment['speaker'] == current_segment['speaker'] and 
                    current_segment['start'] - last_segment['end'] <= self.max_gap):
                    
                    merged_segment = self._merge_segments(last_segment, current_segment)
                    processed[-1] = merged_segment
                    i += 1
                    continue
            
            if current_segment['duration'] < self.min_duration:
                merged_segment = self._merge_with_next(segments, i)
                if merged_segment and merged_segment['duration'] <= self.max_duration:
                    processed.append(merged_segment)
                    i = self._find_next_index(segments, i, merged_segment)
                    continue
            
            if current_segment['duration'] > self.max_duration:
                split_segments = self._split_segment(current_segment)
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
        
        if current['speaker'] != next_seg['speaker']:
            return None
        
        gap = next_seg['start'] - current['end']
        if gap > self.max_gap:
            return None
            
        return self._merge_segments(current, next_seg)
    
    def _split_segment(self, segment: Dict) -> List[Dict]:
        """Split long segment into optimal-sized parts"""
        duration = segment['duration']
        words = segment.get('words', [])
        
        if not words:
            return [segment]
        
        num_parts = max(1, int(duration / self.optimal_duration))
        words_per_part = len(words) // num_parts
        
        parts = []
        for i in range(num_parts):
            start_idx = i * words_per_part
            if i == num_parts - 1:
                end_idx = len(words)
            else:
                end_idx = (i + 1) * words_per_part
            
            part_words = words[start_idx:end_idx]
            part = self._create_segment(part_words, segment['speaker'])
            if part:
                parts.append(part)
        
        return parts if parts else [segment]
    
    def _find_next_index(self, segments: List[Dict], start_index: int, merged_segment: Dict) -> int:
        """Find next index after merged segments"""
        merged_end = merged_segment['end']
        
        for i in range(start_index, len(segments)):
            if segments[i]['end'] > merged_end:
                return i
        
        return len(segments)
    
    def _merge_segments(self, seg1: Dict, seg2: Dict) -> Dict:
        """Merge two segments"""
        combined_words = seg1.get('words', []) + seg2.get('words', [])
        combined_text = f"{seg1['text']} {seg2['text']}"
        
        return {
            'start': seg1['start'],
            'end': seg2['end'],
            'duration': seg2['end'] - seg1['start'],
            'text': combined_text,
            'speaker': seg1['speaker'],
            'word_count': seg1['word_count'] + seg2['word_count'],
            'confidence': (seg1['confidence'] + seg2['confidence']) / 2,
            'words': combined_words
        }
    
    def _extract_sentence_reference(self, segment: Dict) -> Optional[Dict]:
        """Extract a sentence-based reference from segment"""
        words = segment.get('words', [])
        text = segment.get('text', '')
        
        if not words or not text:
            return None
        
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = sentence.split()
            word_count = len(sentence_words)
            
            if not (self.ref_min_words <= word_count <= self.ref_max_words):
                continue
            
            sentence_word_objects = []
            for word_obj in words:
                if word_obj['text'] in sentence_words:
                    sentence_word_objects.append(word_obj)
                    if len(sentence_word_objects) == word_count:
                        break
            
            if len(sentence_word_objects) < word_count:
                continue
            
            ref_start = sentence_word_objects[0]['start'] / 1000.0
            ref_end = sentence_word_objects[-1]['end'] / 1000.0
            ref_duration = ref_end - ref_start
            
            if self.ref_min_duration <= ref_duration <= self.ref_max_duration:
                return {
                    'start': ref_start,
                    'end': ref_end,
                    'duration': ref_duration,
                    'text': sentence,
                    'speaker': segment['speaker'],
                    'word_count': word_count,
                    'confidence': segment['confidence'],
                    'words': sentence_word_objects,
                    'is_reference': True
                }
        
        return None
    
    def select_optimal_references(self, segments: List[Dict], speakers: List[str]) -> Dict[str, Dict]:
        """Select best reference for each speaker based on sentence completion"""
        references = {}
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s['speaker'] == speaker]
            
            if not speaker_segments:
                continue
            
            best_reference = None
            best_score = 0
            
            for segment in speaker_segments:
                sentence_ref = self._extract_sentence_reference(segment)
                
                if sentence_ref:
                    score = self._calculate_reference_score(sentence_ref)
                    if score > best_score:
                        best_score = score
                        best_reference = sentence_ref
            
            if not best_reference:
                speaker_segments.sort(key=lambda x: abs(x['duration'] - 3.5))
                segment = speaker_segments[0]
                
                target_duration = min(self.ref_max_duration, segment['duration'])
                target_words = min(self.ref_max_words, segment['word_count'])
                
                words = segment['words'][:target_words]
                ref_start = words[0]['start'] / 1000.0
                ref_end = words[-1]['end'] / 1000.0
                
                best_reference = {
                    'start': ref_start,
                    'end': ref_end,
                    'duration': ref_end - ref_start,
                    'text': ' '.join(w['text'] for w in words),
                    'speaker': speaker,
                    'word_count': len(words),
                    'confidence': segment['confidence'],
                    'words': words,
                    'is_reference': True
                }
            
            references[speaker] = best_reference
        
        return references
    
    def _calculate_reference_score(self, reference: Dict) -> float:
        """Calculate score for reference quality"""
        duration_score = 1.0 - abs(reference['duration'] - 3.5) / 3.5
        word_count_score = 1.0 - abs(reference['word_count'] - 6) / 6
        confidence_score = reference['confidence']
        
        return (duration_score * 0.4 + word_count_score * 0.3 + confidence_score * 0.3)
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int,
                            output_dir: Path, speakers: List[str], 
                            target_language: str, detected_language: str):
        """Save segments for voice cloning with sentence-based references"""
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
        
        for i, segment in enumerate(segments):
            speaker = segment.get('speaker', 'A')
            speaker_dir = output_dir / f"speaker_{speaker}"
            segments_dir = speaker_dir / "segments"
            reference_dir = speaker_dir / "reference"
            
            segments_dir.mkdir(parents=True, exist_ok=True)
            reference_dir.mkdir(parents=True, exist_ok=True)
            
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_audio = audio[start_sample:end_sample]
            
            audio_filename = f"segment_{i+1:03d}.wav"
            audio_path = segments_dir / audio_filename
            sf.write(audio_path, segment_audio, sr)
            
            original_text = segment.get('text', '').strip()
            if not original_text:
                original_text = f"Segment {i+1} audio content"
            
            english_text = self.transcription_service.format_dialogue_text(
                original_text, speaker, False
            )
            
            metadata = {
                'segment_index': i + 1,
                'audio_file': audio_filename,
                'audio_path': str(audio_path),
                'original_text': original_text,
                'english_text': english_text,
                'speaker': speaker,
                'speaker_index': ord(speaker) - ord('A') + 1,
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'word_count': segment['word_count'],
                'confidence': segment['confidence'],
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
        
        for speaker, reference in references.items():
            speaker_dir = output_dir / f"speaker_{speaker}"
            reference_dir = speaker_dir / "reference"
            
            ref_start_sample = int(reference['start'] * sr)
            ref_end_sample = int(reference['end'] * sr)
            ref_audio = audio[ref_start_sample:ref_end_sample]
            
            reference_audio_path = reference_dir / f"speaker_{speaker}_REFERENCE.wav"
            sf.write(reference_audio_path, ref_audio, sr)
            
            ref_english_text = self.transcription_service.format_dialogue_text(
                reference['text'], speaker, False
            )
            
            reference_metadata = {
                'speaker': speaker,
                'speaker_index': ord(speaker) - ord('A') + 1,
                'reference_audio': f"speaker_{speaker}_REFERENCE.wav",
                'reference_audio_path': str(reference_audio_path),
                'start': reference['start'],
                'end': reference['end'],
                'duration': reference['duration'],
                'original_text': reference['text'],
                'english_text': ref_english_text,
                'word_count': reference['word_count'],
                'confidence': reference['confidence'],
                'is_reference': True,
                'reference_type': 'sentence_based',
                'selection_score': self._calculate_reference_score(reference)
            }
            
            reference_metadata_path = reference_dir / f"speaker_{speaker}_REFERENCE_metadata.json"
            with open(reference_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(reference_metadata, f, ensure_ascii=False, indent=2) 