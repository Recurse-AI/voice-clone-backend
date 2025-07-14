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

logger = logging.getLogger(__name__)


class SegmentManager:
    """Simplified segment manager for voice cloning"""
    
    def __init__(self, transcription_service):
        self.transcription_service = transcription_service
        self.min_duration = 2.0      # Minimum 2 seconds
        self.max_duration = 20.0     # Maximum 20 seconds  
        self.optimal_duration = 11.0 # Optimal 11 seconds
        self.max_gap = 3.0           # Maximum gap tolerance
        self.words_per_chunk = 40    # Words per chunk
    
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
        """Process segments to optimize for 11s duration, allow 2s minimum"""
        if not segments:
            return []
        
        processed = []
        i = 0
        
        while i < len(segments):
            current_segment = segments[i]
            
            if current_segment['duration'] < self.optimal_duration:
                merged_segment = self._merge_with_next(segments, i)
                if merged_segment and merged_segment['duration'] <= self.max_duration:
                    processed.append(merged_segment)
                    i = self._find_next_index(segments, i, merged_segment)
                else:
                    processed.append(current_segment)
                    i += 1
            elif current_segment['duration'] > self.max_duration:
                split_segments = self._split_segment(current_segment)
                processed.extend(split_segments)
                i += 1
            else:
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
    
    def select_optimal_references(self, segments: List[Dict], speakers: List[str]) -> Dict[str, Dict]:
        """Select best reference for each speaker"""
        references = {}
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s['speaker'] == speaker]
            
            if not speaker_segments:
                continue
            
            # Sort by duration (prefer longer segments)
            speaker_segments.sort(key=lambda x: x['duration'], reverse=True)
            
            # Select segment closest to optimal duration
            best_segment = None
            for segment in speaker_segments:
                if self.min_duration <= segment['duration'] <= self.max_duration:
                    best_segment = segment
                    break
            
            if not best_segment:
                best_segment = speaker_segments[0]
            
            references[speaker] = best_segment
        
        return references
    
    def save_optimal_segments(self, segments: List[Dict], audio: np.ndarray, sr: int,
                            output_dir: Path, speakers: List[str], 
                            target_language: str, detected_language: str):
        """Save segments for voice cloning with consistent metadata"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        references = self.select_optimal_references(segments, speakers)
        
        # Save overall metadata
        overall_metadata = {
            "total_segments": len(segments),
            "speakers": speakers,
            "target_language": target_language,
            "detected_language": detected_language,
            "processing_timestamp": str(datetime.now()),
            "reference_selections": {k: v['segment_index'] if 'segment_index' in v else 0 for k, v in references.items()}
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
            
            # Extract audio segment
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Save audio
            audio_filename = f"segment_{i+1:03d}.wav"
            audio_path = segments_dir / audio_filename
            sf.write(audio_path, segment_audio, sr)
            
            # Process text with OpenAI formatting - ensure it always works
            original_text = segment.get('text', '').strip()
            if not original_text:
                original_text = f"Segment {i+1} audio content"
            
            try:
                english_text = self.transcription_service.format_dialogue_text(
                    original_text, speaker, len(speakers) > 1
                )
                # Validate the formatted text
                if not english_text or not english_text.strip():
                    raise ValueError("Empty formatted text")
                if '[S' not in english_text:
                    raise ValueError("Missing speaker tags")
            except Exception as e:
                # Create fallback English text with proper formatting
                speaker_num = ord(speaker) - ord('A') + 1
                words = original_text.split()
                lines = []
                current_line = []
                
                for word in words:
                    current_line.append(word)
                    if len(current_line) >= 10:
                        lines.append(f"[S{speaker_num}] {' '.join(current_line)}")
                        current_line = []
                
                if current_line:
                    lines.append(f"[S{speaker_num}] {' '.join(current_line)}")
                
                english_text = '\n'.join(lines) if lines else f"[S{speaker_num}] {original_text}"
            
            # Create comprehensive metadata
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
                'is_reference': speaker in references and references[speaker] == segment,
                'cloned_audio_file': f"cloned_segment_{i+1:03d}.wav",
                'cloned_audio_path': str(segments_dir / f"cloned_segment_{i+1:03d}.wav"),
                'metadata_complete': True,
                'processing_status': 'ready_for_cloning'
            }
            
            # Save metadata
            metadata_path = segments_dir / f"segment_{i+1:03d}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Save reference if selected
            if speaker in references and references[speaker] == segment:
                reference_audio_path = reference_dir / f"speaker_{speaker}_REFERENCE.wav"
                sf.write(reference_audio_path, segment_audio, sr)
                
                reference_metadata = {
                    'speaker': speaker,
                    'speaker_index': ord(speaker) - ord('A') + 1,
                    'reference_audio': f"speaker_{speaker}_REFERENCE.wav",
                    'reference_audio_path': str(reference_audio_path),
                    'start': segment['start'],
                    'end': segment['end'],
                    'duration': segment['duration'],
                    'original_text': original_text,
                    'english_text': english_text,
                    'word_count': segment['word_count'],
                    'confidence': segment['confidence'],
                    'is_reference': True,
                    'reference_for_segments': [i + 1]
                }
                
                reference_metadata_path = reference_dir / f"speaker_{speaker}_REFERENCE_metadata.json"
                with open(reference_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(reference_metadata, f, ensure_ascii=False, indent=2)
            
            # Update segment with paths
            segment.update({
                'segment_index': i + 1,
                'audio_path': str(audio_path),
                'metadata_path': str(metadata_path),
                'audio_file': audio_filename,
                'english_text': english_text,
                'metadata_complete': True
            }) 