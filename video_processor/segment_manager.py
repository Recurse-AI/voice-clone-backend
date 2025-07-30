"""
Segment Manager Module - Enhanced Smart Segmentation
Handles multiple speakers, silent parts, optimal segment length, and maintains order
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import soundfile as sf

from utils import local_storage

logger = logging.getLogger(__name__)


class SegmentManager:
    """Smart segment manager for optimal audio segmentation"""
    
    def __init__(self, transcription_service):
        self.transcription_service = transcription_service
        
        # Segmentation parameters
        self.min_segment_duration = 3.0    # Minimum segment length (seconds)
        self.max_segment_duration = 11.0   # Maximum segment length (seconds)
        self.optimal_min_duration = 9.0    # Optimal minimum length (seconds)
        self.optimal_max_duration = 11.0   # Optimal maximum length (seconds)
        self.min_silence_duration = 0.5    # Minimum silence to consider (seconds)
        self.speaker_switch_padding = 0.1   # Padding around speaker switches (seconds)
        
    def create_optimal_segments(self, transcript_data: Dict[str, Any], 
                              target_language: str = "English", audio_id: str = None) -> List[Dict[str, Any]]:
        """Create optimal segments from transcript data with smart segmentation"""
        try:
            logger.info("Starting smart segmentation process")
            
            # Store target language for translation
            self._target_language = target_language
            self._audio_id = audio_id
            
            utterances = transcript_data.get("utterances", [])
            speakers = transcript_data.get("speakers", [])
            total_duration = transcript_data.get("audio_duration", 0)
            
            if not utterances:
                logger.warning("No utterances found, creating single segment")
                fallback_segments = self._create_fallback_segments(total_duration, speakers)
                return self._translate_segments_if_needed(fallback_segments, transcript_data, audio_id)
            
            # Step 1: Detect silent parts and speaker boundaries
            timeline_events = self._create_timeline_events(utterances, total_duration)
            
            # Step 2: Create speaker segments with silence handling
            speaker_segments = self._create_speaker_segments(timeline_events, total_duration)
            
            # Step 3: Optimize segment lengths (9-11s)
            optimized_segments = self._optimize_segment_lengths(speaker_segments)
            
            # Step 4: Ensure no audio loss and maintain order
            final_segments = self._ensure_complete_coverage(optimized_segments, total_duration)
            
            # Step 5: Add metadata and validate
            validated_segments = self._validate_and_add_metadata(final_segments, utterances, speakers)
            
            # Step 6: Translate segments for dubbing
            translated_segments = self._translate_segments_if_needed(
                validated_segments, transcript_data, audio_id
            )
            
            logger.info(f"Smart segmentation completed: {len(translated_segments)} segments created")
            
            return translated_segments
            
        except Exception as e:
            logger.error(f"Smart segmentation failed: {str(e)}")
            # Fallback to basic segmentation
            fallback_segments = self._create_fallback_segments(
                transcript_data.get("audio_duration", 0), 
                transcript_data.get("speakers", [])
            )
            # Apply translation to fallback segments too
            try:
                return self._translate_segments_if_needed(fallback_segments, transcript_data, audio_id)
            except:
                return fallback_segments
    
    def _create_timeline_events(self, utterances: List[Dict[str, Any]], 
                              total_duration: float) -> List[Dict[str, Any]]:
        """Create timeline events from utterances to detect speakers and silences"""
        events = []
        
        # Add start event
        events.append({
            "time": 0.0,
            "type": "start",
            "speaker": None
        })
        
        # Process utterances
        for utterance in utterances:
            start_time = utterance["start"]
            end_time = utterance["end"]
            speaker = utterance["speaker"]
            
            # Add utterance start event
            events.append({
                "time": start_time,
                "type": "utterance_start",
                "speaker": speaker,
                "utterance": utterance
            })
            
            # Add utterance end event
            events.append({
                "time": end_time,
                "type": "utterance_end",
                "speaker": speaker,
                "utterance": utterance
            })
        
        # Add end event
        events.append({
            "time": total_duration,
            "type": "end",
            "speaker": None
        })
        
        # Sort events by time
        events.sort(key=lambda x: (x["time"], x["type"] == "utterance_start"))
        
        return events
    
    def _create_speaker_segments(self, timeline_events: List[Dict[str, Any]], 
                               total_duration: float) -> List[Dict[str, Any]]:
        """Create speaker segments with silence handling"""
        segments = []
        current_segment = None
        last_utterance_end = 0.0
        
        for i, event in enumerate(timeline_events):
            event_time = event["time"]
            event_type = event["type"]
            
            # Check for silence gap before this event
            if event_time > last_utterance_end + self.min_silence_duration:
                silence_duration = event_time - last_utterance_end
                
                # Add silence segment if significant
                if silence_duration >= self.min_silence_duration:
                    silence_segment = {
                        "start": last_utterance_end,
                        "end": event_time,
                        "duration": silence_duration,
                        "speaker": "SILENCE",
                        "type": "silence",
                        "text": "",
                        "utterances": [],
                        "confidence": 1.0
                    }
                    segments.append(silence_segment)
            
            if event_type == "utterance_start":
                speaker = event["speaker"]
                utterance = event["utterance"]
                
                # Start new segment or continue existing one
                if (current_segment is None or 
                    current_segment["speaker"] != speaker or 
                    event_time > current_segment["end"] + self.speaker_switch_padding):
                    
                    # Finish previous segment
                    if current_segment:
                        segments.append(current_segment)
                    
                    # Start new segment
                    current_segment = {
                        "start": utterance["start"],
                        "end": utterance["end"],
                        "duration": utterance["end"] - utterance["start"],
                        "speaker": speaker,
                        "type": "speech",
                        "text": utterance["text"],
                        "utterances": [utterance],
                        "confidence": utterance["confidence"]
                    }
                else:
                    # Extend existing segment
                    current_segment["end"] = utterance["end"]
                    current_segment["duration"] = current_segment["end"] - current_segment["start"]
                    current_segment["text"] += " " + utterance["text"]
                    current_segment["utterances"].append(utterance)
                    current_segment["confidence"] = (current_segment["confidence"] + utterance["confidence"]) / 2
                
                last_utterance_end = max(last_utterance_end, utterance["end"])
        
        # Add final segment
        if current_segment:
            segments.append(current_segment)
        
        # Add final silence if exists
        if last_utterance_end < total_duration - self.min_silence_duration:
            final_silence = {
                "start": last_utterance_end,
                "end": total_duration,
                "duration": total_duration - last_utterance_end,
                "speaker": "SILENCE",
                "type": "silence",
                "text": "",
                "utterances": [],
                "confidence": 1.0
            }
            segments.append(final_silence)
        
        return segments
    
    def _optimize_segment_lengths(self, speaker_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize segment lengths to be between 9-11 seconds"""
        optimized_segments = []
        
        for segment in speaker_segments:
            duration = segment["duration"]
            
            # Skip silence segments or segments that are already optimal
            if (segment["type"] == "silence" or 
                (self.optimal_min_duration <= duration <= self.optimal_max_duration)):
                optimized_segments.append(segment)
                continue
            
            # Handle segments that are too short
            if duration < self.min_segment_duration:
                # Try to merge with adjacent segments or extend
                optimized_segments.append(segment)  # Keep as is for now, merge later
                continue
            
            # Handle segments that are too long - split them
            if duration > self.max_segment_duration:
                split_segments = self._split_long_segment(segment)
                optimized_segments.extend(split_segments)
            else:
                optimized_segments.append(segment)
        
        # Second pass: merge short segments
        return self._merge_short_segments(optimized_segments)
    
    def _split_long_segment(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a long segment into optimal-sized chunks"""
        if segment["type"] == "silence":
            return [segment]  # Don't split silence
        
        total_duration = segment["duration"]
        target_duration = self.optimal_max_duration
        
        # Calculate number of segments needed
        num_segments = max(2, int(np.ceil(total_duration / target_duration)))
        actual_segment_duration = total_duration / num_segments
        
        split_segments = []
        utterances = segment["utterances"]
        
        for i in range(num_segments):
            segment_start = segment["start"] + (i * actual_segment_duration)
            segment_end = segment["start"] + ((i + 1) * actual_segment_duration)
            
            # Adjust last segment to exact end
            if i == num_segments - 1:
                segment_end = segment["end"]
            
            # Find utterances for this segment
            segment_utterances = []
            segment_text_parts = []
            
            for utterance in utterances:
                # Check if utterance overlaps with this segment
                if (utterance["start"] < segment_end and utterance["end"] > segment_start):
                    # Calculate overlap
                    overlap_start = max(utterance["start"], segment_start)
                    overlap_end = min(utterance["end"], segment_end)
                    overlap_duration = overlap_end - overlap_start
                    
                    # Include if significant overlap (>50% of utterance or >1 second)
                    utterance_duration = utterance["end"] - utterance["start"]
                    if (overlap_duration > utterance_duration * 0.5 or overlap_duration > 1.0):
                        segment_utterances.append(utterance)
                        segment_text_parts.append(utterance["text"])
            
            # Create segment
            split_segment = {
                "start": segment_start,
                "end": segment_end,
                "duration": segment_end - segment_start,
                "speaker": segment["speaker"],
                "type": segment["type"],
                "text": " ".join(segment_text_parts) if segment_text_parts else f"[Part {i+1} of {num_segments}]",
                "utterances": segment_utterances,
                "confidence": segment["confidence"],
                "is_split": True,
                "split_index": i,
                "total_splits": num_segments
            }
            
            split_segments.append(split_segment)
        
        return split_segments
    
    def _merge_short_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge short segments with adjacent segments"""
        if not segments:
            return segments
        
        merged_segments = []
        i = 0
        
        while i < len(segments):
            current_segment = segments[i].copy()
            
            # Skip if segment is already optimal or is silence
            if (current_segment["duration"] >= self.min_segment_duration or 
                current_segment["type"] == "silence"):
                merged_segments.append(current_segment)
                i += 1
                continue
            
            # Try to merge with next segment
            merged = False
            if i + 1 < len(segments):
                next_segment = segments[i + 1]
                
                # Can merge if same speaker or if total duration is reasonable
                if (current_segment["speaker"] == next_segment["speaker"] or
                    current_segment["duration"] + next_segment["duration"] <= self.max_segment_duration):
                    
                    # Merge segments
                    merged_segment = {
                        "start": current_segment["start"],
                        "end": next_segment["end"],
                        "duration": next_segment["end"] - current_segment["start"],
                        "speaker": current_segment["speaker"] if current_segment["type"] != "silence" else next_segment["speaker"],
                        "type": "speech" if (current_segment["type"] == "speech" or next_segment["type"] == "speech") else "silence",
                        "text": f"{current_segment['text']} {next_segment['text']}".strip(),
                        "utterances": current_segment["utterances"] + next_segment["utterances"],
                        "confidence": (current_segment["confidence"] + next_segment["confidence"]) / 2,
                        "is_merged": True
                    }
                    
                    merged_segments.append(merged_segment)
                    i += 2  # Skip next segment as it's merged
                    merged = True
            
            if not merged:
                # Keep as is
                merged_segments.append(current_segment)
                i += 1
        
        return merged_segments
    
    def _ensure_complete_coverage(self, segments: List[Dict[str, Any]], 
                                 total_duration: float) -> List[Dict[str, Any]]:
        """Ensure no audio loss and complete timeline coverage"""
        if not segments:
            return segments
        
        # Sort segments by start time
        segments.sort(key=lambda x: x["start"])
        
        complete_segments = []
        last_end = 0.0
        
        for segment in segments:
            # Fill gap before segment if exists
            if segment["start"] > last_end + 0.01:  # 10ms tolerance
                gap_duration = segment["start"] - last_end
                
                if gap_duration >= 0.1:  # Only fill gaps >= 100ms
                    gap_segment = {
                        "start": last_end,
                        "end": segment["start"],
                        "duration": gap_duration,
                        "speaker": "UNKNOWN",
                        "type": "gap",
                        "text": "",
                        "utterances": [],
                        "confidence": 0.5,
                        "is_gap_fill": True
                    }
                    complete_segments.append(gap_segment)
            
            complete_segments.append(segment)
            last_end = segment["end"]
        
        # Fill final gap if exists
        if last_end < total_duration - 0.01:
            final_gap = {
                "start": last_end,
                "end": total_duration,
                "duration": total_duration - last_end,
                "speaker": "UNKNOWN",
                "type": "gap",
                "text": "",
                "utterances": [],
                "confidence": 0.5,
                "is_gap_fill": True
            }
            complete_segments.append(final_gap)
        
        return complete_segments
    
    def _validate_and_add_metadata(self, segments: List[Dict[str, Any]], 
                                  utterances: List[Dict[str, Any]], 
                                  speakers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate segments and add metadata"""
        validated_segments = []
        
        for i, segment in enumerate(segments):
            # Add segment index
            segment["segment_index"] = i + 1
            segment["segment_id"] = f"segment_{i+1:03d}"
            
            # Add speaker metadata
            speaker_id = segment["speaker"]
            speaker_info = next((s for s in speakers if s["id"] == speaker_id), None)
            
            if speaker_info:
                segment["speaker_label"] = speaker_info.get("label", f"Speaker {speaker_id}")
                segment["speaker_confidence"] = speaker_info.get("confidence", segment.get("confidence", 0.9))
            else:
                segment["speaker_label"] = f"Speaker {speaker_id}"
                segment["speaker_confidence"] = segment.get("confidence", 0.9)
            
            # Validate timing
            if segment["end"] <= segment["start"]:
                logger.warning(f"Invalid segment timing at index {i}, adjusting")
                segment["end"] = segment["start"] + 0.5  # Set minimum 0.5s instead of 0.1s
                segment["duration"] = 0.5
            
            # Ensure minimum duration for ALL segments (not just speech)
            min_required_duration = 0.5  # Minimum 0.5 seconds for any segment
            if segment["duration"] < min_required_duration:
                logger.info(f"Extending short {segment.get('type', 'unknown')} segment from {segment['duration']:.2f}s to {min_required_duration}s")
                # Extend all short segments to minimum duration instead of skipping
                segment["duration"] = min_required_duration
                segment["end"] = segment["start"] + min_required_duration
            
            validated_segments.append(segment)
        
        return validated_segments
    
    def _create_fallback_segments(self, total_duration: float, 
                                 speakers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create fallback segments when smart segmentation fails"""
        logger.warning("Creating fallback segments")
        
        segments = []
        segment_duration = min(self.optimal_max_duration, max(self.min_segment_duration, total_duration / 10))
        
        num_segments = int(np.ceil(total_duration / segment_duration))
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, total_duration)
            
            # Use first speaker or create default
            speaker = speakers[0] if speakers else {"id": "A", "label": "Speaker A"}
            
            segment = {
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time,
                "speaker": speaker["id"],
                "speaker_label": speaker.get("label", f"Speaker {speaker['id']}"),
                "type": "speech",
                "text": f"Segment {i+1}",
                "utterances": [],
                "confidence": 0.7,
                "segment_index": i + 1,
                "segment_id": f"segment_{i+1:03d}",
                "is_fallback": True
            }
            
            segments.append(segment)
        
        return segments
    
    def save_optimal_segments(self, segments: List[Dict[str, Any]], audio: np.ndarray, 
                            sr: int, output_dir: Path, speakers: List[Dict[str, Any]], 
                            target_language: str, detected_language: str, 
                            original_audio_details: Optional[Dict] = None):
        """Save optimally segmented audio with metadata"""
        try:
            logger.info(f"Saving {len(segments)} optimal segments to {output_dir}")
            
            # Create directory structure
            segments_dir = output_dir / "segments"
            segments_dir.mkdir(parents=True, exist_ok=True)
            
            segment_files = []
            
            for segment in segments:
                # Calculate sample indices
                start_sample = int(segment["start"] * sr)
                end_sample = int(segment["end"] * sr)
                
                # Extract audio segment
                segment_audio = audio[start_sample:end_sample]
                
                # Skip empty segments
                if len(segment_audio) == 0:
                    continue
                
                # Create filenames
                segment_index = segment["segment_index"]
                segment_filename = f"segment_{segment_index:03d}.wav"
                metadata_filename = f"segment_{segment_index:03d}_metadata.json"
                
                segment_path = segments_dir / segment_filename
                metadata_path = segments_dir / metadata_filename
                
                # Save audio segment
                sf.write(str(segment_path), segment_audio, sr)
                
                # Prepare metadata with translation info
                metadata = {
                    "segment_index": segment_index,
                    "segment_id": segment["segment_id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "speaker": segment["speaker"],
                    "speaker_label": segment.get("speaker_label", f"Speaker {segment['speaker']}"),
                    "text": segment["text"],  # This will be translated text if translation occurred
                    "original_text": segment.get("original_text", segment["text"]),  # Original before translation
                    "english_text": segment.get("english_text", segment["text"]),  # For compatibility
                    "type": segment["type"],
                    "confidence": segment["confidence"],
                    "language_code": detected_language,
                    "target_language": target_language,
                    "audio_file": segment_filename,
                    "sample_rate": sr,
                    "channels": 1,
                    "created_at": datetime.now().isoformat(),
                    "original_audio": original_audio_details,
                    "utterances": segment.get("utterances", []),
                    "translation": segment.get("translation", {"translated": False}),
                    "processing_flags": {
                        "is_split": segment.get("is_split", False),
                        "is_merged": segment.get("is_merged", False),
                        "is_gap_fill": segment.get("is_gap_fill", False),
                        "is_fallback": segment.get("is_fallback", False)
                    }
                }
                
                # Save metadata
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                # Store locally as backup
                self._store_segment_locally(segment_audio, metadata, segment_index, sr)
                
                segment_files.append({
                    "audio_file": str(segment_path),
                    "metadata_file": str(metadata_path),
                    "segment": segment
                })
            
            # Save summary
            summary = {
                "total_segments": len(segments),
                "total_duration": sum(s["duration"] for s in segments),
                "speakers": speakers,
                "target_language": target_language,
                "detected_language": detected_language,
                "segmentation_method": "smart_optimal",
                "created_at": datetime.now().isoformat(),
                "original_audio": original_audio_details,
                "segment_files": segment_files
            }
            
            summary_path = output_dir / "segmentation_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved {len(segment_files)} segments")
            
        except Exception as e:
            logger.error(f"Failed to save segments: {str(e)}")
            raise Exception(f"Segment saving failed: {str(e)}")
    
    def _store_segment_locally(self, segment_audio: np.ndarray, metadata: Dict[str, Any], 
                              segment_index: int, sr: int):
        """Store individual segment locally for backup - DISABLED to prevent duplicate storage"""
        # NOTE: Duplicate storage disabled as segments are already stored in voice_cloning directory
        pass
    
    def _translate_segments_if_needed(self, segments: List[Dict[str, Any]], 
                                     transcript_data: Dict[str, Any], audio_id: str) -> List[Dict[str, Any]]:
        """Translate segments if target language is different from detected language"""
        try:
            if not audio_id:
                logger.warning("No audio_id provided, skipping translation")
                return segments
            
            # Get language information
            detected_language = transcript_data.get("language_code", "en")
            target_language = getattr(self, '_target_language', "English")  # Set via method parameter
            
            # Use transcription service for translation
            translated_segments = self.transcription_service.translate_segments_for_dubbing(
                segments, target_language, detected_language, audio_id
            )
            
            logger.info(f"Translation completed for {len(translated_segments)} segments")
            return translated_segments
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            # Return original segments with error metadata
            for segment in segments:
                segment["translation"] = {
                    "translated": False,
                    "error": str(e),
                    "status": "translation_failed"
                }
            return segments
    
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

