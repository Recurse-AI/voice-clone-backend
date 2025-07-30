"""
Simple Segment Manager - AssemblyAI Sentence-based Segmentation
Clean and neat implementation for sentence-wise audio cropping

Usage Example:
    # Initialize (called automatically by base_processor)
    segment_manager = SegmentManager()
    
    # Create segments from AssemblyAI (transcript_data from transcription service)
    segments = segment_manager.create_optimal_segments(transcript_data, target_language="Bengali")
    
    # Save segments with audio cropping (called by base_processor)
    result = segment_manager.save_optimal_segments(
        segments, audio_array, sample_rate, output_dir, 
        speakers_data, target_language, detected_language
    )
    
    # Use with voice cloning
    speech_segments = result["speech_segments_for_cloning"]
    voice_service = get_fish_speech_service()
    for segment in speech_segments:
        cloned_result = voice_service.clone_voice_for_segment(segment)
"""

import json
import logging
from typing import Dict, Any, List
import requests
import numpy as np
import soundfile as sf
from pathlib import Path

logger = logging.getLogger(__name__)


class SegmentManager:
    """Simple segment manager using AssemblyAI sentences API"""
    
    def __init__(self):
        self.silent_threshold = 0.5  # 0.5s+ gaps treated as silent parts
        
    def _fetch_sentences_from_assemblyai(self, transcript_id: str, api_key: str) -> List[Dict]:
        """Fetch sentences from AssemblyAI API"""
        try:
            url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}/sentences"
            headers = {"Authorization": api_key}
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return data.get("sentences", [])
            
        except Exception as e:
            logger.error(f"Failed to fetch sentences from AssemblyAI: {e}")
            return []
    
    def _create_sentence_segments(self, sentences: List[Dict], total_duration: float) -> List[Dict]:
        """Create segments from sentences with silent parts handling and gap coverage"""
        segments = []
        last_end_time = 0.0
        
        # Handle gap from start if first sentence doesn't start at 0
        if sentences and sentences[0]["start"] / 1000.0 > self.silent_threshold:
            first_start = sentences[0]["start"] / 1000.0
            initial_silent = {
                "index": "initial_silent",
                "start": 0.0,
                "end": first_start,
                "duration": first_start,
                "text": "",
                "speaker": "SILENT",
                "confidence": 1.0,
                "type": "silent",
                "words": []
            }
            segments.append(initial_silent)
            last_end_time = first_start
        
        for i, sentence in enumerate(sentences):
            start_ms = sentence["start"]
            end_ms = sentence["end"]
            text = sentence["text"]
            confidence = sentence.get("confidence", 0.0)
            speaker = sentence.get("speaker", "SPEAKER_00")
            
            # Convert to seconds
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            duration = end_sec - start_sec
            
            # Check for gap before this sentence
            if start_sec > last_end_time + self.silent_threshold:
                gap_duration = start_sec - last_end_time
                silent_segment = {
                    "index": f"gap_{i}",
                    "start": last_end_time,
                    "end": start_sec,
                    "duration": gap_duration,
                    "text": "",
                    "speaker": "SILENT",
                    "confidence": 1.0,
                    "type": "silent",
                    "words": []
                }
                segments.append(silent_segment)
            
            # Add sentence segment
            segment = {
                "index": i,
                "start": start_sec,
                "end": end_sec, 
                "duration": duration,
                "text": text,
                "speaker": speaker,
                "confidence": confidence,
                "type": "speech",
                "words": sentence.get("words", [])
            }
            segments.append(segment)
            last_end_time = end_sec
        
        # Handle gap at the end if needed
        if last_end_time < total_duration - self.silent_threshold:
            final_gap = total_duration - last_end_time
            final_silent = {
                "index": "final_silent",
                "start": last_end_time,
                "end": total_duration,
                "duration": final_gap,
                "text": "",
                "speaker": "SILENT", 
                "confidence": 1.0,
                "type": "silent",
                "words": []
            }
            segments.append(final_silent)
        
        return segments
    
    def _crop_audio_segments(self, segments: List[Dict], audio: np.ndarray, sample_rate: int) -> List[Dict]:
        """Crop audio for each segment"""
        processed_segments = []
        
        for segment in segments:
            start_sec = segment["start"]
            end_sec = segment["end"]
            
            # Convert to sample indices
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            
            # Ensure bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            # Extract audio segment
            if segment["type"] == "silent":
                # Create silent audio
                segment_length = end_sample - start_sample
                segment_audio = np.zeros(segment_length, dtype=audio.dtype)
            else:
                # Extract actual audio
                segment_audio = audio[start_sample:end_sample]
            
            # Add audio data to segment
            segment_with_audio = segment.copy()
            segment_with_audio["audio"] = segment_audio
            segment_with_audio["sample_rate"] = sample_rate
            
            processed_segments.append(segment_with_audio)
        
        return processed_segments
    
    def _create_reference_audio_segments(self, segments: List[Dict], audio_id: str) -> List[Dict]:
        """Create reference audio segments for voice cloning from original vocal audio"""
        reference_segments = []
        
        for segment in segments:
            if segment["type"] == "speech" and len(segment.get("audio", [])) > 0:
                # Save reference audio file for voice cloning
                ref_audio_dir = Path(f"segments/{audio_id}/reference")
                ref_audio_dir.mkdir(parents=True, exist_ok=True)
                ref_audio_path = ref_audio_dir / f"ref_segment_{segment['index']:03d}.wav"
                
                # Save reference audio
                sf.write(ref_audio_path, segment["audio"], segment["sample_rate"])
                
                # Create reference segment metadata for voice cloning
                reference_segment = {
                    "index": segment["index"],
                    "audio_id": audio_id,
                    "reference_audio_path": str(ref_audio_path),
                    "sample_rate": segment["sample_rate"],
                    "duration": segment["duration"],
                    "speaker": segment["speaker"],
                    "text": segment["text"],
                    "type": segment["type"]
                }
                reference_segments.append(reference_segment)
        
        return reference_segments
    
    def create_optimal_segments(self, transcript_data: Dict[str, Any], 
                              target_language: str = "English", audio_id: str = None) -> List[Dict[str, Any]]:
        """Create optimal segments using AssemblyAI sentences API"""
        try:
            logger.info("Starting AssemblyAI sentence-based segmentation")
            
            # Get transcript ID and API key from transcript_data
            transcript_id = transcript_data.get("transcript_id")
            api_key = transcript_data.get("api_key")
            total_duration = transcript_data.get("audio_duration", 0)
            
            if not transcript_id or not api_key:
                logger.error("Missing transcript_id or api_key")
                return []
            
            # Fetch sentences from AssemblyAI
            sentences = self._fetch_sentences_from_assemblyai(transcript_id, api_key)
            
            if not sentences:
                logger.warning("No sentences found from AssemblyAI")
                return []
            
            # Create segments from sentences
            segments = self._create_sentence_segments(sentences, total_duration)
            
            logger.info(f"Created {len(segments)} segments from {len(sentences)} sentences")
            
            return segments
            
        except Exception as e:
            logger.error(f"Sentence-based segmentation failed: {str(e)}")
            return []
    
    def prepare_segments_for_voice_cloning(self, segments: List[Dict[str, Any]], 
                                         target_language: str = "English") -> List[Dict[str, Any]]:
        """Prepare speech segments for voice cloning by adding target language translation"""
        speech_segments = []
        
        for segment in segments:
            if segment["type"] == "speech" and segment.get("text", "").strip():
                # Prepare segment metadata for voice cloning
                cloning_segment = segment.copy()
                cloning_segment["target_language"] = target_language
                
                # Add original text as reference text
                cloning_segment["original_text"] = segment["text"]
                
                # TODO: Add translation logic here when translation service is ready
                # For now, keep original text if target is English, otherwise mark for translation
                if target_language.lower() == "english":
                    cloning_segment["translated_text"] = segment["text"]
                else:
                    cloning_segment["translated_text"] = f"[TO_TRANSLATE:{target_language}] {segment['text']}"
                
                speech_segments.append(cloning_segment)
        
        return speech_segments
    
    def save_optimal_segments(self, segments: List[Dict[str, Any]], audio: np.ndarray,
                            sample_rate: int, output_dir, speakers_data: List[Dict], 
                            target_language: str = "English", detected_language: str = "en",
                            original_audio_details = None) -> Dict[str, Any]:
        """Save cropped audio segments and create reference segments"""
        try:
            # Extract audio_id from output_dir path
            if hasattr(output_dir, 'name'):
                audio_id = output_dir.name.replace('segments_', '')
            else:
                audio_id = str(output_dir).split('/')[-1].replace('segments_', '')
                
            logger.info(f"Processing and saving {len(segments)} segments for {audio_id}")
            
            # Crop audio for each segment
            processed_segments = self._crop_audio_segments(segments, audio, sample_rate)
            
            # Create reference audio segments for voice cloning
            reference_segments = self._create_reference_audio_segments(processed_segments, audio_id)
            
            # Prepare speech segments for voice cloning
            speech_segments_for_cloning = self.prepare_segments_for_voice_cloning(reference_segments, target_language)
            
            # Save segments locally
            saved_segments = []
            for i, segment in enumerate(processed_segments):
                if segment.get("audio") is not None:
                    # Save segment audio
                    segment_path = f"segments/{audio_id}/segment_{i:03d}.wav"
                    
                    # Ensure directory exists
                    segment_dir = Path(f"segments/{audio_id}")
                    segment_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save audio file
                    sf.write(segment_path, segment["audio"], sample_rate)
                    
                    # Create segment metadata
                    segment_metadata = {
                        "index": segment["index"],
                        "start": segment["start"],
                        "end": segment["end"], 
                        "duration": segment["duration"],
                        "text": segment["text"],
                        "speaker": segment["speaker"],
                        "confidence": segment.get("confidence", 0.0),
                        "type": segment["type"],
                        "audio_path": segment_path,
                        "sample_rate": sample_rate
                    }
                    
                    saved_segments.append(segment_metadata)
            
            # Save reference segments metadata
            reference_metadata = {
                "audio_id": audio_id,
                "reference_segments": reference_segments,
                "total_segments": len(saved_segments),
                "target_language": target_language,
                "detected_language": detected_language,
                "speakers_data": speakers_data
            }
            
            # Store metadata
            metadata_path = f"segments/{audio_id}/metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    "segments": saved_segments,
                    "reference_data": reference_metadata
                }, f, indent=2)
            
            logger.info(f"Successfully saved {len(saved_segments)} segments")
            
            return {
                "segments": saved_segments,
                "reference_segments": reference_segments,
                "speech_segments_for_cloning": speech_segments_for_cloning,
                "metadata_path": metadata_path
            }
            
        except Exception as e:
            logger.error(f"Failed to save segments: {str(e)}")
            return {"segments": [], "reference_segments": [], "speech_segments_for_cloning": [], "metadata_path": None}