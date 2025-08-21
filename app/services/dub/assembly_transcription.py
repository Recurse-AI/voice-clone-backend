"""
Transcription and Text Processing Module - Enhanced with Speaker Diarization
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import assemblyai as aai
import requests
from openai import OpenAI

from app.config.settings import settings
from .audio_utils import AudioUtils
from app.services.language_service import language_service

logger = logging.getLogger(__name__)

class TranscriptionService:
    """Enhanced transcription service with speaker diarization"""

    # Using centralized language service (removed duplicate mapping)

    @classmethod
    def _normalize_language_code(cls, language: Optional[str]) -> Optional[str]:
        """
        Convert a human-readable language name to a valid AssemblyAI ISO code.
        Uses centralized language service for consistency.
        """
        return language_service.get_language_code_for_transcription(language)
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        self.audio_utils = AudioUtils()
        
        # AssemblyAI configuration for optimal transcription
        self.transcription_config = {
            "speaker_labels": True,           # Enable speaker diarization
            "language_detection": True,       # Auto-detect language
            "punctuate": True,               # Add punctuation
            "format_text": True,             # Format text properly
            "dual_channel": False,           # Single channel processing
            "filter_profanity": False,       # Keep original content
            "word_boost": [],                # No word boosting
            "boost_param": None,             # No boost parameters
            "auto_highlights": False,        # Disable highlights
            "content_safety": False,         # Disable content safety
            "iab_categories": False,         # Disable categorization
            "speech_threshold": 0.1,         # Detect low volume speech
            "disfluencies": True,            # Include uh, um, etc.
            "language_confidence_threshold": 0.1
        }
    
    def transcribe_audio(self, audio_path: str, language_code: Optional[str], 
                        speakers_expected: Optional[int], audio_id: str, 
                        original_duration: float, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio with speaker diarization using AssemblyAI"""
        try:
            logger.info(f"Starting AssemblyAI transcription for {audio_id}")
            
            # Store audio locally for backup
            local_audio_path = self._store_audio_locally(audio_path, audio_id)
            
            # Prepare transcription config
            config = self.transcription_config.copy()
            
            # Set language if specified
            normalized_code = self._normalize_language_code(language_code) if language_code else None
            if normalized_code:
                config["language_code"] = normalized_code
                config["language_detection"] = False
            
            # Set expected speakers if specified
            if speakers_expected and speakers_expected > 1:
                config["speakers_expected"] = min(speakers_expected, 10)  # Max 10 speakers
            
            # Upload audio file to AssemblyAI
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_path, config=aai.TranscriptionConfig(**config))
            
            # Wait for completion with progress tracking
            start_time = time.time()
            while transcript.status in [aai.TranscriptStatus.queued, aai.TranscriptStatus.processing]:
                # 🛡️ Check cancellation if job_id provided
                if job_id:
                    from app.utils.shared_memory import is_job_cancelled
                    if is_job_cancelled(job_id):
                        logger.info(f"🛑 AssemblyAI transcription cancelled for job {job_id}")
                        # Update database status to cancelled before raising exception
                        try:
                            import asyncio
                            from app.utils.unified_status_manager import get_unified_status_manager, ProcessingStatus, JobType
                            manager = get_unified_status_manager()
                            
                            def run_update():
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                try:
                                    loop.run_until_complete(
                                        manager.update_status(job_id, JobType.DUB, ProcessingStatus.CANCELLED, 0, {"error": "Job cancelled by user"})
                                    )
                                finally:
                                    loop.close()
                            
                            import threading
                            threading.Thread(target=run_update, daemon=True).start()
                        except Exception as e:
                            logger.warning(f"Failed to update status during cancellation: {e}")
                        raise Exception("Job cancelled by user")
                
                elapsed = time.time() - start_time
                if elapsed > 600:  # 10 minute timeout
                    raise Exception("Transcription timeout - took longer than 10 minutes")
                
                # Update progress
                self._update_transcription_progress(audio_id, elapsed)
                time.sleep(20)  # Check every 20 seconds
                transcript = transcriber.get_transcript(transcript.id)
            
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Transcription failed: {transcript.error}")
            
            if transcript.status != aai.TranscriptStatus.completed:
                raise Exception(f"Unexpected transcription status: {transcript.status}")
            
            # Process the response
            transcription_result = self._process_assemblyai_response(
                transcript, audio_id, original_duration
            )
            
            # Save transcription results locally
            self._save_transcription_locally(transcription_result, audio_id)
            
            logger.info(f"Transcription completed for {audio_id} - {len(transcription_result['speakers'])} speakers detected")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_id}: {str(e)}")
            raise Exception(f"Transcription service error: {str(e)}")
    
    def get_sentences_and_split_audio(self, audio_url: str, output_dir: str = None, speakers_count: int = 1, source_video_language: str = None, max_sentences: int = None, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get sentences from AssemblyAI API, split long sentences, and split audio into segments"""
        try:
            logger.info(f"Starting sentence extraction and audio splitting for audio: {audio_url}")
            
            # Better URL validation and filename extraction
            if not audio_url or not audio_url.startswith(('http://', 'https://')):
                raise Exception(f"Invalid audio URL: {audio_url}")
            
            # Download audio file first
            audio_filename = audio_url.split('/')[-1]
            # Remove query parameters if any
            audio_filename = audio_filename.split('?')[0]
            if not audio_filename or not audio_filename.strip():
                audio_filename = f"audio_{int(time.time())}.wav"
            elif not audio_filename.endswith(('.wav', '.mp3', '.m4a', '.flac', '.aac')):
                audio_filename = f"{audio_filename}.wav"
            
            # Store downloaded audio inside the provided output directory (or TEMP_DIR if None)
            download_dir = output_dir if output_dir else settings.TEMP_DIR
            os.makedirs(download_dir, exist_ok=True)
            temp_audio_path = os.path.join(download_dir, audio_filename)
            
            download_result = self.audio_utils.download_audio_file(audio_url, temp_audio_path)
            if not download_result["success"]:
                raise Exception(f"Failed to download audio: {download_result['error']}")
            
            # Determine language config
            language_code = self._normalize_language_code(source_video_language)
            language_detection = True if not language_code else False
            
            speakers_count = int(speakers_count) if speakers_count else 1
            
            # Create transcription for the audio
            transcript_result = self._create_transcription_for_audio(temp_audio_path, speakers_count, language_code, language_detection, job_id)
            transcript_id = transcript_result["transcript_id"]
            logger.info(f"Created transcript with ID: {transcript_id}")
            
            # Get sentences from Assembly AI
            sentences_data = self._get_sentences_from_api(transcript_id)
            if isinstance(sentences_data, dict) and "sentences" in sentences_data:
                raw_sentences = sentences_data["sentences"]
            elif isinstance(sentences_data, list):
                raw_sentences = sentences_data
            else:
                raise Exception(f"Unexpected response format from AssemblyAI sentences API: {sentences_data}")
            
            logger.info(f"Found {len(raw_sentences)} raw sentences from Assembly AI")
            
            # Split long sentences (>20 seconds) into smaller segments
            processed_sentences = self._split_long_sentences(raw_sentences, max_duration_ms=20000)
            logger.info(f"After splitting: {len(processed_sentences)} sentences (was {len(raw_sentences)})")
            
            # Apply max limit if specified
            sentences_to_process = processed_sentences[:max_sentences] if max_sentences else processed_sentences
            
            # Convert sentences to segments for audio splitting
            segments_to_split = []
            for i, sentence in enumerate(sentences_to_process):
                segments_to_split.append({
                    "start": sentence["start"],
                    "end": sentence["end"], 
                    "text": sentence["text"],
                    "speaker_label": sentence.get("speaker", sentence["words"][0]["speaker"] if sentence.get("words") else "A")
                })


            
            # Set output directory
            if not output_dir:
                output_dir = os.path.join(settings.TEMP_DIR, f"segments_{transcript_id}")
            
            # Split audio based on sentence timestamps
            split_result = self.audio_utils.split_audio_by_timestamps(
                temp_audio_path, output_dir, segments_to_split
            )
            
            if not split_result["success"]:
                raise Exception(f"Failed to split audio: {split_result['error']}")
            
            logger.info(f"Successfully split audio into {split_result['segments_count']} segments")
            

            
            # Combine split file info with original sentence data (including speaker)
            enhanced_segments = []
            split_files = split_result.get("split_files", [])
            
            for i, split_file in enumerate(split_files):
                if i < len(segments_to_split):
                    enhanced_segment = segments_to_split[i].copy()
                    enhanced_segment.update({
                        "output_path": split_file["output_path"],
                        "duration_ms": split_file["duration_ms"]
                    })
                    enhanced_segments.append(enhanced_segment)
            
            return {
                "success": True,
                "transcript_id": transcript_id,
                "audio_url": audio_url,
                "raw_sentences_count": len(raw_sentences),
                "processed_sentences_count": len(processed_sentences),
                "segments_processed": len(enhanced_segments),
                "original_audio": temp_audio_path,
                "split_result": split_result,
                "segments": enhanced_segments
            }
            
        except Exception as e:
            logger.error(f"Sentence extraction and audio splitting failed: {str(e)}")
            raise Exception(f"Service error: {str(e)}")
    
    def _create_transcription_for_audio(self, audio_path: str, speakers_count: int = 2, language_code: str = None, language_detection: bool = True, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new transcription for audio file with speaker diarization and language control"""
        try:
            transcriber = aai.Transcriber()
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                speakers_expected=speakers_count,
                language_detection=language_detection,
                language_code=None if language_detection else language_code,
                punctuate=True,
                format_text=True
            )
            transcript = transcriber.transcribe(audio_path, config=config)
            
            # Wait for completion
            start_time = time.time()
            while transcript.status in [aai.TranscriptStatus.queued, aai.TranscriptStatus.processing]:
                # 🛡️ Check cancellation if job_id provided
                if job_id:
                    from app.utils.shared_memory import is_job_cancelled
                    if is_job_cancelled(job_id):
                        try:
                            import asyncio
                            from app.utils.unified_status_manager import get_unified_status_manager, ProcessingStatus, JobType
                            manager = get_unified_status_manager()
                            
                            def run_update():
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                try:
                                    loop.run_until_complete(
                                        manager.update_status(job_id, JobType.DUB, ProcessingStatus.CANCELLED, 0, {"error": "Job cancelled by user"})
                                    )
                                finally:
                                    loop.close()
                            
                            import threading
                            threading.Thread(target=run_update, daemon=True).start()
                        except Exception as e:
                            logger.warning(f"Failed to update status during cancellation: {e}")
                        raise Exception("Job cancelled by user")
                
                elapsed = time.time() - start_time
                if elapsed > 300:  # 5 minute timeout
                    raise Exception("Transcription timeout")
                time.sleep(20)
                transcript = transcriber.get_transcript(transcript.id)
            
            if transcript.status != aai.TranscriptStatus.completed:
                raise Exception(f"Transcription failed with status: {transcript.status}")
            
            return {"transcript_id": transcript.id}
            
        except Exception as e:
            raise Exception(f"Failed to create transcription: {str(e)}")
    
   
    
    def _get_sentences_from_api(self, transcript_id: str) -> Dict[str, Any]:
        """Get sentences from AssemblyAI API using direct HTTP request"""
        try:
            url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}/sentences"
            headers = {
                "Authorization": settings.ASSEMBLYAI_API_KEY
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get sentences from API: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse API response: {str(e)}")
    
    def _split_long_sentences(self, sentences: List[Dict[str, Any]], max_duration_ms: int = 20000) -> List[Dict[str, Any]]:
        """Split sentences longer than max_duration_ms into smaller segments based on punctuation"""
        
        # Multi-language punctuation endpoints for sentence splitting
        punctuation_patterns = {
            'period': ['.', '।', '。', '۔', '፡'],  # English, Hindi, Chinese, Urdu, Amharic
            'comma': [',', '،', '、', '，'],        # English, Arabic, Chinese  
            'semicolon': [';', '؛'],               # English, Arabic
            'question': ['?', '؟', '？'],          # English, Arabic, Chinese
            'exclamation': ['!', '！'],            # English, Chinese
            'pipe': ['|']                          # Custom delimiter
        }
        
        def find_split_points(words_list):
            """Find good split points in words list"""
            split_points = []
            
            for i, word in enumerate(words_list):
                word_text = word.get('text', '').strip()
                
                # Check for punctuation at word end only
                for punct_list in punctuation_patterns.values():
                    if any(word_text.endswith(p) for p in punct_list):
                        split_points.append(i + 1)  # Split after punctuation
                        break  # Found punctuation, no need to check other patterns
            
            return split_points
        
        result_sentences = []
        
        for sentence in sentences:
            duration_ms = sentence['end'] - sentence['start']
            
            # If sentence is within limit, keep as is
            if duration_ms <= max_duration_ms:
                result_sentences.append(sentence)
                continue
            
            # Need to split long sentence
            words = sentence.get('words', [])
            if not words:
                result_sentences.append(sentence)
                continue
            
            split_points = find_split_points(words)
            
            # If no good split points found, split by duration evenly
            if not split_points:
                words_per_segment = max(1, len(words) // (duration_ms // max_duration_ms + 1))
                split_points = list(range(words_per_segment, len(words), words_per_segment))
            
            # Create segments
            start_idx = 0
            segment_num = 0
            
            for split_point in split_points + [len(words)]:
                if split_point <= start_idx:
                    continue
                
                segment_words = words[start_idx:split_point]
                if not segment_words:
                    break
                
                segment_start = segment_words[0]['start']
                segment_end = segment_words[-1]['end']
                segment_duration = segment_end - segment_start
                
                # Skip very short segments (less than 1 second)
                if segment_duration < 1000:
                    start_idx = split_point
                    continue
                
                # If this segment is still too long, force split
                if segment_duration > max_duration_ms and len(segment_words) > 1:
                    mid_point = len(segment_words) // 2
                    segment_words = segment_words[:mid_point]
                    split_point = start_idx + mid_point
                    segment_end = segment_words[-1]['end']
                
                segment_text = ' '.join(word['text'] for word in segment_words)
                
                # Get speaker from first word
                speaker = segment_words[0].get('speaker') if segment_words else sentence.get('speaker', 'A')
                
                new_segment = {
                    'text': segment_text,
                    'start': segment_start,
                    'end': segment_end,
                    'words': segment_words,
                    'confidence': sentence.get('confidence', 0.9),
                    'speaker': speaker,
                    'original_sentence_id': sentence.get('id', f"sentence_{segment_num}"),
                    'segment_index': segment_num
                }
                
                result_sentences.append(new_segment)
                start_idx = split_point
                segment_num += 1
                
                # If we've split enough, break
                if segment_end >= sentence['end']:
                    break
        
        return result_sentences


    