"""
Transcription and Text Processing Module - Enhanced with Speaker Diarization
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import assemblyai as aai
import requests
from openai import OpenAI

from config import settings
from utils import local_storage

logger = logging.getLogger(__name__)

class TranscriptionService:
    """Enhanced transcription service with speaker diarization"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        
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
                        original_duration: float) -> Dict[str, Any]:
        """Transcribe audio with speaker diarization using AssemblyAI"""
        try:
            logger.info(f"Starting AssemblyAI transcription for {audio_id}")
            
            # Store audio locally for backup
            local_audio_path = self._store_audio_locally(audio_path, audio_id)
            
            # Prepare transcription config
            config = self.transcription_config.copy()
            
            # Set language if specified
            if language_code:
                config["language_code"] = language_code
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
                elapsed = time.time() - start_time
                if elapsed > 600:  # 10 minute timeout
                    raise Exception("Transcription timeout - took longer than 10 minutes")
                
                # Update progress
                self._update_transcription_progress(audio_id, elapsed)
                time.sleep(5)  # Check every 5 seconds
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
    
    def _store_audio_locally(self, audio_path: str, audio_id: str) -> str:
        """Store audio file locally for backup"""
        try:
            with open(audio_path, 'rb') as f:
                audio_content = f.read()
            
            # Store using local storage system
            filename = f"{audio_id}_vocal_for_transcription.wav"
            local_result = local_storage.store_audio(audio_id, audio_content, filename)
            
            if local_result.get("success"):
                return local_result["local_path"]
            else:
                logger.warning(f"Failed to store audio locally: {local_result.get('error')}")
                return audio_path
                
        except Exception as e:
            logger.warning(f"Failed to store audio locally: {str(e)}")
            return audio_path
    
    def _update_transcription_progress(self, audio_id: str, elapsed: float):
        """Update transcription progress"""
        try:
            from status_manager import status_manager, ProcessingStatus
            
            # Calculate progress based on elapsed time (estimate 3-5 minutes for transcription)
            estimated_duration = 240  # 4 minutes estimate
            progress_percent = min(int((elapsed / estimated_duration) * 20), 20)  # Max 20% for transcription
            base_progress = 30  # Starting from 30%
            
            status_manager.update_status(
                audio_id, 
                ProcessingStatus.PROCESSING, 
                progress=base_progress + progress_percent,
                details={"message": f"Transcribing audio... ({int(elapsed)}s elapsed)"}
            )
        except:
            pass
    
    def _process_assemblyai_response(self, transcript, audio_id: str, 
                                   original_duration: float) -> Dict[str, Any]:
        """Process AssemblyAI response and extract speaker information"""
        
        # Get basic transcript info
        language_code = getattr(transcript, 'language_code', 'en')
        confidence = getattr(transcript, 'confidence', 0.9)
        
        # Extract speakers from utterances
        speakers = {}
        utterances_data = []
        
        if hasattr(transcript, 'utterances') and transcript.utterances:
            for utterance in transcript.utterances:
                speaker_id = utterance.speaker
                
                # Initialize speaker if not seen before
                if speaker_id not in speakers:
                    speakers[speaker_id] = {
                        "id": speaker_id,
                        "label": f"Speaker {speaker_id}",
                        "total_duration": 0,
                        "segments_count": 0,
                        "confidence": 0
                    }
                
                # Process utterance
                utterance_data = {
                    "speaker": speaker_id,
                    "text": utterance.text,
                    "start": utterance.start / 1000.0,  # Convert ms to seconds
                    "end": utterance.end / 1000.0,
                    "confidence": utterance.confidence,
                    "words": []
                }
                
                # Process words within utterance
                if hasattr(utterance, 'words') and utterance.words:
                    for word in utterance.words:
                        word_data = {
                            "text": word.text,
                            "start": word.start / 1000.0,
                            "end": word.end / 1000.0,
                            "confidence": word.confidence,
                            "speaker": word.speaker if hasattr(word, 'speaker') else speaker_id
                        }
                        utterance_data["words"].append(word_data)
                
                utterances_data.append(utterance_data)
                
                # Update speaker stats
                duration = utterance_data["end"] - utterance_data["start"]
                speakers[speaker_id]["total_duration"] += duration
                speakers[speaker_id]["segments_count"] += 1
                speakers[speaker_id]["confidence"] = max(
                    speakers[speaker_id]["confidence"], 
                    utterance_data["confidence"]
                )
        
        # If no utterances with speakers, fall back to words
        if not utterances_data and hasattr(transcript, 'words') and transcript.words:
            logger.info("No utterances found, processing words directly")
            
            # Group words by speaker
            current_speaker = None
            current_utterance = None
            
            for word in transcript.words:
                word_speaker = getattr(word, 'speaker', 'A')
                
                if word_speaker != current_speaker:
                    # Save previous utterance
                    if current_utterance:
                        utterances_data.append(current_utterance)
                        duration = current_utterance["end"] - current_utterance["start"]
                        speakers[current_speaker]["total_duration"] += duration
                        speakers[current_speaker]["segments_count"] += 1
                    
                    # Start new utterance
                    current_speaker = word_speaker
                    if current_speaker not in speakers:
                        speakers[current_speaker] = {
                            "id": current_speaker,
                            "label": f"Speaker {current_speaker}",
                            "total_duration": 0,
                            "segments_count": 0,
                            "confidence": 0
                        }
                    
                    current_utterance = {
                        "speaker": current_speaker,
                        "text": word.text,
                        "start": word.start / 1000.0,
                        "end": word.end / 1000.0,
                        "confidence": word.confidence,
                        "words": [{
                            "text": word.text,
                            "start": word.start / 1000.0,
                            "end": word.end / 1000.0,
                            "confidence": word.confidence,
                            "speaker": word_speaker
                        }]
                    }
                else:
                    # Continue current utterance
                    current_utterance["text"] += " " + word.text
                    current_utterance["end"] = word.end / 1000.0
                    current_utterance["confidence"] = (current_utterance["confidence"] + word.confidence) / 2
                    current_utterance["words"].append({
                        "text": word.text,
                        "start": word.start / 1000.0,
                        "end": word.end / 1000.0,
                        "confidence": word.confidence,
                        "speaker": word_speaker
                    })
            
            # Save last utterance
            if current_utterance:
                utterances_data.append(current_utterance)
                duration = current_utterance["end"] - current_utterance["start"]
                speakers[current_speaker]["total_duration"] += duration
                speakers[current_speaker]["segments_count"] += 1
        
        # Ensure we have at least one speaker
        if not speakers:
            speakers["A"] = {
                "id": "A",
                "label": "Speaker A",
                "total_duration": original_duration,
                "segments_count": 1,
                "confidence": confidence
            }
            
            # Create basic utterance if none exist
            if not utterances_data:
                utterances_data = [{
                    "speaker": "A",
                    "text": transcript.text or "",
                    "start": 0.0,
                    "end": original_duration,
                    "confidence": confidence,
                    "words": []
                }]
        
        return {
            "transcript_id": transcript.id,
            "text": transcript.text,
            "language_code": language_code,
            "confidence": confidence,
            "audio_duration": original_duration,
            "speakers": list(speakers.values()),
            "utterances": utterances_data,
            "metadata": {
                "language_code": language_code,
                "confidence": confidence,
                "processing_time": datetime.now().isoformat(),
                "total_speakers": len(speakers),
                "total_utterances": len(utterances_data)
            }
        }
    
    def _save_transcription_locally(self, transcription_result: Dict[str, Any], audio_id: str):
        """Save transcription results locally for backup"""
        try:
            # Save as JSON
            transcription_json = json.dumps(transcription_result, indent=2, ensure_ascii=False)
            
            local_result = local_storage.store_text(
                audio_id, 
                transcription_json.encode('utf-8'), 
                f"{audio_id}_transcription.json"
            )
            
            if local_result.get("success"):
                logger.info(f"Transcription saved locally: {local_result['local_path']}")
            else:
                logger.warning(f"Failed to save transcription locally: {local_result.get('error')}")
                
        except Exception as e:
            logger.warning(f"Failed to save transcription locally: {str(e)}")
    
    def translate_segments_for_dubbing(self, segments: List[Dict[str, Any]], 
                                      target_language: str, detected_language: str,
                                      audio_id: str) -> List[Dict[str, Any]]:
        """Translate segments using OpenAI with dubbing-optimized prompts"""
        try:
            logger.info(f"Starting OpenAI translation for {len(segments)} segments to {target_language}")
            
            # Skip translation if already in target language
            if self._is_same_language(detected_language, target_language):
                logger.info(f"Detected language ({detected_language}) matches target ({target_language}), skipping translation")
                return self._add_translation_metadata(segments, target_language, "skipped_same_language")
            
            translated_segments = []
            batch_size = 5  # Process segments in batches
            
            for i in range(0, len(segments), batch_size):
                batch = segments[i:i+batch_size]
                
                # Update progress
                progress = int((i / len(segments)) * 100)
                self._update_translation_progress(audio_id, progress, f"Translating batch {i//batch_size + 1}")
                
                # Translate batch
                translated_batch = self._translate_segment_batch(batch, target_language, detected_language)
                translated_segments.extend(translated_batch)
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            
            logger.info(f"Translation completed for {len(translated_segments)} segments")
            return translated_segments
            
        except Exception as e:
            logger.error(f"Translation failed for {audio_id}: {str(e)}")
            # Return original segments with error metadata
            return self._add_translation_metadata(segments, target_language, f"translation_failed: {str(e)}")
    
    def _translate_segment_batch(self, segments: List[Dict[str, Any]], 
                                target_language: str, detected_language: str) -> List[Dict[str, Any]]:
        """Translate a batch of segments using OpenAI"""
        try:
            # Prepare batch for translation
            translation_requests = []
            for segment in segments:
                if segment.get("type") == "speech" and segment.get("text", "").strip():
                    translation_requests.append({
                        "segment_id": segment.get("segment_id", "unknown"),
                        "original_text": segment["text"].strip(),
                        "speaker": segment.get("speaker", "unknown"),
                        "duration": segment.get("duration", 0),
                        "segment_index": segment.get("segment_index", 0)
                    })
            
            if not translation_requests:
                return segments
            
            # Create dubbing-optimized prompt
            prompt = self._create_dubbing_translation_prompt(
                translation_requests, target_language, detected_language
            )
            
            # Call OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Use latest model for best quality
                messages=[
                    {"role": "system", "content": self._get_translation_system_prompt(target_language)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for consistent translation
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            translation_result = json.loads(response.choices[0].message.content)
            
            # Apply translations to segments
            return self._apply_translations_to_segments(segments, translation_result)
            
        except Exception as e:
            logger.error(f"Batch translation failed: {str(e)}")
            return self._add_translation_metadata(segments, target_language, f"batch_translation_failed: {str(e)}")
    
    def _create_dubbing_translation_prompt(self, requests: List[Dict[str, Any]], 
                                          target_language: str, detected_language: str) -> str:
        """Create optimized prompt for dubbing translation"""
        
        # Build context for better translation
        context_info = []
        for req in requests:
            context_info.append({
                "id": req["segment_id"],
                "text": req["original_text"],
                "speaker": req["speaker"],
                "duration": f"{req['duration']:.1f}s"
            })
        
        prompt = f"""
DUBBING TRANSLATION TASK

You are a professional dubbing translator specializing in video content translation from {detected_language} to {target_language}.

CRITICAL REQUIREMENTS FOR DUBBING:
1. NATURAL SPEECH: Translate for spoken dialogue, not written text
2. TIMING MATCH: Keep similar syllable count and speech rhythm 
3. EMOTIONAL PRESERVATION: Maintain the speaker's emotional tone and intent
4. CULTURAL ADAPTATION: Use culturally appropriate expressions
5. LIP-SYNC FRIENDLY: Consider mouth movements and speech patterns
6. CONVERSATIONAL STYLE: Use natural, spoken language (contractions, informal speech)

FISH SPEECH TTS OPTIMIZATION:
- Use clear, pronounceable text
- Add emotional markers when appropriate: (happy), (excited), (sad), (serious), (casual), etc.
- Use natural pauses with commas and periods
- Avoid complex punctuation that might confuse TTS
- Keep sentences conversational and flowing

SEGMENTS TO TRANSLATE:
{json.dumps(context_info, indent=2, ensure_ascii=False)}

INSTRUCTIONS:
1. Translate each segment preserving the speaker's personality and emotion
2. Match the approximate speaking duration (consider syllable count)
3. Use natural, conversational {target_language} 
4. Add appropriate emotion markers for Fish Speech TTS
5. Maintain speaker consistency across segments
6. Consider the video dubbing context (speakers might be characters, presenters, etc.)

RESPONSE FORMAT (JSON):
{{
  "translations": [
    {{
      "segment_id": "segment_001",
      "original_text": "original text",
      "translated_text": "translated text with (emotion) markers",
      "emotion_detected": "happy/sad/neutral/excited/etc",
      "translation_notes": "brief note about translation choices",
      "syllable_match": "similar/shorter/longer"
    }}
  ],
  "translation_summary": {{
    "source_language": "{detected_language}",
    "target_language": "{target_language}",
    "total_segments": {len(requests)},
    "translation_approach": "description of overall approach"
  }}
}}
"""
        return prompt
    
    def _get_translation_system_prompt(self, target_language: str) -> str:
        """Get system prompt for translation"""
        return f"""You are an expert dubbing translator with extensive experience in video localization and voice-over work. 

Your specializations:
- Video dubbing and voice-over translation
- Cross-cultural communication
- Natural speech patterns in {target_language}
- Text-to-Speech optimization
- Emotional tone preservation

You understand that dubbing translation requires:
- Natural spoken language (not literary translation)
- Timing and rhythm consideration
- Cultural context adaptation
- Emotional authenticity
- Technical TTS optimization

Always provide translations that sound natural when spoken aloud and work well with AI voice cloning technology."""
    
    def _apply_translations_to_segments(self, segments: List[Dict[str, Any]], 
                                       translation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply translations to segments"""
        try:
            translations_map = {}
            for translation in translation_result.get("translations", []):
                segment_id = translation.get("segment_id")
                if segment_id:
                    translations_map[segment_id] = translation
            
            translated_segments = []
            for segment in segments:
                segment_copy = segment.copy()
                segment_id = segment.get("segment_id")
                
                if segment_id in translations_map:
                    translation = translations_map[segment_id]
                    
                    # Add translation fields
                    segment_copy["original_text"] = segment.get("text", "")
                    segment_copy["text"] = translation.get("translated_text", segment.get("text", ""))
                    segment_copy["english_text"] = translation.get("translated_text", segment.get("text", ""))
                    
                    # Add translation metadata
                    segment_copy["translation"] = {
                        "translated": True,
                        "emotion_detected": translation.get("emotion_detected", "neutral"),
                        "translation_notes": translation.get("translation_notes", ""),
                        "syllable_match": translation.get("syllable_match", "similar"),
                        "translation_quality": "openai_gpt4o"
                    }
                else:
                    # No translation found, keep original
                    segment_copy["translation"] = {
                        "translated": False,
                        "reason": "no_translation_needed_or_failed"
                    }
                
                translated_segments.append(segment_copy)
            
            return translated_segments
            
        except Exception as e:
            logger.error(f"Failed to apply translations: {str(e)}")
            return self._add_translation_metadata(segments, "unknown", f"apply_translation_failed: {str(e)}")
    
    def _add_translation_metadata(self, segments: List[Dict[str, Any]], 
                                 target_language: str, status: str) -> List[Dict[str, Any]]:
        """Add translation metadata to segments"""
        for segment in segments:
            segment["translation"] = {
                "translated": False,
                "status": status,
                "target_language": target_language,
                "original_text": segment.get("text", ""),
                "english_text": segment.get("text", "")  # Fallback to original
            }
        return segments
    
    def _is_same_language(self, detected: str, target: str) -> bool:
        """Check if detected and target languages are the same"""
        # Normalize language codes
        detected_norm = detected.lower().strip()
        target_norm = target.lower().strip()
        
        # Direct match
        if detected_norm == target_norm:
            return True
        
        # Common variations
        language_mapping = {
            "english": ["en", "eng", "english"],
            "hindi": ["hi", "hin", "hindi"],
            "bengali": ["bn", "ben", "bengali", "bangla"],
            "spanish": ["es", "spa", "spanish"],
            "french": ["fr", "fra", "french"],
            "german": ["de", "deu", "german"],
            "chinese": ["zh", "chi", "chinese", "mandarin"],
            "japanese": ["ja", "jpn", "japanese"],
            "korean": ["ko", "kor", "korean"],
            "arabic": ["ar", "ara", "arabic"]
        }
        
        for lang_group in language_mapping.values():
            if detected_norm in lang_group and target_norm in lang_group:
                return True
        
        return False
    
    def _update_translation_progress(self, audio_id: str, progress: int, message: str):
        """Update translation progress"""
        try:
            from status_manager import status_manager, ProcessingStatus
            
            # Translation is part of processing phase (50-70%)
            base_progress = 50
            translation_progress = int((progress / 100) * 20)  # 20% allocation for translation
            total_progress = base_progress + translation_progress
            
            status_manager.update_status(
                audio_id, 
                ProcessingStatus.PROCESSING, 
                progress=total_progress,
                details={"message": message}
            )
        except:
            pass
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing"""
        # Remove quotes and extra whitespace
        text = re.sub(r'^["\s]*', '', text).strip()
        text = re.sub(r'["\s]*$', '', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
