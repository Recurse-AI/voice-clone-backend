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
            
            # Override AssemblyAI language detection with script-based detection
            combined_text = " ".join([segment.get("text", "") for segment in segments])
            script_detected_language = self._detect_language_from_script(combined_text)
            
            # Use script detection if it differs from AssemblyAI detection
            if script_detected_language != "en" and detected_language == "en":
                logger.info(f"Script analysis detected {script_detected_language}, overriding AssemblyAI detection ({detected_language})")
                detected_language = script_detected_language
            
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
        
        # Fish Speech supported languages output mapping
        language_output_map = {
            # Fish Speech supported languages
            "english": "English alphabet and words only",
            "chinese": "Chinese characters (simplified/traditional) only",
            "japanese": "Japanese characters (Hiragana/Katakana/Kanji) only", 
            "german": "German with Latin alphabet only",
            "french": "French with Latin alphabet only",
            "spanish": "Spanish with Latin alphabet only",
            "korean": "Korean Hangul characters only",
            "arabic": "Arabic script only",
            "russian": "Russian Cyrillic script only",
            "dutch": "Dutch with Latin alphabet only",
            "italian": "Italian with Latin alphabet only", 
            "polish": "Polish with Latin alphabet only",
            "portuguese": "Portuguese with Latin alphabet only",
            # Common additional languages
            "hindi": "Devanagari script (Hindi) only",
            "bengali": "Bengali script only"
        }
        
        target_script_instruction = language_output_map.get(target_language.lower(), f"{target_language} script only")
        
        prompt = f"""
PROFESSIONAL DUBBING TRANSLATION TASK

You are an expert dubbing translator specializing in video content translation from {detected_language} to {target_language}.

CRITICAL DUBBING REQUIREMENTS:
1. OUTPUT SCRIPT: Use {target_script_instruction} - NO mixing of scripts or languages
2. NATURAL SPEECH: Translate for spoken dialogue, not written text  
3. TIMING MATCH: Keep similar syllable count and speaking duration ({sum(req['duration'] for req in requests):.1f}s total)
4. EMOTIONAL PRESERVATION: Maintain speaker's tone, energy, and emotional intent
5. CULTURAL ADAPTATION: Use culturally natural expressions for {target_language} speakers
6. CONVERSATIONAL STYLE: Use natural, spoken language (contractions, informal speech)

FISH SPEECH TTS OPTIMIZATION & TIMING CONTROL:
- Use clear, pronounceable {target_language} text
- Add emotional markers when appropriate: (happy), (excited), (sad), (serious), (casual), (confident), etc.
- Use natural pauses with commas and periods for timing control
- Add strategic extra spaces "   " at sentence endings for duration matching
- Use ellipses "..." for longer pauses when needed
- Control speaking speed with punctuation and spacing
- Match speaking rhythm to original timing ({sum(req['duration'] for req in requests):.1f}s total)

SEGMENTS TO TRANSLATE:
{json.dumps(context_info, indent=2, ensure_ascii=False)}

TRANSLATION GUIDELINES & TIMING CONTROL:
1. Each segment MUST be translated to proper {target_language} using {target_script_instruction}
2. Preserve speaker personality and speaking style
3. **CRITICAL TIMING MATCH**: Each segment must fill its exact duration to avoid silence gaps
4. Add appropriate Fish Speech emotion markers: (happy), (excited), (serious), (casual), etc.
5. Maintain conversational flow between segments
6. Consider video context (podcast, interview, presentation, etc.)

TIMING SYNCHRONIZATION TECHNIQUES:
- If translation is shorter than original: Add strategic extra spaces "   " between words
- Use ellipses "..." for natural pauses and timing extension
- Add descriptive words or filler phrases when culturally appropriate
- Use slower pacing markers like commas, periods for natural delays
- Example: "What was the reason?   The actual reason for their success was...   hard work and one more thing."
- Target: Fill the ENTIRE segment duration with natural-sounding speech

MANDATORY OUTPUT FORMAT (JSON):
{{
  "translations": [
    {{
      "segment_id": "segment_001",
      "original_text": "original text in source language",
      "translated_text": "TRANSLATED TEXT WITH TIMING CONTROL (extra spaces   and pauses... as needed)",
      "emotion_detected": "happy/sad/neutral/excited/serious/casual/etc",
      "translation_notes": "timing strategy used (added spaces/pauses/extended phrases)",
      "duration_match": "extended_to_match/natural_match/compressed",
      "timing_adjustments": "spaces added between words / ellipses for pauses / extended phrasing",
      "estimated_speech_time": "{req['duration']:.1f}s target"
    }}
  ],
  "translation_summary": {{
    "source_language": "{detected_language}",
    "target_language": "{target_language}",
    "target_script": "{target_script_instruction}",
    "total_segments": {len(requests)},
    "total_duration": "{sum(req['duration'] for req in requests):.1f}s",
    "translation_approach": "timing-aware dubbing with duration matching",
    "sync_strategy": "spaces and pauses added to match original video timing"
  }}
}}

IMPORTANT: All translated_text MUST be in {target_language} using {target_script_instruction}. NO exceptions.
"""
        return prompt
    
    def _get_translation_system_prompt(self, target_language: str) -> str:
        """Get system prompt for translation"""
        
        # Fish Speech supported languages script requirements
        script_requirements = {
            # Fish Speech supported languages
            "english": "English alphabet only - no foreign scripts",
            "chinese": "Chinese characters only - simplified/traditional Chinese",
            "japanese": "Japanese characters only - Hiragana/Katakana/Kanji",
            "german": "German with Latin alphabet only - proper German text",
            "french": "French with Latin alphabet only - proper French text", 
            "spanish": "Spanish with Latin alphabet only - proper Spanish text",
            "korean": "Korean Hangul only - proper Korean text",
            "arabic": "Arabic script only - proper Arabic text",
            "russian": "Russian Cyrillic script only - proper Russian text",
            "dutch": "Dutch with Latin alphabet only - proper Dutch text",
            "italian": "Italian with Latin alphabet only - proper Italian text",
            "polish": "Polish with Latin alphabet only - proper Polish text", 
            "portuguese": "Portuguese with Latin alphabet only - proper Portuguese text",
            # Common additional languages
            "hindi": "Devanagari script only - proper Hindi text",
            "bengali": "Bengali script only - proper Bengali text"
        }
        
        script_req = script_requirements.get(target_language.lower(), f"{target_language} script only")
        
        return f"""You are an expert dubbing translator with extensive experience in video localization and voice-over work.

CRITICAL LANGUAGE OUTPUT REQUIREMENT:
- ALL translations MUST use {script_req}
- NO mixing of scripts or languages in output
- Translations must be grammatically correct {target_language}

Your specializations:
- Professional video dubbing translation
- Cross-cultural adaptation for {target_language} audiences  
- Natural speech patterns and conversational flow
- Text-to-Speech optimization for AI voice cloning
- Emotional tone and speaker personality preservation
- Timing-aware translation (matching speaking duration)

Dubbing translation principles:
- Translate for spoken dialogue, not written text
- Use natural, conversational {target_language} 
- Maintain speaker's emotional tone and personality
- **CRITICAL**: Match exact segment duration to prevent silence gaps
- Add strategic spaces and pauses for timing synchronization
- Use timing control: extra spaces "   ", ellipses "...", natural pauses
- Add appropriate emotion markers for TTS enhancement
- Ensure cultural authenticity for {target_language} speakers
- Fill the entire audio duration with natural-sounding speech

MANDATORY: Every translated text must be in proper {target_language} using {script_req}. This is non-negotiable for dubbing quality."""
    
    def _validate_translation_script(self, text: str, target_language: str) -> bool:
        """Validate that translated text uses correct script for target language"""
        if not text:
            return True
        
        target_lang = target_language.lower()
        script_detected = self._detect_language_from_script(text)
        
        # Fish Speech supported languages to script mapping
        expected_scripts = {
            # Fish Speech supported languages
            "english": "en",
            "chinese": "zh", 
            "japanese": "ja",
            "german": "de",     # Uses Latin script
            "french": "fr",     # Uses Latin script
            "spanish": "es",    # Uses Latin script
            "korean": "ko",
            "arabic": "ar",
            "russian": "ru",
            "dutch": "nl",      # Uses Latin script
            "italian": "it",    # Uses Latin script
            "polish": "pl",     # Uses Latin script
            "portuguese": "pt", # Uses Latin script
            # Common additional languages
            "hindi": "hi",
            "bengali": "bn"
        }
        
        expected_script = expected_scripts.get(target_lang, "en")
        
        # For Latin-script languages (English, German, French, Spanish, Dutch, Italian, Polish, Portuguese)
        latin_script_languages = ["english", "german", "french", "spanish", "dutch", "italian", "polish", "portuguese"]
        
        if target_lang in latin_script_languages:
            latin_chars = sum(1 for char in text if char.isalpha() and ord(char) <= 0x024F)
            total_alpha_chars = sum(1 for char in text if char.isalpha())
            
            if total_alpha_chars > 0:
                latin_percentage = (latin_chars / total_alpha_chars) * 100
                return latin_percentage > 80  # At least 80% Latin characters for Latin-script languages
            return True
        
        # For non-Latin script languages, check exact script match
        return script_detected == expected_script
    
    def _optimize_timing_for_tts(self, text: str, target_duration: float) -> str:
        """Simple space-based timing optimization for TTS"""
        if not text or target_duration <= 0:
            return text
        
        # Simple word count estimation: ~2 words/second average speech
        words = text.strip().split()
        estimated_time = len(words) / 2.0
        
        # If text is shorter than target, add strategic spaces for timing
        if estimated_time < target_duration * 0.8:  # If less than 80% of target
            optimized_text = text
            
            # Add extra spaces between words for timing control
            optimized_text = optimized_text.replace(' ', '  ')  # Double spaces
            
            # Add longer pauses at sentence boundaries
            optimized_text = optimized_text.replace('. ', '.   ')
            optimized_text = optimized_text.replace('? ', '?   ')
            optimized_text = optimized_text.replace('! ', '!   ')
            optimized_text = optimized_text.replace(', ', ',  ')
            
            # Add trailing spaces if still short
            if estimated_time < target_duration * 0.6:
                optimized_text = optimized_text + "   "
            
            logger.info(f"Space timing applied: {estimated_time:.1f}s -> target {target_duration:.1f}s")
            return optimized_text
        
        return text
    
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
                    translated_text = translation.get("translated_text", segment.get("text", ""))
                    
                    # Apply timing optimization for better video sync
                    segment_duration = segment.get("duration", 1.0)
                    optimized_text = self._optimize_timing_for_tts(translated_text, segment_duration)
                    
                    # Validate translation script
                    target_language = translation_result.get("translation_summary", {}).get("target_language", "English")
                    is_valid_script = self._validate_translation_script(optimized_text, target_language)
                    
                    if not is_valid_script:
                        logger.warning(f"Translation validation failed for segment {segment_id}: incorrect script for {target_language}")
                        # Fallback to original text if validation fails
                        final_text = segment.get("text", "")
                        translation_quality = "validation_failed_fallback"
                        script_valid = False
                    else:
                        final_text = optimized_text  # Use timing-optimized text
                        translation_quality = "openai_gpt4o_timing_optimized"
                        script_valid = True
                    
                    # Add translation fields
                    segment_copy["original_text"] = segment.get("text", "")
                    segment_copy["text"] = final_text
                    segment_copy["english_text"] = final_text
                    
                    # Add translation metadata with timing information
                    timing_applied = optimized_text != translated_text
                    segment_copy["translation"] = {
                        "translated": script_valid,
                        "script_validation": is_valid_script,
                        "emotion_detected": translation.get("emotion_detected", "neutral"),
                        "translation_notes": translation.get("translation_notes", ""),
                        "duration_match": translation.get("duration_match", "similar"),
                        "timing_adjustments": translation.get("timing_adjustments", "automatic_spacing" if timing_applied else "none"),
                        "estimated_speech_time": f"{segment_duration:.1f}s",
                        "translation_quality": translation_quality,
                        "target_language": target_language,
                        "sync_optimized": True,  # Indicates timing-aware translation
                        "original_duration": segment.get("duration", 0),
                        "timing_optimization_applied": timing_applied,
                        "pre_optimization_text": translated_text if timing_applied else None
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
    
    def _detect_language_from_script(self, text: str) -> str:
        """Detect language based on script/characters in text"""
        if not text:
            return "en"
        
        # Count different script characters (Fish Speech supported languages)
        script_counts = {
            'latin': 0,       # English, German, French, Spanish, Dutch, Italian, Polish, Portuguese
            'chinese': 0,     # Chinese (zh)
            'japanese': 0,    # Japanese (ja) 
            'korean': 0,      # Korean (ko)
            'arabic': 0,      # Arabic (ar)
            'cyrillic': 0,    # Russian (ru)
            'devanagari': 0,  # Hindi (not in Fish Speech list but common)
            'bengali': 0      # Bengali (not in Fish Speech list but common)
        }
        
        for char in text:
            code_point = ord(char)
            # Devanagari script (Hindi) - U+0900 to U+097F
            if 0x0900 <= code_point <= 0x097F:
                script_counts['devanagari'] += 1
            # Arabic script (Arabic) - U+0600 to U+06FF
            elif 0x0600 <= code_point <= 0x06FF:
                script_counts['arabic'] += 1
            # Bengali script - U+0980 to U+09FF
            elif 0x0980 <= code_point <= 0x09FF:
                script_counts['bengali'] += 1
            # Cyrillic script (Russian) - U+0400 to U+04FF
            elif 0x0400 <= code_point <= 0x04FF:
                script_counts['cyrillic'] += 1
            # CJK characters (Chinese)
            elif 0x4E00 <= code_point <= 0x9FFF:  # CJK Unified Ideographs
                script_counts['chinese'] += 1
            # Japanese characters
            elif 0x3040 <= code_point <= 0x309F or 0x30A0 <= code_point <= 0x30FF:  # Hiragana/Katakana
                script_counts['japanese'] += 1
            # Korean characters
            elif 0xAC00 <= code_point <= 0xD7AF:  # Hangul
                script_counts['korean'] += 1
            # Latin script (English, German, French, Spanish, Dutch, Italian, Polish, Portuguese)
            elif char.isalpha() and code_point <= 0x024F:
                script_counts['latin'] += 1
        
        # Determine language based on dominant script
        total_chars = sum(script_counts.values())
        if total_chars == 0:
            return "en"
        
        # Calculate percentages
        script_percentages = {script: (count / total_chars) * 100 
                            for script, count in script_counts.items()}
        
        # If more than 20% non-Latin characters, override AssemblyAI detection
        non_latin_percentage = 100 - script_percentages['latin']
        
        if non_latin_percentage > 20:
            # Find dominant script
            dominant_script = max(script_counts, key=script_counts.get)
            
            script_to_language = {
                'devanagari': 'hi',   # Hindi (common but not in Fish Speech list)
                'arabic': 'ar',       # Arabic (Fish Speech supported)
                'bengali': 'bn',      # Bengali (common but not in Fish Speech list)  
                'cyrillic': 'ru',     # Russian (Fish Speech supported)
                'chinese': 'zh',      # Chinese (Fish Speech supported)
                'japanese': 'ja',     # Japanese (Fish Speech supported)
                'korean': 'ko'        # Korean (Fish Speech supported)
            }
            
            return script_to_language.get(dominant_script, "en")
        
        # For Latin script, default to English (Fish Speech supported)
        # Note: Fish Speech also supports: de, fr, es, nl, it, pl, pt with Latin script
        return "en"
    
    def _is_same_language(self, detected: str, target: str) -> bool:
        """Check if detected and target languages are the same"""
        # Normalize language codes
        detected_norm = detected.lower().strip()
        target_norm = target.lower().strip()
        
        # Direct match
        if detected_norm == target_norm:
            return True
        
        # Fish Speech supported languages with common variations
        language_mapping = {
            "english": ["en", "eng", "english"],
            "chinese": ["zh", "chi", "chinese", "mandarin", "zh-cn", "zh-tw"],
            "japanese": ["ja", "jpn", "japanese"],
            "german": ["de", "deu", "german", "deutsch"],
            "french": ["fr", "fra", "french", "français"],
            "spanish": ["es", "spa", "spanish", "español"],
            "korean": ["ko", "kor", "korean", "한국어"],
            "arabic": ["ar", "ara", "arabic", "العربية"],
            "russian": ["ru", "rus", "russian", "русский"],
            "dutch": ["nl", "nld", "dutch", "nederlands"],
            "italian": ["it", "ita", "italian", "italiano"],
            "polish": ["pl", "pol", "polish", "polski"],
            "portuguese": ["pt", "por", "portuguese", "português"],
            # Common but not Fish Speech supported
            "hindi": ["hi", "hin", "hindi", "हिन्दी"],
            "bengali": ["bn", "ben", "bengali", "bangla", "বাংলা"]
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
