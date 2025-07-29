"""
Transcription and Text Processing Module - Enhanced for OpenVoice
Improved AssemblyAI integration with natural text formatting for voice cloning
"""

import re
import time
from typing import Dict, Any, List, Optional
import assemblyai as aai
from openai import OpenAI
from config import settings
import threading
import logging

logger = logging.getLogger(__name__)

class TranscriptionService:
    """Enhanced transcription service optimized for OpenVoice voice cloning"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        self.translation_cache = {}  # Cache for translations
        self.cache_lock = threading.Lock()  # Thread-safe cache access
        logger.info("🎤 Initialized TranscriptionService for OpenVoice")
    
    def transcribe_audio(self, audio_path: str, language_code: Optional[str] = None, 
                        speakers_expected: Optional[int] = None, audio_id: Optional[str] = None,
                        original_duration: Optional[float] = None) -> Dict[str, Any]:
        """Enhanced transcription using AssemblyAI with better error handling"""
        try:
            logger.info(f"🎯 Starting enhanced transcription for: {audio_path}")
            start_time = time.time()
            
            # Enhanced configuration for better accuracy
            config_params = {
                "speaker_labels": True,
                "punctuate": True,
                "format_text": True,
                "speech_model": aai.SpeechModel.nano,  # Use nano for better speed/accuracy balance
                "boost_param": "high",  # Increase accuracy for common words
                "filter_profanity": False,  # Keep original content
                "redact_pii": False,  # Keep personal information for voice cloning
                "dual_channel": False,  # Process as single channel for voice cloning
                "word_boost": ["um", "uh", "ah", "like", "you know"],  # Boost filler words for natural speech
                "auto_highlights": False  # Disable for voice cloning focus
            }
            
            # Enhanced language detection and configuration
            if language_code and language_code.strip():
                # Validate and normalize language code
                lang_code = language_code.strip().lower()
                valid_languages = {
                    'en': 'en_us', 'english': 'en_us',
                    'es': 'es', 'spanish': 'es', 
                    'fr': 'fr', 'french': 'fr',
                    'de': 'de', 'german': 'de',
                    'it': 'it', 'italian': 'it',
                    'pt': 'pt', 'portuguese': 'pt',
                    'nl': 'nl', 'dutch': 'nl',
                    'ja': 'ja', 'japanese': 'ja',
                    'ko': 'ko', 'korean': 'ko',
                    'zh': 'zh', 'chinese': 'zh',
                    'ar': 'ar', 'arabic': 'ar',
                    'ru': 'ru', 'russian': 'ru',
                    'hi': 'hi', 'hindi': 'hi'
                }
                
                normalized_lang = valid_languages.get(lang_code, lang_code)
                config_params["language_code"] = normalized_lang
                logger.info(f"🌍 Using language: {normalized_lang}")
            else:
                config_params["language_detection"] = True
                logger.info("🌍 Using automatic language detection")
            
            # Enhanced speaker configuration
            if speakers_expected and 1 <= speakers_expected <= 10:
                config_params["speakers_expected"] = speakers_expected
                logger.info(f"👥 Expected speakers: {speakers_expected}")
            else:
                # Auto-detect with reasonable limits
                config_params["speakers_expected"] = 2  # Default to 2 for better detection
                logger.info("👥 Using automatic speaker detection (default: 2)")
            
            # Create config and transcriber
            config = aai.TranscriptionConfig(**config_params)
            transcriber = aai.Transcriber()
            
            logger.info("📡 Submitting to AssemblyAI...")
            transcript = transcriber.transcribe(audio_path, config=config)
            
            if transcript.status == "error":
                raise Exception(f"AssemblyAI transcription failed: {transcript.error}")
            
            transcription_time = time.time() - start_time
            logger.info(f"✅ Transcription completed in {transcription_time:.2f} seconds")
            
            # Store complete AssemblyAI response as JSON metadata
            if audio_id:
                self._save_assemblyai_response(transcript, audio_id)
            
            # Enhanced word extraction with better error handling
            words = self._extract_words_enhanced(transcript)
            speakers = self._extract_speakers_enhanced(words)
            final_language_code = self._get_language_code_enhanced(transcript, language_code)
            
            # Calculate transcribed duration with fallback
            transcribed_duration = self._calculate_transcribed_duration(words, original_duration)
            
            # Enhanced metadata
            metadata = {
                "language_code": final_language_code,
                "speakers_expected": speakers_expected,
                "detected_speakers": len(speakers),
                "transcript_id": transcript.id,
                "transcription_time": transcription_time,
                "transcribed_duration": transcribed_duration,
                "original_duration": original_duration,
                "confidence_avg": self._calculate_average_confidence(words),
                "word_count": len(words),
                "speaker_distribution": self._calculate_speaker_distribution(words)
            }
            
            logger.info(f"📊 Transcription stats: {len(words)} words, {len(speakers)} speakers, {final_language_code} language")
            
            return {
                "text": transcript.text,
                "words": words,
                "speakers": speakers,
                "duration": transcribed_duration,
                "audio_duration": original_duration or transcribed_duration,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {str(e)}")
            raise Exception(f"Enhanced transcription failed: {str(e)}")

    def _extract_words_enhanced(self, transcript) -> List[Dict[str, Any]]:
        """Enhanced word extraction with better error handling and validation"""
        words = []
        
        try:
            # Primary extraction from transcript.words
            if hasattr(transcript, 'words') and transcript.words:
                logger.info(f"📝 Extracting from {len(transcript.words)} transcript words")
                
                for i, word in enumerate(transcript.words):
                    try:
                        word_text = getattr(word, 'text', '').strip()
                        word_start = getattr(word, 'start', 0)
                        word_end = getattr(word, 'end', word_start + 500)
                        word_speaker = self._get_word_speaker_enhanced(word)
                        word_confidence = getattr(word, 'confidence', 0.5)
                        
                        # Validate word data
                        if word_text and word_start >= 0 and word_end > word_start:
                            word_data = {
                                "text": word_text,
                                "start": int(word_start),
                                "end": int(word_end),
                                "speaker": word_speaker,
                                "confidence": float(word_confidence)
                            }
                            words.append(word_data)
                        
                    except Exception as e:
                        logger.warning(f"Skipping invalid word {i}: {e}")
                        continue
            
            # Fallback extraction from utterances
            elif hasattr(transcript, 'utterances') and transcript.utterances:
                logger.info(f"📝 Fallback: Extracting from {len(transcript.utterances)} utterances")
                
                for utterance in transcript.utterances:
                    try:
                        utterance_speaker = getattr(utterance, 'speaker', 'A')
                        utterance_start = getattr(utterance, 'start', 0)
                        utterance_end = getattr(utterance, 'end', utterance_start + 5000)
                        utterance_text = getattr(utterance, 'text', '')
                        utterance_confidence = getattr(utterance, 'confidence', 0.5)
                        
                        # Extract words from utterance
                        if hasattr(utterance, 'words') and utterance.words:
                            for word in utterance.words:
                                word_text = getattr(word, 'text', '').strip()
                                word_start = getattr(word, 'start', utterance_start)
                                word_end = getattr(word, 'end', word_start + 500)
                                word_speaker = getattr(word, 'speaker', utterance_speaker)
                                word_confidence = getattr(word, 'confidence', utterance_confidence)
                                
                                if word_text:
                                    words.append({
                                        "text": word_text,
                                        "start": int(word_start),
                                        "end": int(word_end),
                                        "speaker": word_speaker,
                                        "confidence": float(word_confidence)
                                    })
                        else:
                            # Split utterance text into words
                            word_list = utterance_text.split()
                            if word_list:
                                word_duration = (utterance_end - utterance_start) / len(word_list)
                                
                                for i, word_text in enumerate(word_list):
                                    word_start = utterance_start + (i * word_duration)
                                    word_end = word_start + word_duration
                                    
                                    words.append({
                                        "text": word_text.strip(),
                                        "start": int(word_start),
                                        "end": int(word_end),
                                        "speaker": utterance_speaker,
                                        "confidence": float(utterance_confidence)
                                    })
                    
                    except Exception as e:
                        logger.warning(f"Skipping invalid utterance: {e}")
                        continue
            
            # Final fallback from transcript text
            elif hasattr(transcript, 'text') and transcript.text:
                logger.warning("⚠️ Final fallback: Creating words from transcript text")
                text = transcript.text.strip()
                if text:
                    word_list = text.split()
                    estimated_duration = len(word_list) * 600  # 600ms per word estimate
                    
                    for i, word_text in enumerate(word_list):
                        word_start = i * 600
                        word_end = word_start + 600
                        
                        words.append({
                            "text": word_text.strip(),
                            "start": word_start,
                            "end": word_end,
                            "speaker": "A",
                            "confidence": 0.3  # Low confidence for estimated timing
                        })
            
            logger.info(f"✅ Extracted {len(words)} words successfully")
            return words
            
        except Exception as e:
            logger.error(f"❌ Word extraction failed: {e}")
            return []

    def _get_word_speaker_enhanced(self, word) -> str:
        """Enhanced speaker extraction with better validation"""
        try:
            speaker = getattr(word, 'speaker', None)
            
            # Handle various speaker formats
            if speaker is None or speaker == "null" or speaker == "":
                return "A"
            
            # Convert to string and normalize
            speaker_str = str(speaker).strip()
            
            # Handle numeric speakers (0, 1, 2...) -> (A, B, C...)
            if speaker_str.isdigit():
                speaker_num = int(speaker_str)
                return chr(ord('A') + speaker_num) if speaker_num < 26 else f"S{speaker_num}"
            
            # Handle already formatted speakers
            if speaker_str.upper() in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                return speaker_str.upper()
            
            # Default fallback
            return "A"
            
        except Exception:
            return "A"

    def _extract_speakers_enhanced(self, words: List[Dict[str, Any]]) -> List[str]:
        """Enhanced speaker extraction with better sorting and validation"""
        if not words:
            return ["A"]
        
        speakers = set()
        for word in words:
            speaker = word.get('speaker', 'A')
            if speaker and speaker != "null":
                speakers.add(str(speaker))
        
        if not speakers:
            speakers.add("A")
        
        # Sort speakers alphabetically for consistency
        sorted_speakers = sorted(list(speakers))
        logger.info(f"👥 Detected speakers: {sorted_speakers}")
        return sorted_speakers

    def _get_language_code_enhanced(self, transcript, provided_language_code: Optional[str]) -> str:
        """Enhanced language code detection with better fallback"""
        # Use provided language if available
        if provided_language_code and provided_language_code.strip():
            return provided_language_code.strip()
        
        # Try to get from transcript
        try:
            if hasattr(transcript, 'json_response') and transcript.json_response:
                detected_lang = transcript.json_response.get("language_code", "")
                if detected_lang:
                    return detected_lang
        except Exception:
            pass
        
        # Try direct attribute access
        try:
            if hasattr(transcript, 'language_code'):
                return getattr(transcript, 'language_code', "")
        except Exception:
            pass
        
        # Default fallback
        return "en"
    
    def _calculate_transcribed_duration(self, words: List[Dict], original_duration: Optional[float]) -> float:
        """Calculate transcribed duration with better fallback handling"""
        if not words:
            return original_duration or 0.0
        
        try:
            # Use the last word's end time
            last_word_end = max(word.get('end', 0) for word in words) / 1000.0
            return last_word_end
        except Exception:
            # Fallback to original duration or estimated duration
            return original_duration or len(words) * 0.6  # 600ms per word estimate
    
    def _calculate_average_confidence(self, words: List[Dict]) -> float:
        """Calculate average confidence score"""
        if not words:
            return 0.0
        
        try:
            confidences = [word.get('confidence', 0.0) for word in words]
            return sum(confidences) / len(confidences)
        except Exception:
            return 0.5
    
    def _calculate_speaker_distribution(self, words: List[Dict]) -> Dict[str, int]:
        """Calculate word count per speaker"""
        distribution = {}
        
        for word in words:
            speaker = word.get('speaker', 'A')
            distribution[speaker] = distribution.get(speaker, 0) + 1
        
        return distribution

    def _save_assemblyai_response(self, transcript, audio_id: str):
        """Enhanced AssemblyAI response saving with better error handling"""
        try:
            import json
            from pathlib import Path
            from config import settings
            
            # Create path to segments metadata directory
            temp_dir = Path(settings.TEMP_DIR)
            segments_dir = temp_dir / f"segments_{audio_id}"
            metadata_dir = segments_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            assemblyai_file = metadata_dir / "assemblyai_response.json"
            
            # Enhanced data extraction
            response_data = {}
            
            # Try json_response first
            if hasattr(transcript, 'json_response') and transcript.json_response:
                response_data = transcript.json_response
            else:
                # Fallback: extract key attributes
                response_data = {
                    "id": getattr(transcript, 'id', ''),
                    "status": getattr(transcript, 'status', ''),
                    "text": getattr(transcript, 'text', ''),
                    "confidence": getattr(transcript, 'confidence', 0.0),
                    "language_code": getattr(transcript, 'language_code', ''),
                    "audio_duration": getattr(transcript, 'audio_duration', 0)
                }
            
            # Add processing metadata
            response_data['processing_metadata'] = {
                "processed_at": time.time(),
                "audio_id": audio_id,
                "service_version": "enhanced_v2"
            }
            
            with open(assemblyai_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"💾 Saved enhanced AssemblyAI response to {assemblyai_file}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to save AssemblyAI response: {e}")

    def format_dialogue_text(self, text: str, speaker_data, words_data: List[Dict] = None) -> str:
        """Enhanced translation optimized for OpenVoice with natural formatting"""
        try:
            # Handle speaker data format
            if isinstance(speaker_data, str):
                speaker = speaker_data
                is_multi_speaker = False
                speakers_in_segment = [speaker]
                primary_speaker = speaker
            else:
                is_multi_speaker = speaker_data.get('is_multi_speaker', False)
                speakers_in_segment = speaker_data.get('speakers', ['A'])
                primary_speaker = speaker_data.get('primary_speaker', 'A')
                speaker = primary_speaker
            
            clean_text = text.strip()
            
            # Skip empty text
            if not clean_text:
                return ""
            
            # Check cache first
            with self.cache_lock:
                cache_key = f"{clean_text}_{is_multi_speaker}_openvoice_v1"
                if cache_key in self.translation_cache:
                    return self.translation_cache[cache_key]
            
            # Enhanced translation optimized for OpenVoice
            try:
                if len(speakers_in_segment) > 1:
                    # Multi-speaker format optimized for OpenVoice (natural dialogue)
                    processed_text = self._preprocess_multispeaker_text_enhanced(clean_text, words_data) if words_data else clean_text
                    
                    prompt = f"""Translate to natural English optimized for OpenVoice voice cloning.

TEXT: {processed_text}

RULES for OpenVoice:
- Create natural dialogue without speaker tags or markers
- Each speaker's line should be on a separate line
- Use natural conversational English that flows well when spoken
- Maintain emotional tone and natural speech patterns
- Preserve natural pauses and rhythm
- Keep lines moderately short for better voice synthesis
- Use quotation marks to separate different speakers
- Create realistic conversation flow

EXAMPLE OUTPUT:
"I understand what you're saying about the project timeline."
"Yes, we need to be more realistic about our deadlines."
"Let's discuss this further in our next meeting."

OUTPUT (Natural English dialogue, each speaker on new line):"""
                else:
                    # Single speaker format optimized for OpenVoice
                    prompt = f"""Translate to natural English optimized for OpenVoice voice cloning.

TEXT: {clean_text}

RULES for OpenVoice:
- Create natural, conversational English
- NO speaker tags or special markers needed
- Keep natural conversational flow for voice synthesis
- Maintain emotional tone and speaking style
- Use clear, natural language that flows well when spoken
- Preserve natural speech rhythm and pauses
- Break into natural speech segments if too long
- Optimize for natural voice cloning

EXAMPLE OUTPUT:
This is a natural example of how speech should flow when we're talking about something important, and it should sound conversational and engaging.

OUTPUT (Natural English, optimized for voice synthesis):"""
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert at formatting text for OpenVoice voice cloning. Create natural, conversational English that flows perfectly when spoken aloud. Use natural dialogue format without special tags."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400,
                    temperature=0.1,
                    timeout=45
                )
                
                if response and response.choices:
                    formatted_text = response.choices[0].message.content.strip()
                    
                    # Post-process for OpenVoice optimization
                    formatted_text = self._post_process_for_openvoice(formatted_text)
                    
                    with self.cache_lock:
                        self.translation_cache[cache_key] = formatted_text
                    
                    logger.debug(f"✅ Formatted text for OpenVoice: {formatted_text[:100]}...")
                    return formatted_text
                
                raise ValueError("No valid OpenAI response")
                
            except Exception as openai_error:
                logger.warning(f"⚠️ OpenAI formatting failed: {openai_error}")
                # Fallback to simple formatting
                return self._simple_format_fallback(clean_text, speaker)
                
        except Exception as e:
            logger.error(f"❌ Enhanced text formatting failed for speaker {speaker}: {str(e)}")
            return self._simple_format_fallback(text, speaker if isinstance(speaker_data, str) else "A")
    
    def _preprocess_multispeaker_text_enhanced(self, text: str, words_data: List[Dict]) -> str:
        """Enhanced multi-speaker text preprocessing for OpenVoice"""
        if not words_data:
            return text
        
        # Group words by speaker with better timing analysis
        speaker_segments = []
        current_speaker = None
        current_words = []
        current_start = None
        
        for word_data in words_data:
            word_text = word_data.get('text', '').strip()
            word_speaker = word_data.get('speaker', 'A')
            word_start = word_data.get('start', 0)
            
            if word_text:
                if current_speaker != word_speaker:
                    # Speaker change - save previous segment
                    if current_words and current_speaker:
                        speaker_segments.append({
                            'speaker': current_speaker,
                            'text': ' '.join(current_words),
                            'start': current_start,
                            'word_count': len(current_words)
                        })
                    
                    current_speaker = word_speaker
                    current_words = [word_text]
                    current_start = word_start
                else:
                    current_words.append(word_text)
        
        # Add final segment
        if current_words and current_speaker:
            speaker_segments.append({
                'speaker': current_speaker,
                'text': ' '.join(current_words),
                'start': current_start,
                'word_count': len(current_words)
            })
        
        # Create speaker-marked text with better formatting
        marked_text = ""
        for i, segment in enumerate(speaker_segments):
            speaker_num = ord(segment['speaker']) - ord('A') + 1
            marked_text += f"<SPEAKER{speaker_num}> {segment['text']}"
            if i < len(speaker_segments) - 1:
                marked_text += " "
        
        return marked_text.strip()
    
    def _post_process_for_openvoice(self, text: str) -> str:
        """Post-process formatted text for OpenVoice optimization"""
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove any leftover speaker tags (in case they appear)
        text = re.sub(r'\[S\d+\]\s*', '', text)
        
        # Clean up quotation marks - ensure proper spacing
        text = re.sub(r'"\s*([^"]+)\s*"', r'"\1"', text)
        
        # Ensure natural speech flow with proper punctuation spacing
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure each line ends with proper punctuation
        lines = text.split('\n')
        processed_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.endswith(('.', '!', '?')):
                line += '.'
            if line:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _simple_format_fallback(self, text: str, speaker: str) -> str:
        """Simple fallback formatting for OpenVoice (natural text)"""
        clean_text = text.strip()
        if not clean_text:
            return ""
        
        # Ensure proper punctuation
        if not clean_text.endswith(('.', '!', '?')):
            clean_text += '.'
        
        # Return natural text without speaker tags
        return clean_text
    
    def format_dialogue_batch(self, text_list: List[str], speaker_data: List, words_data_list: List[List[Dict]] = None) -> List[str]:
        """Enhanced batch dialogue processing for OpenVoice"""
        if not text_list:
            return []
        
        # Ensure data consistency
        if not speaker_data:
            speaker_data = ['A'] * len(text_list)
        
        if len(speaker_data) != len(text_list):
            logger.warning(f"⚠️ Speaker data length mismatch, using default speakers")
            speaker_data = ['A'] * len(text_list)
        
        if words_data_list is None:
            words_data_list = [None] * len(text_list)
        elif len(words_data_list) != len(text_list):
            words_data_list = [None] * len(text_list)
        
        # Process with enhanced error handling
        results = []
        for i, (text, speaker, words_data) in enumerate(zip(text_list, speaker_data, words_data_list)):
            try:
                # Create enhanced speaker data format
                if isinstance(speaker, str):
                    enhanced_speaker_data = {
                        'speakers': [speaker],
                        'is_multi_speaker': False,
                        'primary_speaker': speaker
                    }
                else:
                    enhanced_speaker_data = {
                        'speakers': speaker.get('speakers', [speaker.get('primary_speaker', 'A')]),
                        'is_multi_speaker': speaker.get('is_multi_speaker', False),
                        'primary_speaker': speaker.get('primary_speaker', 'A')
                    }
                
                formatted_text = self.format_dialogue_text(text, enhanced_speaker_data, words_data)
                results.append(formatted_text)
                
            except Exception as e:
                logger.error(f"❌ Failed to format text {i+1}: {str(e)}")
                # Enhanced fallback
                fallback_text = self._simple_format_fallback(text, "A")
                results.append(fallback_text)
        
        logger.info(f"✅ Batch processed {len(results)} texts for OpenVoice")
        return results
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning for OpenVoice"""
        # Remove quotes and extra whitespace
        text = re.sub(r'^["\s]*', '', text).strip()
        text = re.sub(r'["\s]*$', '', text)
        
        # Clean up multiple spaces and normalize formatting
        text = re.sub(r'\s+', ' ', text)
        
        # Preserve natural speech patterns (don't over-clean)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
