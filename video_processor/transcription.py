"""
Transcription and Text Processing Module - Simplified
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
    """Simplified transcription service with parallel processing"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        self.translation_cache = {}  # Cache for translations
        self.cache_lock = threading.Lock()  # Thread-safe cache access
    
    def transcribe_audio(self, audio_path: str, language_code: Optional[str] = None, 
                        speakers_expected: Optional[int] = None, audio_id: Optional[str] = None,
                        original_duration: Optional[float] = None) -> Dict[str, Any]:
        """Transcribe audio using AssemblyAI"""
        try:
            logger.info(f"Starting transcription for audio: {audio_path}")
            start_time = time.time()
            
            config_params = {
                "speaker_labels": True,
                "punctuate": True,
                "format_text": True,
                "speech_model": aai.SpeechModel.universal,
                "boost_param": "high"  # Increase accuracy for common words
            }
            
            if language_code and language_code.strip():
                config_params["language_code"] = language_code.strip()
            else:
                config_params["language_detection"] = True
            
            if speakers_expected and 1 <= speakers_expected <= 10:
                config_params["speakers_expected"] = speakers_expected
            
            config = aai.TranscriptionConfig(**config_params)
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_path, config=config)
            
            if transcript.status == "error":
                raise Exception(f"Transcription failed: {transcript.error}")
            
            transcription_time = time.time() - start_time
            logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
            
            # Store complete AssemblyAI response as JSON metadata
            if audio_id:
                self._save_assemblyai_response(transcript, audio_id)
            
            words = self._extract_words(transcript)
            speakers = self._extract_speakers(words)
            final_language_code = self._get_language_code(transcript, language_code)
            
            # Calculate transcribed duration
            transcribed_duration = words[-1]['end'] / 1000 if words else 0
            
            return {
                "text": transcript.text,
                "words": words,
                "speakers": speakers,
                "duration": transcribed_duration,  # Duration of transcribed content only
                "audio_duration": original_duration or transcribed_duration,  # Full original audio duration
                "metadata": {
                    "language_code": final_language_code,
                    "speakers_expected": speakers_expected,
                    "detected_speakers": len(speakers),
                    "transcript_id": transcript.id,
                    "transcription_time": transcription_time,
                    "transcribed_duration": transcribed_duration,
                    "original_duration": original_duration
                }
            }
            
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")

    def _save_assemblyai_response(self, transcript, audio_id: str):
        """Save complete AssemblyAI response as JSON metadata in segments directory"""
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
            
            # Direct conversion to dict using json_response if available
            if hasattr(transcript, 'json_response'):
                response_data = transcript.json_response
            else:
                # Fallback: convert transcript attributes to dict
                response_data = transcript.__dict__
            
            with open(assemblyai_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
                
        except Exception:
            pass

    def _extract_words(self, transcript) -> List[Dict[str, Any]]:
        """Extract words from transcript"""
        words = []
        
        if hasattr(transcript, 'words') and transcript.words:
            for word in transcript.words:
                word_data = {
                    "text": getattr(word, 'text', '').strip(),
                    "start": getattr(word, 'start', 0),
                    "end": getattr(word, 'end', 0),
                    "speaker": self._get_word_speaker(word),
                    "confidence": getattr(word, 'confidence', 0.5)
                }
                
                if word_data["text"]:
                    words.append(word_data)
        
        elif hasattr(transcript, 'utterances') and transcript.utterances:
            for utterance in transcript.utterances:
                utterance_speaker = getattr(utterance, 'speaker', 'A')
                utterance_start = getattr(utterance, 'start', 0)
                utterance_end = getattr(utterance, 'end', 5000)
                utterance_text = getattr(utterance, 'text', '')
                
                if hasattr(utterance, 'words') and utterance.words:
                    for word in utterance.words:
                        word_data = {
                            "text": getattr(word, 'text', '').strip(),
                            "start": getattr(word, 'start', utterance_start),
                            "end": getattr(word, 'end', utterance_start + 500),
                            "speaker": getattr(word, 'speaker', utterance_speaker),
                            "confidence": getattr(word, 'confidence', 0.5)
                        }
                        
                        if word_data["text"]:
                            words.append(word_data)
                else:
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
                                "confidence": 0.5
                            })
        
        return words

    def _get_word_speaker(self, word) -> str:
        """Get speaker from word"""
        speaker = getattr(word, 'speaker', None)
        if speaker is None or speaker == "null":
            return "A"
        return str(speaker)

    def _extract_speakers(self, words: List[Dict[str, Any]]) -> List[str]:
        """Extract unique speakers from words"""
        if not words:
            return ["A"]
        
        speakers = set()
        for word in words:
            speaker = word.get('speaker', 'A')
            if speaker and speaker != "null":
                speakers.add(str(speaker))
        
        if not speakers:
            speakers.add("A")
        
        return sorted(list(speakers))

    def _get_language_code(self, transcript, language_code: Optional[str]) -> str:
        """Get final language code"""
        if language_code and language_code.strip():
            return language_code.strip()
        
        try:
            return transcript.json_response.get("language_code", "")
        except:
            return ""
    
    def _preprocess_multispeaker_text(self, text: str, words_data: List[Dict]) -> str:
        """Pre-process text to add speaker indicators for better OpenAI handling"""
        if not words_data:
            return text
        
        # Group words by speaker in chronological order
        processed_parts = []
        current_speaker = None
        current_words = []
        
        for word_data in words_data:
            word_text = word_data.get('text', '').strip()
            word_speaker = word_data.get('speaker', 'A')
            
            if word_text:
                if current_speaker != word_speaker:
                    # Speaker change - save previous group
                    if current_words:
                        processed_parts.append({
                            'speaker': current_speaker,
                            'text': ' '.join(current_words)
                        })
                        current_words = []
                    current_speaker = word_speaker
                
                current_words.append(word_text)
        
        # Add final group
        if current_words:
            processed_parts.append({
                'speaker': current_speaker,
                'text': ' '.join(current_words)
            })
        
        # Create speaker-marked text
        marked_text = ""
        for part in processed_parts:
            speaker_num = ord(part['speaker']) - ord('A') + 1
            marked_text += f"<SPEAKER{speaker_num}> {part['text']} "
        
        return marked_text.strip()
    
    def format_dialogue_text(self, text: str, speaker_data, words_data: List[Dict] = None) -> str:
        """Enhanced translation to English with clean [S1]/[S2] speaker tags and proper line breaking"""
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
            
            clean_text = self._clean_text(text)
            if not clean_text.strip():
                raise ValueError(f"No text available for speaker {speaker}")
            
            # Check cache first
            with self.cache_lock:
                cache_key = f"{clean_text.strip()}_{is_multi_speaker}_colab_v2"
                if cache_key in self.translation_cache:
                    return self.translation_cache[cache_key]
            
            # Enhanced translation and formatting with strict line breaking
            try:
                if len(speakers_in_segment) > 1:
                    # Multi-speaker format with strict line breaking
                    processed_text = self._preprocess_multispeaker_text(clean_text, words_data) if words_data else clean_text
                    
                    prompt = f"""Translate to natural English with clean speaker tags and strict line breaking.

TEXT: {processed_text}

RULES:
- Always start with [S1] tag at the beginning
- Use [S2], [S3] etc. for different speakers
- NO tags for continuation lines of same speaker  
- MAXIMUM 9 words per line - VERY IMPORTANT
- If line has more than 9 words, break to new line
- Natural conversational English
- No quotes in output
- Keep simple and clear

EXAMPLE OUTPUT:
[S1] Hello how are you doing
today my friend
[S2] I am fine thanks for
asking about my health
[S1] That's good to hear from you

OUTPUT (English with clean speaker tags and max 9 words per line):"""
                else:
                    # Single speaker format with strict line breaking
                    prompt = f"""Translate to natural English with strict line breaking.

TEXT: {clean_text}

RULES:
- Start with [S1] tag only at beginning
- NO additional speaker tags needed
- MAXIMUM 9 words per line - VERY IMPORTANT
- If line has more than 9 words, break to new line
- Natural conversational English
- No quotes in output
- Keep simple and clear

EXAMPLE OUTPUT:
[S1] Hello this is example text that
demonstrates proper line breaking with
maximum nine words per line always

OUTPUT (English with [S1] tag and max 9 words per line):"""
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Translate with clean speaker formatting and strict line breaking (max 9 words per line) optimized for Dia voice cloning."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1,
                    timeout=45
                )
                
                if response and response.choices:
                    formatted_text = response.choices[0].message.content.strip()
                    formatted_text = self._clean_and_format_with_line_breaking(formatted_text)
                    
                    with self.cache_lock:
                        self.translation_cache[cache_key] = formatted_text
                    return formatted_text
                
                raise ValueError("No valid OpenAI response")
                
            except Exception as openai_error:
                logger.warning(f"OpenAI formatting failed: {openai_error}")
                # Enhanced fallback with proper line breaking
                return self._enhanced_fallback_with_line_breaking(clean_text, is_multi_speaker)
                
        except Exception as e:
            raise ValueError(f"Enhanced text formatting failed for speaker {speaker}: {str(e)}")
    
    def _clean_and_format_with_line_breaking(self, text: str) -> str:
        """Clean formatted text and ensure proper line breaking (max 9 words per line)"""
        # Remove quotes and extra whitespace
        text = re.sub(r'^["\s]*', '', text).strip()
        text = re.sub(r'["\s]*$', '', text)
        
        if not text:
            raise ValueError("Empty response from OpenAI")
        
        # Ensure proper speaker tag format
        if not re.search(r'\[S\d+\]', text):
            text = f"[S1] {text}"
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Apply strict line breaking (max 9 words per line)
        formatted_text = self._apply_strict_line_breaking(text)
        
        return formatted_text
    
    def _apply_strict_line_breaking(self, text: str) -> str:
        """Apply strict line breaking with maximum 9 words per line"""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            words = line.split()
            if len(words) <= 9:
                # Line is fine as is
                formatted_lines.append(line)
            else:
                # Break line into chunks of max 9 words
                current_line = []
                speaker_tag = None
                
                for word in words:
                    # Check if word is a speaker tag
                    if word.startswith('[S') and ']' in word:
                        # If we have accumulated words, save them first
                        if current_line:
                            formatted_lines.append(' '.join(current_line))
                            current_line = []
                        speaker_tag = word
                        current_line.append(word)
                    else:
                        current_line.append(word)
                        
                        # If we hit 9 words, start a new line
                        if len(current_line) >= 9:
                            formatted_lines.append(' '.join(current_line))
                            current_line = []
                
                # Add remaining words
                if current_line:
                    formatted_lines.append(' '.join(current_line))
        
        return '\n'.join(formatted_lines)
    
    def _enhanced_fallback_with_line_breaking(self, text: str, is_multi_speaker: bool = False) -> str:
        """Enhanced fallback formatting with strict line breaking"""
        try:
            # Simple translation first
            translation_response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Translate to natural English only. Keep it simple."},
                    {"role": "user", "content": f"Translate this to natural English: {text}"}
                ],
                max_tokens=150,
                temperature=0.1,
                timeout=30
            )
            
            if translation_response and translation_response.choices:
                english_text = translation_response.choices[0].message.content.strip()
                
                # Apply simple formatting
                if not re.search(r'\[S\d+\]', english_text):
                    english_text = f"[S1] {english_text}"
                
                # Apply strict line breaking
                formatted_text = self._apply_strict_line_breaking(english_text)
                
                return formatted_text
            else:
                raise ValueError("Translation failed")
                
        except Exception as e:
            # Ultimate fallback with line breaking
            logger.error(f"Enhanced fallback failed: {e}")
            cleaned = self._clean_text(text)
            if not cleaned.startswith('[S'):
                cleaned = f"[S1] {cleaned}"
            
            # Apply line breaking to fallback too
            return self._apply_strict_line_breaking(cleaned)
    
    def format_dialogue_batch(self, text_list: List[str], speaker_data: List, words_data_list: List[List[Dict]] = None) -> List[str]:
        """Simplified batch dialogue processing"""
        if not text_list:
            return []
        
        # Simplified speaker data handling
        if not speaker_data:
            speaker_data = ['A'] * len(text_list)
        
        # Ensure speaker_data matches text_list length
        if len(speaker_data) != len(text_list):
            logger.warning(f"Speaker data length mismatch, using default speakers")
            speaker_data = ['A'] * len(text_list)
        
        # Ensure words_data_list has same length
        if words_data_list is None:
            words_data_list = [None] * len(text_list)
        elif len(words_data_list) != len(text_list):
            words_data_list = [None] * len(text_list)
        
        # Process each text with simplified logic
        results = []
        for i, (text, speaker, words_data) in enumerate(zip(text_list, speaker_data, words_data_list)):
            try:
                # Create simple speaker data format
                simple_speaker_data = {
                    'speakers': [speaker] if isinstance(speaker, str) else speaker.get('speakers', [speaker]),
                    'is_multi_speaker': False,
                    'primary_speaker': speaker if isinstance(speaker, str) else speaker.get('primary_speaker', 'A')
                }
                
                formatted_text = self.format_dialogue_text(text, simple_speaker_data, words_data)
                results.append(formatted_text)
                
            except Exception as e:
                logger.error(f"Failed to format text {i+1}: {str(e)}")
                results.append(text)  # Use original text as fallback
        
        return results
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing"""
        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text)
        
        # Clean text
        text = text.strip()
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text