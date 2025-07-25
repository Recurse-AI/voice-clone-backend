"""
Transcription and Text Processing Module - Simplified
"""

import re
import time
from typing import Dict, Any, List, Optional
import assemblyai as aai
from openai import OpenAI
from config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
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
                        speakers_expected: Optional[int] = None, audio_id: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio using AssemblyAI"""
        try:
            print(f"Starting transcription for audio: {audio_path}")
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
            print(f"Transcription completed in {transcription_time:.2f} seconds")
            
            # Store complete AssemblyAI response as JSON metadata
            if audio_id:
                self._save_assemblyai_response(transcript, audio_id)
            
            words = self._extract_words(transcript)
            speakers = self._extract_speakers(words)
            final_language_code = self._get_language_code(transcript, language_code)
            
            return {
                "text": transcript.text,
                "words": words,
                "speakers": speakers,
                "duration": words[-1]['end'] / 1000 if words else 0,
                "metadata": {
                    "language_code": final_language_code,
                    "speakers_expected": speakers_expected,
                    "detected_speakers": len(speakers),
                    "transcript_id": transcript.id,
                    "transcription_time": transcription_time
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
        """Simple translation to English with clean [S1]/[S2] speaker tags"""
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
            
            # Pre-process multi-speaker text with speaker indicators
            if is_multi_speaker and words_data:
                processed_text = self._preprocess_multispeaker_text(clean_text, words_data)
            else:
                processed_text = clean_text
            
            # Check cache
            with self.cache_lock:
                cache_key = f"{processed_text.strip()}_{is_multi_speaker}"
                if cache_key in self.translation_cache:
                    return self.translation_cache[cache_key]
            
            # Simple translation and formatting
            try:
                if len(speakers_in_segment) > 1:
                    # Multi-speaker format with pre-processed speaker indicators
                    prompt = f"""Translate to natural English with proper speaker tags.

TEXT WITH SPEAKER MARKERS: {processed_text}

RULES:
- Always start segment with [S1] tag regardless of which speaker starts
- Use [S2], [S3] etc. for subsequent speakers within the same segment
- NO tags for continuation lines of same speaker
- 3-9 words per line optimal for best Dia performance
- Natural conversational English
- No quotes in output
- Each line should be clear and simple
- This will used for Dia model voice cloning

EXAMPLE INPUT: <SPEAKER1> Hello how are you doing today <SPEAKER2> I am fine thanks for asking <SPEAKER1> That's good to hear
EXAMPLE OUTPUT: 
[S1] Hello how are you
doing today
[S2] I am fine thanks
for asking
[S3] That's good to hear

OUTPUT (English with speaker tags - always start with [S1]):"""
                else:
                    # Single speaker format
                    prompt = f"""Translate to natural English with speaker tag.

TEXT: {clean_text}

RULES:
- Start with [S1] tag only at beginning
- NO tags for continuation lines
- 3-9 words per line optimal for best Dia performance
- Natural conversational English
- No quotes in output
- Keep it simple and clear

EXAMPLE OUTPUT:
[S1] Hello this is example text
yes I understand perfectly now
great let's continue the discussion

OUTPUT (English with [S1] tag only at start):"""
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Translate with clean speaker formatting. Keep it simple."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1,
                    timeout=45  # Increased timeout
                )
                
                if response and response.choices:
                    formatted_text = response.choices[0].message.content.strip()
                    formatted_text = self._clean_formatted_text_simple(formatted_text)
                    
                    with self.cache_lock:
                        self.translation_cache[cache_key] = formatted_text
                    return formatted_text
                
                raise ValueError("No valid OpenAI response")
                
            except Exception:
                # Simple fallback translation
                return self._simple_translate_and_format(clean_text, is_multi_speaker)
                
        except Exception as e:
            raise ValueError(f"Text formatting failed for speaker {speaker}: {str(e)}")
    
    def _clean_formatted_text_simple(self, text: str) -> str:
        """Simple cleanup of formatted text"""
        # Remove quotes
        text = re.sub(r'^["\s]*', '', text).strip()
        text = re.sub(r'["\s]*$', '', text)
        
        if not text:
            raise ValueError("Empty response from OpenAI")
        
        # Ensure speaker tag
        if '[S' not in text:
            text = f"[S1] {text}"
        
        return text
    
    def _simple_translate_and_format(self, text: str, is_multi_speaker: bool = False) -> str:
        """Simple translation with basic formatting"""
        try:
            translation_response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Translate to natural English only."},
                    {"role": "user", "content": f"Translate: {text}"}
                ],
                max_tokens=150,
                temperature=0.1,
                timeout=30  # Increased timeout
            )
            
            if translation_response and translation_response.choices:
                english_text = translation_response.choices[0].message.content.strip()
                # Simple format - just add [S1] tag
                if '[S' not in english_text:
                    english_text = f"[S1] {english_text}"
                return english_text
            else:
                raise ValueError("Translation failed")
        except Exception as e:
            raise ValueError(f"Translation failed: {str(e)}")
    

    
    def format_dialogue_batch(self, text_list: List[str], speaker_data: List, words_data_list: List[List[Dict]] = None) -> List[str]:
        """Process multiple dialogue texts in parallel - enhanced for multi-speaker"""
        if not text_list:
            logger.warning("format_dialogue_batch: Empty text_list provided")
            return []
        
        logger.info(f"format_dialogue_batch: Processing {len(text_list)} texts")
        logger.info(f"format_dialogue_batch: Speaker data type: {type(speaker_data)}, length: {len(speaker_data) if speaker_data else 0}")
        
        # Handle both old format (speaker_list) and new format (speaker_data_list)
        if speaker_data and isinstance(speaker_data[0], dict):
            speaker_data_list = speaker_data
            logger.info("format_dialogue_batch: Using new format speaker data")
        else:
            # Old single speaker format - convert to new format
            speaker_data_list = []
            for speaker in speaker_data:
                speaker_data_list.append({
                    'speakers': [speaker],
                    'is_multi_speaker': False,
                    'primary_speaker': speaker
                })
            logger.info(f"format_dialogue_batch: Converted old format, created {len(speaker_data_list)} speaker data items")
        
        if len(text_list) != len(speaker_data_list):
            logger.error(f"format_dialogue_batch: Length mismatch - text_list: {len(text_list)}, speaker_data_list: {len(speaker_data_list)}")
            logger.error(f"format_dialogue_batch: Text list: {text_list}")
            logger.error(f"format_dialogue_batch: Speaker data: {speaker_data}")
            return []
        
        # Ensure words_data_list has same length
        if words_data_list is None:
            words_data_list = [None] * len(text_list)
        elif len(words_data_list) != len(text_list):
            words_data_list = [None] * len(text_list)
        
        # Use ThreadPoolExecutor for parallel processing with timeout
        with ThreadPoolExecutor(max_workers=min(4, len(text_list))) as executor:
            # Submit all tasks with words_data
            future_to_index = {}
            for i, (text, speaker_data_item, words_data) in enumerate(zip(text_list, speaker_data_list, words_data_list)):
                future = executor.submit(self.format_dialogue_text, text, speaker_data_item, words_data)
                future_to_index[future] = i
            
            # Collect results in order with timeout
            results = [''] * len(text_list)
            try:
                for future in as_completed(future_to_index, timeout=120):  # 2 minute timeout total
                    index = future_to_index[future]
                    try:
                        results[index] = future.result(timeout=60)  # 1 minute per translation
                    except Exception as e:
                        logger.error(f"Translation failed for text at index {index}: {str(e)}")
                        results[index] = ""
            except Exception as e:
                logger.error(f"Translation batch timeout or error: {str(e)}")
                # Fill remaining empty results
                for i in range(len(results)):
                    if not results[i]:
                        results[i] = ""
        
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