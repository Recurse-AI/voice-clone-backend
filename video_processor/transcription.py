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
            return transcript.json_response.get("language_code", "en")
        except:
            return "en"
    
    def format_dialogue_text(self, text: str, speaker_data, is_multi_speaker: bool = False) -> str:
        """Translate text to English and format with proper speaker tags - supports both old and new format"""
        try:
            # Handle both old format (speaker string) and new format (speaker_data dict)
            if isinstance(speaker_data, str):
                # Old format - single speaker
                speaker = speaker_data
                is_multi_speaker = False
                speakers_in_segment = [speaker]
                primary_speaker = speaker
            else:
                # New format - speaker_data dict
                is_multi_speaker = speaker_data.get('is_multi_speaker', False)
                speakers_in_segment = speaker_data.get('speakers', ['A'])
                primary_speaker = speaker_data.get('primary_speaker', 'A')
                speaker = primary_speaker  # For backward compatibility
            
            # First clean the text
            clean_text = self._clean_text(text)
            
            if not clean_text.strip():
                raise ValueError(f"No text available for speaker {speaker}")
            
            # Check cache first for repeated translations
            with self.cache_lock:
                cache_key = f"{clean_text.strip()}_{is_multi_speaker}_{len(speakers_in_segment)}"
                if cache_key in self.translation_cache:
                    return self.translation_cache[cache_key]
            
            # Try OpenAI translation and formatting
            try:
                if is_multi_speaker and len(speakers_in_segment) > 1:
                    # Multi-speaker prompt with [S1], [S2] tags
                    speaker_tags = ", ".join([f"[S{i+1}]" for i in range(len(speakers_in_segment))])
                    
                    prompt = f"""Translate to natural English with proper speaker tags for voice cloning.

TEXT: "{clean_text}"

FORMAT RULES:
- Use {speaker_tags} for {len(speakers_in_segment)} speakers
- 3-7 words per line maximum
- Start each speaker turn with speaker tag
- Keep sentences complete when possible
- Natural, conversational English

EXAMPLE:
[S1] Hello there friend.
How are you today?
[S2] I'm doing great.
Thanks for asking.
[S1] That's wonderful news.

OUTPUT (English with speaker tags):"""
                else:
                    # Single speaker prompt
                    prompt = f"""Translate to natural English for voice cloning.

TEXT: "{clean_text}"

FORMAT RULES:
- Start with [S1] tag once
- 3-7 words per line maximum
- Natural, conversational English
- Keep sentences complete when possible

EXAMPLE:
[S1] Hello there friend.
How are you today?
I'm glad to hear.

OUTPUT (English with [S1] tag):"""
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional translator. Follow the format rules exactly. Translate to natural English with proper speaker tags for voice cloning."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400, 
                    temperature=0.1,  
                    timeout=30 
                )
                
                if response and response.choices:
                    formatted_text = response.choices[0].message.content.strip()
                    # Minimal cleanup - trust OpenAI output
                    formatted_text = self._clean_formatted_text(formatted_text, is_multi_speaker)
                    
                    # Cache the result
                    with self.cache_lock:
                        self.translation_cache[cache_key] = formatted_text
                    return formatted_text
                
                # If no valid response, use fallback
                raise ValueError("No valid OpenAI response")
                
            except Exception as openai_error:
                # Use simple fallback translation and formatting
                return self._simple_translate_and_format(clean_text, is_multi_speaker)
                
        except Exception as e:
            # No ultimate fallback - raise error if everything fails
            raise ValueError(f"Text formatting failed for speaker {speaker}: {str(e)}")
    
    def format_dialogue_batch(self, text_list: List[str], speaker_data: List) -> List[str]:
        """Process multiple dialogue texts in parallel - supports both old format and new multi-speaker format"""
        if not text_list:
            return []
        
        # Handle both old format (speaker_list) and new format (speaker_data_list)
        if speaker_data and isinstance(speaker_data[0], dict):
            # New multi-speaker format
            speaker_data_list = speaker_data
        else:
            # Old single speaker format - convert to new format
            speaker_data_list = []
            for speaker in speaker_data:
                speaker_data_list.append({
                    'speakers': [speaker],
                    'is_multi_speaker': False,
                    'primary_speaker': speaker
                })
        
        if len(text_list) != len(speaker_data_list):
            return []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(4, len(text_list))) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, (text, speaker_data_item) in enumerate(zip(text_list, speaker_data_list)):
                future = executor.submit(self.format_dialogue_text, text, speaker_data_item, False)
                future_to_index[future] = i
            
            # Collect results in order
            results = [''] * len(text_list)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"Translation failed for text at index {index}: {str(e)}")
                    results[index] = f"[S1] {text_list[index]}"
        
        return results
    

    
    def _simple_translate_and_format(self, text: str, is_multi_speaker: bool = False) -> str:
        """Simple translation and formatting - no fallback"""
        try:
            # Simple translation attempt
            translation_response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Translate to natural English only. No formatting."},
                    {"role": "user", "content": f"Translate to English: {text}"}
                ],
                max_tokens=150,
                temperature=0.1,
                timeout=5
            )
            
            if translation_response and translation_response.choices:
                english_text = translation_response.choices[0].message.content.strip()
                return self._simple_format_text(english_text, is_multi_speaker)
            else:
                # If translation fails, raise error
                raise ValueError("Translation service returned no response")
                
        except Exception as e:
            # No fallback - raise error if translation fails
            raise ValueError(f"Translation failed: {str(e)}")
    
    def _clean_formatted_text(self, text: str, is_multi_speaker: bool = False) -> str:
        """Clean formatted text from OpenAI response - minimal processing, trust OpenAI output"""
        # Remove extra quotation marks only
        text = re.sub(r'^["\s]*', '', text)
        text = re.sub(r'["\s]*$', '', text)
        
        # Basic cleanup and ensure at least one speaker tag
        text = text.strip()
        if not text:
            raise ValueError("Empty response from OpenAI")
        
        # If no speaker tag found, add [S1] at beginning
        if '[S' not in text:
            text = f"[S1] {text}"
        
        return text
    
    def _simple_format_text(self, text: str, is_multi_speaker: bool = False) -> str:
        """Simple text formatting with speaker tags - minimal processing"""
        if not text.strip():
            raise ValueError("Text has no content to format")
        
        # Simple format - just add [S1] if no speaker tag exists
        text = text.strip()
        if '[S' not in text:
            text = f"[S1] {text}"
        
        return text
    
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