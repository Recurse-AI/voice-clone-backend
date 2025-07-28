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
        """Enhanced translation to English with clean formatting and consistent reference text"""
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
                cache_key = f"{clean_text.strip()}_{is_multi_speaker}_v4_reference_match"
                if cache_key in self.translation_cache:
                    return self.translation_cache[cache_key]
            
            # Enhanced translation with reference code style formatting
            try:
                if len(speakers_in_segment) > 1:
                    # Multi-speaker format following reference patterns
                    processed_text = self._preprocess_multispeaker_text(clean_text, words_data) if words_data else clean_text
                    
                    prompt = f"""Translate to natural English with clean speaker tags for voice cloning.

TEXT: {processed_text}

RULES:
- Always start with [S1] tag at the beginning
- Use [S2], [S3] etc. for different speakers
- Put EACH speaker on a NEW LINE - very important
- Keep lines natural and clear for voice synthesis
- Natural conversational English
- Lines should follow natural speech patterns
- Maintain speaker consistency throughout
- Try to keep the text as close to the original as possible
- Don't make any line too long - break into multiple lines if text is long
- Try to keep single line around 9 words or less

EXAMPLE OUTPUT:
[S1] I will take care of all the cookies in a minute
[S2] Just gather all the information you want about the cookies

OUTPUT (English with clean speaker tags, each speaker on new line):"""
                else:
                    # Single speaker format following reference patterns
                    prompt = f"""Translate to natural English for voice cloning synthesis.

TEXT: {clean_text}

RULES:
- Start with [S1] tag only at beginning
- NO additional speaker tags needed
- Keep lines natural for voice synthesis
- Natural conversational English
- Lines should follow natural speech patterns
- Maintain consistency for voice cloning
- Keep the original meaning and emotion
- Try to keep the text as close to the original as possible
- Don't make any line too long - break into multiple lines if text is long
- Try to keep the text as close to the original as possible and don't make it too long
- Try to keep single line around 9 words or less

EXAMPLE OUTPUT:
[S1] Hello this is an example of natural speech
that flows well for voice synthesis

OUTPUT (English with [S1] tag, proper line breaks):"""
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Translate with clean formatting optimized for voice cloning. Use proper line breaks. Each speaker should be on a new line. Don't put everything on one single line."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.1,
                    timeout=45
                )
                
                if response and response.choices:
                    formatted_text = response.choices[0].message.content.strip()
                    formatted_text = self._clean_and_format_reference_style(formatted_text)
                    
                    with self.cache_lock:
                        self.translation_cache[cache_key] = formatted_text
                    return formatted_text
                
                raise ValueError("No valid OpenAI response")
                
            except Exception as openai_error:
                logger.warning(f"OpenAI formatting failed: {openai_error}")
                # Enhanced fallback with reference style
                return self._enhanced_fallback_reference_style(clean_text, is_multi_speaker)
                
        except Exception as e:
            raise ValueError(f"Enhanced text formatting failed for speaker {speaker}: {str(e)}")
    
    def _clean_and_format_reference_style(self, text: str) -> str:
        """Clean formatted text following reference code patterns"""
        # Remove quotes and extra whitespace
        text = re.sub(r'^["\s]*', '', text).strip()
        text = re.sub(r'["\s]*$', '', text)
        
        
        if not text:
            raise ValueError("Empty response from OpenAI")
        
        # Ensure proper speaker tag format
        if not re.search(r'\[S\d+\]', text):
            text = f"[S1] {text}"
        
        # Clean up multiple spaces and normalize formatting
        text = re.sub(r'\s+', ' ', text)
        
        # Apply reference-style line breaking (natural speech patterns)
        formatted_text = self._apply_reference_line_breaking(text)
        
        return formatted_text
    
    def _apply_reference_line_breaking(self, text: str) -> str:
        """Apply line breaking following reference code natural speech patterns with proper speaker separation"""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line has speaker tag
            speaker_match = re.match(r'(\[S\d+\])\s*(.*)', line)
            if speaker_match:
                speaker_tag = speaker_match.group(1)
                content = speaker_match.group(2)
                
                if content:
                    # Each speaker should be on its own line with content
                    formatted_lines.append(f"{speaker_tag} {content}")
                else:
                    formatted_lines.append(speaker_tag)
            else:
                # Continuation line without speaker tag - keep on separate line
                formatted_lines.append(line)
        
        # Ensure we don't have everything on one line
        result = '\n'.join(formatted_lines)
        
        # Additional check: if result has multiple [S tags but no newlines, force line breaks
        if result.count('[S') > 1 and '\n' not in result:
            # Split on speaker tags and rejoin with newlines
            parts = re.split(r'(\[S\d+\])', result)
            new_lines = []
            current_line = ""
            
            for part in parts:
                if re.match(r'\[S\d+\]', part):
                    if current_line.strip():
                        new_lines.append(current_line.strip())
                    current_line = part
                else:
                    current_line += part
            
            if current_line.strip():
                new_lines.append(current_line.strip())
            
            result = '\n'.join(new_lines)
        
        return result
    
    def _enhanced_fallback_reference_style(self, text: str, is_multi_speaker: bool = False) -> str:
        """Enhanced fallback formatting following reference patterns"""
        try:
            # Simple translation first
            translation_response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Translate to natural English only. Keep it simple and natural for voice synthesis. Use proper line breaks."},
                    {"role": "user", "content": f"Translate this to natural English: {text}"}
                ],
                max_tokens=150,
                temperature=0.1,
                timeout=30
            )
            
            if translation_response and translation_response.choices:
                english_text = translation_response.choices[0].message.content.strip()
                
                # Remove single quotes
                english_text = english_text.replace("'", "")
                english_text = english_text.replace("'", "")
                english_text = english_text.replace("'", "")
                
                # Apply simple formatting
                if not re.search(r'\[S\d+\]', english_text):
                    english_text = f"[S1] {english_text}"
                
                # Apply reference-style formatting
                formatted_text = self._apply_reference_line_breaking(english_text)
                
                return formatted_text
            else:
                raise ValueError("Translation failed")
                
        except Exception as e:
            # Ultimate fallback with reference formatting
            logger.error(f"Enhanced fallback failed: {e}")
            cleaned = self._clean_text(text)
            
            # Remove single quotes from fallback too
            cleaned = cleaned.replace("'", "")
            cleaned = cleaned.replace("'", "")
            cleaned = cleaned.replace("'", "")
            
            if not cleaned.startswith('[S'):
                cleaned = f"[S1] {cleaned}"
            
            # Apply reference formatting to fallback too
            return self._apply_reference_line_breaking(cleaned)
    
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
        """Clean text for processing - remove quotes, normalize spacing"""
        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Remove single quotes completely for clean formatting
        text = text.replace("'", "")
        text = text.replace("'", "")
        text = text.replace("'", "")
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text)
        
        # Clean text
        text = text.strip()
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text