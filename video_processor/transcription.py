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
    
    def format_dialogue_text(self, text: str, speaker: str, is_multi_speaker: bool = False) -> str:
        """Translate text to English and format into multi-line dialogue with speaker tags - No fallback"""
        try:
            # First clean the text
            clean_text = self._clean_text(text)
            
            if not clean_text.strip():
                raise ValueError(f"No text available for speaker {speaker}")
            
            # Check cache first for repeated translations
            with self.cache_lock:
                cache_key = f"{clean_text.strip()}_{speaker}"
                if cache_key in self.translation_cache:
                    return self.translation_cache[cache_key]
            
            # Try OpenAI translation and formatting
            try:
                # For voice cloning, always use [S1] for individual segments
                speaker_num = 1
                
                prompt = f"""You are a professional dubbing translator. Convert this text to natural Engligh.

CRITICAL REQUIREMENTS:
- Translate to natural, conversational English suitable for dubbing
- Try to keep 7-15 words per line when possible, adjust if needed for natural flow
- Don't break lines in the middle of a sentence - keep full sentences together when possible
- Use [S1] tag ONLY ONCE at the beginning (single speaker segment)
- Continue with new lines WITHOUT repeating the speaker tag
- Only add non-verbal sounds when truly necessary and natural to the content (don't force them)
- Preserve emotional context and speaking style
- Each line should be naturally speakable
- Match the original's rhythm and pacing
- Break at natural speech pauses

Original text: "{clean_text}"

DUBBING FORMAT EXAMPLE:
[S1] That's really amazing work you did there.
I'm very impressed with the quality.
The attention to detail is excellent.
Everything looks professional and polished.

Convert to natural English dubbing format:"""
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional dubbing translator specializing in voice cloning dialogue. Create natural, speakable English text with emotional context. Try to keep 7-15 words per line when possible, adjust if needed for natural flow. Don't break lines in the middle of a sentence - keep full sentences together when possible. Only add non-verbal sounds when truly necessary and natural to the content. IMPORTANT: Use [S1] tag ONLY ONCE at the beginning for each segment, then continue with plain lines."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=320, 
                    temperature=0.1,  
                    timeout=30 
                )
                
                if response and response.choices:
                    formatted_text = response.choices[0].message.content.strip()
                    # Clean up the response
                    formatted_text = self._clean_formatted_text(formatted_text)
                    
                    # Validate the formatted text has speaker tags and English content
                    if formatted_text and '[S' in formatted_text:
                        # Cache the result
                        with self.cache_lock:
                            self.translation_cache[cache_key] = formatted_text
                        return formatted_text
                    else:
                        raise ValueError("Invalid OpenAI response format")
                
                # If no valid response, use fallback
                raise ValueError("No valid OpenAI response")
                
            except Exception as openai_error:
                # Use simple fallback translation and formatting
                return self._simple_translate_and_format(clean_text)
                
        except Exception as e:
            # No ultimate fallback - raise error if everything fails
            raise ValueError(f"Text formatting failed for speaker {speaker}: {str(e)}")
    
    def format_dialogue_batch(self, text_list: List[str], speaker_list: List[str]) -> List[str]:
        """Process multiple dialogue texts in parallel for better performance"""
        if not text_list or len(text_list) != len(speaker_list):
            return []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(4, len(text_list))) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, (text, speaker) in enumerate(zip(text_list, speaker_list)):
                future = executor.submit(self.format_dialogue_text, text, speaker, False)
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
    
    def _simple_translate_and_format(self, text: str) -> str:
        """Simple translation and formatting - no fallback"""
        try:
            # Simple translation attempt
            translation_response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Translate to natural English only. No formatting."},
                    {"role": "user", "content": f"Translate to English: {text}"}
                ],
                max_tokens=100,
                temperature=0.1,
                timeout=5
            )
            
            if translation_response and translation_response.choices:
                english_text = translation_response.choices[0].message.content.strip()
                return self._simple_format_text(english_text)
            else:
                # If translation fails, raise error
                raise ValueError("Translation service returned no response")
                
        except Exception as e:
            # No fallback - raise error if translation fails
            raise ValueError(f"Translation failed: {str(e)}")
    
    def _clean_formatted_text(self, text: str) -> str:
        """Clean formatted text from OpenAI response - handles new format with speaker tags only once"""
        # Remove extra quotation marks
        text = re.sub(r'^["\s]*', '', text)
        text = re.sub(r'["\s]*$', '', text)
        
        # Ensure proper line breaks
        lines = text.split('\n')
        cleaned_lines = []
        has_speaker_tag = False
        
        for line in lines:
            line = line.strip()
            if line:
                # Keep lines with speaker tags and plain lines after a speaker tag
                if '[S' in line:
                    cleaned_lines.append(line)
                    has_speaker_tag = True
                elif has_speaker_tag and line:
                    # Plain line after a speaker tag - keep it
                    cleaned_lines.append(line)
        
        if not cleaned_lines:
            raise ValueError("Text could not be cleaned and formatted")
        
        # Ensure at least one line has a speaker tag
        if not any('[S' in line for line in cleaned_lines):
            cleaned_lines[0] = f"[S1] {cleaned_lines[0]}"
        
        return '\n'.join(cleaned_lines)
    
    def _simple_format_text(self, text: str) -> str:
        """Simple text formatting with speaker tag only once - no fallback"""
        words = text.split()
        if not words:
            raise ValueError("Text has no words to format")
        
        lines = []
        current_line = []
        first_line = True
        
        for word in words:
            current_line.append(word)
            
            # Break at sentence end if we have enough words (7+)
            if len(current_line) >= 7 and word.endswith(('.', '!', '?')):
                if first_line:
                    lines.append(f"[S1] {' '.join(current_line)}")
                    first_line = False
                else:
                    lines.append(' '.join(current_line))
                current_line = []
            # Force break if too long (15+ words)
            elif len(current_line) >= 15:
                if first_line:
                    lines.append(f"[S1] {' '.join(current_line)}")
                    first_line = False
                else:
                    lines.append(' '.join(current_line))
                current_line = []
        
        if current_line:
            if first_line:
                lines.append(f"[S1] {' '.join(current_line)}")
            else:
                lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
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