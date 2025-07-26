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
            return transcript.json_response.get("language_code", "")
        except:
            return ""
    
    def format_dialogue_text(self, text: str, speaker_data, is_multi_speaker: bool = False) -> str:
        """Translate text to English with optimized formatting for Dia model performance"""
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
            
            # Get optimal format based on text length
            word_count = len(clean_text.split())
            optimal_format = self._get_optimal_format_for_dia(word_count, is_multi_speaker)
            
            # Check cache
            with self.cache_lock:
                cache_key = f"{clean_text.strip()}_{is_multi_speaker}_{len(speakers_in_segment)}_dia_{optimal_format['strategy']}"
                if cache_key in self.translation_cache:
                    return self.translation_cache[cache_key]
            
            # Create optimized prompt
            try:
                if is_multi_speaker and len(speakers_in_segment) > 1:
                    speaker_tags = ", ".join([f"[S{i+1}]" for i in range(len(speakers_in_segment))])
                    prompt = f"""Translate to natural English with Dia-optimized formatting.

TEXT: "{clean_text}"

RULES:
- Use {speaker_tags} for {len(speakers_in_segment)} speakers
- {optimal_format['words_per_line']} words per line
- Target {optimal_format['target_lines']} total lines
- Natural, conversational English

EXAMPLE:
[S1] Hello there my dear friend today.
How are you doing right now?
[S2] I'm doing quite well, thank you.
Life has been treating me well.
[S1] That's wonderful news to hear today.

OUTPUT (English with speaker tags):"""
                else:
                    prompt = f"""Translate to natural English with Dia-optimized formatting.

TEXT: "{clean_text}"

RULES:
- Start with [S1] tag once
- {optimal_format['words_per_line']} words per line
- Target {optimal_format['target_lines']} total lines
- Natural, conversational English
- {optimal_format['explanation']}

OUTPUT (English with [S1] tag):"""
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Translate with optimal Dia voice model formatting. Follow formatting rules exactly."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400, 
                    temperature=0.05,
                    timeout=30 
                )
                
                if response and response.choices:
                    formatted_text = response.choices[0].message.content.strip()
                    formatted_text = self._clean_formatted_text_for_dia(formatted_text)
                    
                    with self.cache_lock:
                        self.translation_cache[cache_key] = formatted_text
                    return formatted_text
                
                raise ValueError("No valid OpenAI response")
                
            except Exception:
                return self._simple_translate_and_format_for_dia(clean_text, optimal_format)
                
        except Exception as e:
            raise ValueError(f"Text formatting failed for speaker {speaker}: {str(e)}")
    
    def _get_optimal_format_for_dia(self, word_count: int, is_multi_speaker: bool) -> Dict:
        """Get optimal formatting parameters for Dia model"""
        if word_count <= 10:
            return {
                'words_per_line': '8-12',
                'target_lines': '1',
                'strategy': 'single_line',
                'explanation': 'Single line for short text'
            }
        elif word_count <= 25:
            return {
                'words_per_line': '7-10', 
                'target_lines': '2-3',
                'strategy': 'balanced',
                'explanation': '2-3 lines with balanced words'
            }
        elif word_count <= 50:
            return {
                'words_per_line': '8-12',
                'target_lines': '3-5', 
                'strategy': 'controlled',
                'explanation': 'Limit to 5 lines max'
            }
        else:
            return {
                'words_per_line': '10-15',
                'target_lines': '4-6',
                'strategy': 'condensed', 
                'explanation': 'Condensed for consistency'
            }
    
    def _clean_formatted_text_for_dia(self, text: str) -> str:
        """Clean formatted text for Dia optimization"""
        # Basic cleanup
        text = re.sub(r'^["\s]*', '', text).strip()
        text = re.sub(r'["\s]*$', '', text)
        
        if not text:
            raise ValueError("Empty response from OpenAI")
        
        # Ensure speaker tag
        if '[S' not in text:
            text = f"[S1] {text}"
        
        # Simple line validation - merge very short lines (relaxed rule)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        fixed_lines = []
        
        for line in lines:
            if line.startswith('[S'):
                # Count words after speaker tag
                words_after_tag = line.split()[1:] if len(line.split()) > 1 else []
                # Only merge if line has just 1 word (relaxed from 1-2)
                if len(words_after_tag) == 1 and fixed_lines and not fixed_lines[-1].startswith('[S'):
                    fixed_lines[-1] += f" {' '.join(words_after_tag)}"
                    continue
                fixed_lines.append(line)
            else:
                # Merge very short non-speaker lines
                if len(line.split()) == 1 and fixed_lines:
                    fixed_lines[-1] += f" {line}"
                else:
                    fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _simple_translate_and_format_for_dia(self, text: str, optimal_format: Dict) -> str:
        """Simple translation with Dia formatting"""
        try:
            translation_response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Translate to natural English only."},
                    {"role": "user", "content": f"Translate: {text}"}
                ],
                max_tokens=150,
                temperature=0.1,
                timeout=5
            )
            
            if translation_response and translation_response.choices:
                english_text = translation_response.choices[0].message.content.strip()
                return self._format_text_for_dia(english_text, optimal_format)
            else:
                raise ValueError("Translation failed")
        except Exception as e:
            raise ValueError(f"Translation failed: {str(e)}")
    
    def _format_text_for_dia(self, text: str, optimal_format: Dict) -> str:
        """Format text for Dia model"""
        if not text.strip():
            raise ValueError("No content to format")
        
        words = text.split()
        
        if optimal_format['strategy'] == 'single_line':
            return f"[S1] {text}"
        
        # Multi-line formatting
        target_words_per_line = 8  # Default
        if '-' in optimal_format['words_per_line']:
            min_words, max_words = map(int, optimal_format['words_per_line'].split('-'))
            target_words_per_line = (min_words + max_words) // 2
        
        lines = []
        i = 0
        first_line = True
        
        while i < len(words):
            if first_line:
                line_words = ['[S1]']
                first_line = False
            else:
                line_words = []
            
            words_to_add = min(target_words_per_line, len(words) - i)
            line_words.extend(words[i:i + words_to_add])
            i += words_to_add
            
            lines.append(' '.join(line_words))
        
        return '\n'.join(lines)
    
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