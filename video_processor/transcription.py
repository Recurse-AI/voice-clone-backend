"""
Transcription and Text Processing Module - Simplified
"""

import re
from typing import Dict, Any, List, Optional
import assemblyai as aai
from openai import OpenAI
from config import settings


class TranscriptionService:
    """Simplified transcription service"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
    
    def transcribe_audio(self, audio_path: str, language_code: Optional[str] = None, 
                        speakers_expected: Optional[int] = None, audio_id: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio using AssemblyAI"""
        try:
            config_params = {
                "speaker_labels": True,
                "punctuate": True,
                "format_text": True,
                "speech_model": aai.SpeechModel.universal
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
                    "transcript_id": transcript.id
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
        """Translate text to English and format into multi-line dialogue with speaker tags - Advanced for Dubbing"""
        try:
            # First clean the text
            clean_text = self._clean_text(text)
            
            if not clean_text.strip():
                return f"[S{ord(speaker) - ord('A') + 1}] No text available"
            
            # Try OpenAI translation and formatting
            try:
                speaker_num = ord(speaker) - ord('A') + 1
                
                if is_multi_speaker:
                    prompt = f"""You are a professional dubbing translator. Convert this text to natural English dialogue for voice cloning/dubbing. 

CRITICAL REQUIREMENTS:
- Translate to natural, conversational English suitable for dubbing
- Try to keep 7-15 words per line when possible, adjust if needed for natural flow
- Don't break lines in the middle of a sentence - keep full sentences together when possible
- Use [S1], [S2], etc. for speaker identification
- Current speaker is {speaker}, use [S{speaker_num}] for this speaker
- Only add non-verbal sounds when truly necessary and natural to the content (don't force them)
- Preserve emotional context and speaking style
- Each line should be naturally speakable
- Match the original's rhythm and pacing
- If speaker changes detected, use appropriate tags

Original text: "{clean_text}"

DUBBING FORMAT EXAMPLE:
[S1] Hey there how's it going today?
[S1] How is your work progressing?
[S2] Pretty good, just working on some projects.
[S2] Everything is going smoothly right now.

Convert to natural English dubbing format:"""
                else:
                    prompt = f"""You are a professional dubbing translator. Convert this text to natural English dialogue for voice cloning/dubbing.

CRITICAL REQUIREMENTS:
- Translate to natural, conversational English suitable for dubbing
- Try to keep 7-15 words per line when possible, adjust if needed for natural flow
- Don't break lines in the middle of a sentence - keep full sentences together when possible
- Use [S{speaker_num}] tag for all lines (single speaker)
- Only add non-verbal sounds when truly necessary and natural to the content (don't force them)
- Preserve emotional context and speaking style
- Each line should be naturally speakable
- Match the original's rhythm and pacing
- Break at natural speech pauses

Original text: "{clean_text}"

DUBBING FORMAT EXAMPLE:
[S{speaker_num}] That's really amazing work you did there.
[S{speaker_num}] I'm very impressed with the quality.
[S{speaker_num}] The attention to detail is excellent.

Convert to natural English dubbing format:"""
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional dubbing translator specializing in voice cloning dialogue. Create natural, speakable English text with emotional context. Try to keep 7-15 words per line when possible, adjust if needed for natural flow. Don't break lines in the middle of a sentence - keep full sentences together when possible. Only add non-verbal sounds when truly necessary and natural to the content."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.2,
                    timeout=15
                )
                
                if response and response.choices:
                    formatted_text = response.choices[0].message.content.strip()
                    # Clean up the response
                    formatted_text = self._clean_formatted_text(formatted_text)
                    
                    # Validate the formatted text has speaker tags and English content
                    if formatted_text and '[S' in formatted_text:
                        return formatted_text
                    else:
                        raise ValueError("Invalid OpenAI response format")
                
                # If no valid response, use fallback
                raise ValueError("No valid OpenAI response")
                
            except Exception as openai_error:
                # Use simple fallback translation and formatting
                return self._simple_translate_and_format(clean_text, speaker)
                
        except Exception as e:
            # Ultimate fallback
            speaker_tag = f"[S{ord(speaker) - ord('A') + 1}]"
            return f"{speaker_tag} {text.strip()}" if text.strip() else f"{speaker_tag} Audio segment"
    
    def _simple_translate_and_format(self, text: str, speaker: str) -> str:
        """Simple fallback translation and formatting"""
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
                return self._simple_format_text(english_text, speaker)
            else:
                # If translation fails, format original text
                return self._simple_format_text(text, speaker)
                
        except Exception:
            # Final fallback
            return self._simple_format_text(text, speaker)
    
    def _clean_formatted_text(self, text: str) -> str:
        """Clean formatted text from OpenAI response"""
        # Remove extra quotation marks
        text = re.sub(r'^["\s]*', '', text)
        text = re.sub(r'["\s]*$', '', text)
        
        # Ensure proper line breaks
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and '[S' in line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines) if cleaned_lines else f"[S1] {text}"
    
    def _simple_format_text(self, text: str, speaker: str) -> str:
        """Simple fallback text formatting"""
        words = text.split()
        if not words:
            return f"[S1] {text}"
        
        lines = []
        current_line = []
        speaker_tag = f"[S{ord(speaker) - ord('A') + 1}]"
        
        for word in words:
            current_line.append(word)
            
            # Break at sentence end if we have enough words (7+)
            if len(current_line) >= 7 and word.endswith(('.', '!', '?')):
                lines.append(f"{speaker_tag} {' '.join(current_line)}")
                current_line = []
            # Force break if too long (15+ words)
            elif len(current_line) >= 15:
                lines.append(f"{speaker_tag} {' '.join(current_line)}")
                current_line = []
        
        if current_line:
            lines.append(f"{speaker_tag} {' '.join(current_line)}")
        
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