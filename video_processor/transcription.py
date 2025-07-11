"""
Transcription and Text Processing Module

Handles AssemblyAI integration, text translation, and formatting for voice cloning.
"""

import re
from typing import Dict, Any, List, Optional
import assemblyai as aai
from openai import OpenAI
from config import settings


class TranscriptionService:
    """Service for audio transcription and text processing"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
    
    def transcribe_audio(self, audio_path: str, language_code: Optional[str] = None, 
                        speakers_expected: Optional[int] = None) -> Dict[str, Any]:
        """
        Transcribe audio using AssemblyAI Universal model
        
        Args:
            audio_path: Path to audio file
            language_code: Language code (e.g., "en", "es", "fr", "de", "hi", "ja", "zh") - None for auto-detection
            speakers_expected: Expected number of speakers (1-10)
        """
        try:
            # Build transcription config with universal model
            config_params = {
                "speaker_labels": True,
                "auto_chapters": False,
                "punctuate": True,
                "format_text": True,
                "speech_model": aai.SpeechModel.universal
            }
            
            # Handle language detection vs manual language code
            if language_code and language_code.strip():
                config_params["language_code"] = language_code.strip()
            else:
                config_params["language_detection"] = True
                config_params["language_confidence_threshold"] = 0.1  # Reduced from 0.5 to 0.1 for better language detection
            
            # Add speaker count if specified
            if speakers_expected and 1 <= speakers_expected <= 10:
                config_params["speakers_expected"] = speakers_expected
            
            # Create transcription config
            config = aai.TranscriptionConfig(**config_params)
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_path, config=config)
            
            if transcript.status == "error":
                raise Exception(f"AssemblyAI transcription failed: {transcript.error}")
            
            # Extract words with proper speaker labels and safe attribute access
            words = []
            if hasattr(transcript, 'words') and transcript.words:
                for word in transcript.words:
                    try:
                        # Safely access word attributes with fallbacks
                        word_data = {
                            "text": getattr(word, 'text', ''),
                            "start": getattr(word, 'start', 0),
                            "end": getattr(word, 'end', 0),
                            "speaker": getattr(word, 'speaker', 'A') if hasattr(word, 'speaker') and word.speaker else "A",
                            "confidence": getattr(word, 'confidence', 0.5)
                        }
                        
                        # Skip empty words
                        if word_data["text"].strip():
                            words.append(word_data)
                    except Exception:
                        # Skip problematic words
                        continue
            
            # If no words were extracted, create fallback from text
            if not words and transcript.text:
                words = [{
                    "text": transcript.text,
                    "start": 0,
                    "end": 5000,  # Default 5 seconds
                    "speaker": "A",
                    "confidence": 0.5
                }]
            
            # Get unique speakers and sort them
            speakers = sorted(list(set(word["speaker"] for word in words))) if words else ["A"]
            
            # Get the final language code
            final_language_code = language_code if language_code and language_code.strip() else transcript.json_response.get("language_code", "en")
            
            return {
                "text": transcript.text,
                "words": words,
                "speakers": speakers,
                "duration": words[-1]['end'] / 1000 if words else 0,
                "metadata": {
                    "language_code": final_language_code,
                    "language_detection_used": not (language_code and language_code.strip()),
                    "language_confidence": transcript.json_response.get("language_confidence", 0.0),
                    "speakers_expected": speakers_expected,
                    "detected_speakers": len(speakers)
                }
            }
            
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")
    
    def translate_text_clean(self, text: str) -> str:
        """Clean translation optimized for voice cloning"""
        try:
            prompt = f"""Convert this text to natural English for voice cloning:

"{text}"

REQUIREMENTS:
- Clear, conversational English
- Simple vocabulary and sentence structure
- Natural speech rhythm and flow
- Remove filler words and hesitations
- Maintain emotional tone and intent
- Add non-verbal tags ONLY when contextually appropriate from this list:
  (laughs), (chuckles), (sighs), (gasps), (coughs), (clears throat), (inhales), (exhales), (humming), (sneezes), (whistles)
- Do NOT force non-verbals into every sentence
- Use them sparingly and naturally

Return only the clean English text:"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert translator for voice cloning. Return only clean, natural English text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.2
            )
            
            if response and response.choices:
                result = response.choices[0].message.content.strip()
                result = re.sub(r'^["\s]*', '', result)
                result = re.sub(r'["\s]*$', '', result)
                return result
            return text
                
        except Exception as e:
            return text
    
    def format_dia_text(self, english_text: str, speaker: str, all_speakers: List[str]) -> str:
        """Format text for Dia model with proper speaker handling"""
        # Clean text
        cleaned_text = self._clean_text(english_text)
        
        if len(all_speakers) == 1:
            return f"[S1] {cleaned_text}"
        else:
            # Multi-speaker: use proper speaker mapping
            try:
                speaker_idx = all_speakers.index(speaker) + 1
                return f"[S{speaker_idx}] {cleaned_text}"
            except ValueError:
                return f"[S1] {cleaned_text}"
    
    def _clean_text(self, text: str) -> str:
        """Clean text for Dia model"""
        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper sentence ending
        text = text.strip()
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text 