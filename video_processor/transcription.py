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
                        speakers_expected: Optional[int] = 1) -> Dict[str, Any]:
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
                "auto_chapters": True,
                "punctuate": True,
                "format_text": True,
                "speech_model": aai.SpeechModel.universal
            }
            
            # Handle language detection vs manual language code
            if language_code and language_code.strip():
                # Manual language code specified
                config_params["language_code"] = language_code.strip()
            else:
                # Auto-detect language
                config_params["language_detection"] = True
                config_params["language_confidence_threshold"] = 0.1  # Low threshold for better detection
            
            # Add speaker count if specified
            if speakers_expected and 1 <= speakers_expected <= 10:
                config_params["speakers_expected"] = speakers_expected
            
            # Create transcription config
            config = aai.TranscriptionConfig(**config_params)
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_path, config=config)
            
            if transcript.status == "error":
                raise Exception(f"AssemblyAI transcription failed: {transcript.error}")
            
            words = []
            for word in transcript.words:
                words.append({
                    "text": word.text,
                    "start": word.start,
                    "end": word.end,
                    "speaker": word.speaker if hasattr(word, 'speaker') else "A",
                    "confidence": word.confidence
                })
            
            speakers = sorted(list(set(word.get("speaker", "A") for word in words)))
            
            # Get the final language code (either provided or detected)
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
    
    def get_voice_cloning_prompt(self, text: str) -> str:
        """Generate optimized prompt for voice cloning"""
        return f"""Convert this text to natural English for high-quality voice cloning. Add appropriate non-verbal tags when contextually suitable.

SOURCE: "{text}"

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

CLEAN ENGLISH WITH NATURAL NON-VERBALS:"""
    
    def translate_text_clean(self, text: str) -> str:
        """Clean translation optimized for voice cloning"""
        try:
            prompt = self.get_voice_cloning_prompt(text)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert translator specializing in voice cloning optimization. Return only clean, natural English text."},
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
    
    def validate_dia_text_length(self, text: str) -> bool:
        """Validate text length for Dia guidelines (5-20 seconds optimal)"""
        word_count = len(text.split())
        estimated_seconds = word_count / 17  # More accurate estimate
        return 5 <= estimated_seconds <= 20
    
    def estimate_audio_duration(self, text: str) -> float:
        """Estimate audio duration from text (following Dia guidelines)"""
        word_count = len(text.split())
        estimated_tokens = word_count / 0.75
        estimated_seconds = estimated_tokens / 86
        return estimated_seconds
    
    def format_dia_text(self, english_text: str, speaker: str, all_speakers: List[str], 
                       segment_index: int = 0, is_last_segment: bool = False) -> str:
        """Format text for Dia model with optimal speaker handling"""
        # Clean and optimize text for Dia model
        cleaned_text = self._optimize_text_for_dia(english_text)
        
        if len(all_speakers) == 1:
            # Single speaker: always use [S1]
            return f"[S1] {cleaned_text}"
        else:
            # Multi-speaker: use speaker mapping
            try:
                speaker_idx = all_speakers.index(speaker) + 1
                return f"[S{speaker_idx}] {cleaned_text}"
            except ValueError:
                # Fallback to S1 if speaker not found
                return f"[S1] {cleaned_text}"
    
    def _optimize_text_for_dia(self, text: str) -> str:
        """Optimize text for Dia model performance"""
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