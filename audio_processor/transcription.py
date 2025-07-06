"""
Transcription and Text Processing Module

Handles AssemblyAI integration, text translation, and formatting for voice cloning.
"""

import re
from typing import Dict, Any, List
import assemblyai as aai
from openai import OpenAI
from config import settings


class TranscriptionService:
    """Service for audio transcription and text processing"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using AssemblyAI with speaker diarization"""
        try:
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                auto_chapters=True,
                punctuate=True,
                format_text=True,
                language_code="en"
            )
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_path, config=config)
            
            if transcript.status == aai.TranscriptionStatus.error:
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
            
            speakers = list(set(word.get("speaker", "A") for word in words))
            
            return {
                "text": transcript.text,
                "words": words,
                "speakers": speakers,
                "duration": words[-1]['end'] / 1000 if words else 0
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
        """Format text for Dia model following official guidelines"""
        sorted_speakers = sorted(all_speakers)
        
        if len(all_speakers) == 1:
            # Single speaker: always use [S1]
            formatted_text = f"[S1] {english_text}"
        else:
            # Multi-speaker: ensure proper alternation
            if segment_index == 0:
                speaker_tag = "[S1]"
            else:
                speaker_tag = "[S1]" if segment_index % 2 == 0 else "[S2]"
            
            formatted_text = f"{speaker_tag} {english_text}"
        
        # Add speaker tag at end for better audio quality (Dia guideline)
        if is_last_segment:
            if len(all_speakers) == 1:
                end_tag = "[S1]"
            else:
                current_tag = "[S1]" if segment_index % 2 == 0 else "[S2]"
                end_tag = "[S2]" if current_tag == "[S1]" else "[S1]"
            
            if not formatted_text.endswith(end_tag):
                formatted_text += f" {end_tag}"
        
        return formatted_text 