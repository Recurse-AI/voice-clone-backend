"""OpenAI Service for text translation and generation"""

import logging
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config.settings import settings

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        try:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.is_available = True
            logger.info("OpenAI service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI service: {e}")
            self.client = None
            self.is_available = False
    
    def translate_dubbing_batch(self, segments: List[Dict], target_language: str, batch_size: int = 10) -> List[str]:
        if not self.is_available:
            return [f"[Translation Error]" for _ in segments]
        
        all_dubbed = []
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            
            prompt_lines = [f"[{idx+1}] (Target duration: {seg['duration_ms']} ms) {seg['text']}" for idx, seg in enumerate(batch)]
            joined_texts = "\n".join(prompt_lines)
            
            system_prompt = (
                f"You are assisting in creating dubbing scripts for the Fish Audio OpenAudio-S1 TTS model.\n"
                f"Translate each input segment into {target_language} (keeping meaning accurate).\n"
                f"Constraints for every translated segment:\n"
                f"1. Try to match the target duration (given in ms) — if you need to lengthen, prefer inserting extra *spaces* between words rather than adding new words.\n"
                f"2. Use the correct alphabet/script for {target_language}; never mix English letters unless the original word is a proper noun or acronym.\n"
                f"3. You MAY optionally use the Fish-Audio emotion/tone markers like (excited), (sad), (whispering) etc. **only** when that better reflects the original intent. Place the marker at the very beginning of the sentence.\n"
                f"4. Do NOT add any explanatory text, numbering, or comments — only the final translated sentences.\n"
                f"5. Return the translated segments in the same order, separated by ||| on a single line."
            )
            
            user_prompt = (
                "Translate each segment below. Return only the translated segments, in order, separated by |||.\n"
                "Segments (with target duration):\n"
                f"{joined_texts}"
            )
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2048
                )
                output = response.choices[0].message.content.strip()
                translated_segments = [seg.strip() for seg in output.split("|||")]
                all_dubbed.extend(translated_segments)

            except Exception as e:
                logger.warning(f"OpenAI translation failed for batch {i//batch_size + 1}: {e}")
                # Simple retry for connection issues
                if "ConnectionResetError" in str(type(e)) or "connection" in str(e).lower():
                    try:
                        time.sleep(1)
                        response = self.client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=2048
                        )
                        output = response.choices[0].message.content.strip()
                        translated_segments = [seg.strip() for seg in output.split("|||")]
                        all_dubbed.extend(translated_segments)
                    except Exception as retry_e:
                        logger.error(f"OpenAI retry failed: {retry_e}")
                        all_dubbed.extend([f"[Translation Error]" for _ in range(len(batch))])
                else:
                    all_dubbed.extend([f"[Translation Error]" for _ in range(len(batch))])
        
        return all_dubbed
    
    def regenerate_text_with_prompt(self, original_text: str, target_language: str, custom_prompt: str) -> str:
        if not self.is_available:
            return f"[Error] {original_text}"
        
        try:
            system_prompt = (
                f"You are a professional dubbing script writer. "
                f"Rewrite the given text in {target_language} according to the specific instructions provided. "
                f"Keep the meaning accurate but adapt the style based on the prompt. "
                f"Return only the rewritten text, nothing else."
            )
            
            user_prompt = (
                f"Instructions: {custom_prompt}\n"
                f"Original text: {original_text}\n"
                f"Language: {target_language}\n"
                f"Rewrite this text following the instructions:"
            )
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception:
            return f"[Error] {original_text}"
    
    def check_availability(self) -> bool:
        """Check if OpenAI service is available - uses cached status to avoid unnecessary API calls"""
        return self.is_available and self.client is not None


# Global service instance
_openai_service = None

def get_openai_service() -> OpenAIService:
    global _openai_service
    if _openai_service is None:
        _openai_service = OpenAIService()
    return _openai_service

def initialize_openai_service() -> bool:
    try:
        service = get_openai_service()
        return service.is_available
    except Exception:
        return False
