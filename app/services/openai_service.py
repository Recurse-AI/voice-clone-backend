"""Simple OpenAI Service for independent worker usage"""

import logging
from typing import List, Dict
from openai import OpenAI
from app.config.settings import settings

logger = logging.getLogger(__name__)

class OpenAIService:
    """Simple OpenAI service - each worker creates its own instance"""

    def __init__(self):
        self.client = None
        self.is_available = False

        try:
            api_key = settings.OPENAI_API_KEY
            if api_key and api_key.strip():
                self.client = OpenAI(api_key=api_key)
                self.is_available = True
                logger.info("OpenAI service ready")
            else:
                logger.warning("OpenAI API key not configured")
        except Exception as e:
            logger.warning(f"OpenAI init failed: {e}")

    def translate_dubbing_batch(self, segments: List[Dict], target_language: str, batch_size: int = 10) -> List[str]:
        """Simple synchronous translation"""
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
                f"CRITICAL CONSTRAINTS (MUST FOLLOW EXACTLY):\n"
                f"1. Try to match the target duration (given in ms) — if you need to lengthen, prefer inserting extra *spaces* between words rather than adding new words.\n"
                f"2. Use the correct alphabet/script for {target_language}; never mix English letters unless the original word is a proper noun or acronym.\n"
                f"3. You MAY optionally use the Fish-Audio emotion/tone markers like (excited), (sad), (whispering) etc. **only** when that better reflects the original intent. Place the marker at the very beginning of the sentence.\n"
                f"4. ABSOLUTELY FORBIDDEN: Do NOT include ANY segment markers, numbers in brackets like [1], [2], numbers, explanatory text, or comments.\n"
                f"5. Return ONLY the translated text segments in the same order, separated by ||| on a single line. NOTHING ELSE."
            )

            user_prompt = f"Translate each segment below. Return only the translated segments, in order, separated by |||.\nSegments (with target duration):\n{joined_texts}"

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
                logger.warning(f"OpenAI translation failed: {e}")
                all_dubbed.extend([f"[Translation Error]" for _ in range(len(batch))])

        return all_dubbed

    def regenerate_text_with_prompt(self, original_text: str, target_language: str, custom_prompt: str) -> str:
        """Simple text regeneration"""
        if not self.is_available:
            return f"[Error] {original_text}"

        try:
            system_prompt = f"You are a professional dubbing script writer. Rewrite the given text in {target_language} according to the specific instructions provided. Keep the meaning accurate but adapt the style based on the prompt. Return only the rewritten text, nothing else."

            user_prompt = f"Instructions: {custom_prompt}\nOriginal text: {original_text}\nLanguage: {target_language}\nRewrite this text following the instructions:"

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

        except Exception as e:
            logger.warning(f"OpenAI regeneration failed: {e}")
            return f"[Error] {original_text}"
