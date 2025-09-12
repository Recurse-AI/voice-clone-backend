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
        """Simple clean translation with better prompt"""
        if not self.is_available:
            return [f"[Translation Error]" for _ in segments]

        all_dubbed = []

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            prompt_lines = [f"[{idx+1}] {seg['text']}" for idx, seg in enumerate(batch)]
            joined_texts = "\n".join(prompt_lines)

            system_prompt = (
                f"You are producing dubbing text for Fish Audio OpenAudio-S1 (Fish-Speech).\n"
                f"Translate each numbered segment to {target_language}.\n"
                f"Rules:\n"
                f"1) Translate meaning accurately (no creative additions).\n"
                f"2) Use proper {target_language} script only; keep proper nouns in original when appropriate.\n"
                f"3) Keep outputs natural and concise. DO NOT repeat words/syllables; collapse long onomatopoeia or loops into a brief phrase such as 'repeated clicking sound' or 'again and again'.\n"
                f"4) You MAY optionally use OpenAudio tone markers like (excited), (sad), (whispering), (angry), (soft tone) at the very beginning IF it clearly matches intent.\n"
                f"5) NEVER omit any segment. Do NOT include segment numbers or explanations.\n"
                f"6) Return EXACTLY {len(batch)} translations separated by |||, with no empty items."
            )

            user_prompt = f"Translate these {len(batch)} segments:\n{joined_texts}"

            try:
                response = self.client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=4096
                )
                
                output = response.choices[0].message.content.strip()
                translated_segments = [seg.strip() for seg in output.split("|||")]
                
                # Ensure count matches
                while len(translated_segments) < len(batch):
                    translated_segments.append("[Translation Error]")
                
                all_dubbed.extend(translated_segments[:len(batch)])

            except Exception as e:
                logger.warning(f"OpenAI translation failed: {e}")
                all_dubbed.extend([f"[Translation Error]" for _ in batch])

        return all_dubbed

    def regenerate_text_with_prompt(self, original_text: str, target_language: str, custom_prompt: str) -> str:
        """Simple text regeneration"""
        if not self.is_available:
            return f"[Error] {original_text}"

        try:
            system_prompt = f"You are a professional dubbing script writer. Rewrite the given text in {target_language} according to the specific instructions provided. Keep the meaning accurate but adapt the style based on the prompt. Return only the rewritten text, nothing else."

            user_prompt = f"Instructions: {custom_prompt}\nOriginal text: {original_text}\nLanguage: {target_language}\nRewrite this text following the instructions:"

            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=500
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"OpenAI regeneration failed: {e}")
            return f"[Error] {original_text}"
