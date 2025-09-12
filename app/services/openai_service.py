"""Simple OpenAI Service for independent worker usage"""

import logging
from typing import List, Dict
from openai import OpenAI
from app.config.settings import settings
import concurrent.futures
import threading

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

    def translate_dubbing_batch(self, segments: List[Dict], target_language: str, max_workers: int = 8) -> List[str]:
        """Fast parallel translation with threading - no batch calls"""
        if not self.is_available:
            return ["" for _ in segments]

        # Use threading for parallel individual calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all segments for parallel processing
            future_to_index = {
                executor.submit(self._translate_single_segment, seg['text'], target_language): i 
                for i, seg in enumerate(segments)
            }
            
            # Collect results in order
            results = [""] * len(segments)
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.warning(f"Segment {index} failed: {e}")
                    results[index] = ""
            
            return results

    def _translate_single_segment(self, text: str, target_language: str) -> str:
        """Thread-safe individual translation with comprehensive prompt"""
        if not text or not text.strip():
            return ""
        
        # Comprehensive system prompt for high-quality results
        system_prompt = (
            f"You are producing dubbing text for Fish Audio OpenAudio-S1 (Fish-Speech).\n"
            f"Translate this segment to {target_language}.\n"
            f"Rules:\n"
            f"1) Translate meaning accurately (no creative additions).\n"
            f"2) Use proper {target_language} script only; keep proper nouns in original when appropriate.\n"
            f"3) Keep outputs natural and concise. DO NOT repeat words/syllables; collapse long onomatopoeia or loops into a brief phrase such as 'repeated clicking sound' or 'again and again'.\n"
            f"4) You MAY optionally use OpenAudio tone markers like (excited), (sad), (whispering), (angry), (soft tone) at the very beginning IF it clearly matches intent.\n"
            f"5) Return only the translation, no explanations or notes."
        )
        
        # Try with GPT-5 (2 attempts)
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    max_completion_tokens=500
                )
                result = response.choices[0].message.content.strip()
                if result:
                    return result
            except Exception as e:
                logger.warning(f"GPT-5 attempt {attempt+1} failed for '{text[:30]}...': {e}")
                if attempt == 0:
                    continue  # Retry once
        
        # Try with GPT-4o as backup with same comprehensive prompt
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=500
            )
            result = response.choices[0].message.content.strip()
            if result:
                return result
        except Exception as e:
            logger.warning(f"GPT-4o backup failed for '{text[:30]}...': {e}")
        
        # Return original text if all else fails (better than empty)
        return text[:100] if text.strip() else ""

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
