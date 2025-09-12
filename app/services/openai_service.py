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

    def translate_dubbing_batch(self, segments: List[Dict], target_language: str, batch_size: int = 8) -> List[str]:
        """Fast translation with retry system - no fallbacks"""
        if not self.is_available:
            return ["" for _ in segments]

        all_dubbed = []
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            
            # Try batch first, then retry individually if needed
            result = self._translate_with_retry(batch, target_language)
            all_dubbed.extend(result)

        return all_dubbed

    def _translate_with_retry(self, batch: List[Dict], target_language: str) -> List[str]:
        """Retry-based translation: batch first, individual retry if needed"""
        
        # Attempt 1: Batch translation
        try:
            return self._try_batch_translation(batch, target_language)
        except Exception as e:
            logger.warning(f"Batch failed, retrying individually: {e}")
        
        # Attempt 2: Individual translations with retry
        results = []
        for segment in batch:
            result = self._try_individual_translation(segment['text'], target_language)
            results.append(result)
        
        return results

    def _try_batch_translation(self, batch: List[Dict], target_language: str) -> List[str]:
        """Clean batch translation with validation"""
        prompt_lines = [f"[{idx+1}] {seg['text']}" for idx, seg in enumerate(batch)]
        joined_texts = "\n".join(prompt_lines)

        system_prompt = (
            f"Translate each numbered segment to {target_language}.\n"
            f"Rules:\n"
            f"1) Translate meaning accurately (no creative additions).\n"
            f"2) Use proper {target_language} script only; keep proper nouns in original when appropriate.\n"
            f"3) Keep outputs natural and concise. DO NOT repeat words/syllables; collapse long onomatopoeia or loops into a brief phrase such as 'repeated clicking sound' or 'again and again'.\n"
            f"4) You MAY optionally use OpenAudio tone markers like (excited), (sad), (whispering), (angry), (soft tone) at the very beginning IF it clearly matches intent.\n"
            f"5) NEVER omit any segment. Do NOT include segment numbers or explanations.\n"
            f"6) Return EXACTLY {len(batch)} translations separated by |||, with no empty items."
        )

        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate these {len(batch)} segments:\n{joined_texts}"}
            ],
            max_completion_tokens=4096
        )
        
        output = response.choices[0].message.content.strip()
        result = [seg.strip() for seg in output.split("|||")]
        
        # Validate count
        if len(result) != len(batch):
            raise ValueError(f"Count mismatch: expected {len(batch)}, got {len(result)}")
        
        return result

    def _try_individual_translation(self, text: str, target_language: str) -> str:
        """Clean individual translation with retry"""
        if not text or not text.strip():
            return ""
        
        # Try with GPT-5
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": f"Translate to {target_language}. Be concise."},
                        {"role": "user", "content": text}
                    ],
                    max_completion_tokens=500
                )
                result = response.choices[0].message.content.strip()
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Translation attempt {attempt+1} failed: {e}")
                if attempt == 0:
                    continue  # Retry once
        
        # Try with GPT-4o as backup
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"Translate to {target_language}."},
                    {"role": "user", "content": text}
                ],
                max_tokens=500
            )
            result = response.choices[0].message.content.strip()
            if result:
                return result
        except Exception as e:
            logger.warning(f"GPT-4o backup failed: {e}")
        
        # Final attempt: return meaningful response for empty case
        return "" if not text.strip() else text[:100]

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
