import logging
import json
import re
from typing import List, Dict, Any, Optional
from app.services.language_service import language_service
from app.config.settings import settings

logger = logging.getLogger(__name__)

class AISegmentationService:
    def __init__(self, settings_override=None):
        self.max_tokens = 16384
        self._openai_client = None
        self.settings = settings_override if settings_override is not None else settings
        
    def create_optimal_segments_and_dub(
        self,
        transcription_data: Dict[str, Any],
        target_language: str,
        preserve_segments: bool = False
    ) -> List[Dict[str, Any]]:
        try:
            if transcription_data.get("segments"):
                segments = transcription_data["segments"]
            elif transcription_data.get("sentences"):
                segments = transcription_data["sentences"]
            else:
                logger.error(f"No segments or sentences found in transcription data. Keys available: {list(transcription_data.keys())}")
                raise ValueError("No segments or sentences found in transcription data")
            
            if not segments:
                logger.error(f"Empty segments list received. Transcription data success: {transcription_data.get('success')}, language: {transcription_data.get('language')}")
                raise ValueError("Segments list is empty - transcription may have failed")
            
            logger.info(f"Processing {len(segments)} segments from transcription data")
            
            source_language = transcription_data.get("language", "auto_detect")
            source_language_code = language_service.normalize_language_input(source_language)
            target_language_code = language_service.normalize_language_input(target_language)
            
            is_same_language = source_language_code == target_language_code
            if is_same_language:
                logger.info(f"🎯 SAME LANGUAGE DETECTED: source={source_language_code}, target={target_language_code}")
            else:
                logger.info(f"🌍 TRANSLATION NEEDED: source={source_language_code} → target={target_language_code}")
            
            if preserve_segments:
                logger.info(f"REDUB: Preserving {len(segments)} segments, translating text only")
                return self._translate_existing_segments(segments, target_language_code, is_same_language)
            
            logger.info(f"FRESH DUBBING: Full AI segmentation + translation for {len(segments)} segments")
            combined_text = []
            for seg in segments:
                text = seg.get("text", "").strip() or seg.get("original_text", "").strip()
                if text:
                    start_ms = int(seg.get("start", 0))
                    end_ms = int(seg.get("end", 0))
                    
                   
                    # Convert to seconds for AI prompt
                    start_s = start_ms / 1000.0
                    end_s = end_ms / 1000.0
                    
                    combined_text.append({
                        "text": text,
                        "start": round(start_s, 3),
                        "end": round(end_s, 3),
                        "duration": round(end_s - start_s, 3),
                        "start_ms": start_ms,
                        "end_ms": end_ms
                    })
            
            if not combined_text:
                return []
            
            logger.info(f"Processing {len(combined_text)} segments in chunks")
            logger.info(f"✅ TIMING STANDARD: All inputs converted to milliseconds, AI gets seconds for better understanding")
            return self._process_in_chunks(combined_text, target_language, is_same_language, preserve_segments=False)
        except Exception as e:
            logger.error(f"CRITICAL ERROR in create_optimal_segments_and_dub: {str(e)}")
            logger.error(f"Transcription data structure: {transcription_data}")
            raise e

    def _build_segmentation_and_dubbing_prompt(self, segments: List[Dict], target_language: str, is_same_language: bool = False, preserve_segments: bool = False) -> str:
        
        def get_language_name(code):
            lang_names = {
                'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
                'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean',
                'ar': 'Arabic', 'hi': 'Hindi', 'bn': 'Bengali', 'ur': 'Urdu', 'tr': 'Turkish',
                'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian'
            }
            return lang_names.get(code, code.title())
        
        target_lang_name = get_language_name(target_language)
        
        if is_same_language:
            translation_instructions = f"""SAME LANGUAGE PROCESSING:
- Source and target language are IDENTICAL ({target_lang_name})
- PRESERVE the original text EXACTLY as is
- Copy the original_text to dubbed_text WITHOUT any modifications
- CORRUPTION FIX: If original text has repetitive/corrupted patterns, clean it first then copy"""
        else:
            translation_instructions = f"""TRANSLATION TO {target_lang_name.upper()}:
- Translate EVERY WORD into proper {target_lang_name}
- NEVER change sentence structure or meaning
- Keep EXACT same meaning and tone as original
- Only make minimal changes when {target_lang_name} grammar requires it
- CORRUPTION FIX: Clean any repetitive/corrupted patterns before translating
- NEVER translate corrupted text patterns - fix them first"""
        
        if preserve_segments:
            if is_same_language:
                processing_instruction = f"SAME LANGUAGE REDUB - Keep dubbed_text IDENTICAL to original_text"
                example_dubbed = "exact same text as original_text"
            else:
                processing_instruction = f"DIFFERENT LANGUAGE REDUB - Translate dubbed_text to {target_lang_name}"
                example_dubbed = f"proper translation in {target_lang_name}"
            
            return f"""REDUB MODE - EXACT PRESERVATION:

MANDATORY RULES:
1. Output EXACTLY {len(segments)} segments (1:1 mapping)
2. Keep start/end timing exactly as input
3. Keep original_text exactly as input  
4. {processing_instruction}
5. No merging, no splitting, no changes to structure

{translation_instructions}

INPUT SEGMENTS:
{json.dumps(segments, ensure_ascii=False, indent=2)}

OUTPUT JSON FORMAT:
{{
  "segments": [
    {{
      "id": "seg_001",
      "start": 0.080,
      "end": 4.560,
      "original_text": "exact text from input",
      "dubbed_text": "{example_dubbed}"
    }}
  ]
}}

CRITICAL: Must output exactly {len(segments)} segments as valid JSON. {processing_instruction}. Remove corruption but keep meaningful content."""
        else:
            return f"""FRESH DUBBING MODE:

🚨 ABSOLUTE RULE: Every output segment MUST be ≤15.0 seconds duration. NO EXCEPTIONS.

If any input segment is >15s, you MUST split it into multiple shorter segments.
Calculate: (end_time - start_time) MUST be ≤15.0 for every output segment.

RULES:
1. Check each segment duration: (end - start) ≤ 15.0 seconds
2. Split long segments at natural speech breaks
3. Merge very short segments if total ≤ 15.0s
4. Fix corrupted/repetitive text by extracting meaningful content
5. Use all input content exactly once
6. Output segment count can be different from input count

{translation_instructions}

INPUT: {json.dumps(segments, ensure_ascii=False, indent=2)}

OUTPUT JSON:
{{
  "segments": [
    {{
      "id": "seg_001", 
      "start": 0.080,
      "end": 4.560,
      "original_text": "meaningful text",
      "dubbed_text": "{target_lang_name} translation"
    }}
  ]
}}

🚨 VERIFY: Every segment duration (end-start) ≤ 15.0 seconds before output. NO corrupted or repetitive text patterns."""
    
    def _format_segments_with_translation(self, ai_segments: List[Dict], global_segment_index_start: int = 0) -> List[Dict[str, Any]]:
        formatted_segments = []
        
        for idx, seg in enumerate(ai_segments):
            start_s = float(seg.get("start", 0))
            end_s = float(seg.get("end", 0))
            
            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)
            duration_ms = end_ms - start_ms
            
            dubbed_text = seg.get("dubbed_text", "").strip()
            dubbed_text = re.sub(r'\[\w+:\s*([^\]]+)\]', r'\1', dubbed_text)
            dubbed_text = re.sub(r'\[[^\]]+\]', '', dubbed_text).strip()
            
            global_segment_index = global_segment_index_start + len(formatted_segments)
            
            formatted_segments.append({
                "id": seg.get("id", f"seg_{global_segment_index+1:03d}"),
                "segment_index": global_segment_index,
                "start": start_ms,
                "end": end_ms,
                "duration_ms": duration_ms,
                "original_text": seg.get("original_text", "").strip(),
                "dubbed_text": dubbed_text,
                "voice_cloned": False,
                "original_audio_file": None,
                "cloned_audio_file": None
            })
        
        return formatted_segments
    
    def _translate_existing_segments(self, segments: List[Dict], target_language_code: str, is_same_language: bool = False) -> List[Dict[str, Any]]:
        logger.info(f"REDUB MODE: Translating {len(segments)} segments to {target_language_code}")
        
        if is_same_language:
            logger.info(f"Same language redub - preserving original text")
            formatted_segments = []
            for idx, seg in enumerate(segments):
                original_text = seg.get("original_text", seg.get("text", "")).strip()
                formatted_segments.append({
                    "id": seg.get("id", f"seg_{idx+1:03d}"),
                    "segment_index": idx,
                    "start": int(seg.get("start", 0)),
                    "end": int(seg.get("end", 0)),
                    "duration_ms": int(seg.get("end", 0)) - int(seg.get("start", 0)),
                    "original_text": original_text,
                    "dubbed_text": original_text,
                    "voice_cloned": False,
                    "original_audio_file": None,
                    "cloned_audio_file": None
                })
            return formatted_segments
        
        return self._translate_segments_preserve_timing(segments, target_language_code)
    
    def _translate_segments_preserve_timing(self, segments: List[Dict], target_language_code: str) -> List[Dict[str, Any]]:
        chunk_size = 25
        all_results = []
        
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i + chunk_size]
            chunk_number = i//chunk_size + 1
            total_chunks = (len(segments) + chunk_size - 1)//chunk_size
            
            logger.info(f"Translating chunk {chunk_number}/{total_chunks} ({len(chunk)} segments)")
            
            segments_for_translation = []
            for idx, seg in enumerate(chunk):
                original_text = seg.get("original_text", seg.get("text", "")).strip()
                start_ms = int(seg.get("start", 0))
                end_ms = int(seg.get("end", 0))
                
                segments_for_translation.append({
                    "id": seg.get("id", f"seg_{i+idx+1:03d}"),
                    "start": start_ms,
                    "end": end_ms,
                    "original_text": original_text
                })
            
            prompt = self._build_translation_only_prompt(segments_for_translation, target_language_code)
            
            try:
                response = self._get_openai_client().chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a professional translator. Your only job is to translate text while preserving exact segment timing."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=8192,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                ai_response = response.choices[0].message.content.strip()
                result = json.loads(ai_response)
                
                for seg_result in result.get("segments", []):
                    original_seg = next((s for s in chunk if s.get("id") == seg_result.get("id")), None)
                    if original_seg:
                        dubbed_text = seg_result.get("dubbed_text", "").strip()
                        
                        all_results.append({
                            "id": seg_result.get("id"),
                            "segment_index": len(all_results),
                            "start": int(original_seg.get("start", 0)),
                            "end": int(original_seg.get("end", 0)),
                            "duration_ms": int(original_seg.get("end", 0)) - int(original_seg.get("start", 0)),
                            "original_text": seg_result.get("original_text", ""),
                            "dubbed_text": dubbed_text,
                            "voice_cloned": False,
                            "original_audio_file": None,
                            "cloned_audio_file": None
                        })
                
                logger.info(f"Translated chunk {chunk_number}: {len(result.get('segments', []))} segments")
                
            except Exception as e:
                logger.error(f"Translation chunk failed: {str(e)}")
                for idx, seg in enumerate(chunk):
                    original_text = seg.get("original_text", seg.get("text", "")).strip()
                    all_results.append({
                        "id": seg.get("id", f"seg_{i+idx+1:03d}"),
                        "segment_index": len(all_results),
                        "start": int(seg.get("start", 0)),
                        "end": int(seg.get("end", 0)),
                        "duration_ms": int(seg.get("end", 0)) - int(seg.get("start", 0)),
                        "original_text": original_text,
                        "dubbed_text": original_text,
                        "voice_cloned": False,
                        "original_audio_file": None,
                        "cloned_audio_file": None
                    })
        
        logger.info(f"REDUB translation complete: {len(all_results)} segments translated")
        return all_results
    
    def _build_translation_only_prompt(self, segments: List[Dict], target_language_code: str) -> str:
        return f"""You are a professional translator. Translate the text content while preserving EXACT timing and structure.

INPUT SEGMENTS:
{json.dumps(segments, ensure_ascii=False, indent=2)}

STRICT TRANSLATION RULES:
- Translate ONLY the original_text to {self._get_language_name(target_language_code)}
- PRESERVE exact start/end timing from input - DO NOT change timestamps
- PRESERVE exact segment IDs from input  
- DO NOT merge, split, or modify segments
- DO NOT change segment structure or timing
- Provide literal, accurate translation of each text
- Output MUST have same number of segments as input
- Each output segment must match input segment by ID and timing

OUTPUT FORMAT (JSON):
{{
  "segments": [
    {{
      "id": "[same as input]",
      "original_text": "[same as input]", 
      "dubbed_text": "[translation in {self._get_language_name(target_language_code)}]"
    }}
  ]
}}

CRITICAL: Output exactly {len(segments)} segments. Do NOT change timing, merge, or split segments."""
    
    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
        return self._openai_client
    
    def _process_in_chunks(self, segments: List[Dict], target_language: str, is_same_language: bool = False, preserve_segments: bool = False) -> List[Dict[str, Any]]:
        chunk_size = 10
        all_results = []
        
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i + chunk_size]
            chunk_number = i//chunk_size + 1
            total_chunks = (len(segments) + chunk_size - 1)//chunk_size
            
            logger.info(f"Processing chunk {chunk_number}/{total_chunks} ({len(chunk)} segments)")
            
            target_lang_code = language_service.normalize_language_input(target_language)
            
            if preserve_segments:
                chunk_context = f"""
REDUB CHUNK {chunk_number}/{total_chunks}:
- Input: {len(chunk)} segments → Required output: EXACTLY {len(chunk)} segments  
- Start segment IDs from seg_{len(all_results)+1:03d}
- Time range: {chunk[0].get('start', 0):.3f}s to {chunk[-1].get('end', 0):.3f}s
- MANDATORY: Process each input segment as exactly one output segment
- PRESERVE exact timing and original_text from input
- TRANSLATE only the dubbed_text field
- CORRUPTION: Clean any corrupted patterns in original_text before copying/translating
"""
            else:
                chunk_context = f"""
FRESH DUBBING CHUNK {chunk_number}/{total_chunks}:
- Input: {len(chunk)} segments for intelligent optimization
- Start segment IDs from seg_{len(all_results)+1:03d}  
- Time range: {chunk[0].get('start', 0):.3f}s to {chunk[-1].get('end', 0):.3f}s
- OPTIMIZE: Merge/split segments for better voice cloning (3-8s ideal)
- MANDATORY: All segments ≤ 15.0 seconds duration
- COVER ALL content exactly once - no gaps, no repetitions
- DO NOT repeat content from previous chunks
- CORRUPTION: Clean repetitive/corrupted patterns automatically
- ANTI-LOOP: Never output the same phrase multiple times in this chunk
"""
            
            target_lang_name = self._get_language_name(target_lang_code)
            prompt = chunk_context + self._build_segmentation_and_dubbing_prompt(chunk, target_lang_name, is_same_language, preserve_segments)
            
            model = "gpt-4o"
            
            try:
                response = self._get_openai_client().chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"You are an expert audio dubbing AI with advanced corruption detection. {'REDUB MODE: Preserve exact timing and structure, 1:1 mapping, only translate dubbed_text.' if preserve_segments else 'FRESH MODE: Intelligent segmentation - merge short segments, split long segments at sentence breaks, target 3-8 seconds for optimal voice cloning.'} CRITICAL: All segments must be ≤15 seconds duration. CORRUPTION HANDLING: If input text contains repetitive patterns (xxx, ..., aaaa, repeated words/phrases), corrupted transcriptions, meaningless symbols, or placeholder text - AUTOMATICALLY extract only the meaningful content. Remove repetitive artifacts but preserve actual speech. Never output corrupted text patterns. Never repeat the same phrase multiple times. Always provide clean, meaningful text in proper target language alphabet. Skip segments that are purely corrupted/meaningless."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                ai_response = response.choices[0].message.content.strip()
                result = json.loads(ai_response)
                
                ai_segments = result.get("segments", [])
                
                if preserve_segments:
                    expected_segments = len(chunk)
                    actual_segments = len(ai_segments)
                    if actual_segments != expected_segments:
                        logger.error(f"🚨 REDUB SEGMENT COUNT VIOLATION: Expected {expected_segments} segments, got {actual_segments}")
                        logger.error(f"   Input chunk size: {len(chunk)}, AI output size: {len(ai_segments)}")
                        raise ValueError(f"REDUB mode violated 1:1 mapping requirement: expected {expected_segments}, got {actual_segments}")
                    logger.info(f"REDUB: Chunk processed with exact 1:1 mapping: {len(ai_segments)} segments")
                else:
                    logger.info(f"FRESH DUBBING: Chunk optimized from {len(chunk)} input → {len(ai_segments)} output segments")
                
                global_segment_index_start = len(all_results)
                chunk_segments = self._format_segments_with_translation(
                    ai_segments, 
                    global_segment_index_start
                )
                all_results.extend(chunk_segments)
                
            except Exception as e:
                logger.error(f"Chunk processing FAILED: {str(e)}")
                raise e
        
        logger.info(f"Total processed: {len(all_results)} segments from {len(segments)} input segments")
        return all_results

    def _get_language_name(self, code):
        lang_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
            'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean',
            'ar': 'Arabic', 'hi': 'Hindi', 'bn': 'Bengali', 'ur': 'Urdu', 'tr': 'Turkish',
            'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian'
        }
        return lang_names.get(code, code.title())


def get_ai_segmentation_service() -> AISegmentationService:
    return AISegmentationService()