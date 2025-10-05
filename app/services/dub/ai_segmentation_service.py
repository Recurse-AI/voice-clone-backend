import logging
import json
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            translation_instructions = f"""SAME LANGUAGE PROCESSING ({target_lang_name}):
- Source and target are IDENTICAL - NO translation needed
- CLEAN corrupted/repetitive patterns first
- Copy cleaned meaningful text to dubbed_text 
- If purely corrupted, use '[unclear audio]' in {target_lang_name}
- NEVER output repetitive patterns like "000,000..." or "aaa..."
- Extract only meaningful speech content
- ADD SPEECH MARKERS: Enhance with OpenAudio S1 markers when context requires: (laughing) (excited) (whispering) (angry) (sad) (hesitating) etc."""
        else:
            translation_instructions = f"""PROFESSIONAL TRANSLATION TO {target_lang_name.upper()} - ZERO TOLERANCE FOR MIXING:
- ABSOLUTE RULE: 100% {target_lang_name} ONLY - NO Spanish/German/French/Italian mixing
- CORRUPTION AUTO-CLEAN: Before translating, automatically detect and remove:
  * Long repetitive numbers (40,000,000,000,000...)
  * Repeated character sequences (aaaaaaa, xxxxx...)
  * Looping words/phrases (juice juice juice, do it do it...)
  * Symbol spam or transcription artifacts (@#$%, [[[, +++...)
  * Placeholder text or corrupted patterns
- MEANINGFUL EXTRACTION: Find the actual speech content within corrupted input
- CLEAN TRANSLATION: Translate only the meaningful parts to natural {target_lang_name}
- LANGUAGE PURITY: Every single word must be proper {target_lang_name} vocabulary
- FALLBACK PROTOCOL: Purely corrupted segments → '[unclear audio]' in {target_lang_name}
- PRESERVE INTENT: Keep original meaning/tone of actual speech content
- SPEECH CONTROL: Add OpenAudio S1 markers ONLY when truly needed: (laughing) (excited) (angry) (sad) (whispering) (shouting) (hesitating) (scared) (surprised) etc."""
        
        if preserve_segments:
            if is_same_language:
                processing_instruction = f"SAME LANGUAGE REDUB - Clean corruption then copy to dubbed_text"
                example_dubbed = "cleaned meaningful text (same as original after corruption removal)"
            else:
                processing_instruction = f"TRANSLATION REDUB - Clean corruption then translate to {target_lang_name}"
                example_dubbed = f"professional {target_lang_name} translation of cleaned text"
            
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
6. If possible, split segments by speaker changes
7. Output segment count can be different from input count

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

🚨 CORRUPTION EXAMPLES TO DETECT & CLEAN:
- "40,000,000,000,000,000..." → Extract meaningful part or use "[unclear audio]"
- "juice juice juice juice..." → "juice" (single occurrence) 
- "aaaaaaaaaaaaaaa..." → Remove entirely or extract meaningful content
- "कर दो कर दो कर दो" → "कर दो" (clean repetition)

🚨 LANGUAGE MIXING PREVENTION:
- Target: English → ALL dubbed_text in English (never Spanish/German/French)
- Target: French → ALL dubbed_text in French (never English/Spanish/German) 
- NEVER output mixed languages in the same response

🚨 FINAL VERIFICATION: Every segment duration (end-start) ≤ 15.0 seconds. ZERO corruption patterns."""
    
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
    
    def _translate_chunk_worker(self, args) -> tuple:
        import time
        i, chunk, target_language_code = args
        chunk_number = i + 1
        
        segments_for_translation = []
        for idx, seg in enumerate(chunk):
            original_text = seg.get("original_text", seg.get("text", "")).strip()
            segments_for_translation.append({
                "id": seg.get("id", f"seg_{idx+1:03d}"),
                "start": int(seg.get("start", 0)),
                "end": int(seg.get("end", 0)),
                "original_text": original_text
            })
        
        prompt = self._build_translation_only_prompt(segments_for_translation, target_language_code)
        
        time.sleep(0.5)
        
        try:
            response = self._get_openai_client().responses.create(
                model="gpt-5-mini",
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": f"You are an expert translator. MANDATORY: 1) ALL translations in {self._get_language_name(target_language_code)} ONLY. 2) CORRUPTION DETECTION: Clean repetitive patterns. 3) EXTRACT meaningful speech only. 4) Ensure natural {self._get_language_name(target_language_code)} output. OUTPUT FORMAT: Return ONLY valid JSON with 'segments' array."}]},
                    {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
                ],
                text={"verbosity": "low"},
                reasoning={"effort": "minimal"},
                max_output_tokens=8192
            )
            
            ai_response = response.output_text.strip()
            result = json.loads(ai_response)
            
            chunk_results = []
            for seg_result in result.get("segments", []):
                original_seg = next((s for s in chunk if s.get("id") == seg_result.get("id")), None)
                if original_seg:
                    chunk_results.append({
                        "id": seg_result.get("id"),
                        "start": int(original_seg.get("start", 0)),
                        "end": int(original_seg.get("end", 0)),
                        "duration_ms": int(original_seg.get("end", 0)) - int(original_seg.get("start", 0)),
                        "original_text": seg_result.get("original_text", ""),
                        "dubbed_text": seg_result.get("dubbed_text", "").strip(),
                        "voice_cloned": False,
                        "original_audio_file": None,
                        "cloned_audio_file": None
                    })
            
            logger.info(f"✅ Translated chunk {chunk_number}: {len(chunk_results)} segments")
            return (i, chunk_results, True)
            
        except Exception as e:
            logger.error(f"❌ Translation chunk {chunk_number} failed: {str(e)}")
            fallback_results = []
            for seg in chunk:
                original_text = seg.get("original_text", seg.get("text", "")).strip()
                fallback_results.append({
                    "id": seg.get("id"),
                    "start": int(seg.get("start", 0)),
                    "end": int(seg.get("end", 0)),
                    "duration_ms": int(seg.get("end", 0)) - int(seg.get("start", 0)),
                    "original_text": original_text,
                    "dubbed_text": original_text,
                    "voice_cloned": False,
                    "original_audio_file": None,
                    "cloned_audio_file": None
                })
            return (i, fallback_results, False)
    
    def _translate_segments_preserve_timing(self, segments: List[Dict], target_language_code: str) -> List[Dict[str, Any]]:
        chunk_size = 10
        chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
        total_chunks = len(chunks)
        
        logger.info(f"🚀 Translating {len(segments)} segments in {total_chunks} chunks with 5 parallel workers (GPT-5-mini)")
        
        chunk_results = [None] * total_chunks
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._translate_chunk_worker, (i, chunk, target_language_code)): i
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(futures):
                chunk_idx, results, success = future.result()
                chunk_results[chunk_idx] = results
        
        all_results = []
        for chunk_result in chunk_results:
            if chunk_result:
                for seg in chunk_result:
                    seg["segment_index"] = len(all_results)
                    all_results.append(seg)
        
        logger.info(f"🎯 REDUB: Translated {len(all_results)} segments")
        return all_results
    
    def _build_translation_only_prompt(self, segments: List[Dict], target_language_code: str) -> str:
        target_lang_name = self._get_language_name(target_language_code)
        return f"""ADVANCED TRANSLATION WITH CORRUPTION HANDLING - ZERO TOLERANCE POLICY

TARGET LANGUAGE: {target_lang_name} (EXCLUSIVE - NO OTHER LANGUAGES ALLOWED)

INPUT SEGMENTS:
{json.dumps(segments, ensure_ascii=False, indent=2)}

CORRUPTION AUTO-DETECTION & ELIMINATION:
- SCAN for these corruption patterns and REMOVE automatically:
  * Repetitive number sequences (40,000,000,000,000... or similar)
  * Character spam (aaaaaaa, xxxxx, +++++)
  * Word/phrase loops (juice juice juice, do it do it do it)
  * Symbol artifacts (@#$%, [[[, ----, corrupted transcription marks)
  * Meaningless placeholder text or garbled content
- EXTRACT ONLY the actual meaningful speech content
- CLEAN before processing - never translate corrupted patterns
- FALLBACK: If segment is purely corrupted → "[unclear audio]" in {target_lang_name}

TRANSLATION PROTOCOL:
- LANGUAGE PURITY: 100% {target_lang_name} vocabulary - NEVER mix Spanish/German/French/etc
- STRUCTURE: Maintain exact timing, segment count, and IDs from input  
- QUALITY: Natural, professional {target_lang_name} with proper grammar
- CONSISTENCY: Every dubbed_text must be pure {target_lang_name}
- MAPPING: EXACTLY {len(segments)} output segments (1:1 with input)
- SPEECH MARKERS: Add emotion/tone markers when context demands: (laughing) (excited) (angry) (whispering) (shouting) (sad) (hesitating) etc.

OUTPUT SPECIFICATION (JSON):
{{
  "segments": [
    {{
      "id": "[identical to input]",
      "original_text": "[corruption-cleaned input text]", 
      "dubbed_text": "[professional {target_lang_name} translation ONLY]"
    }}
  ]
}}

VALIDATION CHECKLIST:
✓ Exactly {len(segments)} segments output
✓ ALL dubbed_text in {target_lang_name} (no language mixing)  
✓ Zero corruption patterns in output
✓ Preserved timing and segment IDs
✓ Natural, meaningful translations"""
    
    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
        return self._openai_client
    
    def _process_in_chunks(self, segments: List[Dict], target_language: str, is_same_language: bool = False, preserve_segments: bool = False) -> List[Dict[str, Any]]:
        import time
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
            
            time.sleep(0.5)
            
            try:
                response = self._get_openai_client().responses.create(
                    model="gpt-5-mini",
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": f"You are an expert audio dubbing AI with ADVANCED CORRUPTION DETECTION and LANGUAGE CONSISTENCY ENFORCEMENT. MODE: {'REDUB - Preserve exact timing/structure, 1:1 mapping, translate dubbed_text only' if preserve_segments else 'FRESH - Intelligent segmentation, merge short segments, split long ones, target 3-8s for optimal voice cloning'}. CRITICAL RULES: 1) All segments must be ≤15 seconds. 2) TARGET LANGUAGE: {target_lang_name} - ALL dubbed_text must be in {target_lang_name} EXCLUSIVELY, NEVER mix with Spanish/German/French/Italian/etc. 3) CORRUPTION AUTO-DETECTION: Identify and clean these patterns automatically: • Repetitive numbers (40,000,000,000...) • Repeated characters (aaaaaaa...) • Repeated words/phrases (juice juice juice...) • Meaningless symbol sequences • Corrupted transcription artifacts • Placeholder text patterns. 4) EXTRACT meaningful speech from corrupted input, ignore artifacts. 5) LANGUAGE PURITY: Each segment must be 100% {target_lang_name}, no mixing. 6) FALLBACK: For purely corrupted segments, use '[unclear audio]' in {target_lang_name}. 7) ANTI-LOOP: Never repeat phrases within/across segments. OUTPUT FORMAT: Return ONLY valid JSON with 'segments' array."}]},
                        {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
                    ],
                    text={"verbosity": "low"},
                    reasoning={"effort": "minimal"},
                    max_output_tokens=self.max_tokens
                )
                
                ai_response = response.output_text.strip()
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