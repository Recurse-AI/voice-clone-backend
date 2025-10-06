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
                    
                    segment_data = {
                        "text": text,
                        "start": round(start_s, 3),
                        "end": round(end_s, 3),
                        "duration": round(end_s - start_s, 3),
                        "start_ms": start_ms,
                        "end_ms": end_ms
                    }
                    
                    # Include speaker tag if available
                    if "speaker" in seg and seg["speaker"]:
                        segment_data["speaker"] = seg["speaker"]
                    
                    combined_text.append(segment_data)
            
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

MANDATORY 1:1 MAPPING:
1. Output EXACTLY {len(segments)} segments (one output per input)
2. PRESERVE exact start/end timing from input
3. PRESERVE exact original_text from input
4. {processing_instruction}
5. NO merging, NO splitting, NO structure changes

TRANSLATION QUALITY:
- Make dubbed_text sound natural in target language
- Preserve emotional tone and speaker emphasis
- Adapt punctuation for target language conventions
- Convert idioms/expressions to cultural equivalents
- Maintain conversation flow and context
- Clean corruption while keeping meaningful content

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
      "speaker": "SPEAKER_00",
      "original_text": "exact text from input (preserved)",
      "dubbed_text": "{example_dubbed} (natural translation)"
    }}
  ]
}}

🚨 MANDATORY SPEAKER PRESERVATION:
- MUST preserve "speaker" field from input to output EXACTLY
- If input segment has "speaker": "SPEAKER_00", output MUST include "speaker": "SPEAKER_00"
- Speaker tags are CRITICAL for voice consistency - NEVER drop them

CORRUPTION HANDLING:
- Clean repetitive patterns before translating
- Extract meaningful content from corrupted input
- Use "[unclear audio]" only for completely unrecoverable segments
- Leverage context from surrounding segments when possible

CRITICAL: Must output exactly {len(segments)} segments as valid JSON. {processing_instruction}."""
        else:
            return f"""FRESH DUBBING MODE:

🚨 ABSOLUTE RULE: Every output segment MUST be ≤15.0 seconds duration. NO EXCEPTIONS.

SEGMENTATION STRATEGY:
1. Calculate each segment: (end - start) MUST be ≤15.0 seconds
2. SPLIT long segments at:
   • Sentence endings (period, question mark, exclamation)
   • Clause boundaries (commas, semicolons)
   • Natural pauses in speech flow
   • Speaker transitions
3. MERGE short segments (< 2s) if combined ≤15.0s
4. IDEAL target: 3-8 seconds per segment for best voice quality
5. Output segment count can differ from input count

QUALITY REQUIREMENTS:
- Use ALL input content exactly once (no gaps, no duplicates)
- Preserve emotional tone and speaker intent
- Adapt cultural idioms/expressions naturally
- Maintain conversation flow and context
- Fix corrupted text by extracting meaningful parts

{translation_instructions}

INPUT: {json.dumps(segments, ensure_ascii=False, indent=2)}

OUTPUT JSON:
{{
  "segments": [
    {{
      "id": "seg_001", 
      "start": 0.080,
      "end": 4.560,
      "speaker": "SPEAKER_00",
      "original_text": "meaningful clean text",
      "dubbed_text": "natural {target_lang_name} translation"
    }}
  ]
}}

🚨 MANDATORY SPEAKER PRESERVATION:
- MUST preserve "speaker" field from input to output EXACTLY
- If input segment has "speaker": "SPEAKER_00", output MUST include "speaker": "SPEAKER_00"
- If input has no speaker, output can omit or set null
- Speaker tags are CRITICAL for voice consistency - NEVER drop them
- Consider speaker changes when creating segments
- Keep same-speaker dialogue together when natural

🚨 CORRUPTION AUTO-CLEAN EXAMPLES:
- "40,000,000,000..." → Extract meaningful part or "[unclear audio]"
- "juice juice juice..." → "juice" (deduplicate)
- "aaaaaaaaa..." → Remove entirely
- "कर दो कर दो कर दो" → "कर दो" (clean repetition)
- Partial corruption → Use context from nearby segments to infer meaning

🚨 LANGUAGE PURITY:
- Target: {target_lang_name} → 100% {target_lang_name} in ALL dubbed_text
- NEVER mix languages (no English in French, no Spanish in English, etc.)
- Cultural adaptation: Convert idioms to target culture equivalents

FINAL CHECK: Every segment ≤15.0s ✓ No corruption ✓ Natural speech ✓"""
    
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
                "cloned_audio_file": None,
                "speaker": seg.get("speaker")
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
                    "cloned_audio_file": None,
                    "speaker": seg.get("speaker")
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
                        "cloned_audio_file": None,
                        "speaker": original_seg.get("speaker")
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
                    "cloned_audio_file": None,
                    "speaker": seg.get("speaker")
                })
            return (i, fallback_results, False)
    
    def _translate_segments_preserve_timing(self, segments: List[Dict], target_language_code: str) -> List[Dict[str, Any]]:
        chunk_size = 10
        chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
        total_chunks = len(chunks)
        
        logger.info(f"🚀 Translating {len(segments)} segments in {total_chunks} chunks with {settings.OPENAI_PARALLEL_WORKERS} parallel workers (GPT-5-mini)")
        
        chunk_results = [None] * total_chunks
        
        with ThreadPoolExecutor(max_workers=settings.OPENAI_PARALLEL_WORKERS) as executor:
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
        return f"""PROFESSIONAL TRANSLATION WITH CORRUPTION HANDLING

TARGET LANGUAGE: {target_lang_name} (100% exclusive - no language mixing)

INPUT SEGMENTS:
{json.dumps(segments, ensure_ascii=False, indent=2)}

CORRUPTION AUTO-CLEAN:
Detect and remove these patterns automatically:
• Repetitive sequences (40,000,000... or juice juice juice...)
• Character spam (aaaaaaa, xxxxx, +++++)
• Symbol artifacts (@#$%, [[[, ----)
• Meaningless/garbled content
• Transcription errors

ACTION: Extract meaningful speech only, ignore corruption artifacts
FALLBACK: Use "[unclear audio]" for completely corrupted segments

TRANSLATION REQUIREMENTS:
1. LANGUAGE PURITY: 100% {target_lang_name} - never mix other languages
2. STRUCTURE: Preserve exact timing, segment count, and IDs (1:1 mapping)
3. QUALITY: Natural, professional speech that sounds native
4. EMOTION: Preserve speaker's tone and emphasis
5. CULTURAL: Adapt idioms/expressions to target culture
6. CONTEXT: Use surrounding segments to infer unclear parts
7. CONSISTENCY: Keep terminology consistent across all segments
8. OUTPUT: EXACTLY {len(segments)} segments

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
- Input: {len(chunk)} segments → Output: EXACTLY {len(chunk)} segments (1:1 mapping)
- Start segment IDs from seg_{len(all_results)+1:03d}
- Time range: {chunk[0].get('start', 0):.3f}s to {chunk[-1].get('end', 0):.3f}s

MANDATORY:
- Process each input segment as exactly one output segment
- PRESERVE exact timing (start/end) and original_text from input
- TRANSLATE only the dubbed_text field to sound natural
- Maintain emotional tone and emphasis from original speech
- Adapt punctuation and phrasing for target language conventions
- Clean corruption patterns while preserving meaningful content
"""
            else:
                chunk_context = f"""
FRESH DUBBING CHUNK {chunk_number}/{total_chunks}:
- Input: {len(chunk)} segments for intelligent optimization
- Start segment IDs from seg_{len(all_results)+1:03d}  
- Time range: {chunk[0].get('start', 0):.3f}s to {chunk[-1].get('end', 0):.3f}s

OPTIMIZATION GOALS:
- IDEAL segment length: 3-8 seconds (optimal for voice cloning quality)
- MAXIMUM segment length: 15.0 seconds (strict limit)
- Merge very short segments if combined duration ≤15s
- Split long segments intelligently at natural breaks

SPLIT PRIORITY (when breaking long segments):
1. Sentence boundaries (highest priority)
2. Clause boundaries (commas, conjunctions)
3. Natural pauses in speech
4. Speaker changes (if applicable)
5. Breathing points for voice actors

QUALITY RULES:
- COVER ALL content exactly once - no gaps, no overlaps, no repetitions
- DO NOT repeat content from previous chunks
- CLEAN all corruption patterns automatically
- PRESERVE emotional flow and conversation context
- Keep single speaker's continuous thought together when possible
"""
            
            target_lang_name = self._get_language_name(target_lang_code)
            prompt = chunk_context + self._build_segmentation_and_dubbing_prompt(chunk, target_lang_name, is_same_language, preserve_segments)
            
            time.sleep(0.5)
            
            try:
                response = self._get_openai_client().responses.create(
                    model="gpt-5-mini",
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": f"""ROLE: Expert dubbing AI specializing in natural speech adaptation and corruption detection

MODE: {'REDUB - Preserve exact timing and structure (1:1 mapping)' if preserve_segments else 'FRESH - Intelligent segmentation for optimal voice cloning'}

CORE RULES:
1. Duration: All segments ≤15 seconds maximum
2. Language: {target_lang_name} ONLY - no mixing with other languages
3. Emotion: Preserve speaker's tone, emphasis, and emotional intent
4. Natural: Output must sound natural in {target_lang_name}, not word-for-word translation

CORRUPTION AUTO-CLEAN:
• Repetitive patterns (40,000,000... or juice juice juice...) → Extract once or mark unclear
• Repeated characters (aaaaaaa...) → Remove entirely
• Meaningless sequences → Use '[unclear audio]' in {target_lang_name}
• Partial corruption → Extract meaningful parts using context clues

QUALITY GUIDELINES:
• Cultural adaptation: Convert idioms/expressions naturally for target culture
• Consistency: Keep terminology consistent across segments
• Punctuation: Adapt to {target_lang_name} conventions
• Context: Use surrounding segments to infer unclear audio when possible

OUTPUT: Valid JSON with 'segments' array only"""}]},
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