import logging
import json
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
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
        preserve_segments: bool = False,
        num_speakers: Optional[int] = None
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
            
            logger.info(f"🌍 TRANSLATION: source={source_language_code} → target={target_language_code}")
            
            if preserve_segments:
                logger.info(f"REDUB: Preserving {len(segments)} segments, translating text only")
                return self._translate_existing_segments(segments, target_language_code)
            
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
            return self._process_in_chunks(combined_text, target_language, preserve_segments=False, num_speakers=num_speakers)
        except Exception as e:
            logger.error(f"CRITICAL ERROR in create_optimal_segments_and_dub: {str(e)}")
            logger.error(f"Transcription data structure: {transcription_data}")
            raise e

    def _build_segmentation_and_dubbing_prompt(self, segments: List[Dict], target_language: str, preserve_segments: bool = False) -> str:
        target_lang_name = language_service.get_language_name(target_language)
        
        translation_instructions = f"""TRANSLATION TO {target_lang_name}:
- 100% {target_lang_name} only (same-language dubbing is valid for voice cloning/clarity)
- Natural conversational style
- Fix typos in original_text (keep meaning)
- Clean corrupted text
- Adapt idioms naturally"""
        
        if preserve_segments:
            processing_instruction = f"TRANSLATION REDUB - Clean corruption then translate to {target_lang_name}"
            example_dubbed = f"professional {target_lang_name} translation of cleaned text"
            
            return f"""REDUB MODE: Edit existing segments

RULES:
1. Output EXACTLY {len(segments)} segments (1 input = 1 output)
2. PRESERVE timing (start/end) from input
3. PRESERVE original_text from input (no changes)
4. {processing_instruction}
5. NO merging, NO splitting
6. PRESERVE speaker field exactly

{translation_instructions}

INPUT:
{json.dumps(segments, ensure_ascii=False, indent=2)}

OUTPUT:
{{
  "segments": [
    {{
      "id": "<from_input>",
      "start": <from_input>,
      "end": <from_input>,
      "speaker": "<from_input>",
      "original_text": "<from_input_no_changes>",
      "dubbed_text": "{example_dubbed}"
    }}
  ]
}}

✓ Exactly {len(segments)} segments
✓ Same timing as input
✓ {processing_instruction}"""
        else:
            return f"""DUBBING TASK: Translate audio transcription to {target_lang_name}

🚨 CRITICAL RULES:
1. USE ALL INPUT: Every word from input MUST appear in output (no content loss)
2. PRESERVE TIMING: Keep input start/end times (only split if segment >15s)
3. MAX DURATION: Each output segment ≤15.0 seconds
4. NO OVERLAPS: segment[i].start ≥ segment[i-1].end
5. PRESERVE SPEAKER: Copy exact speaker field from input to output
6. MANDATORY FIELDS: Every segment MUST have both original_text AND dubbed_text (never empty)

SEGMENTATION:
- If input segment ≤15s → Keep as-is (1 input = 1 output)
- If input segment >15s → Split at sentence/clause boundaries
- MERGE only if segments are very short (<2s) AND same speaker
- When splitting: All parts get same speaker as input
- When merging: Only merge if same speaker

{translation_instructions}

INPUT SEGMENTS:
{json.dumps(segments, ensure_ascii=False, indent=2)}

OUTPUT FORMAT:
{{
  "segments": [
    {{
      "id": "seg_001",
      "start": <input_start>,
      "end": <input_end or split_point>,
      "speaker": "<exact_speaker_from_input>",
      "original_text": "<corrected_input_text_NEVER_EMPTY>",
      "dubbed_text": "{target_lang_name} translation_NEVER_EMPTY"
    }}
  ]
}}

✓ Use ALL input content
✓ Preserve timing unless splitting >15s segments
✓ No overlaps in timestamps
✓ Keep speaker tags exact
✓ Both original_text and dubbed_text MUST be filled (never empty strings)"""
    
    def _format_segments_with_translation(self, ai_segments: List[Dict], global_segment_index_start: int = 0, preserve_original_timing: bool = False, original_segments: List[Dict] = None) -> List[Dict[str, Any]]:
        formatted_segments = []
        
        if preserve_original_timing and original_segments:
            logger.info(f"🔒 REDUB: Using original manifest timing and speakers for {len(ai_segments)} segments (ignoring AI timing)")
        
        for idx, seg in enumerate(ai_segments):
            if preserve_original_timing and original_segments and idx < len(original_segments):
                start_ms = int(original_segments[idx].get("start_ms", original_segments[idx].get("start", 0)))
                end_ms = int(original_segments[idx].get("end_ms", original_segments[idx].get("end", 0)))
                original_speaker = original_segments[idx].get("speaker")
            else:
                start_s = float(seg.get("start", 0))
                end_s = float(seg.get("end", 0))
                start_ms = int(start_s * 1000)
                end_ms = int(end_s * 1000)
                original_speaker = seg.get("speaker")
            
            if formatted_segments:
                prev_end_ms = formatted_segments[-1]["end"]
                if start_ms < prev_end_ms:
                    logger.warning(f"AI generated overlap: segment {idx} starts at {start_ms}ms but previous ends at {prev_end_ms}ms")
                    start_ms = prev_end_ms
                    if start_ms >= end_ms:
                        end_ms = start_ms + 100
                    logger.warning(f"Fixed overlap: adjusted segment to start at {start_ms}ms")
            
            duration_ms = end_ms - start_ms
            
            original_text = seg.get("original_text", "").strip()
            dubbed_text = seg.get("dubbed_text", "").strip()
            dubbed_text = re.sub(r'\[\w+:\s*([^\]]+)\]', r'\1', dubbed_text)
            dubbed_text = re.sub(r'\[[^\]]+\]', '', dubbed_text).strip()
            
            if not original_text or not dubbed_text:
                seg_id = seg.get("id", f"seg_{idx}")
                logger.error(f"AI returned malformed segment {seg_id}: {json.dumps(seg, ensure_ascii=False)}")
                raise ValueError(f"AI failed: segment {seg_id} has empty text (original: '{original_text}', dubbed: '{dubbed_text}')")
            
            global_segment_index = global_segment_index_start + len(formatted_segments)
            
            formatted_segments.append({
                "id": seg.get("id", f"seg_{global_segment_index+1:03d}"),
                "segment_index": global_segment_index,
                "start": start_ms,
                "end": end_ms,
                "duration_ms": duration_ms,
                "original_text": original_text,
                "dubbed_text": dubbed_text,
                "voice_cloned": False,
                "original_audio_file": None,
                "cloned_audio_file": None,
                "speaker": original_speaker
            })
        
        return formatted_segments
    
    def _translate_existing_segments(self, segments: List[Dict], target_language_code: str) -> List[Dict[str, Any]]:
        logger.info(f"REDUB MODE: Translating {len(segments)} segments to {target_language_code}")
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
        
        try:
            response = self._call_openai_with_retry(
                model="gpt-5-mini",
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": f"You are an expert translator. MANDATORY: 1) ALL translations in {self._get_language_name(target_language_code)} ONLY. 2) Use COLLOQUIAL/INFORMAL language (everyday conversation style). 3) Use STANDARD DIALECT and COMMON/POPULAR words. 4) FULLY LOCALIZE all cultural references. 5) CORRUPTION DETECTION: Clean repetitive patterns. 6) EXTRACT meaningful speech only. 7) Sound AUTHENTIC like native speakers, NOT textbook translation. OUTPUT FORMAT: Return ONLY valid JSON with 'segments' array."}]},
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
                    dubbed_text = seg_result.get("dubbed_text", "").strip()
                    
                    
                    chunk_results.append({
                        "id": seg_result.get("id"),
                        "start": int(original_seg.get("start", 0)),
                        "end": int(original_seg.get("end", 0)),
                        "duration_ms": int(original_seg.get("end", 0)) - int(original_seg.get("start", 0)),
                        "original_text": seg_result.get("original_text", ""),
                        "dubbed_text": dubbed_text,
                        "voice_cloned": False,
                        "original_audio_file": None,
                        "cloned_audio_file": None,
                        "speaker": original_seg.get("speaker")
                    })
            
            logger.info(f"✅ Translated chunk {chunk_number}: {len(chunk_results)} segments")
            return (i, chunk_results)
            
        except Exception as e:
            logger.error(f"❌ Translation chunk {chunk_number} failed: {str(e)}")
            raise ValueError(f"AI translation failed for chunk {chunk_number}: {str(e)}")
    
    def _translate_segments_preserve_timing(self, segments: List[Dict], target_language_code: str) -> List[Dict[str, Any]]:
        chunk_size = self.settings.AI_SEGMENTATION_CHUNK_SIZE
        chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]
        total_chunks = len(chunks)
        
        logger.info(f"🚀 Translating {len(segments)} segments in {total_chunks} chunks (size={chunk_size}) with {self.settings.OPENAI_PARALLEL_WORKERS} parallel workers (GPT-5-mini)")
        
        chunk_results = [None] * total_chunks
        
        with ThreadPoolExecutor(max_workers=self.settings.OPENAI_PARALLEL_WORKERS) as executor:
            futures = {
                executor.submit(self._translate_chunk_worker, (i, chunk, target_language_code)): i
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(futures):
                chunk_idx, results = future.result()
                chunk_results[chunk_idx] = results
        
        all_results = []
        for chunk_result in chunk_results:
            if chunk_result:
                for seg in chunk_result:
                    seg["segment_index"] = len(all_results)
                    all_results.append(seg)
        
        for i in range(1, len(all_results)):
            if all_results[i]["start"] < all_results[i-1]["end"]:
                gap = (all_results[i-1]["end"] - all_results[i]["start"]) // 2
                all_results[i-1]["end"] -= gap
                all_results[i-1]["duration_ms"] = all_results[i-1]["end"] - all_results[i-1]["start"]
                all_results[i]["start"] = all_results[i-1]["end"]
                all_results[i]["duration_ms"] = all_results[i]["end"] - all_results[i]["start"]
                logger.warning(f"REDUB TRANSLATE: Fixed overlap between segments - adjusted by {gap}ms")
        
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

ACTION: Extract meaningful speech only, clean corruption artifacts

TRANSLATION REQUIREMENTS:
1. LANGUAGE PURITY: 100% {target_lang_name} - never mix other languages
2. STRUCTURE: Preserve exact timing, segment count, and IDs (1:1 mapping)
3. NATURAL NATIVE SPEECH: Sound like everyday conversation by native speakers
   - Use COLLOQUIAL/INFORMAL language (how people talk in daily life)
   - Use STANDARD DIALECT (neutral, understood by all speakers)
   - Prefer COMMON/POPULAR vocabulary over formal words
   - Avoid literal translation - adapt meaning naturally
   - Make it AUTHENTIC, not textbook style
4. EMOTION: Preserve speaker's tone and emphasis
5. FULL CULTURAL LOCALIZATION:
   - Convert ALL cultural references to {target_lang_name} culture equivalents
   - Adapt idioms/metaphors to expressions familiar in target culture
   - Replace foreign examples/names with local alternatives when relevant
   - Use popular expressions native to {target_lang_name} speakers
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
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def _call_openai_with_retry(self, **kwargs):
        """OpenAI API call with exponential backoff retry logic"""
        success = False
        tokens = 0
        try:
            response = self._get_openai_client().responses.create(**kwargs)
            success = True
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
            return response
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                logger.warning(f"⚠️ Rate limit hit, retrying with exponential backoff...")
            else:
                logger.error(f"OpenAI API error: {error_msg}")
            raise
        finally:
            import asyncio
            from app.services.analytics_service import AnalyticsService
            try:
                asyncio.create_task(AnalyticsService.track_api_call("system", "openai", tokens=tokens, success=success))
            except:
                pass
    
    def _process_in_chunks(self, segments: List[Dict], target_language: str, preserve_segments: bool = False, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        chunk_size = self.settings.AI_SEGMENTATION_CHUNK_SIZE
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
- Expected speakers: {num_speakers if num_speakers else 'auto-detect'}
- Start segment IDs from seg_{len(all_results)+1:03d}  
- Time range: {chunk[0].get('start', 0):.3f}s to {chunk[-1].get('end', 0):.3f}s

🚨 SPEAKER PRESERVATION (MANDATORY):
- Each input segment has a "speaker" field (e.g., SPEAKER_00, SPEAKER_01)
- COPY the exact speaker value from input to output - NEVER change it
- When splitting: ALL resulting segments keep the SAME speaker as source
- When merging: ONLY merge if SAME speaker (NEVER merge different speakers)
- Keep speaker boundaries distinct - this is critical for voice consistency

OPTIMIZATION GOALS:
- IDEAL segment length: 3-8 seconds (optimal for voice cloning quality)
- MAXIMUM segment length: 15.0 seconds (strict limit)
- Merge very short segments if SAME speaker and combined duration ≤15s
- Split long segments intelligently at natural breaks
- When merging: Combine BOTH original_text AND dubbed_text
- When splitting: Divide BOTH original_text AND dubbed_text proportionally

SPLIT RULES (when breaking long segments):
- BOTH original_text AND dubbed_text must be split proportionally
- Each split segment must have corresponding portions of BOTH texts
- Split at natural boundaries to maintain meaning in BOTH languages

SPLIT EXAMPLE:
Input: start=0.0s, end=12.5s (12.5s duration - too long)
       original_text: "Hello everyone, welcome to the show. Today we'll discuss AI."
       dubbed_text: "সবাইকে স্বাগতম, আমাদের শোতে। আজ আমরা AI নিয়ে আলোচনা করব।"
       
Split → Segment 1: start=0.0s, end=6.2s
                    original_text: "Hello everyone, welcome to the show."
                    dubbed_text: "সবাইকে স্বাগতম, আমাদের শোতে।"
        Segment 2: start=6.2s, end=12.5s
                    original_text: "Today we'll discuss AI."
                    dubbed_text: "আজ আমরা AI নিয়ে আলোচনা করব।"

SPLIT PRIORITY:
1. Sentence boundaries (highest priority)
2. Clause boundaries (commas, conjunctions)
3. Natural pauses in speech
4. Speaker changes (NEVER merge across different speakers)
5. Breathing points for voice actors

QUALITY RULES:
- COVER ALL content exactly once - no gaps, no overlaps, no repetitions
- DO NOT repeat content from previous chunks
- CLEAN all corruption patterns automatically
- PRESERVE emotional flow and conversation context
- Keep single speaker's continuous thought together when possible
- NEVER merge segments with different speakers

CONTENT PRESERVATION (CRITICAL):
- MUST cover ALL input text exactly once - no gaps, no missing content
- EVERY segment MUST have both original_text, dubbed_text AND speaker filled
- NO empty segments allowed - all text must be meaningful
- If {num_speakers if num_speakers else 'auto'} speakers detected, preserve speaker distinction across segments
- Split/merge intelligently but NEVER drop content or speaker info
- Validate output: all input content AND speaker tags must appear in output segments
"""
            
            target_lang_name = self._get_language_name(target_lang_code)
            prompt = chunk_context + self._build_segmentation_and_dubbing_prompt(chunk, target_lang_name, preserve_segments)
            
            try:
                response = self._call_openai_with_retry(
                    model="gpt-5-mini",
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": f"""ROLE: Expert dubbing AI specializing in natural speech adaptation and corruption detection

MODE: {'REDUB - Preserve exact timing and structure (1:1 mapping)' if preserve_segments else 'FRESH - Intelligent segmentation for optimal voice cloning'}

🚨 TOP PRIORITY - SPEAKER PRESERVATION:
- EVERY input segment has a speaker field (SPEAKER_00, SPEAKER_01, etc.)
- YOU MUST COPY the exact speaker from input to output - NEVER change it
- When splitting: ALL output segments inherit the speaker from original input
- When merging: ONLY merge segments with SAME speaker, never different speakers
- NEVER default to SPEAKER_00 - use the actual speaker value from input
- Multiple speakers detected: {num_speakers if num_speakers else 'auto'} - keep them distinct

CRITICAL RULES:
1. Speaker: PRESERVE exact speaker from input (highest priority)
2. Duration: All segments ≤15 seconds maximum
3. Language: {target_lang_name} ONLY - no mixing with other languages
4. Content Coverage: ALL input text must appear in output - NO empty segments
5. MANDATORY: Every segment MUST have both original_text AND dubbed_text filled (never empty)
6. Emotion: Preserve speaker's tone, emphasis, and emotional intent
7. NATURAL NATIVE SPEECH: Sound like everyday native speakers
   - Use COLLOQUIAL/INFORMAL expressions (daily conversation style)
   - Use STANDARD DIALECT (neutral, widely understood)
   - Prefer COMMON/POPULAR words over formal vocabulary
   - Adapt naturally, NOT word-for-word literal translation
   - Make output AUTHENTIC like real {target_lang_name} speakers

CORRUPTION AUTO-CLEAN:
• Repetitive patterns (40,000,000... or juice juice juice...) → Extract once
• Repeated characters (aaaaaaa...) → Remove entirely
• Partial corruption → Extract meaningful parts using context

QUALITY GUIDELINES:
• Cultural Localization: Convert cultural references to {target_lang_name} equivalents
• Consistency: Keep terminology consistent across segments
• Punctuation: Adapt to {target_lang_name} conventions
• Context: Use surrounding segments to infer meaning

OUTPUT: Valid JSON with 'segments' array only (EVERY segment MUST have: speaker, original_text, dubbed_text - all filled/never empty)"""}]},
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
                    global_segment_index_start,
                    preserve_original_timing=preserve_segments,
                    original_segments=chunk if preserve_segments else None
                )
                all_results.extend(chunk_segments)
                
            except Exception as e:
                logger.error(f"Chunk processing FAILED: {str(e)}")
                raise e
        
        logger.info(f"Total processed: {len(all_results)} segments from {len(segments)} input segments")
        return all_results

    def translate_text(self, text: str, target_language: str, source_language: str = "auto") -> str:
        """Translate a portion of text using OpenAI"""
        try:
            is_same_language = source_language.lower().strip() == target_language.lower().strip()
            
            if is_same_language:
                return text.strip()
            
            prompt = f"""PROFESSIONAL TRANSLATION TO {target_language.upper()}:

TEXT TO TRANSLATE: {text}

REQUIREMENTS:
- 100% {target_language} ONLY - no language mixing
- NATURAL NATIVE SPEECH style (colloquial/informal)
- Use STANDARD DIALECT and COMMON vocabulary
- FULL CULTURAL LOCALIZATION of references
- Sound AUTHENTIC like native speakers
- Clean any corruption patterns automatically

OUTPUT: Return ONLY the translated text, nothing else."""
            
            response = self._call_openai_with_retry(
                model="gpt-5-mini",
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": f"You are a professional translator. Translate the given text to {target_language} naturally. Return ONLY the translated text."}]},
                    {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
                ],
                text={"verbosity": "low"},
                reasoning={"effort": "minimal"},
                max_output_tokens=500
            )
            
            result = response.output_text.strip()
            return result if result else text
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return text

    def translate_contextual_segment(self, segments: List[str], target_segment_index: int, target_language: str, source_language: str = "auto") -> str:
        """Translate a specific segment with context from all segments"""
        try:
            if not segments or target_segment_index < 0 or target_segment_index >= len(segments):
                raise ValueError(f"Invalid segment index {target_segment_index} for {len(segments)} segments")
            
            target_text = segments[target_segment_index]
            is_same_language = source_language.lower().strip() == target_language.lower().strip()
            
            if is_same_language:
                return target_text.strip()
            
            # Build context from all segments
            context_parts = []
            for i, segment in enumerate(segments):
                marker = " ← TRANSLATE THIS" if i == target_segment_index else ""
                context_parts.append(f"Segment {i+1}: {segment}{marker}")
            
            context = "\n".join(context_parts)
            
            prompt = f"""CONTEXTUAL TRANSLATION TO {target_language.upper()}:

FULL CONVERSATION CONTEXT:
{context}

INSTRUCTIONS:
- Translate ONLY the marked segment (Segment {target_segment_index + 1})
- Keep the translation CONSISTENT with the overall conversation flow
- Maintain the same tone, style, and meaning as the other segments
- Ensure the translated segment fits naturally in the conversation context
- Use NATURAL NATIVE SPEECH style (colloquial/informal)
- Apply FULL CULTURAL LOCALIZATION

OUTPUT: Return ONLY the translated segment text, nothing else."""
            
            response = self._call_openai_with_retry(
                model="gpt-5-mini",
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": f"You are a professional translator specializing in contextual translation. Translate the specified segment to {target_language} while maintaining conversation consistency. Return ONLY the translated text."}]},
                    {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
                ],
                text={"verbosity": "low"},
                reasoning={"effort": "minimal"},
                max_output_tokens=500
            )
            
            result = response.output_text.strip()
            return result if result else target_text
            
        except Exception as e:
            logger.error(f"Contextual translation failed: {str(e)}")
            return segments[target_segment_index] if segments else ""

    def _get_language_name(self, code):
        return language_service.get_language_name(code)


def get_ai_segmentation_service() -> AISegmentationService:
    return AISegmentationService()