import logging
import json
import re
from typing import List, Dict, Any, Optional
from app.services.language_service import language_service

logger = logging.getLogger(__name__)

class AISegmentationService:
    def __init__(self):
        self.max_tokens = 16384
        self._openai_client = None
        
    def create_optimal_segments_and_dub(
        self,
        transcription_data: Dict[str, Any],
        target_language: str,
        preserve_segments: bool = False
    ) -> List[Dict[str, Any]]:
        """Create optimal segments AND translate using AI in one call"""
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
            
            # Extract source language from transcription data
            source_language = transcription_data.get("language", "auto_detect")
            source_language_code = language_service.normalize_language_input(source_language)
            target_language_code = language_service.normalize_language_input(target_language)
            
            # Check if source and target languages are the same
            is_same_language = source_language_code == target_language_code
            if is_same_language:
                logger.info(f"🎯 SAME LANGUAGE DETECTED: source={source_language_code}, target={target_language_code} - will preserve original text")
            else:
                logger.info(f"🌍 TRANSLATION NEEDED: source={source_language_code} → target={target_language_code}")
            
            # Simple logic:
            # - If preserve_segments = True = REDUB (keep segments, only translate text)
            # - Otherwise = FRESH DUBBING (full AI segmentation and translation)
            
            if preserve_segments:
                logger.info(f"REDUB: Preserving {len(segments)} segments, translating text only")
                return self._translate_existing_segments(segments, target_language_code, is_same_language)
            
            # Process through AI to generate new dubbed_text with segmentation
            logger.info(f"FRESH DUBBING: Full AI segmentation + translation for {len(segments)} segments")
            combined_text = []
            for seg in segments:
                # Handle both transcription segments ("text") and manifest segments ("original_text")
                text = seg.get("text", "").strip() or seg.get("original_text", "").strip()
                if text:
                    start_ms = int(seg.get("start", 0))
                    end_ms = int(seg.get("end", 0))
                    
                    # Validate timing (milliseconds)
                    if start_ms >= end_ms or end_ms - start_ms < 100:
                        logger.warning(f"Invalid segment timing: start={start_ms}ms, end={end_ms}ms, skipping")
                        continue
                    
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
            
            # Process all segments in chunks
            logger.info(f"Processing {len(combined_text)} segments in chunks")
            logger.info(f"✅ TIMING STANDARD: All inputs converted to milliseconds, AI gets seconds for better understanding")
            return self._process_in_chunks(combined_text, target_language, is_same_language)
        except Exception as e:
            logger.error(f"CRITICAL ERROR in create_optimal_segments_and_dub: {str(e)}")
            logger.error(f"Transcription data structure: {transcription_data}")
            raise e

    
    def _build_segmentation_and_dubbing_prompt(self, segments: List[Dict], target_language: str, is_same_language: bool = False) -> str:
        
        if is_same_language:
            translation_instructions = f"""SAME LANGUAGE PROCESSING:
- Source and target language are IDENTICAL ({target_language})
- PRESERVE the original text EXACTLY as is - NO translation needed
- Copy the original_text to dubbed_text WITHOUT any modifications
- Do NOT attempt to translate, paraphrase, or change the text in any way
- Keep the EXACT wording, punctuation, and formatting"""
        else:
            translation_instructions = f"""STRICT TRANSLATION REQUIREMENTS - LITERAL TRANSLATION ONLY:
- Translate EVERY SINGLE WORD into proper {target_language} - NO PARAPHRASING
- NEVER change sentence structure, word order, or meaning
- NEVER add, remove, or substitute words unless grammatically required for {target_language}
- NEVER use synonyms, alternatives, or "improved" versions - translate LITERALLY
- Keep EXACT same meaning, tone, and sentence structure as original
- Maintain original word count and phrasing as closely as possible
- Only make minimal changes when target language grammar absolutely requires it"""
        
        return f"""You are a professional dubbing AI. Your job: Create SMART, UNIQUE audio segments and provide REAL content.

INPUT TRANSCRIPTION:
{json.dumps(segments, ensure_ascii=False, indent=2)}

CRITICAL DUPLICATE PREVENTION:
- Each output segment MUST be UNIQUE - no duplicates allowed
- Use DIFFERENT start/end times for each segment
- NEVER repeat the same text content across multiple segments
- Each segment must cover DIFFERENT portions of the input
- Assign UNIQUE IDs to each segment (seg_001, seg_002, etc.)

CRITICAL PRESERVATION RULES - MINIMAL CHANGES ONLY:
1. PRESERVE ORIGINAL TIMING: Keep start/end times exactly as provided in input
2. PRESERVE ORIGINAL TEXT: Keep original_text exactly as provided - no modifications
3. ONLY TRANSLATE: Create proper dubbed_text in target language with correct alphabet
4. SPLIT ONLY IF MANDATORY: Only split segments that exceed 12.0 seconds duration
5. NO MERGING: Do not merge segments unless absolutely critical for quality
6. ONE-TO-ONE MAPPING: Each input segment should typically become one output segment
7. MINIMAL OPTIMIZATION: Preserve the natural segmentation from transcription
8. SENTENCE INTEGRITY: If splitting is required (>12s), split at natural sentence breaks
9. COVER ALL input content exactly once - no repetition, no omissions
10. DURATION ENFORCEMENT: Only modify timing if segment exceeds 12 seconds

{translation_instructions}
- NEVER use generic phrases like "Translated content for segment X"
- NEVER use placeholder text or lazy translations

BANNED PHRASES - DO NOT USE THESE:
- "Translated content for segment"
- "English translation"
- "Content for segment"
- Any generic placeholder text
- Any lazy or meaningless translations

WHAT I WANT VS WHAT I DON'T WANT:
✅ GOOD: Proper meaningful translation of the actual content
❌ BAD: "Translated content for segment 1"
❌ BAD: "English translation 2" 
❌ BAD: "Content for segment X"
❌ BAD: Any generic placeholder text
❌ BAD: Duplicate segments with same timing or content

OUTPUT FORMAT (JSON):
{{
  "segments": [
    {{
      "id": "seg_001",
      "start": 0.080,
      "end": 4.560,
      "original_text": "[exact text from input]",
      "dubbed_text": "[actual meaningful translation in {target_language}]"
    }}
  ]
}}

TIMING FORMAT RULES - PRESERVE ORIGINAL:
- INPUT times are in SECONDS for easier understanding (e.g., 145.320 = 145.32 seconds from video start)
- OUTPUT times must MATCH INPUT times exactly (preserve original timing)
- ONLY modify timing if input segment exceeds 12.0 seconds (split required)
- Use 3 decimal places for precision (e.g., 145.320, not 145.32)
- System Note: Input data is converted from milliseconds to seconds for AI processing
- PRESERVE example: input start: 145.320, end: 149.870 → output start: 145.320, end: 149.870
- SPLIT example: input 15s segment → split into 2 segments of ~7.5s each
- FORBIDDEN: Changing timing of segments that are ≤12.0 seconds

CRITICAL RULES - BREAK THESE AND YOU FAIL:
1. PRESERVE EXACTLY: Keep original start, end, and original_text from input unchanged
2. ONLY TRANSLATE dubbed_text: Translate to target language with proper alphabet
3. NO lazy translations like "Translated content for segment X"
4. MINIMAL CHANGES: Only modify segments if they exceed 12.0 seconds duration
5. Every dubbed_text must be ACTUAL TRANSLATION of the original_text
6. PRESERVE STRUCTURE: Maintain the natural segmentation from transcription
7. NEVER skip any input content - process every input segment
8. TRANSLATE the actual meaning, not just word-for-word
9. Make it sound natural in {target_language}
10. orginal_text and dubbed_text should be plain text, no [], no formatting, no language markers, no explanations, no notes, no nothing.
11. NO DUPLICATE SEGMENTS - each segment must be unique in timing and content
12. Sequential IDs starting from seg_001, seg_002, etc.
13. Cover ALL input content exactly ONCE - no repetition, no gaps
14. dubbed_text should be in target language, sometimes target language could be the same as original language, in that case, use the original language text.
15. TARGET LANGUAGE SHOULD BE IN WRITE IN TARGET LANGUAGE APLHABET, LIKE ENGLISH SHOULD BE IN ENGLISH APLHABET, BENGALI SHOULD BE IN BENGALI APLHABET, ETC.
16. DURATION CHECK: Only split if segment exceeds 12.0 seconds

VALIDATION CHECKLIST - Your output must pass:
✓ Original start/end times preserved from input (unless splitting >12s segments)
✓ Original text preserved exactly from input
✓ Real translations in proper target language alphabet (no placeholder text)
✓ All input segments processed (close to 1:1 input/output ratio)
✓ Sequential unique IDs (seg_001, seg_002...)
✓ No overlapping segments if splits were needed
✓ MANDATORY: ALL segments ≤ 12.0 seconds duration
✓ All input content covered exactly once

YOUR JOB: PRESERVE the input segments' timing and original text exactly as provided. ONLY translate the dubbed_text to proper target language with correct alphabet. Only split segments if they exceed 12 seconds. Provide REAL translations that sound natural and convey the actual meaning. ENSURE NO DUPLICATES.

IF YOU USE PLACEHOLDER TEXT, LAZY TRANSLATIONS, MODIFY ORIGINAL TIMING/TEXT, OR CREATE DUPLICATES, YOU HAVE COMPLETELY FAILED."""
    
    def _format_segments_with_translation(self, ai_segments: List[Dict], global_segment_index_start: int = 0) -> List[Dict[str, Any]]:
        """Format segments with translation included and enforce duration limits"""
        from app.config.settings import settings
        formatted_segments = []
        # Use the configured max segment duration, but cap at 12 seconds for voice quality
        max_duration_ms = min(settings.WHISPER_MAX_SEG_SECONDS * 1000, 12000)  # 12 seconds max in milliseconds
        
        for idx, seg in enumerate(ai_segments):
            start_s = float(seg.get("start", 0))
            end_s = float(seg.get("end", 0))
            
            # Convert to milliseconds
            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)
            duration_ms = end_ms - start_ms
            
            # ENFORCE 12-second limit - this is critical
            if duration_ms > max_duration_ms:
                logger.warning(f"⚠️  DURATION VIOLATION: Segment {seg.get('id', 'unknown')} exceeds 12s limit: {duration_ms/1000:.2f}s")
                logger.warning(f"   Text: {seg.get('original_text', '')[:100]}...")
                # Truncate to 12 seconds - this is a safety measure
                end_ms = start_ms + max_duration_ms
                duration_ms = max_duration_ms
                logger.warning(f"   🔧 Auto-corrected to 12.0s duration")
            
            # Validate minimum duration
            if duration_ms < 1000:  # Less than 1 second
                logger.warning(f"⚠️  SHORT SEGMENT: {seg.get('id', 'unknown')} is only {duration_ms}ms")
            
            # Clean and validate dubbed_text from AI
            dubbed_text = seg.get("dubbed_text", "").strip()
            # Remove [English: ...] or similar formatting
            dubbed_text = re.sub(r'\[\w+:\s*([^\]]+)\]', r'\1', dubbed_text)
            dubbed_text = re.sub(r'\[[^\]]+\]', '', dubbed_text).strip()
            
            # Use global segment index to maintain sequential numbering across chunks
            global_segment_index = global_segment_index_start + len(formatted_segments)
            
            formatted_seg = {
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
            }
            formatted_segments.append(formatted_seg)
        
        # Log summary of duration validation
        over_limit = sum(1 for seg in formatted_segments if seg["duration_ms"] > max_duration_ms)
        if over_limit > 0:
            logger.error(f"🚨 CRITICAL: {over_limit} segments exceeded 12s limit and were corrected")
        
        logger.info(f"Formatted {len(formatted_segments)} valid segments (global index start: {global_segment_index_start})")
        return formatted_segments
    
    
    def _translate_existing_segments(self, segments: List[Dict], target_language_code: str, is_same_language: bool = False) -> List[Dict[str, Any]]:
        """Translate existing segments without changing timing or structure - for redubs"""
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
        
        # Translate segments in chunks while preserving timing
        return self._translate_segments_preserve_timing(segments, target_language_code)
    
    def _translate_segments_preserve_timing(self, segments: List[Dict], target_language_code: str) -> List[Dict[str, Any]]:
        """Translate segments while preserving exact timing and structure"""
        chunk_size = 25
        all_results = []
        
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i + chunk_size]
            chunk_number = i//chunk_size + 1
            total_chunks = (len(segments) + chunk_size - 1)//chunk_size
            
            logger.info(f"Translating chunk {chunk_number}/{total_chunks} ({len(chunk)} segments)")
            
            # Build translation-only prompt
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
                
                # Format results preserving original timing
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
                # Fallback: preserve original text
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
        """Build prompt for translation-only (no segmentation changes)"""
        return f"""You are a professional translator. Translate the text content while preserving EXACT timing and structure.

INPUT SEGMENTS:
{json.dumps(segments, ensure_ascii=False, indent=2)}

STRICT TRANSLATION RULES:
- Translate ONLY the original_text to {target_language_code}
- PRESERVE exact start/end timing from input - DO NOT change timestamps
- PRESERVE exact segment IDs from input  
- DO NOT merge, split, or modify segments
- DO NOT change segment structure or timing
- Provide literal, accurate translation of each text
- Output MUST have same number of segments as input
- Each output segment must match input segment by ID and timing

TARGET LANGUAGE: {target_language_code}
- Write dubbed_text in proper {target_language_code} alphabet/script
- Use natural, fluent {target_language_code} while staying faithful to meaning
- Maintain original tone and style

OUTPUT FORMAT (JSON):
{{
  "segments": [
    {{
      "id": "[same as input]",
      "original_text": "[same as input]", 
      "dubbed_text": "[translation in {target_language_code}]"
    }}
  ]
}}

CRITICAL: Output exactly {len(segments)} segments. Do NOT change timing, merge, or split segments."""
    
    def _get_openai_client(self):
        """Get or create OpenAI client (reusable)"""
        if self._openai_client is None:
            from openai import OpenAI
            from app.config.settings import settings
            self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._openai_client
    
    def _process_in_chunks(self, segments: List[Dict], target_language: str, is_same_language: bool = False) -> List[Dict[str, Any]]:
        """Process large segment lists in chunks to avoid API limits"""
        chunk_size = 25  # Process 25 segments at a time
        all_results = []
        
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i + chunk_size]
            chunk_number = i//chunk_size + 1
            total_chunks = (len(segments) + chunk_size - 1)//chunk_size
            
            logger.info(f"Processing chunk {chunk_number}/{total_chunks} ({len(chunk)} segments)")
            
            target_lang_code = language_service.normalize_language_input(target_language)
            
            # Add chunk context to prevent AI from generating duplicates
            chunk_context = f"""
CHUNK PROCESSING INFO:
- This is chunk {chunk_number} of {total_chunks}
- Start your segment IDs from seg_{len(all_results)+1:03d}
- You are processing segments from time {chunk[0].get('start', 0):.3f}s to {chunk[-1].get('end', 0):.3f}s
- Time range: {chunk[-1].get('end', 0) - chunk[0].get('start', 0):.1f} seconds of content
- PRESERVATION APPROACH: Keep original timing and text intact for all segments
- Expected output: Close to {len(chunk)} segments (1:1 mapping unless splitting >12s segments)
- ONLY modify segments if they exceed 12.0 seconds duration
- DO NOT repeat any content from previous chunks
- Process ONLY the content in this chunk
- Preserve exact timings from input: {chunk[0].get('start', 0):.3f}s to {chunk[-1].get('end', 0):.3f}s
"""
            
            prompt = chunk_context + self._build_segmentation_and_dubbing_prompt(chunk, target_lang_code, is_same_language)
            
            model = "gpt-4o"
            
            try:
                response = self._get_openai_client().chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert in audio dubbing translation. CRITICAL: PRESERVE original timing and text exactly as provided. ONLY translate dubbed_text to target language with proper alphabet. NO segment can exceed 12.0 seconds duration. Only split segments if they exceed 12s. Maintain 1:1 input/output mapping. Process content in chunks and avoid duplicates."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                ai_response = response.choices[0].message.content.strip()
                result = json.loads(ai_response)
                
                # Pass the current global segment index start to maintain sequential numbering
                global_segment_index_start = len(all_results)
                chunk_segments = self._format_segments_with_translation(
                    result.get("segments", []), 
                    global_segment_index_start
                )
                all_results.extend(chunk_segments)
                
                logger.info(f"Chunk processed: {len(chunk_segments)} segments returned")
                
            except Exception as e:
                logger.error(f"Chunk processing failed: {str(e)}")
                # Continue with other chunks
                continue
        
        logger.info(f"Total processed: {len(all_results)} segments from {len(segments)} input segments")
        return all_results


def get_ai_segmentation_service() -> AISegmentationService:
    return AISegmentationService()
