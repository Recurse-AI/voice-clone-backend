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
            
            logger.info(f"Processing {len(combined_text)} segments in chunks")
            logger.info(f"✅ TIMING STANDARD: All inputs converted to milliseconds, AI gets seconds for better understanding")
            return self._process_in_chunks(combined_text, target_language, is_same_language, preserve_segments=False)
        except Exception as e:
            logger.error(f"CRITICAL ERROR in create_optimal_segments_and_dub: {str(e)}")
            logger.error(f"Transcription data structure: {transcription_data}")
            raise e

    

    def _build_segmentation_and_dubbing_prompt(self, segments: List[Dict], target_language: str, is_same_language: bool = False, preserve_segments: bool = False) -> str:
        
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
        
        if preserve_segments:
            # REDUB MODE: Strict 1:1 mapping, preserve exact timing and structure
            return f"""REDUB MODE - EXACT PRESERVATION:

MANDATORY RULES:
1. Output EXACTLY {len(segments)} segments (1:1 mapping)
2. Keep start/end timing exactly as input
3. Keep original_text exactly as input  
4. Only translate dubbed_text to {target_language}
5. No merging, no splitting, no changes to structure

{translation_instructions}

INPUT SEGMENTS:
{json.dumps(segments, ensure_ascii=False, indent=2)}

OUTPUT FORMAT:
{{
  "segments": [
    {{
      "id": "seg_001",
      "start": 0.080,
      "end": 4.560,
      "original_text": "exact text from input",
      "dubbed_text": "proper translation in {target_language}"
    }}
  ]
}}

CRITICAL: Must output exactly {len(segments)} segments with proper translations."""
        else:
            # FRESH DUBBING MODE: Intelligent segmentation allowed
            return f"""FRESH DUBBING - INTELLIGENT SEGMENTATION:

SEGMENTATION RULES:
1. SMART MERGING: Combine short segments (under 2 seconds) for better voice quality
2. MANDATORY SPLITTING: Split segments over 12 seconds at natural sentence breaks  
3. OPTIMAL DURATION: Target 3-8 seconds per segment for best voice cloning
4. COMPLETE COVERAGE: Use all input content exactly once - no gaps, no repeats
5. NATURAL BREAKS: Split at sentence boundaries, not mid-sentence
6. MAX DURATION: NO segment can exceed 12.0 seconds - STRICT LIMIT

{translation_instructions}
- Provide REAL meaningful translations in {target_language}
- NO placeholder text or lazy translations

INPUT SEGMENTS:
{json.dumps(segments, ensure_ascii=False, indent=2)}

OUTPUT FORMAT:
{{
  "segments": [
    {{
      "id": "seg_001",
      "start": 0.080, 
      "end": 4.560,
      "original_text": "combined/optimized text from input",
      "dubbed_text": "proper translation in {target_language}"
    }}
  ]
}}

CRITICAL REQUIREMENTS:
✓ All segments ≤ 12.0 seconds duration (MANDATORY)
✓ Real translations in {target_language} with proper alphabet
✓ Natural sentence boundaries preserved
✓ Optimal voice cloning segment lengths (3-8s ideal)"""
    
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
            
            # ENFORCE 12-second limit - CRITICAL FAILURE if violated
            if duration_ms > max_duration_ms:
                logger.error(f"🚨 CRITICAL DURATION VIOLATION: Segment {seg.get('id', 'unknown')} exceeds 12s limit: {duration_ms/1000:.2f}s")
                logger.error(f"   Text: {seg.get('original_text', '')[:100]}...")
                logger.error(f"   This indicates AI failed to properly split long segments during optimization")
                raise ValueError(f"AI generated segment exceeding 12s limit: {duration_ms/1000:.2f}s - AI must split long segments at sentence breaks")
            
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
    
    def _process_in_chunks(self, segments: List[Dict], target_language: str, is_same_language: bool = False, preserve_segments: bool = False) -> List[Dict[str, Any]]:
        """Process large segment lists in chunks to avoid API limits"""
        chunk_size = 10  # Process 10 segments at a time to avoid JSON parsing issues
        all_results = []
        
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i + chunk_size]
            chunk_number = i//chunk_size + 1
            total_chunks = (len(segments) + chunk_size - 1)//chunk_size
            
            logger.info(f"Processing chunk {chunk_number}/{total_chunks} ({len(chunk)} segments)")
            
            target_lang_code = language_service.normalize_language_input(target_language)
            
            # Add chunk context based on mode
            if preserve_segments:
                chunk_context = f"""
REDUB CHUNK {chunk_number}/{total_chunks}:
- Input: {len(chunk)} segments → Required output: EXACTLY {len(chunk)} segments  
- Start segment IDs from seg_{len(all_results)+1:03d}
- Time range: {chunk[0].get('start', 0):.3f}s to {chunk[-1].get('end', 0):.3f}s
- MANDATORY: Process each input segment as exactly one output segment
- PRESERVE exact timing and original_text from input
- TRANSLATE only the dubbed_text field
"""
            else:
                chunk_context = f"""
FRESH DUBBING CHUNK {chunk_number}/{total_chunks}:
- Input: {len(chunk)} segments for intelligent optimization
- Start segment IDs from seg_{len(all_results)+1:03d}  
- Time range: {chunk[0].get('start', 0):.3f}s to {chunk[-1].get('end', 0):.3f}s
- OPTIMIZE: Merge/split segments for better voice cloning (3-8s ideal)
- MANDATORY: All segments ≤ 12.0 seconds duration
- COVER ALL content exactly once - no gaps, no repetitions
- DO NOT repeat content from previous chunks
"""
            
            prompt = chunk_context + self._build_segmentation_and_dubbing_prompt(chunk, target_lang_code, is_same_language, preserve_segments)
            
            model = "gpt-4o"
            
            try:
                response = self._get_openai_client().chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"You are an expert audio dubbing AI. {'REDUB MODE: Preserve exact timing and structure, 1:1 mapping, only translate dubbed_text.' if preserve_segments else 'FRESH MODE: Intelligent segmentation - merge short segments, split long segments at sentence breaks, target 3-8 seconds for optimal voice cloning.'} CRITICAL: All segments must be ≤12 seconds duration. Provide real translations in target language with proper alphabet. No placeholder text."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.1,  # Lower temperature for more consistent output
                    response_format={"type": "json_object"}
                )
                
                ai_response = response.choices[0].message.content.strip()
                result = json.loads(ai_response)
                
                ai_segments = result.get("segments", [])
                
                if preserve_segments:
                    # REDUB MODE: Enforce strict 1:1 mapping
                    expected_segments = len(chunk)
                    actual_segments = len(ai_segments)
                    if actual_segments != expected_segments:
                        logger.error(f"🚨 REDUB SEGMENT COUNT VIOLATION: Expected {expected_segments} segments, got {actual_segments}")
                        logger.error(f"   Input chunk size: {len(chunk)}, AI output size: {len(ai_segments)}")
                        raise ValueError(f"REDUB mode violated 1:1 mapping requirement: expected {expected_segments}, got {actual_segments}")
                    logger.info(f"REDUB: Chunk processed with exact 1:1 mapping: {len(ai_segments)} segments")
                else:
                    # FRESH DUBBING: Allow intelligent segmentation
                    logger.info(f"FRESH DUBBING: Chunk optimized from {len(chunk)} input → {len(ai_segments)} output segments")
                
                # Pass the current global segment index start to maintain sequential numbering
                global_segment_index_start = len(all_results)
                chunk_segments = self._format_segments_with_translation(
                    ai_segments, 
                    global_segment_index_start
                )
                all_results.extend(chunk_segments)
                
            except Exception as e:
                logger.error(f"Chunk processing FAILED: {str(e)}")
                raise e  # Fail entire process - no fallback behavior
        
        logger.info(f"Total processed: {len(all_results)} segments from {len(segments)} input segments")
        return all_results


def get_ai_segmentation_service() -> AISegmentationService:
    return AISegmentationService()
