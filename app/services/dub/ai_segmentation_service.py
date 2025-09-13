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
        edited_map: Dict[str, str] = None
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
            # - If edited_map has data = RESUME (same language, use existing dubbed_text)
            # - If edited_map is empty = FRESH DUBBING or REDUB (regenerate dubbed_text via AI)
            
            if edited_map:
                logger.info(f"RESUME: Using existing translations ({len(edited_map)} edited texts)")
                return self._use_manifest_segments_and_translations(segments, edited_map)
            
            # Process through AI to generate new dubbed_text
            logger.info(f"AI TRANSLATION: Processing {len(segments)} segments for new target language")
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

SMART SEGMENTATION RULES:
1. MERGE short segments (under 2 seconds) with adjacent ones for better voice cloning
2. SPLIT long segments (over 12 seconds) at natural sentence breaks
3. Target 3-8 seconds per segment for optimal voice quality
4. NEVER exceed 15 seconds per segment (voice quality degrades)
5. Keep complete thoughts together - don't break mid-sentence
6. Ensure smooth timing transitions - no gaps between segments
7. COVER ALL input content exactly once - no repetition, no omissions
8. MINIMUM segment duration: 1.0 seconds (1000ms) - NO segments shorter than this
9. Use realistic timestamps - segments cannot be only milliseconds long

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

TIMING FORMAT RULES - STANDARDIZED:
- INPUT times are in SECONDS for easier understanding (e.g., 145.320 = 145.32 seconds from video start)
- OUTPUT times must be in SECONDS (e.g., 4.560 = 4.56 seconds)
- Minimum segment duration: 1.0 seconds (difference between end and start)
- Maximum segment duration: 15.0 seconds  
- Target segment duration: 3-8 seconds
- Use 3 decimal places for precision (e.g., 145.320, not 145.32)
- System Note: Input data is converted from milliseconds to seconds for AI processing
- REALISTIC example: start: 145.320, end: 149.870 (duration: 4.55 seconds)
- INVALID example: start: 145.320, end: 145.328 (duration: 0.008 seconds - TOO SHORT)

CRITICAL RULES - BREAK THESE AND YOU FAIL:
1. NO lazy translations like "Translated content for segment X"
2. MERGE short segments intelligently (don't just copy each line from SRT)
3. Provide REAL meaningful translations in proper {target_language}
4. Every dubbed_text must be ACTUAL TRANSLATION of the original_text
5. Use SMART segmentation - optimize segment lengths for voice cloning
6. NEVER skip any input content
7. TRANSLATE the actual meaning, not just word-for-word
8. Make it sound natural in {target_language}
9. orginal_text and dubbed_text should be plain text, no [], no formatting, no language markers, no explanations, no notes, no nothing.
10. NO DUPLICATE SEGMENTS - each segment must be unique in timing and content
11. Sequential IDs starting from seg_001, seg_002, etc.
12. Cover ALL input content exactly ONCE - no repetition, no gaps
13. dubbed_text should be in target language, sometimes target language could be the same as original language, in that case, use the original language text.
14. TARGET LANGUAGE SHOULD BE IN WRITE IN TARGET LANGUAGE APLHABET, LIKE ENGLISH SHOULD BE IN ENGLISH APLHABET, BENGALI SHOULD BE IN BENGALI APLHABET, ETC.

VALIDATION CHECKLIST - Your output must pass:
✓ All segments have unique start/end times
✓ No overlapping segments 
✓ No duplicate text content
✓ All input content is covered exactly once
✓ Sequential unique IDs (seg_001, seg_002...)
✓ Real translations (no placeholder text)
✓ Optimal segment durations (3-8 seconds target)

YOUR JOB: Take the input segments, intelligently combine/split them for optimal voice cloning, then provide REAL translations that sound natural and convey the actual meaning of what was said in target language. ENSURE NO DUPLICATES.

IF YOU USE PLACEHOLDER TEXT, LAZY TRANSLATIONS, OR CREATE DUPLICATES, YOU HAVE COMPLETELY FAILED."""
    
    def _format_segments_with_translation(self, ai_segments: List[Dict], global_segment_index_start: int = 0) -> List[Dict[str, Any]]:
        """Format segments with translation included"""
        formatted_segments = []
        
        for idx, seg in enumerate(ai_segments):
            start_s = float(seg.get("start", 0))
            end_s = float(seg.get("end", 0))
            
            # Convert to milliseconds
            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)
            duration_ms = end_ms - start_ms
            
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
    
    def _use_manifest_segments_and_translations(self, manifest_segments: List[Dict], edited_map: Dict[str, str]) -> List[Dict[str, Any]]:
        """Use existing segments from manifest for redub/resume scenarios
        
        Manifest segments come pre-normalized with timing in milliseconds format.
        Example: {"start": 503, "end": 5000, "duration_ms": 4497}
        """
        formatted_segments = []
        
        logger.info(f"Processing {len(manifest_segments)} manifest segments with {len(edited_map)} edited texts")
        
        for idx, seg in enumerate(manifest_segments):
            seg_id = seg.get("id", f"seg_{idx+1:03d}")
            start_ms = seg.get("start", 0)
            end_ms = seg.get("end", 0)
            duration_ms = seg.get("duration_ms", end_ms - start_ms)
            
            original_text = seg.get("original_text", seg.get("text", "")).strip()
            dubbed_text = seg.get("dubbed_text", "").strip()
            
            # Use edited text if available, otherwise fallback to original
            if not dubbed_text and seg_id in edited_map:
                dubbed_text = edited_map[seg_id]
            if not dubbed_text:
                dubbed_text = original_text
                
            formatted_segments.append({
                "id": seg_id,
                "segment_index": idx,
                "start": start_ms,
                "end": end_ms,
                "duration_ms": duration_ms,
                "original_text": original_text,
                "dubbed_text": dubbed_text,
                "voice_cloned": seg.get("voice_cloned", False),
                "original_audio_file": seg.get("original_audio_file"),
                "cloned_audio_file": seg.get("cloned_audio_file")
            })
        
        logger.info(f"Formatted {len(formatted_segments)} segments for redub/resume processing")
        return formatted_segments
    
    def _get_openai_client(self):
        """Get or create OpenAI client (reusable)"""
        if self._openai_client is None:
            from openai import OpenAI
            from app.config.settings import settings
            self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._openai_client
    
    def _process_in_chunks(self, segments: List[Dict], target_language: str, is_same_language: bool = False) -> List[Dict[str, Any]]:
        """Process large segment lists in chunks to avoid API limits"""
        chunk_size = 100  # Process 100 segments at a time
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
- Expected output: 5-15 merged segments with realistic durations (1-15 seconds each)
- DO NOT repeat any content from previous chunks
- Process ONLY the content in this chunk
- Ensure your output timings are within the input range: {chunk[0].get('start', 0):.3f}s to {chunk[-1].get('end', 0):.3f}s
"""
            
            prompt = chunk_context + self._build_segmentation_and_dubbing_prompt(chunk, target_lang_code, is_same_language)
            
            model = "gpt-4o"
            
            try:
                response = self._get_openai_client().chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert in audio segmentation and S1 voice cloning dubbing translation. You process content in chunks and must avoid any duplicates."},
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
