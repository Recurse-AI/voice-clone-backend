import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def parse_srt_to_whisperx_format(srt_file_path: str) -> Dict[str, Any]:
    """Parse SRT file and convert to WhisperX format"""
    try:
        content = _read_file_with_encoding(srt_file_path)
        if not content:
            return {"success": False, "error": "Could not read SRT file"}
        
        content = _normalize_content(content)
        segments = _parse_srt_content(content)
        segments = _merge_short_adjacent_segments(segments)
        
        if not segments:
            return {"success": False, "error": "No valid subtitle segments found"}
        
        logger.info(f"Parsed {len(segments)} segments from SRT file")
        
        return {
            "success": True,
            "segments": segments,
            "sentences": segments,
            "language": "auto"
        }
        
    except Exception as e:
        logger.error(f"Failed to parse SRT file {srt_file_path}: {e}")
        return {
            "success": False,
            "error": f"SRT parsing failed: {str(e)}"
        }

def _read_file_with_encoding(file_path: str) -> str:
    """Try reading file with different encodings"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    
    return ""

def _normalize_content(content: str) -> str:
    """Normalize content by removing BOM and standardizing line endings"""
    if content.startswith('\ufeff'):
        content = content[1:]
    
    content = re.sub(r'\r\n|\r', '\n', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()

def _parse_srt_content(content: str) -> List[Dict[str, Any]]:
    """Parse SRT content with pattern matching"""
    segments = []
    
    patterns = [
        r'(\d+)\s*\n\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*\n(.*?)(?=\n\s*\d+\s*\n|\Z)',
        r'(\d+)\s*\n+\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*\n+(.*?)(?=\n+\s*\d+\s*\n|\Z)',
        r'(\d+)\s*\n\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*\n(.*?)(?=\d+\s*\n|\Z)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
        if matches:
            break
    
    if not matches:
        return []
    
    for match in matches:
        try:
            subtitle_id, start_time, end_time, text = match
            
            start_seconds = _time_to_seconds(start_time.strip())
            end_seconds = _time_to_seconds(end_time.strip())
            
            if start_seconds >= end_seconds:
                continue
            
            clean_text = _clean_subtitle_text(text)
            if not clean_text:
                continue
            
            segment = {
                "id": f"srt_{subtitle_id}",
                "text": clean_text,
                "start": start_seconds,
                "end": end_seconds
            }
            
            segments.append(segment)
            
        except Exception:
            continue
    
    return segments

def _merge_short_adjacent_segments(segments: List[Dict[str, Any]], max_chars: int = 220, max_duration_s: float = 8.0, max_gap_ms: int = 250, max_segments: int = 4) -> List[Dict[str, Any]]:
    if not segments:
        return []
    
    def _has_sentence_end(text: str) -> bool:
        return bool(re.search(r'[।.!?؟؛।৷\n]$', text.strip()))
    
    def _has_speaker_change(text1: str, text2: str) -> bool:
        patterns = [r'^-\s+', r'^\w+:', r'^\[\w+\]', r'♪.*♪', r'\(.*\)']
        return any(re.search(p, text2.strip()) for p in patterns)
    
    def _should_break_segment(current_text: str, next_text: str, gap_ms: int) -> bool:
        if gap_ms > 800:
            return True
        if _has_sentence_end(current_text) and gap_ms > 150:
            return True
        if _has_speaker_change(current_text, next_text):
            return True
        return False
    
    merged = []
    current = None
    merged_count = 0
    
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start_s = float(seg.get("start", 0.0))
        end_s = float(seg.get("end", 0.0))
        if start_s >= end_s:
            continue
        start_ms = int(round(start_s * 1000))
        end_ms = int(round(end_s * 1000))
        
        if current is None:
            current = {"id": seg.get("id"), "text": text, "start": start_s, "end": end_s}
            merged_count = 1
            continue
            
        gap_ms = start_ms - int(round(current["end"] * 1000))
        combined_text = (current["text"] + " " + text).strip()
        combined_dur_ms = end_ms - int(round(current["start"] * 1000))
        
        should_break = _should_break_segment(current["text"], text, gap_ms)
        
        can_merge = (not should_break and
                    gap_ms <= max_gap_ms and 
                    len(combined_text) <= max_chars and 
                    combined_dur_ms <= int(max_duration_s * 1000) and
                    merged_count < max_segments)
        
        if can_merge:
            current["text"] = combined_text
            current["end"] = end_s
            merged_count += 1
        else:
            merged.append(current)
            current = {"id": seg.get("id"), "text": text, "start": start_s, "end": end_s}
            merged_count = 1
            
    if current:
        merged.append(current)
    return merged

def _clean_subtitle_text(text: str) -> str:
    """Clean subtitle text removing formatting and normalizing whitespace"""
    if not text:
        return ""
    
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    text = re.sub(r'\\[nN]', ' ', text)
    text = re.sub(r'\\[^\\]*', '', text)
    text = re.sub(r'^\s*[\[\(][^)\]]*[\]\)]\s*:?\s*', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def _time_to_seconds(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    time_str = time_str.replace(',', '.')
    time_parts = time_str.split(':')
    
    if len(time_parts) != 3:
        raise ValueError(f"Invalid time format: {time_str}")
    
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds_ms = float(time_parts[2])
    
    if not (0 <= hours <= 99 and 0 <= minutes <= 59 and 0 <= seconds_ms < 60):
        raise ValueError(f"Time values out of range: {time_str}")
    
    total_seconds = hours * 3600 + minutes * 60 + seconds_ms
    return round(total_seconds, 3)
