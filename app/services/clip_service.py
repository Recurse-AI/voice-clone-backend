import os
import re
import json
import time
import subprocess
import requests
from typing import Dict, Any, List
from app.config.settings import settings
from app.services.r2_service import R2Service
from app.repositories.clip_repository import ClipRepository
from app.subtitles import build_ass_from_words
from scripts.render_subtitles import render as render_video_with_subtitles

class ClipService:
    def __init__(self):
        self.r2 = R2Service()
        self.repo = ClipRepository()
    
    def _ffmpeg(self):
        return os.environ.get("FFMPEG_PATH", "ffmpeg")
    
    def _time_to_seconds(self, s: str) -> float:
        if re.match(r"^\d+(\.\d+)?$", s):
            return float(s)
        s = s.replace(",", ".")
        parts = s.split(":")
        parts = [float(x) for x in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return float(s)
    
    def download_video(self, url: str, local_path: str) -> Dict[str, Any]:
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {'success': True, 'local_path': local_path}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def trim_video(self, src: str, ss: float, to: float, out_path: str):
        dur = max(0.0, to - ss)
        cmd = [self._ffmpeg(), "-y", "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", src, "-c", "copy", "-avoid_negative_ts", "make_zero", out_path]
        subprocess.run(cmd, check=True)
    
    def extract_audio(self, video_path: str, audio_path: str):
        cmd = [self._ffmpeg(), "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
        subprocess.run(cmd, check=True)
    
    def parse_srt(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read().strip()
        entries = []
        for block in re.split(r"\n\s*\n", data):
            lines = [l for l in block.splitlines() if l.strip()]
            if len(lines) < 2:
                continue
            idx_line = 1 if re.match(r"^\d+$", lines[0].strip()) else 0
            if idx_line + 1 >= len(lines):
                continue
            ts = lines[idx_line]
            m = re.search(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})", ts)
            if not m:
                continue
            start = self._time_to_seconds(m.group(1))
            end = self._time_to_seconds(m.group(2))
            text = " ".join(lines[idx_line + 1:]).strip()
            text = re.sub(r"<[^>]+>", "", text)
            entries.append({"start": start, "end": end, "text": text})
        return entries
    
    def srt_to_words(self, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        words = []
        for s in sentences:
            toks = s["text"].split()
            if not toks:
                continue
            span = max(0.01, s["end"] - s["start"])
            step = span / max(1, len(toks))
            t = s["start"]
            for tok in toks:
                words.append({"text": tok, "start": int(t * 1000), "end": int(min(s["end"], t + step) * 1000)})
                t += step
        return words
    
    def transcribe_assemblyai(self, audio_path: str) -> Dict[str, Any]:
        headers = {"authorization": settings.ASSEMBLYAI_API_KEY}
        with open(audio_path, "rb") as f:
            up = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f)
            up.raise_for_status()
        audio_url = up.json()["upload_url"]
        
        payload = {"audio_url": audio_url, "speaker_labels": False, "language_detection": True, "punctuate": True, "format_text": True}
        r = requests.post("https://api.assemblyai.com/v2/transcript", headers={**headers, "content-type": "application/json"}, data=json.dumps(payload))
        r.raise_for_status()
        tid = r.json()["id"]
        
        while True:
            g = requests.get(f"https://api.assemblyai.com/v2/transcript/{tid}", headers=headers)
            g.raise_for_status()
            st = g.json()
            if st.get("status") in {"completed", "error"}:
                if st.get("status") == "error":
                    raise RuntimeError(st.get("error", "transcription failed"))
                return st
            time.sleep(2)
    
    def segment_openai(self, transcript: str, expected_duration: float) -> Dict[str, Any]:
        prompt = {
            "role": "system",
            "content": (
                "You are a video content editor for short-form content. Identify highly engaging segments where EACH segment should be around the target duration. "
                "The target duration is PER SEGMENT (not total). Find as many quality segments as exist—you do NOT need to cover the full video. "
                "Return STRICT JSON: {segments:[{start:number,end:number,reason:string,ratings:{hook:number,flow:number,value:number,trend:number}}],overall:{score:number,out_of:number,grade:string}}. "
                "All ratings MUST be 0-100. overall.out_of MUST be 100. Segments must be sorted, non-overlapping, and within [0,T]."
            )
        }

        max_chars = 12000
        safe_transcript = transcript if len(transcript) <= max_chars else (transcript[:max_chars] + "\n...[truncated]")

        user_content = f"Target duration PER SEGMENT: ~{expected_duration}s (can be shorter or longer based on content quality). Transcript:\n{safe_transcript}"
        user = {"role": "user", "content": user_content}
        body = {"model": "gpt-4o-mini", "messages": [prompt, user], "response_format": {"type": "json_object"}, "temperature": 0.2}
        r = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}", "Content-Type": "application/json"}, data=json.dumps(body))
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    
    def cut_segment(self, base_clip: str, ss: float, to: float, out_path: str):
        cmd = [self._ffmpeg(), "-y", "-ss", f"{ss:.3f}", "-to", f"{to:.3f}", "-i", base_clip, "-c", "copy", out_path]
        subprocess.run(cmd, check=True)
    
    def extract_sentences(self, text: str) -> List[Dict[str, Any]]:
        sentences = re.split(r'(?<=[.!?।])\s+', text)
        result = []
        for sentence in sentences:
            if sentence.strip():
                words = re.findall(r'\b\w+\b', sentence)
                result.append({"sentence": sentence.strip(), "words": words})
        return result
    
    def resize_video(self, video_path: str, out_path: str, preset: str = "reels"):
        size_map = {"reels": (1080, 1920), "square": (1080, 1080), "landscape": (1920, 1080), "fourfive": (1080, 1350)}
        w, h = size_map.get(preset, (1080, 1920))
        
        fc = (
            f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"boxblur=20:1,crop={w}:{h}[bg];"
            f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vout]"
        )
        
        cmd = [
            self._ffmpeg(), '-y', '-i', video_path,
            '-filter_complex', fc,
            '-map', '[vout]', '-map', '0:a?',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'copy',
            out_path
        ]
        subprocess.run(cmd, check=True)
    
    def render_subtitles(self, video_path: str, words: List[Dict[str, Any]], out_path: str, style: str = "karaoke", preset: str = "reels", font: str = None, font_size: int = None, wpl: int = None):
        size_map = {"reels": (1080, 1920), "square": (1080, 1080), "landscape": (1920, 1080), "fourfive": (1080, 1350)}
        resolution = size_map.get(preset, (1080, 1920))
        auto_fs = int(min(resolution) * 0.15)
        
        kwargs = {
            "style": style or "karaoke",
            "resolution": resolution,
            "font_size": font_size or auto_fs,
            "max_words_per_line": wpl or 3,
        }
        if font:
            kwargs["font_name"] = font
        
        ass_text = build_ass_from_words(words, **kwargs)
        ass_path = video_path.replace(".mp4", ".ass")
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_text)

        fonts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "fonts"))
        render_video_with_subtitles(video_path, ass_path, out_path, fontsdir=fonts_dir, size=resolution)
