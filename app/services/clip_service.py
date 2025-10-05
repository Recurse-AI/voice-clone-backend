import os
import re
import json
import time
import subprocess
import requests
from typing import Dict, Any, List
from openai import OpenAI
from app.config.settings import settings
from app.services.r2_service import R2Service
from app.repositories.clip_repository import ClipRepository
from app.subtitles import build_ass_from_words
from scripts.render_subtitles import render as render_video_with_subtitles

class ClipService:
    def __init__(self):
        self.r2 = R2Service()
        self.repo = ClipRepository()
        self._openai_client = None
    
    def _get_openai_client(self) -> OpenAI:
        if not self._openai_client:
            self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._openai_client
    
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
        
        payload = {"audio_url": audio_url, "speech_model": "best", "speaker_labels": False, "language_detection": True, "punctuate": True, "format_text": True}
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
    
    def get_video_duration(self, video_path: str) -> float:
        cmd = [self._ffmpeg().replace("ffmpeg", "ffprobe"), "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
        return 0.0

    def segment_openai(self, transcript: str, expected_duration: float, video_duration: float) -> Dict[str, Any]:
        import time
        
        
        prompt = {
            "role": "system",
            "content": (
                "You are an ELITE viral content curator. Your mission: Extract ONLY the absolute BEST moments.\n\n"
                f"📹 VIDEO: {video_duration:.2f}s ({video_duration/60:.1f} minutes)\n"
                f"⏱️ User preference: ~{expected_duration:.0f}s clips (FLEXIBLE - not strict)\n\n"
                "🎯 YOUR DECISION: You decide how many clips (1-5 max) based PURELY on quality.\n"
                "   • 12-min video with only 1 excellent moment? Return 1 clip.\n"
                "   • 20-min video with 5 excellent moments? Return all 5.\n"
                "   • Poor quality content? Return 1 BEST available moment (minimum).\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "🔥 QUALITY-FIRST RULES (NO COMPROMISE):\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "1. MINIMUM QUALITY THRESHOLD:\n"
                "   ✓ PREFER clips scoring ≥80/100\n"
                "   ✓ ALWAYS return at least 1 clip (pick the best available)\n"
                "   ✓ If multiple clips score ≥80, include them (up to 5 max)\n"
                "   ✓ If nothing scores ≥80, return the single BEST moment you can find\n"
                "   ✓ Better to return 1 amazing clip than 5 mediocre ones\n\n"
                "2. SCORING CRITERIA (Each 0-100):\n"
                "   • hook: Instant attention grab (first 3s impact)\n"
                "   • flow: Narrative completeness & pacing\n"
                "   • value: Educational/Entertainment/Emotional impact\n"
                "   • trend: Viral potential & shareability\n"
                "   OVERALL SCORE = (hook×0.35 + flow×0.25 + value×0.25 + trend×0.15)\n\n"
                "3. PERFECT CLIP ANATOMY:\n"
                "   • Duration: 20-75s (FLEXIBLE range based on content)\n"
                f"   • Timestamps: MUST be within [0, {video_duration:.2f}]s\n"
                f"   • Try to aim around {expected_duration:.0f}s BUT prioritize natural boundaries\n"
                "   • EXAMPLE: User wants 30s clips but you find amazing 75s moment? USE IT!\n"
                "   • EXAMPLE: User wants 60s clips but perfect moment is 25s? USE IT!\n"
                "   • Structure:\n"
                "     - Opens with STRONG hook (question/statement/action)\n"
                "     - Contains COMPLETE thought/story/argument\n"
                "     - Ends with SATISFYING conclusion or cliffhanger\n"
                "   • NO mid-sentence cuts, NO abrupt endings\n\n"
                "4. CONTENT QUALITY MARKERS:\n"
                "   ✓ Emotional peaks (surprise, laughter, insight)\n"
                "   ✓ Unique insights or controversial takes\n"
                "   ✓ Storytelling with clear beginning-middle-end\n"
                "   ✓ Quotable moments or memorable phrases\n"
                "   ✓ Visual or conceptual 'aha!' moments\n"
                "   ✗ Filler, rambling, repetitive content\n"
                "   ✗ Setup without payoff\n"
                "   ✗ Generic or obvious statements\n\n"
                "5. STRATEGIC SELECTION:\n"
                "   • Prioritize DIVERSE content types across clips\n"
                "   • Avoid overlapping topics/themes\n"
                "   • Each clip should stand alone perfectly\n"
                "   • If transcript quality is poor, return FEWER clips\n\n"
                "6. MINIMUM GUARANTEE:\n"
                "   • ALWAYS return at least 1 clip (frontend requirement)\n"
                "   • If nothing scores ≥80, pick the BEST available moment\n"
                "   • Mark low-quality content with honest score (<80)\n"
                "   • Explain quality limitations in 'reason' field\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📊 OUTPUT FORMAT:\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                '{"segments":[{"start":0,"end":0,"reason":"Why this is EXCEPTIONAL","ratings":{"hook":0,"flow":0,"value":0,"trend":0,"overall":0}}],"overall":{"score":0,"out_of":100,"grade":"A+/A/B/C/F","quality_assessment":"Brief analysis"}}'
            )
        }

        max_chars = 12000
        safe_transcript = transcript if len(transcript) <= max_chars else (transcript[:max_chars] + "\n...[truncated]")

        user_content = (
            f"🎬 ANALYZE THIS {video_duration/60:.1f}-MINUTE VIDEO:\n\n"
            f"User prefers: ~{expected_duration:.0f}s clips (but FLEXIBILITY allowed for quality)\n"
            "You can create 1-5 clips - YOUR CHOICE based on quality.\n\n"
            "🚨 CRITICAL RULES:\n"
            "1. MUST return at least 1 clip (frontend requirement)\n"
            "2. PREFER clips scoring ≥80/100, but return best available if needed\n"
            "3. Natural content boundaries > exact duration match\n"
            "4. Complete moments > hitting target length\n\n"
            "Decision examples:\n"
            f"• User wants {expected_duration:.0f}s but perfect moment is 75s? → Use 75s clip!\n"
            f"• User wants {expected_duration:.0f}s but best moment is 25s? → Use 25s clip!\n"
            "• Found 3 clips scoring ≥80? → Return all 3\n"
            "• Found 1 clip scoring ≥80? → Return 1 clip\n"
            "• Nothing scores ≥80? → Return 1 BEST available moment (mark honest score)\n\n"
            "Remember: Quality & completeness > matching preferred duration.\n\n"
            f"TRANSCRIPT:\n{safe_transcript}"
        )
        user = {"role": "user", "content": user_content}
        
        time.sleep(0.5)
        
        response = self._get_openai_client().responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": [{"type": "text", "text": prompt["content"]}]},
                {"role": "user", "content": [{"type": "text", "text": user_content}]}
            ],
            text={"verbosity": "medium"},
            reasoning={"effort": "low"},
            temperature=0.2,
            max_output_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.output_text.strip())
    
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
