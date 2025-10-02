from typing import List, Dict, Literal, Tuple
import unicodedata
import os
import json

AssStyle = Literal["simple", "karaoke"]

# Load effects from JSON files
def _load_effects():
    effects = {}
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    styles_dir = os.path.join(base_dir, "styles")
    
    if os.path.exists(styles_dir):
        for filename in os.listdir(styles_dir):
            if filename.endswith(".json"):
                style_name = filename[:-5].replace("_", " ").lower()
                filepath = os.path.join(styles_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    effects[style_name] = json.load(f)
    
    return effects

_EFFECTS = _load_effects()

# Styles to hide because they are too similar to stronger options
_STYLE_BLACKLIST = {
    "neon glow",          # prefer neon outline / modern pop
    "shadow pulse",       # prefer drop bounce
    "gradient reveal",    # prefer overlay gradient
    "wave liquid",        # prefer green_orange_swim
    "glass rainbow",      # prefer glass neon / glass frost
    "bold minimal",       # prefer solid bold
    "split reveal",       # prefer slide highlight
    "sunset glow",        # overlaps overlay gradient
}

# Remove blacklisted styles from loaded effects
for _name in list(_EFFECTS.keys()):
    if _name in _STYLE_BLACKLIST:
        _EFFECTS.pop(_name, None)

# Curated aliases mapping user-friendly labels to visual styles
_STYLE_ALIASES = {
    # High-energy / defaults
    "karaoke": "modern pop",

    # Brand/style labels
    "beasty": "neon glow",
    "deep diver": "electric blue",
    "youshaei": "solid bold",
    "pod p": "overlay gradient",
    "mozi": "cinematic gold",
    "popline": "neon glow",
    "glitch infinite": "glitch distortion",
    "seamless bounce": "drop bounce",
    "baby earthquake": "shadow pulse",
    "blur switch": "glass frost",
    "highlighter box": "slide highlight",
    "simple": "solid bold",
    "think media": "cinematic gold",
    "focus": "gradient reveal",
    "blur in": "glass frost",
    "with backdrop": "overlay gradient",
    "soft landing": "wave liquid",
    "baby steps": "typewriter",
    "grow": "gradient reveal",
    "breathe": "wave liquid",
    "minimal": "normal",
}


def _resolve_style(style: str) -> str:
    name = (style or "").strip().lower()
    return _STYLE_ALIASES.get(name, name)

def get_effect_names() -> List[str]:
    return list(_EFFECTS.keys())

def get_effect_config(name: str) -> Dict:
    return _EFFECTS.get(name.lower(), {})

def get_alias_names() -> List[str]:
    return list(_STYLE_ALIASES.keys())

def resolve_style_name(name: str) -> str:
    return _resolve_style(name)


def build_ass_from_words(
    words: List[Dict],
    style: AssStyle = "karaoke",
    resolution: Tuple[int, int] = (1080, 1920),  # 9:16 by default
    font_name: str = "Montserrat",
    font_size: int = 200,
    gap_ms: int = 500,
    max_words_per_line: int = 3,
    max_lines: int = 2,
    letter_spacing: float = 0.0,
    page_colors: List[Tuple[str, str]] | None = None,
) -> str:
    pages = _group_words_into_pages(words, gap_ms, max_words_per_line, max_lines)
    
    # Auto-detect font based on text content
    sample_text = " ".join(w.get("text", "") for w in words[:10])  # Sample first 10 words
    selected_font = _auto_select_font(sample_text, font_name)
    
    header = _ass_header(resolution, selected_font, font_size, letter_spacing)
    # Resolve aliases and normalize before event rendering
    resolved_style = _resolve_style(style)
    events = _ass_events(pages, resolved_style, max_words_per_line, page_colors)
    return header + "\n" + events


def _create_animation_for_word(anim_type: str, word_params: dict, effect_config: dict) -> str:
    """Creates an animation tag block for a single word."""
    off_s, mid, off_e = word_params['off_s'], word_params['mid'], word_params['off_e']
    c1_eff, c2_eff = word_params['c1_eff'], word_params['c2_eff']
    blur_t1, blur_t2 = word_params['blur_t1'], word_params['blur_t2']
    alpha_t, glitch_t = word_params['alpha_t'], word_params['glitch_t']
    scale = effect_config.get("scale", 103)

    if anim_type == "fade":
        return f"\\t({off_s},{mid},\\alpha&H00&)\\t({mid},{off_e},\\1c{c2_eff}){blur_t1}{blur_t2}{glitch_t}"
    
    if anim_type == "slide":
        return f"\\t({off_s},{mid},\\fscx100\\fscy100\\fsp0\\alpha&H00&)\\t({mid},{off_e},\\1c{c2_eff}){blur_t1}{blur_t2}{glitch_t}"

    if anim_type == "bounce":
        over = max(104, scale + 4)
        settle = scale
        return f"\\t({off_s},{mid},\\fscx{over}\\fscy{over}\\alpha&H00&)\\t({mid},{off_e},\\fscx{settle}\\fscy{settle}\\1c{c1_eff}){blur_t1}{blur_t2}{alpha_t}{glitch_t}"

    if anim_type == "rise":
        return f"\\t({off_s},{mid},\\fscy100\\alpha&H00&)\\t({mid},{off_e},\\1c{c2_eff}){blur_t1}{blur_t2}"

    if anim_type == "float":
        # Enhanced floating animation with gentle drift and settle
        duration = off_e - off_s
        drift_start = max(off_s, mid - 80)
        drift_mid1 = drift_start + duration // 6
        drift_mid2 = drift_start + duration // 3
        drift_mid3 = drift_start + duration // 2
        drift_end = min(off_e, mid + 80)
        
        # Create gentle floating motion with scale and position changes
        float_t = (
            f"\\t({drift_start},{drift_mid1},\\fscx102\\fscy103\\frz1)"
            f"\\t({drift_mid1},{drift_mid2},\\fscx98\\fscy97\\frz-1.5)"
            f"\\t({drift_mid2},{drift_mid3},\\fscx101\\fscy102\\frz0.5)"
            f"\\t({drift_mid3},{drift_end},\\fscx100\\fscy100\\frz0)"
        )
        
        # Add gentle blur effect for dreamy floating
        float_blur = f"\\t({off_s},{mid-20},\\blur2)\\t({mid-20},{mid+20},\\blur1)\\t({mid+20},{off_e},\\blur0)"
        
        return f"\\t({off_s},{mid},\\alpha&H00&)\\t({mid},{off_e},\\1c{c2_eff}){blur_t1}{blur_t2}{float_t}{float_blur}"

    if anim_type == "swim":
        g1 = max(off_s, mid - 60)
        g2 = min(off_e, mid + 60)
        swim_t = (
            f"\\t({g1},{g1+30},\\frz3\\fscx101)"
            f"\\t({g1+30},{g2-30},\\frz-3\\fscx99)"
            f"\\t({g2-30},{g2},\\frz0\\fscx100)"
        )
        return f"\\t({off_s},{mid},\\alpha&H00&)\\t({mid},{off_e},\\1c{c2_eff}){blur_t1}{blur_t2}{swim_t}"
    
    if anim_type == "expand":
        small = 70
        return (
            f"\\t({off_s},{mid},\\fscx{small}\\fscy{small}\\alpha&H00&)"
            f"\\t({mid},{off_e},\\fscx{scale}\\fscy{scale}\\1c{c2_eff}){blur_t1}{blur_t2}{alpha_t}{glitch_t}"
        )
    
    if anim_type == "drift":
        # Very gentle floating with slow drift and soft landing
        duration = off_e - off_s
        phase1 = off_s + duration // 4
        phase2 = off_s + duration // 2
        phase3 = off_s + (3 * duration) // 4
        
        # Gentle drift with very subtle movements
        drift_t = (
            f"\\t({off_s},{phase1},\\fscx101\\fscy101\\frz0.8\\blur1)"
            f"\\t({phase1},{phase2},\\fscx99\\fscy100\\frz-0.5\\blur0.5)"
            f"\\t({phase2},{phase3},\\fscx100\\fscy101\\frz0.3\\blur0.8)"
            f"\\t({phase3},{off_e},\\fscx100\\fscy100\\frz0\\blur0)"
        )
        
        return f"\\t({off_s},{mid},\\alpha&H00&)\\t({mid},{off_e},\\1c{c2_eff}){blur_t1}{blur_t2}{drift_t}"

    if anim_type == "zoom_and_pan":
        over = max(104, scale)
        settle = scale
        return (
            f"\\t({off_s},{mid},\\fscx{over}\\fscy{over}\\alpha&H00&)"
            f"\\t({mid},{off_e},\\fscx{settle}\\fscy{settle}\\1c{c2_eff})"
        )

    if anim_type == "none":
        return ""

    # Default to "pop" animation
    return f"\\t({off_s},{mid},\\fscx{scale}\\fscy{scale}\\1c{c1_eff})\\t({mid},{off_e},\\fscx{scale}\\fscy{scale}\\1c{c2_eff}){blur_t1}{blur_t2}{alpha_t}{glitch_t}"


def _ass_header(resolution: Tuple[int, int], font_name: str, font_size: int, letter_spacing: float) -> str:
    x, y = resolution
    
    # Use family names; let fontconfig resolve variants and fallback
    family_mapping = {
        "Montserrat-Bold": "Montserrat",
        "Montserrat-Regular": "Montserrat",
        "Poppins-Bold": "Poppins",
        "Poppins-Regular": "Poppins",
        "Poppins-Medium": "Poppins",
        "Lato-Bold": "Lato",
        "Lato-Regular": "Lato",
        "Roboto-Bold": "Roboto",
        "Roboto-Regular": "Roboto",
        "Roboto-Medium": "Roboto",
        "Roboto-Light": "Roboto",
        "NotoSansDevanagari-Bold": "Noto Sans Devanagari",
        "NotoSansDevanagari-Regular": "Noto Sans Devanagari",
    }
    # If text requires Indic, force Noto Sans Devanagari regardless of user choice
    family = family_mapping.get(font_name, font_name or "Montserrat")
    
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {x}\n"
        f"PlayResY: {y}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Simple,{family},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,{letter_spacing},0,1,6,3,2,60,60,140,1\n"
        f"Style: Karaoke,{family},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,{letter_spacing},0,1,7,3,2,60,60,140,1\n"
        f"Style: KaraokeBox,{family},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H40000000,1,0,0,0,100,100,{letter_spacing},0,3,0,0,2,60,60,160,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _ass_events(pages: List[List[Dict]], style: str, words_per_line: int, page_colors: List[Tuple[str, str]] | None = None) -> str:
    rows: List[str] = []
    for page_idx, page_words in enumerate(pages):
        start_ms = page_words[0]["start"]
        end_ms = page_words[-1]["end"]
        start = _format_ass_time(start_ms)
        end = _format_ass_time(end_ms)
        style_l = style.lower()
        cfg = _EFFECTS.get(style_l, {})
        if style_l == "karaoke":
            text = _animated_text_multiline(page_words, start_ms, words_per_line)
            style_display = cfg.get("style_name") or "Karaoke"
            style_used = "Karaoke"
        elif style_l == "simple":
            text = _simple_text(page_words)
            style_display = cfg.get("style_name") or "Simple"
            style_used = "Simple"
        else:
            c1_override = None
            c2_override = None
            # Allow styles to opt-out from adaptive overrides
            lock_colors = cfg.get("lock_colors", 0) == 1
            if (not lock_colors) and page_colors and page_idx < len(page_colors):
                c1_override, c2_override = page_colors[page_idx]
            text = _effect_text_multiline(style_l, page_words, start_ms, words_per_line, c1_override=c1_override, c2_override=c2_override)
            # Display name but use base header style to avoid missing style warnings
            style_display = cfg.get("style_name") or style_l.title()
            style_used = "Karaoke"
        # Per-effect alignment/box, optional per-style font, and disable auto-wrap
        line_pre = cfg.get("line_pre", "\\an2")
        font_override = cfg.get("font")
        if font_override and "\\fn" not in line_pre:
            # Auto-detect family based on content; if Indic, force Noto Sans Devanagari
            page_text = " ".join(w.get("text", "") for w in page_words)
            detected = _auto_select_font(page_text, font_override)
            family_alias = {
                "Montserrat-Bold": "Montserrat",
                "Montserrat-Regular": "Montserrat",
                "Poppins-Bold": "Poppins",
                "Poppins-Regular": "Poppins",
                "Poppins-Medium": "Poppins",
                "Lato-Bold": "Lato",
                "Lato-Regular": "Lato",
                "Roboto-Bold": "Roboto",
                "Roboto-Regular": "Roboto",
                "Roboto-Medium": "Roboto",
                "Roboto-Light": "Roboto",
                "NotoSansDevanagari": "Noto Sans Devanagari",
                "NotoSansDevanagari-Bold": "Noto Sans Devanagari",
                "NotoSansDevanagari-Regular": "Noto Sans Devanagari",
            }
            family = family_alias.get(detected, detected)
            line_pre = f"{line_pre}\\fn{family}"
        # Optional explicit outline/shadow thickness
        if "\\bord" not in line_pre and cfg.get("bord") is not None:
            try:
                b = int(cfg.get("bord"))
                line_pre = f"{line_pre}\\bord{b}"
            except Exception:
                pass
        if "\\shad" not in line_pre and cfg.get("shad") is not None:
            try:
                s = int(cfg.get("shad"))
                line_pre = f"{line_pre}\\shad{s}"
            except Exception:
                pass
        # Auto-contrast outline if effect didn't specify one
        if "\\3c&" not in line_pre:
            c1 = cfg.get("c1", "&H00FFFFFF&")
            rr = int(c1[3:5], 16)
            gg = int(c1[5:7], 16)
            bb = int(c1[7:9], 16)
            lum = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb
            outline = "&H00000000&" if lum > 140 else "&H00FFFFFF&"
            line_pre = f"{line_pre}\\bord6\\shad3\\3c{outline}\\4c&H64000000&"
        text = "{\\q2" + line_pre + "}" + text
        rows.append(
            f"Dialogue: 0,{start},{end},{style_used},,0,0,60,,{text}"
        )
    return "\n".join(rows)


def _detect_language_type(text: str) -> str:
    """Detect language type for character limit optimization"""
    if not text:
        return "latin"
    
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or
                                       '\u3040' <= c <= '\u309f' or
                                       '\u30a0' <= c <= '\u30ff' or
                                       '\uac00' <= c <= '\ud7af')
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06ff')
    devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097f')
    bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09ff')
    
    total_chars = len(text)
    if total_chars == 0:
        return "latin"
    
    if cjk_chars / total_chars > 0.3:
        return "cjk"
    elif arabic_chars / total_chars > 0.3:
        return "arabic"
    elif (devanagari_chars + bengali_chars) / total_chars > 0.3:
        return "indic"
    return "latin"

def _auto_select_font(text: str, default_font: str) -> str:
    """Use user's font for Latin, otherwise force a universal fallback family."""
    lang_type = _detect_language_type(text)
    if lang_type != "latin":
        return "Noto Sans Devanagari"
    return default_font or "Montserrat"

def _get_char_limit(text: str) -> int:
    """Get optimal character limit based on language"""
    lang_type = _detect_language_type(text)
    limits = {"cjk": 18, "arabic": 30, "indic": 35, "latin": 40}
    return limits.get(lang_type, 40)

def _group_words_into_pages(
    words: List[Dict], gap_ms: int, words_per_line: int, max_lines: int
) -> List[List[Dict]]:
    blocks: List[List[Dict]] = []
    cur: List[Dict] = []
    for i, w in enumerate(words):
        if not cur:
            cur = [w]
            continue
        prev = words[i - 1]
        if w["start"] - prev["end"] > gap_ms:
            blocks.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        blocks.append(cur)

    pages: List[List[Dict]] = []
    for blk in blocks:
        page: List[Dict] = []
        char_count = 0
        
        for word in blk:
            word_text = word.get("text", "")
            word_len = len(word_text)
            test_len = char_count + word_len + (1 if char_count > 0 else 0)
            
            if not page:
                char_limit = _get_char_limit(word_text)
                page.append(word)
                char_count = word_len
            elif test_len <= char_limit:
                page.append(word)
                char_count = test_len
            else:
                pages.append(page)
                page = [word]
                char_count = word_len
                char_limit = _get_char_limit(word_text)
        
        if page:
            pages.append(page)
    
    return pages


def _animated_text_multiline(
    words: List[Dict], base_start_ms: int, words_per_line: int
) -> str:
    parts: List[str] = []
    for idx, w in enumerate(words, start=1):
        dur_cs = max(1, int((w["end"] - w["start"]) / 10))
        off_s = max(0, w["start"] - base_start_ms)
        off_e = max(off_s + 1, w["end"] - base_start_ms)
        mid = off_s + max(1, (off_e - off_s) // 2)
        word = _sanitize_text(w["text"])
        parts.append(
            f"{{\\k{dur_cs}\\b1\\fscx100\\fscy100\\1c&H00FFFFFF&"
            f"\\t({off_s},{mid},\\fscx103\\fscy103\\1c&H00FFA500&)"
            f"\\t({mid},{off_e},\\fscx103\\fscy103\\1c&H00FFD700&)"
            f"\\t({off_e},{off_e+1},\\fscx100\\fscy100\\1c&H00FFFFFF&)}}{word}"
        )
        if idx % words_per_line == 0 and idx != len(words):
            parts.append("\\N")
        else:
            parts.append(" ")
    return ("".join(parts)).strip()


def _effect_text_multiline(effect: str, words: List[Dict], base_start_ms: int, words_per_line: int, c1_override: str | None = None, c2_override: str | None = None) -> str:
    cfg = _EFFECTS.get(effect, {})
    scale = cfg.get("scale", 103)
    c1 = c1_override or cfg.get("c1", "&H00FFFFFF&")
    c2 = c2_override or cfg.get("c2", "&H00FFFFFF&")
    blur_peak = cfg.get("blur", 0)
    alpha_in = cfg.get("alpha", 0) == 1
    glitch = cfg.get("glitch", 0) == 1
    anim_default = cfg.get("anim", "pop")  # pop | fade | slide | bounce | rise | float | swim | zoom_and_pan
    uppercase = cfg.get("uppercase", 0) == 1
    bold_default = 1 if cfg.get("bold") is None else (1 if int(cfg.get("bold")) != 0 else 0)
    italic_default = 0 if cfg.get("italic") is None else (1 if int(cfg.get("italic")) != 0 else 0)
    parts: List[str] = []
    for idx, w in enumerate(words, start=1):
        dur_cs = max(1, int((w["end"] - w["start"]) / 10))
        off_s = max(0, w["start"] - base_start_ms)
        off_e = max(off_s + 1, w["end"] - base_start_ms)
        mid = off_s + max(1, (off_e - off_s) // 2)
        raw_text = w["text"].upper() if uppercase else w["text"]
        word = _sanitize_text(raw_text)
        # Optional alternating per-line colors and bold
        line_group = ((idx - 1) // max(1, words_per_line)) % 2
        c1_line = cfg.get("line1_c") if line_group == 0 else cfg.get("line2_c")
        c2_line = cfg.get("line1_c") if line_group == 0 else cfg.get("line2_c")
        c1_eff = c1_line or c1
        c2_eff = c2_line or c2
        if line_group == 0:
            bold_flag = cfg.get("line1_bold")
        else:
            bold_flag = cfg.get("line2_bold")
        if bold_flag is None:
            bold_flag = bold_default
        ital_flag = cfg.get("line1_italic") if line_group == 0 else cfg.get("line2_italic")
        if ital_flag is None:
            ital_flag = italic_default
        btag = "\\b1" if int(bold_flag) != 0 else "\\b0"
        itag = "\\i1" if int(ital_flag) != 0 else "\\i0"
        blur_t1 = f"\\t({off_s},{mid},\\blur{blur_peak})" if blur_peak else ""
        blur_t2 = f"\\t({mid},{off_e},\\blur0)" if blur_peak else ""
        alpha_pre = "\\alpha&H80&" if alpha_in else ""
        alpha_t = f"\\t({off_s},{mid},\\alpha&H00&)" if alpha_in else ""
        glitch_t = ""
        if glitch:
            g1 = max(off_s, mid - 40)
            g2 = min(off_e, mid + 40)
            glitch_t = (
                f"\\t({g1},{g1+15},\\fscx95)"
                f"\\t({g1+15},{g1+30},\\fscx105)"
                f"\\t({g2-30},{g2-15},\\fscx95)"
                f"\\t({g2-15},{g2},\\fscx103)"
            )

        # Choose animation per line if configured
        anim = anim_default
        if line_group == 0 and cfg.get("line1_anim"):
            anim = cfg.get("line1_anim")
        elif line_group == 1 and cfg.get("line2_anim"):
            anim = cfg.get("line2_anim")

        initial_transform = ""
        if anim == "slide":
            initial_transform = "\\fscx90\\fscy90\\fsp2"
        elif anim == "bounce":
            initial_transform = "\\fscx95\\fscy95"
        elif anim == "rise":
            initial_transform = "\\fscx100\\fscy85"
        elif anim == "none":
            initial_transform = "\\fscx100\\fscy100"

        word_params = {
            'off_s': off_s, 'mid': mid, 'off_e': off_e,
            'c1_eff': c1_eff, 'c2_eff': c2_eff,
            'blur_t1': blur_t1, 'blur_t2': blur_t2,
            'alpha_t': alpha_t, 'glitch_t': glitch_t,
        }
        
        animation_tags = _create_animation_for_word(anim, word_params, cfg)

        base_color = "&H00FFFFFF&" if anim in ["pop", "bounce"] else c1_eff
        if anim == "none":
            base_color = c1_eff
        
        parts.append(
            f"{{\\k{dur_cs}{btag}{itag}{initial_transform}{alpha_pre}\\1c{base_color}"
            f"{animation_tags}}}{word}"
        )
        if idx % words_per_line == 0 and idx != len(words):
            parts.append("\\N")
        else:
            parts.append(" ")
    return ("".join(parts)).strip()


def build_srt_from_words(
    words: List[Dict],
    gap_ms: int = 500,
    max_words_per_line: int = 3,
    max_lines: int = 2,
) -> str:
    pages = _group_words_into_pages(words, gap_ms, max_words_per_line, max_lines)
    rows: List[str] = []
    for i, page_words in enumerate(pages, start=1):
        start_ms = page_words[0]["start"]
        end_ms = page_words[-1]["end"]
        text = _simple_text(page_words)
        rows.append(str(i))
        rows.append(_format_srt_time(start_ms) + " --> " + _format_srt_time(end_ms))
        rows.append(text)
        rows.append("")
    return "\n".join(rows)


def _format_srt_time(ms: int) -> str:
    total_ms = int(round(ms))
    s, ms_part = divmod(total_ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms_part:03d}"


def _simple_text(words: List[Dict]) -> str:
    return _sanitize_text(" ".join(w["text"] for w in words))


def _format_ass_time(ms: int) -> str:
    total_cs = int(round(ms / 10))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _sanitize_text(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.translate({
        0x200B: None,  # ZERO WIDTH SPACE
        0x200C: None,  # ZERO WIDTH NON-JOINER
        0x200D: None,  # ZERO WIDTH JOINER
        0x2060: None,  # WORD JOINER
        0xFEFF: None,  # ZERO WIDTH NO-BREAK SPACE
    })
    t = (
        t.replace("\u2018", "'")
         .replace("\u2019", "'")
         .replace("\u201C", '"')
         .replace("\u201D", '"')
    )
    t = (
        t.replace("\\", "\\\\")
         .replace("{", "(")
         .replace("}", ")")
         .replace("\n", " ")
         .strip()
    )
    return t


