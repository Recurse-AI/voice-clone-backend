import os
import subprocess
from app.utils.ffmpeg_helper import get_ffmpeg_path

def render(video_path: str, ass_path: str, output_path: str, fontsdir: str = None, size: tuple[int, int] = None):
    """
    Render video with ASS subtitles using ffmpeg (cross-platform, no path escaping in filter).
    Uses cwd to locate the ASS file and env vars for fonts.
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    abs_ass = os.path.abspath(ass_path)
    ass_dir = os.path.dirname(abs_ass) or "."
    ass_name = os.path.basename(abs_ass)
    
    # Configure fonts via environment variables (works on Windows/Linux/Mac)
    env = os.environ.copy()
    if fontsdir:
        abs_fonts = os.path.abspath(fontsdir)
        env['FONTCONFIG_PATH'] = abs_fonts
        env['FONTCONFIG_FILE'] = os.path.join(abs_fonts, 'fonts.conf')
    
    # Always set fontsdir for libass to find fonts
    env['LIBASS_FONTS_DIR'] = fontsdir if fontsdir else '/usr/share/fonts'
    
    # Build filter chain using only the filename (no drive letters in filter)
    if size:
        w, h = size
    else:
        # Default to vertical reels size if not provided
        w, h = 1080, 1920

    # Build a blur background composition and apply subtitles
    # Escape font directory path for FFmpeg filter
    fontsdir_param = ""
    if fontsdir:
        fonts_escaped = abs_fonts.replace('\\', '\\\\').replace(':', '\\:')
        fontsdir_param = f":fontsdir={fonts_escaped}"
    
    fc = (
        f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"boxblur=20:1,crop={w}:{h}[bg];"
        f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base];"
        f"[base]subtitles={ass_name}{fontsdir_param}[vout]"
    )

    cmd = [
        ffmpeg,
        '-y',
        '-i', video_path,
        '-filter_complex', fc,
        '-map', '[vout]',
        '-map', '0:a?',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'copy',
        output_path,
    ]

    # Run with cwd set to the ASS directory so the filter finds the file without absolute path
    subprocess.run(cmd, check=True, env=env, cwd=ass_dir)
