import os
import subprocess
from app.utils.ffmpeg_helper import get_ffmpeg_path

def render(video_path: str, ass_path: str, output_path: str, fontsdir: str = None, size: tuple[int, int] = None):
    """
    Render video with ASS subtitles using ffmpeg with direct font loading.
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    abs_ass = os.path.abspath(ass_path)
    ass_dir = os.path.dirname(abs_ass) or "."
    ass_name = os.path.basename(abs_ass)
    
    # Build filter
    if size:
        w, h = size
    else:
        w, h = 1080, 1920
    
    # Use subtitles filter with direct fontsdir parameter
    if fontsdir:
        abs_fonts = os.path.abspath(fontsdir)
        fonts_escaped = abs_fonts.replace('\\', '/').replace(':', '\\:')
        fc = (
            f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"boxblur=20:1,crop={w}:{h}[bg];"
            f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base];"
            f"[base]subtitles={ass_name}:fontsdir={fonts_escaped}[vout]"
        )
    else:
        fc = (
            f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"boxblur=20:1,crop={w}:{h}[bg];"
            f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base];"
            f"[base]subtitles={ass_name}[vout]"
        )

    cmd = [
        ffmpeg,
        '-y',
        '-threads', '0',
        '-i', video_path,
        '-filter_complex', fc,
        '-map', '[vout]',
        '-map', '0:a?',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-tune', 'fastdecode',
        '-x264-params', 'ref=1:bframes=0:subme=1:me_range=4',
        '-c:a', 'copy',
        output_path,
    ]

    subprocess.run(cmd, check=True, cwd=ass_dir)
