import os
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.subtitles.ass_builder import build_ass_from_words
from app.utils.ffmpeg_helper import get_ffmpeg_path

def create_sample_video(output_path: str, duration: int = 3, width: int = 1080, height: int = 1920):
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, '-y',
        '-f', 'lavfi',
        '-i', f'color=c=0x2D3748:s={width}x{height}:d={duration}',
        '-f', 'lavfi',
        '-i', f'anullsrc=r=44100:cl=stereo',
        '-t', str(duration),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def create_gif_from_video(video_path: str, gif_path: str, width: int = 480):
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, '-y',
        '-i', video_path,
        '-vf', f'fps=10,scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        '-loop', '0',
        gif_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def main():
    output_dir = Path(__file__).parent.parent / "assets" / "demos"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_video = output_dir / "base_video.mp4"
    
    print("Creating base video...")
    create_sample_video(str(base_video))
    
    sample_words = [
        {"text": "This", "start": 100, "end": 400},
        {"text": "is", "start": 450, "end": 650},
        {"text": "a", "start": 700, "end": 800},
        {"text": "normal", "start": 850, "end": 1250},
        {"text": "subtitle", "start": 1300, "end": 1700},
        {"text": "style", "start": 1750, "end": 2100},
    ]
    
    print("Creating normal subtitle demo...")
    normal_video = output_dir / "normal_subtitle_video.mp4"
    ass_text = build_ass_from_words(
        sample_words,
        style="normal",
        resolution=(1080, 1920),
        font_name="Montserrat-Bold",
        font_size=110,
        max_words_per_line=3
    )
    
    ass_path = output_dir / "normal_subtitle.ass"
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_text)
    
    fonts_dir = Path(__file__).parent.parent / "assets" / "fonts"
    env = os.environ.copy()
    env['ASS_FONTSDIR'] = str(fonts_dir.absolute())
    env['FONTCONFIG_PATH'] = str(fonts_dir.absolute())
    
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, '-y',
        '-i', str(base_video),
        '-vf', f'subtitles={ass_path.name}',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'copy',
        str(normal_video)
    ]
    subprocess.run(cmd, check=True, env=env, cwd=str(output_dir))
    
    print("Converting to GIF...")
    normal_gif = output_dir / "normal_subtitle_demo.gif"
    create_gif_from_video(str(normal_video), str(normal_gif))
    
    print("Creating no subtitle demo...")
    none_gif = output_dir / "no_subtitle_demo.gif"
    create_gif_from_video(str(base_video), str(none_gif))
    
    print(f"\nDemo GIFs created successfully!")
    print(f"Normal subtitle: {normal_gif}")
    print(f"No subtitle: {none_gif}")
    
    os.remove(base_video)
    os.remove(normal_video)
    os.remove(ass_path)

if __name__ == "__main__":
    main()

