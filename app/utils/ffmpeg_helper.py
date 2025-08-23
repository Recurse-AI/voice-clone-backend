"""
Centralized FFmpeg Path Helper
"""
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def get_ffmpeg_path():
    """Get FFmpeg executable path - centralized function"""
    
    # Check if ffmpeg is in PATH
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ FFmpeg found in PATH")
            return 'ffmpeg'
    except FileNotFoundError:
        logger.warning("❌ FFmpeg not found in PATH")
    except subprocess.TimeoutExpired:
        logger.warning("❌ FFmpeg check timed out")
    
    # Common Windows paths
    windows_paths = [
        r"C:\Program Files\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
    ]
    
    for path in windows_paths:
        if os.path.exists(path):
            logger.info(f"✅ FFmpeg found at: {path}")
            return path
    return None


def verify_ffmpeg():
    """Verify FFmpeg is available and working"""
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        return False, "FFmpeg not found"
    
    try:
        result = subprocess.run([ffmpeg_path, '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            return True, f"FFmpeg available: {version_line}"
        else:
            return False, f"FFmpeg execution failed: {result.stderr}"
    except Exception as e:
        return False, f"FFmpeg verification error: {str(e)}"
