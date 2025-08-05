"""
Constants for Export Video Module
All hardcoded values should be defined here for maintainability
"""

# Audio Constants
AUDIO_SAMPLE_RATE = 44100
AUDIO_DEFAULT_VOLUME = 1.0
AUDIO_CHANNELS = 2

# Video Constants  
DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_VIDEO_HEIGHT = 1080
DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_FORMAT = "mp4"

# Export Quality Constants
EXPORT_QUALITY_LOW = "low"
EXPORT_QUALITY_MEDIUM = "medium" 
EXPORT_QUALITY_HIGH = "high"

BITRATE_LOW = "1000k"
BITRATE_MEDIUM = "2000k"
BITRATE_HIGH = "5000k"

# Canvas/Overlay Constants
DEFAULT_OVERLAY_WIDTH = 300
DEFAULT_OVERLAY_HEIGHT = 300
DEFAULT_TEXT_OVERLAY_HEIGHT = 100

# Subtitle Constants
SUBTITLE_FONT_SIZE = 18
SUBTITLE_MARGIN_BOTTOM = 30
SUBTITLE_OUTLINE_WIDTH = 3

# Network Constants
HTTP_TIMEOUT_SHORT = 30      # For images/short downloads
HTTP_TIMEOUT_MEDIUM = 60     # For subtitles/text files  
HTTP_TIMEOUT_LONG = 120      # For video/audio files

# Processing Constants
DEFAULT_CHUNK_SIZE = 8192
MAX_PROCESSING_RETRIES = 3

# Voice Segment Processing Constants
VOICE_SEGMENT_VOLUME = 1.0  # Full volume for voice clarity
INSTRUMENT_VOLUME_RATIO = 0.15  # Soft background volume for instruments

# File Extensions
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".m4a"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]

# Color Constants (for subtitles)
COLOR_WHITE = "&H00ffffff"
COLOR_BLACK = "&H00000000"

# Video Download Constants
DOWNLOAD_TIMEOUT = 300  # 5 minutes timeout for video downloads
DOWNLOAD_TEMP_DIR = "./tmp/downloads"
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB max video size
DEFAULT_VIDEO_QUALITY = "best[height<=720]"  # Default to 720p max

# Supported Video Download Sites
SUPPORTED_DOWNLOAD_SITES = [
    "youtube.com", "youtu.be", "vimeo.com", "facebook.com", 
    "twitter.com", "tiktok.com", "instagram.com", "reddit.com"
]

# Cleanup Constants
CLEANUP_DELAY = 60  # Wait 60 seconds before cleanup
TEMP_FILE_RETENTION = 3600  # Keep temp files for 1 hour max 