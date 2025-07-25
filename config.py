import os
import torch
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    API_TITLE: str = "Voice Cloning API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Production Voice Cloning API with Dia Model"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 150 * 1024 * 1024  # 150MB
    ALLOWED_AUDIO_FORMATS: list = [".wav", ".mp3", ".flac", ".m4a"]
    TEMP_DIR: str = os.getenv("TEMP_DIR", "./tmp/voice_cloning")
    LOGS_DIR: str = os.getenv("LOGS_DIR", "./logs")
    
    # Dia Model Configuration
    DIA_MODEL_REPO: str = "nari-labs/Dia-1.6B-0626"
    DIA_DEVICE: str = "cuda" if (os.getenv("CUDA_AVAILABLE") and torch.cuda.is_available()) else "cpu"
    DIA_COMPUTE_DTYPE: str = "float16"  # float16 for CUDA, float32 for CPU
    DIA_USE_TORCH_COMPILE: bool = False
    DIA_VERBOSE: bool = False
    
    # Dia Generation Parameters (optimized based on official examples)
    DIA_MAX_TOKENS: int = 3072
    DIA_CFG_SCALE: float = 3.0  # Optimized from official examples (was 3.0)
    DIA_TEMPERATURE: float = 1.3  # Optimized from official examples (was 1.3)
    DIA_TOP_P: float = 0.95  # Optimized from official examples (was 0.95)
    DIA_CFG_FILTER_TOP_K: int = 35  # Optimized from official
    
    # Processing Configuration
    DEFAULT_SEED: Optional[int] = 42  # Fixed seed for consistency
    BASE_SEED: int = 42  # Base seed for speaker-specific seeds
    SPEAKER_SEED_OFFSET: int = 1000  # Offset for speaker-specific seeds
    
    # Performance Optimization Settings
    BATCH_SIZE: int = 4  # Process segments in batches
    MAX_SEGMENT_DURATION: float = 15.0  # Maximum segment duration for optimal processing
    ENABLE_FAST_MODE: bool = True  # Enable fast processing mode

    # Audio Length Adjustment Settings
    USE_SPEED_ADJUSTMENT: bool = True  # Enable time stretching when needed
    AUDIO_SPEED_FACTOR: float = 1.0  # Not used when time stretching is automatic
    ENABLE_AUDIO_PADDING: bool = True  # Enable padding for any remaining gaps
    PARALLEL_PROCESSING: bool = False  # Disable parallel processing to avoid GPU memory issues
    
    # Audio Adjustment Method
    PRESERVE_PITCH: bool = True  # Preserve pitch when time-stretching
    HIGH_QUALITY_STRETCH: bool = True  # Use high-quality time stretching algorithms
    
    # Stretch Ratio Limits (to preserve audio quality)
    MAX_STRETCH_RATIO: float = 1.1  # Maximum 10% slower (reduced from 20%)
    MIN_STRETCH_RATIO: float = 0.9  # Maximum 10% faster (reduced from 20%)
    

    
    # R2 Bucket Configuration
    R2_ACCESS_KEY_ID: str = os.getenv("R2_ACCESS_KEY_ID", "")
    R2_SECRET_ACCESS_KEY: str = os.getenv("R2_SECRET_ACCESS_KEY", "")
    R2_BUCKET_NAME: str = os.getenv("R2_BUCKET_NAME", "")
    R2_ENDPOINT_URL: str = os.getenv("R2_ENDPOINT", os.getenv("R2_ENDPOINT_URL", ""))
    R2_REGION: str = os.getenv("R2_REGION", "auto")
    R2_BASE_PATH: str = "voice-cloning"
    R2_PUBLIC_URL: str = os.getenv("R2_PUBLIC_URL", "")
    

    
    # MongoDB Configuration
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # AssemblyAI Configuration
    ASSEMBLYAI_API_KEY: str = os.getenv("ASSEMBLYAI_API_KEY", "")
    
    # RunPod Configuration for Audio Separation
    API_ACCESS_TOKEN: str = os.getenv("API_ACCESS_TOKEN", "")
    RUNPOD_ENDPOINT_ID: str = os.getenv("RUNPOD_ENDPOINT_ID", "")
    RUNPOD_TIMEOUT: int = int(os.getenv("RUNPOD_TIMEOUT", "1800000"))
    
    # Processing Options
    ENABLE_SUBTITLES: bool = True
    ENABLE_INSTRUMENTS: bool = True
    
    # Response Configuration
    INCLUDE_PROCESSING_DETAILS: bool = True
    INCLUDE_METADATA: bool = True
    
    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}

# Global settings instance
settings = Settings() 