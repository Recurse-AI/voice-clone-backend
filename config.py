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
    API_DESCRIPTION: str = "Production Voice Cloning API with OpenVoice Model (MIT Licensed)"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 150 * 1024 * 1024  # 150MB
    ALLOWED_AUDIO_FORMATS: list = [".wav", ".mp3", ".flac", ".m4a"]
    TEMP_DIR: str = os.getenv("TEMP_DIR", "./tmp/voice_cloning")
    LOGS_DIR: str = os.getenv("LOGS_DIR", "./logs")
    
    # Local Storage Settings
    LOCAL_STORAGE_DIR: str = os.getenv('LOCAL_STORAGE_DIR', './tmp/local_storage')
    LOCAL_STORAGE_RETENTION_HOURS: int = int(os.getenv('LOCAL_STORAGE_RETENTION_HOURS', '1'))
    
    # OpenVoice Model Configuration (MIT License - Dedicated Voice Cloning)
    OPENVOICE_MODEL_PATH: str = os.getenv("OPENVOICE_MODEL_PATH", "./models/openvoice")
    OPENVOICE_DEVICE: str = "cuda" if (os.getenv("CUDA_AVAILABLE") and torch.cuda.is_available()) else "cpu"
    OPENVOICE_COMPILE: bool = True  # Enable torch.compile for better performance
    OPENVOICE_VERBOSE: bool = False

    # OpenVoice Generation Parameters (Optimized for voice cloning)
    OPENVOICE_MAX_LENGTH: int = 8192  # Maximum audio length
    OPENVOICE_TEMPERATURE: float = 0.7  # Balance creativity and consistency  
    OPENVOICE_TOP_P: float = 0.8  # Nucleus sampling
    OPENVOICE_REPETITION_PENALTY: float = 1.2  # Avoid repetition
    OPENVOICE_USE_AUTOCAST: bool = True  # Mixed precision for CUDA

    # Voice Cloning Settings (OpenVoice specific)
    OPENVOICE_SAMPLE_RATE: int = 24000  # OpenVoice native sample rate
    OPENVOICE_MIN_REFERENCE_SECONDS: float = 3.0  # Minimum reference audio length
    OPENVOICE_MAX_REFERENCE_SECONDS: float = 30.0  # Maximum reference audio length  
    OPENVOICE_CHUNK_LENGTH: float = 30.0  # Maximum chunk length for processing
    OPENVOICE_MAX_TEXT_LENGTH: int = 200  # Maximum text length for voice cloning (characters)

    # Voice Consistency Settings
    OPENVOICE_USE_GLOBAL_SEED: bool = True  # Use single seed for consistency
    OPENVOICE_DEFAULT_SEED: int = 42  # Default seed value
    OPENVOICE_MEMORY_CLEANUP_FREQUENCY: int = 1  # Clean memory after processing
    OPENVOICE_MAX_RETRIES: int = 3  # Retry count for failed generations

    # Style Control Settings (OpenVoice advanced features)
    OPENVOICE_ENABLE_EMOTION_CONTROL: bool = True  # Enable emotion control
    OPENVOICE_DEFAULT_EMOTION: str = "neutral"  # Default emotion
    OPENVOICE_ENABLE_ACCENT_CONTROL: bool = True  # Enable accent control
    OPENVOICE_ENABLE_RHYTHM_CONTROL: bool = True  # Enable rhythm/pace control

    # Performance Settings (GPU optimized)
    OPENVOICE_BATCH_SIZE: int = 1  # Batch size for generation
    OPENVOICE_NUM_WORKERS: int = 4  # Number of workers for data loading
    
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
    
    # Processing Configuration
    DEFAULT_SEED: Optional[int] = 42  # Fixed seed for consistency
    BASE_SEED: int = 42  # Base seed for speaker-specific seeds
    SPEAKER_SEED_OFFSET: int = 1000  # Offset for speaker-specific seeds
    USE_SPEED_ADJUSTMENT: bool = True  # Enable time stretching when needed
    
    # Processing Options
    ENABLE_SUBTITLES: bool = True
    ENABLE_INSTRUMENTS: bool = True
    
    # Response Configuration
    INCLUDE_PROCESSING_DETAILS: bool = True
    INCLUDE_METADATA: bool = True
    
    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}

# Global settings instance
settings = Settings() 