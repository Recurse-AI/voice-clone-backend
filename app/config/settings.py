import os
import torch
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    API_TITLE: str = "ClearVocals API"
    API_VERSION: str = "1.0.0" 
    API_DESCRIPTION: str = "Voice Cloning and Audio Processing API"
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "2"))  # Multiple workers with optimized R2 handling
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    NODE_ENV: str = os.getenv("NODE_ENV", "development")
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "https://clearvocals.io")
    PUBLIC_HOST: str = os.getenv("PUBLIC_HOST", "clearvocals.io")
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 1024 * 1024 * 1024  # 1GB
    ALLOWED_AUDIO_FORMATS: list = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]
    TEMP_DIR: str = os.getenv("TEMP_DIR", "./tmp")
    
    # Local Storage Settings
    LOCAL_STORAGE_DIR: str = os.getenv('LOCAL_STORAGE_DIR', './tmp/local_storage')
    LOCAL_STORAGE_RETENTION_HOURS: int = int(os.getenv('LOCAL_STORAGE_RETENTION_HOURS', '1'))
    
    # R2 Bucket Configuration
    R2_ACCESS_KEY_ID: str = os.getenv("R2_ACCESS_KEY_ID", "")
    R2_SECRET_ACCESS_KEY: str = os.getenv("R2_SECRET_ACCESS_KEY", "")
    R2_BUCKET_NAME: str = os.getenv("R2_BUCKET_NAME", "music-separator")
    R2_ENDPOINT_URL: str = os.getenv("R2_ENDPOINT_URL", "https://8e8c5686fdafcf9560f1babda0e7e460.r2.cloudflarestorage.com")
    R2_REGION: str = os.getenv("R2_REGION", "auto")
    R2_BASE_PATH: str = os.getenv("R2_BASE_PATH", os.getenv("R2_BUCKET_NAME", "music-separator"))
    R2_PUBLIC_URL: str = os.getenv("R2_PUBLIC_URL", "")
    
    # MongoDB Configuration
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    DB_NAME: str = os.getenv("DB_NAME", "music-separator")
    
    # Security & Authentication
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "qK8pR7xT5vZ2wY9sA1bC3dE4fG6hJ0mL2nP3rS5tU7vX9yZ")
    ALGORITHM: str = "RS256"
    SINGIN_TOKEN_EXPIRES: int = 7
    JWT_ALGORITHM: str = "HS256"
    
    # Email Configuration
    EMAIL_HOST_USER: str = os.getenv("EMAIL_HOST_USER", "")
    EMAIL_HOST_PASSWORD: str = os.getenv("EMAIL_HOST_PASSWORD", "")
    EMAIL_FROM: str = os.getenv("EMAIL_FROM", "")
    RESET_PASSWORD_EXPIRES: int = 3600000  # 1 hour in ms
    EMAIL_VERIFICATION_EXPIRES: int = 86400000  # 24 hours in ms
    
    # Google OAuth
    GOOGLE_CLIENT_ID: Optional[str] = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET: Optional[str] = os.getenv("GOOGLE_CLIENT_SECRET")
    
    # Stripe Configuration
    STRIPE_SECRET_KEY: Optional[str] = os.getenv("STRIPE_SECRET_KEY")
    STRIPE_WEBHOOK_SECRET: Optional[str] = os.getenv("STRIPE_WEBHOOK_SECRET")
    STRIPE_PREMIUM_MONTHLY_PRICE_ID: Optional[str] = os.getenv("STRIPE_PREMIUM_MONTHLY_PRICE_ID")
    STRIPE_PREMIUM_YEARLY_PRICE_ID: Optional[str] = os.getenv("STRIPE_PREMIUM_YEARLY_PRICE_ID")
    STRIPE_PRO_MONTHLY_PRICE_ID: Optional[str] = os.getenv("STRIPE_PRO_MONTHLY_PRICE_ID")
    STRIPE_PRO_YEARLY_PRICE_ID: Optional[str] = os.getenv("STRIPE_PRO_YEARLY_PRICE_ID")
    PAY_AS_YOU_GO_PRICE_ID: Optional[str] = os.getenv("PAY_AS_YOU_GO_PRICE_ID")
    METER_EVENT: Optional[str] = os.getenv("METER_EVENT")
    TIME_CYCLE: int = int(os.getenv("TIME_CYCLE", "30"))
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # HuggingFace Configuration
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
    # WhisperX Configuration
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "large-v2")  # Options: tiny, base, small, medium, large-v2, large-v3
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "auto")  # auto, float16, int8
    WHISPER_ALIGNMENT_DEVICE: str = os.getenv("WHISPER_ALIGNMENT_DEVICE", "cpu")
    WHISPER_POOL_SIZE: int = int(os.getenv("WHISPER_POOL_SIZE", "3"))
    WHISPER_MODEL_TIMEOUT: int = int(os.getenv("WHISPER_MODEL_TIMEOUT", "300"))
    
    PYTORCH_CUDA_ALLOC_CONF: str = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
    
    # PyTorch configuration (set in runpod_setup.sh)
    
    # Fish Speech Configuration
    FISH_SPEECH_LOW_MEMORY: bool = os.getenv("FISH_SPEECH_LOW_MEMORY", "false").lower() == "true"
    FISH_SPEECH_COMPILE: bool = os.getenv("FISH_SPEECH_COMPILE", "true").lower() == "true"
    FISH_SPEECH_CHECKPOINT: str = os.getenv("FISH_SPEECH_CHECKPOINT", "checkpoints/openaudio-s1-mini")
    FISH_SPEECH_DECODER: str = os.getenv("FISH_SPEECH_DECODER", "checkpoints/openaudio-s1-mini/codec.pth")
    FISH_SPEECH_DEVICE: str = os.getenv("FISH_SPEECH_DEVICE", "auto")  # auto, cuda, cpu
    FISH_SPEECH_PRECISION: str = os.getenv("FISH_SPEECH_PRECISION", "auto")  # auto, float16, float32
    FISH_SPEECH_MAX_BATCH_SIZE: int = int(os.getenv("FISH_SPEECH_MAX_BATCH_SIZE", "8"))  # Increased for better GPU utilization
    FISH_SPEECH_CHUNK_SIZE: int = int(os.getenv("FISH_SPEECH_CHUNK_SIZE", "200"))  # Larger chunks for efficiency
    
    LOAD_AI_MODELS: bool = os.getenv("LOAD_AI_MODELS", "false").lower() == "true"
    
    # RunPod Configuration for Audio Separation
    API_ACCESS_TOKEN: str = os.getenv("API_ACCESS_TOKEN", "")
    RUNPOD_ENDPOINT_ID: str = os.getenv("RUNPOD_ENDPOINT_ID", "")
    RUNPOD_TIMEOUT: int = int(os.getenv("RUNPOD_TIMEOUT", "1800000"))
    
    # Processing Configuration
    # Voice cloning configuration
    MAX_REFERENCE_SECONDS: int = int(os.getenv('MAX_REFERENCE_SECONDS', '20'))  # Maximum seconds of reference audio to feed into TTS
    # FFmpeg Configuration
    FFMPEG_USE_GPU: bool = bool(int(os.getenv('FFMPEG_USE_GPU', '0')))  # 1 to enable GPU (NVENC)
    
    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}

# Global settings instance
settings = Settings()
