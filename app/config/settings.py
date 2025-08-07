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
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    NODE_ENV: str = "development"
    BACKEND_URL: str = "http://localhost:8000"
    FRONTEND_URL: str = "https://clearvocals.io"
    PUBLIC_HOST: str = "clearvocals.io"
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 1024 * 1024 * 1024  # 1GB
    ALLOWED_AUDIO_FORMATS: list = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]
    TEMP_DIR: str = os.getenv("TEMP_DIR", "./tmp/voice_cloning")
    
    # Local Storage Settings
    LOCAL_STORAGE_DIR: str = os.getenv('LOCAL_STORAGE_DIR', './tmp/local_storage')
    LOCAL_STORAGE_RETENTION_HOURS: int = int(os.getenv('LOCAL_STORAGE_RETENTION_HOURS', '1'))
    
    # R2 Bucket Configuration
    R2_ACCESS_KEY_ID: str = os.getenv("R2_ACCESS_KEY_ID", "")
    R2_SECRET_ACCESS_KEY: str = os.getenv("R2_SECRET_ACCESS_KEY", "")
    R2_BUCKET_NAME: str = os.getenv("R2_BUCKET_NAME", "")
    R2_ENDPOINT_URL: str = os.getenv("R2_ENDPOINT_URL", "")
    R2_REGION: str = os.getenv("R2_REGION", "auto")
    R2_BASE_PATH: str = "voice-cloning"
    R2_PUBLIC_URL: str = os.getenv("R2_PUBLIC_URL", "")
    
    # MongoDB Configuration
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    DB_NAME: str = "clearvocals"
    
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
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # AssemblyAI Configuration
    ASSEMBLYAI_API_KEY: str = os.getenv("ASSEMBLYAI_API_KEY", "")
    
    # RunPod Configuration for Audio Separation
    API_ACCESS_TOKEN: str = os.getenv("API_ACCESS_TOKEN", "")
    RUNPOD_ENDPOINT_ID: str = os.getenv("RUNPOD_ENDPOINT_ID", "")
    RUNPOD_TIMEOUT: int = int(os.getenv("RUNPOD_TIMEOUT", "1800000"))
    
    # Processing Configuration
    # Voice cloning configuration
    MAX_REFERENCE_SECONDS: int = int(os.getenv('MAX_REFERENCE_SECONDS', '20'))  # Maximum seconds of reference audio to feed into TTS
    # FFmpeg Configuration
    FFMPEG_USE_GPU: bool = bool(int(os.getenv('FFMPEG_USE_GPU', '1')))  # 1 to enable GPU (NVENC)
    
    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}

# Global settings instance
settings = Settings()
