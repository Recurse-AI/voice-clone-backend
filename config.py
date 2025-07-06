import os
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
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_AUDIO_FORMATS: list = [".wav", ".mp3", ".flac", ".m4a"]
    TEMP_DIR: str = "/tmp/voice_cloning"
    
    # Model Configuration
    DIA_MODEL_REPO: str = "nari-labs/Dia-1.6B-0626"
    DIA_DEVICE: str = "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu"
    
    # Processing Configuration
    DEFAULT_SEED: Optional[int] = None
    DEFAULT_TEMPERATURE: float = 1.3
    DEFAULT_CFG_SCALE: float = 3.0
    DEFAULT_TOP_P: float = 0.95
    
    # R2 Bucket Configuration
    R2_ACCESS_KEY_ID: str = os.getenv("R2_ACCESS_KEY_ID", "")
    R2_SECRET_ACCESS_KEY: str = os.getenv("R2_SECRET_ACCESS_KEY", "")
    R2_BUCKET_NAME: str = os.getenv("R2_BUCKET_NAME", "")
    R2_ENDPOINT_URL: str = os.getenv("R2_ENDPOINT_URL", "")
    R2_REGION: str = os.getenv("R2_REGION", "auto")
    R2_BASE_PATH: str = "voice-cloning"
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # AssemblyAI Configuration
    ASSEMBLYAI_API_KEY: str = os.getenv("ASSEMBLYAI_API_KEY", "")
    
    # Processing Options
    ENABLE_SUBTITLES: bool = True
    ENABLE_INSTRUMENTS: bool = True
    
    # Response Configuration
    INCLUDE_PROCESSING_DETAILS: bool = True
    INCLUDE_METADATA: bool = True
    
    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}

# Global settings instance
settings = Settings() 