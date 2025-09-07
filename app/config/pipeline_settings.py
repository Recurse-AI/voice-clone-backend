import os

class PipelineSettings:
    # === ORCHESTRATION WORKERS (RunPod Managed) ===
    # These control non-VRAM worker coordination
    MAX_DUB_ORCHESTRATION_WORKERS: int = int(os.getenv("MAX_DUB_ORCHESTRATION_WORKERS", "4"))  # Job coordination workers
    MAX_SEPARATION_WORKERS: int = int(os.getenv("MAX_SEPARATION_WORKERS", "2"))      # Separation job workers
    MAX_SEPARATION_JOBS: int = int(os.getenv("MAX_SEPARATION_JOBS", "2"))            # RunPod separation API calls
    MAX_DUBBING_JOBS: int = int(os.getenv("MAX_DUBBING_JOBS", "3"))                  # Final video processing
    
    
    # === VRAM SERVICE WORKERS (16GB VRAM Optimized) ===
    # Reduced for 16GB VRAM - prevent OOM errors
    MAX_WHISPERX_SERVICE_WORKERS: int = int(os.getenv("MAX_WHISPERX_SERVICE_WORKERS", "1"))  # 1 worker for 16GB
    MAX_FISH_SPEECH_SERVICE_WORKERS: int = int(os.getenv("MAX_FISH_SPEECH_SERVICE_WORKERS", "1"))
    
    # === LEGACY SETTINGS (Deprecated) ===
    MAX_TRANSCRIPTION_JOBS: int = 1      # Use MAX_WHISPERX_SERVICE_WORKERS instead
    MAX_VOICE_CLONING_JOBS: int = 1      # Use MAX_FISH_SPEECH_SERVICE_WORKERS instead
    
    # Existing Redis Keys (keep unchanged)
    REDIS_DUB_ACTIVE: str = "dub:active"
    REDIS_DUB_STAGE: str = "dub:stage"
    REDIS_PRIORITY_QUEUE: str = "dub:priority"
    REDIS_RESUME_JOBS: str = "dub:resume"
    
    # New Redis Keys for Service Workers
    REDIS_WHISPERX_QUEUE: str = "service:whisperx:queue"
    REDIS_WHISPERX_ACTIVE: str = "service:whisperx:active"
    REDIS_WHISPERX_RESULTS: str = "service:whisperx:results"
    
    REDIS_FISH_SPEECH_QUEUE: str = "service:fish_speech:queue"
    REDIS_FISH_SPEECH_ACTIVE: str = "service:fish_speech:active"
    REDIS_FISH_SPEECH_RESULTS: str = "service:fish_speech:results"
    
    # Service Worker Settings
    SERVICE_WORKER_TIMEOUT: int = int(os.getenv("SERVICE_WORKER_TIMEOUT", "1800"))  # 30 minutes
    SERVICE_RESULT_TIMEOUT: int = int(os.getenv("SERVICE_RESULT_TIMEOUT", "3600"))  # 1 hour
    
    # Feature Flags
    USE_WHISPERX_SERVICE_WORKER: bool = os.getenv("USE_WHISPERX_SERVICE_WORKER", "true").lower() == "true"
    USE_FISH_SPEECH_SERVICE_WORKER: bool = os.getenv("USE_FISH_SPEECH_SERVICE_WORKER", "true").lower() == "true"
    
    JOB_TIMEOUT: int = 10800

pipeline_settings = PipelineSettings()
