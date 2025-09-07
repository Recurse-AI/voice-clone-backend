import os

class PipelineSettings:
    DUB_CONCURRENCY_LIMIT: int = int(os.getenv("DUB_CONCURRENCY_LIMIT", "10"))  # Overall pipeline concurrency
    MAX_TRANSCRIPTION_JOBS: int = int(os.getenv("MAX_TRANSCRIPTION_JOBS", "2"))  # 2 WhisperX jobs parallel
    MAX_VOICE_CLONING_JOBS: int = int(os.getenv("MAX_VOICE_CLONING_JOBS", "2"))  # 2 jobs parallel - improved GPU utilization
    MAX_SEPARATION_JOBS: int = int(os.getenv("MAX_SEPARATION_JOBS", "2"))  # 2 separation API limit  
    MAX_DUBBING_JOBS: int = int(os.getenv("MAX_DUBBING_JOBS", "5"))  # 5 dubbing jobs parallel
    
    BATCH_SEPARATION_SIZE: int = int(os.getenv("BATCH_SEPARATION_SIZE", "2"))  # Max 2 separation jobs
    BATCH_DUBBING_SIZE: int = int(os.getenv("BATCH_DUBBING_SIZE", "5"))  # Up to 5 dubbing jobs  
    BATCH_UPLOAD_SIZE: int = int(os.getenv("BATCH_UPLOAD_SIZE", "3"))  # Multiple uploads
    BATCH_TIMEOUT: int = int(os.getenv("BATCH_TIMEOUT", "5"))  # Dynamic - low timeout for immediate processing
    
    REDIS_DUB_ACTIVE: str = "dub:active"
    REDIS_DUB_STAGE: str = "dub:stage"
    REDIS_PRIORITY_QUEUE: str = "dub:priority"
    REDIS_RESUME_JOBS: str = "dub:resume"
    REDIS_BATCH_QUEUE: str = "dub:batch"
    
    JOB_TIMEOUT: int = 10800

pipeline_settings = PipelineSettings()
