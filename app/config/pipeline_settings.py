import os

class PipelineSettings:
    DUB_CONCURRENCY_LIMIT: int = int(os.getenv("DUB_CONCURRENCY_LIMIT", "6"))
    MAX_TRANSCRIPTION_JOBS: int = int(os.getenv("MAX_TRANSCRIPTION_JOBS", "3"))  # 3 WhisperX jobs parallel
    MAX_VOICE_CLONING_JOBS: int = int(os.getenv("MAX_VOICE_CLONING_JOBS", "1"))
    
    BATCH_SEPARATION_SIZE: int = int(os.getenv("BATCH_SEPARATION_SIZE", "3"))
    BATCH_DUBBING_SIZE: int = int(os.getenv("BATCH_DUBBING_SIZE", "4"))
    BATCH_UPLOAD_SIZE: int = int(os.getenv("BATCH_UPLOAD_SIZE", "2"))
    BATCH_TIMEOUT: int = int(os.getenv("BATCH_TIMEOUT", "30"))
    VOICE_CLONE_BATCH_SIZE: int = int(os.getenv("VOICE_CLONE_BATCH_SIZE", "3"))  # 3 segments parallel processing
    VOICE_CLONE_PARALLEL_WORKERS: int = int(os.getenv("VOICE_CLONE_PARALLEL_WORKERS", "3"))  # 3 parallel workers
    
    REDIS_DUB_ACTIVE: str = "dub:active"
    REDIS_DUB_STAGE: str = "dub:stage"
    REDIS_PRIORITY_QUEUE: str = "dub:priority"
    REDIS_RESUME_JOBS: str = "dub:resume"
    REDIS_BATCH_QUEUE: str = "dub:batch"
    
    JOB_TIMEOUT: int = 10800

pipeline_settings = PipelineSettings()
