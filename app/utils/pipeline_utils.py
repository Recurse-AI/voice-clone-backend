from typing import Optional
from redis import Redis
from app.config.pipeline_settings import pipeline_settings

def get_redis_client() -> Optional[Redis]:
    try:
        from app.queue.queue_manager import queue_manager
        return queue_manager._get_redis_client()
    except Exception:
        return None

def mark_dub_job_active(job_id: str, stage: str = "initialization") -> bool:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False
            
        redis_client.sadd(pipeline_settings.REDIS_DUB_ACTIVE, job_id)
        redis_client.setex(f"{pipeline_settings.REDIS_DUB_STAGE}:{job_id}", pipeline_settings.JOB_TIMEOUT, stage)
        redis_client.sadd(f"{pipeline_settings.REDIS_DUB_STAGE}:{stage}", job_id)
        
        return True
    except Exception:
        return False

def mark_dub_job_inactive(job_id: str) -> bool:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False
            
        current_stage = redis_client.get(f"{pipeline_settings.REDIS_DUB_STAGE}:{job_id}")
        if current_stage:
            current_stage = current_stage.decode('utf-8')
            redis_client.srem(f"{pipeline_settings.REDIS_DUB_STAGE}:{current_stage}", job_id)
            
        redis_client.srem(pipeline_settings.REDIS_DUB_ACTIVE, job_id)
        redis_client.delete(f"{pipeline_settings.REDIS_DUB_STAGE}:{job_id}")
        
        return True
    except Exception:
        return False

def update_dub_job_stage(job_id: str, new_stage: str) -> bool:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False
            
        old_stage = redis_client.get(f"{pipeline_settings.REDIS_DUB_STAGE}:{job_id}")
        if old_stage:
            old_stage = old_stage.decode('utf-8')
            redis_client.srem(f"{pipeline_settings.REDIS_DUB_STAGE}:{old_stage}", job_id)
        
        redis_client.setex(f"{pipeline_settings.REDIS_DUB_STAGE}:{job_id}", pipeline_settings.JOB_TIMEOUT, new_stage)
        redis_client.sadd(f"{pipeline_settings.REDIS_DUB_STAGE}:{new_stage}", job_id)
        
        return True
    except Exception:
        return False

def get_active_dub_jobs_count() -> int:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return 0
            
        return redis_client.scard(pipeline_settings.REDIS_DUB_ACTIVE)
    except Exception:
        return 0

def get_stage_jobs_count(stage: str) -> int:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return 0
            
        return redis_client.scard(f"{pipeline_settings.REDIS_DUB_STAGE}:{stage}")
    except Exception:
        return 0

def can_start_dub_job() -> bool:
    # No artificial limit - VRAM bottlenecks controlled by service workers
    # Jobs can start as long as workers are available
    return True

def can_start_stage(stage: str) -> bool:
    current_count = get_stage_jobs_count(stage)
    
    if stage == "separation":
        return current_count < pipeline_settings.MAX_SEPARATION_JOBS
    elif stage == "transcription":
        return current_count < pipeline_settings.MAX_TRANSCRIPTION_JOBS
    elif stage == "voice_cloning":
        return current_count < pipeline_settings.MAX_VOICE_CLONING_JOBS
    elif stage == "dubbing":
        return current_count < pipeline_settings.MAX_DUBBING_JOBS
    else:
        return True

def mark_resume_job(job_id: str) -> bool:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False
            
        redis_client.sadd(pipeline_settings.REDIS_RESUME_JOBS, job_id)
        redis_client.expire(pipeline_settings.REDIS_RESUME_JOBS, pipeline_settings.JOB_TIMEOUT)
        
        return True
    except Exception:
        return False

def is_resume_job(job_id: str) -> bool:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False
            
        return redis_client.sismember(pipeline_settings.REDIS_RESUME_JOBS, job_id)
    except Exception:
        return False

def remove_resume_job(job_id: str) -> bool:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False
            
        redis_client.srem(pipeline_settings.REDIS_RESUME_JOBS, job_id)
        
        return True
    except Exception:
        return False

def can_start_stage_with_priority(stage: str, job_id: str) -> bool:
    import logging
    logger = logging.getLogger(__name__)
    
    is_resume = is_resume_job(job_id)
    if is_resume:
        result = can_start_stage(stage)
        if not result:
            current_count = get_stage_jobs_count(stage)
            logger.warning(f"🔍 Resume job {job_id} blocked: {stage} has {current_count} active jobs")
        return result
    
    result = can_start_stage(stage)
    if not result:
        current_count = get_stage_jobs_count(stage)
        max_allowed = getattr(pipeline_settings, f'MAX_{stage.upper()}_JOBS', 1)
        logger.warning(f"🔍 Job {job_id} blocked: {stage} has {current_count}/{max_allowed} jobs")
    
    return result

# Removed complex waiting queue functions - service workers handle queuing

def get_resume_jobs_for_stage(stage: str) -> int:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return 0
            
        resume_jobs = redis_client.smembers(pipeline_settings.REDIS_RESUME_JOBS)
        count = 0
        
        for job_id in resume_jobs:
            job_id = job_id.decode('utf-8')
            job_stage = redis_client.get(f"{pipeline_settings.REDIS_DUB_STAGE}:{job_id}")
            if job_stage and job_stage.decode('utf-8') == stage:
                count += 1
                
        return count
    except Exception:
        return 0



def get_pipeline_performance_metrics() -> dict:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return {}
        
        stages = ["initialization", "separation", "transcription", "dubbing", "review_prep", "voice_cloning", "upload"]
        stage_counts = {}
        
        for stage in stages:
            stage_counts[stage] = get_stage_jobs_count(stage)
        
        active_jobs = get_active_dub_jobs_count()
        resume_jobs = redis_client.scard(pipeline_settings.REDIS_RESUME_JOBS)
        
        # Calculate bottlenecks
        bottleneck_stage = max(stage_counts.items(), key=lambda x: x[1])[0] if stage_counts else "none"
        
        # Calculate efficiency based on service worker utilization
        total_stage_jobs = sum(stage_counts.values())
        # Efficiency based on VRAM worker utilization (main bottleneck)
        vram_efficiency = (
            get_stage_jobs_count('transcription') + get_stage_jobs_count('voice_cloning')
        ) / 2.0  # 2 VRAM workers max
        efficiency = min(100, vram_efficiency * 100)
        
        return {
            "active_jobs": active_jobs,
            "resume_jobs": resume_jobs,
            "stage_distribution": stage_counts,
            "bottleneck_stage": bottleneck_stage,
            "pipeline_efficiency": round(efficiency, 1),
            "capacity_utilization": {
                "whisperx_service": f"{get_stage_jobs_count('transcription')}/{pipeline_settings.MAX_WHISPERX_SERVICE_WORKERS}",
                "fish_speech_service": f"{get_stage_jobs_count('voice_cloning')}/{pipeline_settings.MAX_FISH_SPEECH_SERVICE_WORKERS}",
                "separation": f"{get_stage_jobs_count('separation')}/{pipeline_settings.MAX_SEPARATION_JOBS}",
                "total_active_jobs": active_jobs
            }
        }
    except Exception:
        return {"error": "Failed to get metrics"}

# Removed unused optimization functions - handled by service workers

def handle_pipeline_overflow(current_load: int) -> dict:
    try:
        # No artificial overflow limit - service workers manage VRAM bottlenecks
        # Monitor VRAM worker queues instead
        whisperx_queue_size = get_stage_jobs_count('transcription')
        fish_speech_queue_size = get_stage_jobs_count('voice_cloning')
        
        max_queue_size = 10  # Alert if queues get too long
        if whisperx_queue_size <= max_queue_size and fish_speech_queue_size <= max_queue_size:
            return {"status": "normal", "action": "none"}
        
        overflow_count = max(whisperx_queue_size, fish_speech_queue_size) - max_queue_size
        
        if overflow_count <= 5:
            return {
                "status": "manageable_overflow",
                "action": "temporary_queue_expansion",
                "overflow_jobs": overflow_count,
                "estimated_delay": f"{overflow_count * 3}-{overflow_count * 5} minutes"
            }
        
        elif overflow_count <= 15:
            return {
                "status": "high_overflow", 
                "action": "queue_optimization_required",
                "overflow_jobs": overflow_count,
                "estimated_delay": f"{overflow_count * 2}-{overflow_count * 4} minutes",
                "recommendation": "Optimize queue processing"
            }
        
        else:
            return {
                "status": "critical_overflow",
                "action": "graceful_degradation",
                "overflow_jobs": overflow_count,
                "estimated_delay": f"{overflow_count * 4}-{overflow_count * 8} minutes",
                "recommendation": "Consider load balancing or user notification"
            }
            
    except Exception:
        return {"status": "error", "action": "fallback_processing"}


# Service Worker Management Functions
def can_start_service_worker(service_type: str) -> bool:
    """Check if service worker can start (for serial processing)"""
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False
        
        from app.config.pipeline_settings import pipeline_settings
        
        if service_type == "whisperx":
            active_key = pipeline_settings.REDIS_WHISPERX_ACTIVE
            max_workers = pipeline_settings.MAX_WHISPERX_SERVICE_WORKERS
        elif service_type == "fish_speech":
            active_key = pipeline_settings.REDIS_FISH_SPEECH_ACTIVE
            max_workers = pipeline_settings.MAX_FISH_SPEECH_SERVICE_WORKERS
        else:
            return False
        
        active_count = redis_client.scard(active_key)
        return active_count < max_workers
        
    except Exception:
        return False


def mark_service_worker_active(service_type: str, worker_id: str) -> bool:
    """Mark service worker as active"""
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False
        
        from app.config.pipeline_settings import pipeline_settings
        
        if service_type == "whisperx":
            active_key = pipeline_settings.REDIS_WHISPERX_ACTIVE
        elif service_type == "fish_speech":
            active_key = pipeline_settings.REDIS_FISH_SPEECH_ACTIVE
        else:
            return False
        
        redis_client.sadd(active_key, worker_id)
        redis_client.expire(active_key, pipeline_settings.SERVICE_WORKER_TIMEOUT)
        
        return True
        
    except Exception:
        return False


def mark_service_worker_inactive(service_type: str, worker_id: str) -> bool:
    """Mark service worker as inactive"""
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False

        from app.config.pipeline_settings import pipeline_settings

        if service_type == "whisperx":
            active_key = pipeline_settings.REDIS_WHISPERX_ACTIVE
        elif service_type == "fish_speech":
            active_key = pipeline_settings.REDIS_FISH_SPEECH_ACTIVE
        elif service_type == "cpu_whisperx":
            active_key = pipeline_settings.REDIS_CPU_WHISPERX_ACTIVE
        elif service_type == "cpu_fish_speech":
            active_key = pipeline_settings.REDIS_CPU_FISH_SPEECH_ACTIVE
        else:
            return False

        redis_client.srem(active_key, worker_id)

        return True
        
    except Exception:
        return False


def store_service_result(service_type: str, request_id: str, result_data: dict) -> bool:
    """Store service worker result in Redis"""
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False
        
        import json
        from app.config.pipeline_settings import pipeline_settings
        
        if service_type == "whisperx":
            results_key = f"{pipeline_settings.REDIS_WHISPERX_RESULTS}:{request_id}"
        elif service_type == "fish_speech":
            results_key = f"{pipeline_settings.REDIS_FISH_SPEECH_RESULTS}:{request_id}"
        else:
            return False
        
        redis_client.setex(
            results_key,
            pipeline_settings.SERVICE_RESULT_TIMEOUT,
            json.dumps(result_data)
        )
        
        return True
        
    except Exception:
        return False


def get_service_result(service_type: str, request_id: str) -> dict:
    """Get service worker result from Redis"""
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return {"error": "Redis not available"}
        
        import json
        from app.config.pipeline_settings import pipeline_settings
        
        if service_type == "whisperx":
            results_key = f"{pipeline_settings.REDIS_WHISPERX_RESULTS}:{request_id}"
        elif service_type == "fish_speech":
            results_key = f"{pipeline_settings.REDIS_FISH_SPEECH_RESULTS}:{request_id}"
        else:
            return {"error": "Invalid service type"}
        
        result_data = redis_client.get(results_key)
        if result_data:
            return json.loads(result_data.decode('utf-8'))
        else:
            return {"error": "Result not found"}
        
    except Exception as e:
        return {"error": str(e)}


def wait_for_service_result(service_type: str, request_id: str, timeout: int = 1800) -> dict:
    """Wait for service worker result with timeout"""
    import time
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        result = get_service_result(service_type, request_id)
        if "error" not in result or result["error"] != "Result not found":
            return result
        time.sleep(2)  # Check every 2 seconds
    
    return {"error": "Timeout waiting for result"}


def cleanup_service_result(service_type: str, request_id: str) -> bool:
    """Cleanup service worker result from Redis"""
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False

        from app.config.pipeline_settings import pipeline_settings

        if service_type == "whisperx":
            results_key = f"{pipeline_settings.REDIS_WHISPERX_RESULTS}:{request_id}"
        elif service_type == "fish_speech":
            results_key = f"{pipeline_settings.REDIS_FISH_SPEECH_RESULTS}:{request_id}"
        elif service_type == "cpu_whisperx":
            results_key = f"{pipeline_settings.REDIS_CPU_WHISPERX_RESULTS}:{request_id}"
        elif service_type == "cpu_fish_speech":
            results_key = f"{pipeline_settings.REDIS_CPU_FISH_SPEECH_RESULTS}:{request_id}"
        else:
            return False

        redis_client.delete(results_key)
        return True

    except Exception:
        return False
