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
    current_count = get_active_dub_jobs_count()
    return current_count < pipeline_settings.DUB_CONCURRENCY_LIMIT

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
    
    # Do not block regular jobs just because resume jobs exist.
    # Let capacity control via can_start_stage handle admission.
    
    result = can_start_stage(stage)
    if not result:
        current_count = get_stage_jobs_count(stage)
        max_allowed = getattr(pipeline_settings, f'MAX_{stage.upper()}_JOBS', 1)
        logger.warning(f"🔍 Job {job_id} blocked: {stage} has {current_count}/{max_allowed} jobs")
    
    return result

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

def get_batchable_jobs(stage: str, min_batch_size: int = 2) -> list:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return []
            
        # Get jobs currently in this stage
        jobs_in_stage = redis_client.smembers(f"{pipeline_settings.REDIS_DUB_STAGE}:{stage}")
        stage_jobs = [job.decode('utf-8') for job in jobs_in_stage]
        
        # Get active jobs waiting for this stage
        active_jobs = redis_client.smembers(pipeline_settings.REDIS_DUB_ACTIVE)
        waiting_jobs = [
            job_bytes.decode('utf-8') for job_bytes in active_jobs
            if redis_client.get(f"{pipeline_settings.REDIS_DUB_STAGE}:{job_bytes.decode('utf-8')}")
            and redis_client.get(f"{pipeline_settings.REDIS_DUB_STAGE}:{job_bytes.decode('utf-8')}").decode('utf-8') == stage
        ]
        
        # Combine unique jobs
        all_jobs = list(set(stage_jobs + waiting_jobs))
        
        return all_jobs[:min_batch_size] if len(all_jobs) >= min_batch_size else []
        
    except Exception:
        return []

def can_batch_separation_requests() -> bool:
    separation_jobs = get_batchable_jobs("separation", pipeline_settings.BATCH_SEPARATION_SIZE)
    return len(separation_jobs) >= pipeline_settings.BATCH_SEPARATION_SIZE

def can_batch_dubbing_requests() -> bool:
    dubbing_jobs = get_batchable_jobs("dubbing", pipeline_settings.BATCH_DUBBING_SIZE)
    return len(dubbing_jobs) >= pipeline_settings.BATCH_DUBBING_SIZE

def can_batch_upload_requests() -> bool:
    upload_jobs = get_batchable_jobs("upload", pipeline_settings.BATCH_UPLOAD_SIZE)
    return len(upload_jobs) >= pipeline_settings.BATCH_UPLOAD_SIZE

def should_wait_for_batch(stage: str, job_id: str) -> bool:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False
        
        # Check if there are other jobs in this stage to batch with
        stage_jobs_count = get_stage_jobs_count(stage)
        if stage_jobs_count < 2:
            return False  # Don't wait if no other jobs to batch with
            
        wait_key = f"batch_wait:{stage}:{job_id}"
        existing = redis_client.get(wait_key)
        
        if not existing:
            redis_client.setex(wait_key, pipeline_settings.BATCH_TIMEOUT, "waiting")
            return True
            
        return False
    except Exception:
        return False

def execute_separation_batch(job_ids: list) -> dict:
    try:
        import uuid
        batch_id = str(uuid.uuid4())[:8]
        
        redis_client = get_redis_client()
        if redis_client:
            redis_client.setex(f"batch:separation:{batch_id}", pipeline_settings.JOB_TIMEOUT, ",".join(job_ids))
        
        return {"batch_id": batch_id, "jobs": job_ids, "status": "processing"}
    except Exception:
        return {"batch_id": None, "jobs": job_ids, "status": "failed"}

def execute_dubbing_batch(job_ids: list) -> dict:
    try:
        import uuid
        batch_id = str(uuid.uuid4())[:8]
        
        redis_client = get_redis_client()
        if redis_client:
            redis_client.setex(f"batch:dubbing:{batch_id}", pipeline_settings.JOB_TIMEOUT, ",".join(job_ids))
        
        return {"batch_id": batch_id, "jobs": job_ids, "status": "processing"}
    except Exception:
        return {"batch_id": None, "jobs": job_ids, "status": "failed"}

def mark_jobs_as_batched(job_ids: list, batch_id: str) -> bool:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return False
            
        for job_id in job_ids:
            redis_client.setex(f"batch:{job_id}", pipeline_settings.JOB_TIMEOUT, batch_id)
            
        return True
    except Exception:
        return False

def get_batch_efficiency_stats() -> dict:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return {}
            
        return {
            "separation_batchable": len(get_batchable_jobs("separation", pipeline_settings.BATCH_SEPARATION_SIZE)),
            "dubbing_batchable": len(get_batchable_jobs("dubbing", pipeline_settings.BATCH_DUBBING_SIZE)),
            "upload_batchable": len(get_batchable_jobs("upload", pipeline_settings.BATCH_UPLOAD_SIZE)),
            "review_prep_batchable": len(get_batchable_jobs("review_prep", 2))
        }
    except Exception:
        return {}

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
        
        # Calculate efficiency
        total_stage_jobs = sum(stage_counts.values())
        efficiency = min(100, (active_jobs / max(1, pipeline_settings.DUB_CONCURRENCY_LIMIT)) * 100)
        
        return {
            "active_jobs": active_jobs,
            "resume_jobs": resume_jobs,
            "stage_distribution": stage_counts,
            "bottleneck_stage": bottleneck_stage,
            "pipeline_efficiency": round(efficiency, 1),
            "batch_opportunities": get_batch_efficiency_stats(),
            "capacity_utilization": {
                "transcription": f"{get_stage_jobs_count('transcription')}/{pipeline_settings.MAX_TRANSCRIPTION_JOBS}",
                "voice_cloning": f"{get_stage_jobs_count('voice_cloning')}/{pipeline_settings.MAX_VOICE_CLONING_JOBS}",
                "total_pipeline": f"{active_jobs}/{pipeline_settings.DUB_CONCURRENCY_LIMIT}"
            }
        }
    except Exception:
        return {"error": "Failed to get metrics"}

def detect_pipeline_bottlenecks() -> list:
    try:
        metrics = get_pipeline_performance_metrics()
        bottlenecks = []
        
        # Check GPU stage bottlenecks
        if metrics.get("stage_distribution", {}).get("transcription", 0) > 2:
            bottlenecks.append("transcription_queue_buildup")
            
        if metrics.get("stage_distribution", {}).get("voice_cloning", 0) > 2:
            bottlenecks.append("voice_cloning_queue_buildup")
        
        # Check batch efficiency
        batch_stats = metrics.get("batch_opportunities", {})
        if batch_stats.get("separation_batchable", 0) >= pipeline_settings.BATCH_SEPARATION_SIZE:
            bottlenecks.append("separation_batch_ready")
            
        if batch_stats.get("dubbing_batchable", 0) >= pipeline_settings.BATCH_DUBBING_SIZE:
            bottlenecks.append("dubbing_batch_ready")
        
        # Check capacity utilization
        if metrics.get("pipeline_efficiency", 0) < 70:
            bottlenecks.append("low_pipeline_efficiency")
            
        return bottlenecks
    except Exception:
        return []

def get_optimal_next_job(stage: str) -> Optional[str]:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return None
        
        # Priority 1: Resume jobs in voice_cloning stage
        if stage == "voice_cloning":
            resume_jobs = redis_client.smembers(pipeline_settings.REDIS_RESUME_JOBS)
            for job_id_bytes in resume_jobs:
                job_id = job_id_bytes.decode('utf-8')
                job_stage_bytes = redis_client.get(f"{pipeline_settings.REDIS_DUB_STAGE}:{job_id}")
                if job_stage_bytes and job_stage_bytes.decode('utf-8') == "voice_cloning":
                    return job_id
        
        # Priority 2: Jobs ready for this stage
        stage_jobs = redis_client.smembers(f"{pipeline_settings.REDIS_DUB_STAGE}:{stage}")
        if stage_jobs:
            return list(stage_jobs)[0].decode('utf-8')
        
        return None
    except Exception:
        return None

def optimize_stage_transitions() -> dict:
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return {}
        
        optimizations = []
        
        # Check if transcription slot is free and jobs are waiting
        if get_stage_jobs_count("transcription") < pipeline_settings.MAX_TRANSCRIPTION_JOBS:
            next_job = get_optimal_next_job("transcription")
            if next_job:
                optimizations.append({
                    "action": "move_to_transcription",
                    "job_id": next_job,
                    "priority": "high"
                })
        
        # Check if voice cloning slot is free
        if get_stage_jobs_count("voice_cloning") < pipeline_settings.MAX_VOICE_CLONING_JOBS:
            next_job = get_optimal_next_job("voice_cloning")
            if next_job:
                optimizations.append({
                    "action": "move_to_voice_cloning", 
                    "job_id": next_job,
                    "priority": "high"
                })
        
        # Check for batch opportunities
        bottlenecks = detect_pipeline_bottlenecks()
        for bottleneck in bottlenecks:
            if "batch_ready" in bottleneck:
                stage_name = bottleneck.replace("_batch_ready", "")
                optimizations.append({
                    "action": f"execute_{stage_name}_batch",
                    "priority": "medium"
                })
        
        return {
            "optimizations": optimizations,
            "total_opportunities": len(optimizations)
        }
    except Exception:
        return {}

def handle_pipeline_overflow(current_load: int) -> dict:
    try:
        if current_load <= pipeline_settings.DUB_CONCURRENCY_LIMIT:
            return {"status": "normal", "action": "none"}
        
        overflow_count = current_load - pipeline_settings.DUB_CONCURRENCY_LIMIT
        
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
                "action": "batch_optimization_required",
                "overflow_jobs": overflow_count,
                "estimated_delay": f"{overflow_count * 2}-{overflow_count * 4} minutes",
                "recommendation": "Enable aggressive batching"
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
