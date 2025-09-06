#!/usr/bin/env python3
"""
Simple RQ Workers Starter - Cross Platform Compatible
"""
import os
import sys
import redis
import logging
from rq import Worker, Queue, SimpleWorker

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def register_job_functions():
    """Register all job functions for the workers"""
    try:
        # Import job functions 
        from app.queue.dub_tasks import process_video_dub_task, process_redub_task
        from app.queue.separation_tasks import process_audio_separation_task
        from app.queue.billing_tasks import process_billing_task
        
        logger.info("Job functions registered successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to register job functions: {e}")
        return False

def initialize_ai_models():
    """Initialize AI models if LOAD_AI_MODELS is true"""
    from app.config.settings import settings
    
    if not settings.LOAD_AI_MODELS:
        logger.info("AI models initialization skipped (LOAD_AI_MODELS=false)")
        return
    
    logger.info("🚀 Loading heavy AI models in worker...")
    
    # Initialize OpenAI for dub worker
    try:
        from app.services.openai_service import initialize_openai_service
        initialize_openai_service()
        logger.info("✅ OpenAI ready")
    except Exception as e:
        logger.warning(f"OpenAI failed: {str(e)[:50]}")
    
    # Initialize FishSpeech
    try:
        from app.services.dub.fish_speech_service import initialize_fish_speech
        initialize_fish_speech()
        logger.info("✅ FishSpeech preloaded")
    except Exception as e:
        logger.warning(f"FishSpeech failed: {str(e)[:50]}")

    # Initialize WhisperX
    try:
        from app.services.dub.whisperx_transcription import initialize_whisperx_transcription
        initialize_whisperx_transcription()
        logger.info("✅ WhisperX preloaded")
    except Exception as e:
        logger.warning(f"WhisperX failed: {str(e)[:50]}")
    
    logger.info("🎯 AI models initialization completed")

def start_worker(queue_name: str, worker_name: str, redis_url: str = "redis://127.0.0.1:6379"):
    try:
        import time
        unique_worker_name = f"{worker_name}_{int(time.time())}"
        
        if not register_job_functions():
            logger.error("Failed to register job functions")
            return False
        
        # Initialize AI models for this worker
        initialize_ai_models()
        
        redis_conn = redis.Redis.from_url(redis_url)
        redis_conn.ping()
        
        queue = Queue(queue_name, connection=redis_conn)
        worker = SimpleWorker([queue], connection=redis_conn, name=unique_worker_name)
        
        # Optimized worker settings for faster queue processing
        worker.work(with_scheduler=True, burst=False)
        
    except KeyboardInterrupt:
        logger.info(f"Worker {unique_worker_name} stopped by user")
        return True
    except Exception as e:
        logger.error(f"Worker {unique_worker_name} error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to start worker based on command line arguments"""
    if len(sys.argv) < 3:
        print("Usage: python workers_starter.py <queue_name> <worker_name> [redis_url]")
        print("Examples:")
        print("  python workers_starter.py dub_queue dub_worker_1")
        print("  python workers_starter.py separation_queue sep_worker_1")
        print("  python workers_starter.py billing_queue billing_worker_1")
        sys.exit(1)
    
    queue_name = sys.argv[1]
    worker_name = sys.argv[2]
    redis_url = sys.argv[3] if len(sys.argv) > 3 else "redis://127.0.0.1:6379"
    
    # Start the worker
    success = start_worker(queue_name, worker_name, redis_url)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
