#!/usr/bin/env python3
"""
Check RQ workers and queue status
"""
import redis
import logging
from rq import Worker, Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_workers_status():
    """Check status of all workers and queues"""
    try:
        # Connect to Redis
        r = redis.Redis(host='127.0.0.1', port=6379, db=0)
        r.ping()
        logger.info("✅ Redis connection successful")
        
        # Check workers
        workers = Worker.all(connection=r)
        logger.info(f"📊 Found {len(workers)} active workers:")
        
        for worker in workers:
            status = "🟢 ACTIVE" if worker.state == 'busy' else "🟡 IDLE"
            logger.info(f"  - {worker.name}: {status} (Queue: {worker.queue_names()})")
        
        # Check queues (including service worker queues)
        queues = ['dub_queue', 'separation_queue', 'billing_queue',
                 'whisperx_service_queue', 'fish_speech_service_queue',
                 'cpu_whisperx_service_queue', 'cpu_fish_speech_service_queue',
                 'video_processing_queue']
        logger.info("\n📋 Queue Status:")

        gpu_jobs = 0
        cpu_jobs = 0

        for queue_name in queues:
            try:
                queue = Queue(queue_name, connection=r)
                job_count = len(queue)
                workers_count = Worker.count(queue=queue)

                # Track GPU vs CPU load
                if 'cpu_' in queue_name:
                    cpu_jobs += job_count
                elif queue_name in ['whisperx_service_queue', 'fish_speech_service_queue']:
                    gpu_jobs += job_count

                status_emoji = "🟢" if job_count > 0 else "🟡" if workers_count > 0 else "🔴"
                logger.info(f"  - {queue_name}: {status_emoji} {job_count} jobs, {workers_count} workers")
            except Exception as e:
                logger.error(f"  - {queue_name}: ERROR - {e}")

        # Load balancing status
        logger.info("\n⚖️ Load Balancing Status:")
        logger.info(f"  - GPU Jobs: {gpu_jobs} (Fast processing)")
        logger.info(f"  - CPU Jobs: {cpu_jobs} (Overflow handling)")
        logger.info(f"  - Load Ratio: {gpu_jobs} GPU : {cpu_jobs} CPU")

        # Performance insights
        if gpu_jobs > 2:
            logger.info("  ⚠️  GPU overloaded - CPU handling overflow")
        elif cpu_jobs > gpu_jobs:
            logger.info("  ✅ Balanced load distribution")
        else:
            logger.info("  🎯 GPU has capacity for more jobs")
        
        # Redis info
        info = r.info()
        logger.info(f"\n💾 Redis: {info['used_memory_human']} memory, {info['connected_clients']} clients")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    check_workers_status()
