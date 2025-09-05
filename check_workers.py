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
        
        # Check queues
        queues = ['dub_queue', 'separation_queue', 'billing_queue']
        logger.info("\n📋 Queue Status:")
        
        for queue_name in queues:
            try:
                queue = Queue(queue_name, connection=r)
                job_count = len(queue)
                workers_count = Worker.count(queue=queue)
                logger.info(f"  - {queue_name}: {job_count} jobs, {workers_count} workers")
            except Exception as e:
                logger.error(f"  - {queue_name}: ERROR - {e}")
        
        # Redis info
        info = r.info()
        logger.info(f"\n💾 Redis: {info['used_memory_human']} memory, {info['connected_clients']} clients")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    check_workers_status()
