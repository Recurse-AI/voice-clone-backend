#!/usr/bin/env python3
"""
Force cleanup all RQ workers
"""
import redis
import logging
from rq import Worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_all_workers():
    """Force cleanup all workers"""
    try:
        # Connect to Redis
        r = redis.Redis(host='127.0.0.1', port=6379, db=0)
        r.ping()
        
        # Get all workers
        workers = Worker.all(connection=r)
        logger.info(f"Found {len(workers)} workers to cleanup")
        
        for worker in workers:
            try:
                logger.info(f"Cleaning worker: {worker.name}")
                worker.register_death()
                logger.info(f"✅ Cleaned: {worker.name}")
            except Exception as e:
                logger.error(f"❌ Error cleaning {worker.name}: {e}")
        
        logger.info("✅ All workers cleaned")
        
        # Clear all queues
        queues = ['dub_queue', 'separation_queue', 'billing_queue']
        for queue_name in queues:
            try:
                queue_length = r.llen(f"rq:queue:{queue_name}")
                if queue_length > 0:
                    r.delete(f"rq:queue:{queue_name}")
                    logger.info(f"✅ Cleared queue: {queue_name} ({queue_length} jobs)")
            except Exception as e:
                logger.error(f"❌ Error clearing {queue_name}: {e}")
        
        logger.info("🎉 Cleanup completed!")
        
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")

if __name__ == "__main__":
    cleanup_all_workers()
