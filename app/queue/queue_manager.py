"""
Queue Manager - Clean RQ queue management
Handles queue setup, worker management, and task enqueueing
"""
import logging
import os
from typing import Optional
from rq import Queue, Worker
from redis import Redis
from redis.exceptions import ConnectionError
from app.config.settings import settings

logger = logging.getLogger(__name__)


class QueueManager:
    """
    Clean queue manager with single responsibility:
    - Setup and validate Redis connections
    - Manage RQ queues
    - Enqueue tasks with proper error handling
    """
    
    def __init__(self):
        self._redis_client = None
        self._dub_queue = None
        self._separation_queue = None
        self._billing_queue = None
    
    def _get_redis_client(self) -> Optional[Redis]:
        """Get Redis client with connection validation"""
        if self._redis_client is None:
            try:
                redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
                self._redis_client = Redis.from_url(redis_url)
                
                # Test connection
                self._redis_client.ping()
                logger.info(f"✅ Redis connected: {redis_url}")
                
            except ConnectionError as e:
                logger.error(f"❌ Redis connection failed: {e}")
                self._redis_client = None
                
        return self._redis_client
    
    def get_dub_queue(self) -> Optional[Queue]:
        """Get dub queue with lazy initialization"""
        if self._dub_queue is None:
            redis_client = self._get_redis_client()
            if redis_client:
                self._dub_queue = Queue("dub_queue", connection=redis_client)
                logger.info("✅ Dub queue initialized")
        
        return self._dub_queue
    
    def get_separation_queue(self) -> Optional[Queue]:
        """Get separation queue with lazy initialization"""
        if self._separation_queue is None:
            redis_client = self._get_redis_client()
            if redis_client:
                self._separation_queue = Queue("separation_queue", connection=redis_client)
                logger.info("✅ Separation queue initialized")
        
        return self._separation_queue
    
    def get_billing_queue(self) -> Optional[Queue]:
        """Get billing queue with lazy initialization"""
        if self._billing_queue is None:
            redis_client = self._get_redis_client()
            if redis_client:
                self._billing_queue = Queue("billing_queue", connection=redis_client)
                logger.info("✅ Billing queue initialized")
        
        return self._billing_queue
    
    def enqueue_separation_task(self, job_id: str, runpod_request_id: str, 
                              user_id: str, duration_seconds: float) -> bool:
        """Enqueue separation task with error handling"""
        try:
            queue = self.get_separation_queue()
            if not queue:
                logger.error("❌ Separation queue not available")
                return False
            
            from app.workers.separation_worker import enqueue_separation_task
            
            job = queue.enqueue(
                enqueue_separation_task,
                job_id, runpod_request_id, user_id, duration_seconds,
                job_timeout='1h'  # 1 hour timeout
            )
            
            logger.info(f"✅ Enqueued separation task: {job_id} (RQ job: {job.id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to enqueue separation task {job_id}: {e}")
            return False
    
    def enqueue_dub_task(self, request_dict: dict, user_id: str) -> bool:
        """Enqueue dub task with error handling"""
        try:
            queue = self.get_dub_queue()
            if not queue:
                logger.error("❌ Dub queue not available")
                return False
            
            from app.workers.dub_worker import enqueue_dub_task
            
            job = queue.enqueue(
                enqueue_dub_task,
                request_dict, user_id,
                job_timeout='2h'  # 2 hour timeout for dub jobs
            )
            
            logger.info(f"✅ Enqueued dub task: {request_dict.get('job_id')} (RQ job: {job.id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to enqueue dub task: {e}")
            return False
    
    def enqueue_billing_task(self, operation: str, **kwargs) -> Optional[bool]:
        """Enqueue billing task to main process"""
        try:
            queue = self.get_billing_queue()
            if not queue:
                logger.error("❌ Billing queue not available")
                return False
            
            from app.queue.billing_tasks import process_billing_task
            
            job = queue.enqueue(
                process_billing_task,
                operation, kwargs,
                job_timeout='30s'  # Quick billing operations
            )
            
            logger.info(f"✅ Enqueued billing task: {operation} (RQ job: {job.id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to enqueue billing task: {e}")
            return False
    
    def get_queue_info(self) -> dict:
        """Get queue status information"""
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                return {"error": "Redis not available"}
            
            dub_queue = self.get_dub_queue()
            separation_queue = self.get_separation_queue()
            
            billing_queue = self.get_billing_queue()
            
            info = {
                "redis_connected": True,
                "dub_queue": {
                    "length": len(dub_queue) if dub_queue else 0,
                    "workers": Worker.count(queue=dub_queue) if dub_queue else 0
                },
                "separation_queue": {
                    "length": len(separation_queue) if separation_queue else 0,
                    "workers": Worker.count(queue=separation_queue) if separation_queue else 0
                },
                "billing_queue": {
                    "length": len(billing_queue) if billing_queue else 0,
                    "workers": Worker.count(queue=billing_queue) if billing_queue else 0
                }
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get queue info: {e}")
            return {"error": str(e)}
    
    def check_health(self) -> bool:
        """Check if queue system is healthy"""
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                return False
            
            # Test Redis connection
            redis_client.ping()
            
            # Check if queues are accessible
            dub_queue = self.get_dub_queue()
            separation_queue = self.get_separation_queue()
            billing_queue = self.get_billing_queue()
            
            return all([dub_queue is not None, separation_queue is not None, billing_queue is not None])
            
        except Exception as e:
            logger.error(f"Queue health check failed: {e}")
            return False


# Global queue manager instance
queue_manager = QueueManager()


# Convenience functions for backward compatibility
def get_dub_queue() -> Optional[Queue]:
    """Get dub queue"""
    return queue_manager.get_dub_queue()


def get_separation_queue() -> Optional[Queue]:
    """Get separation queue"""
    return queue_manager.get_separation_queue()


def get_billing_queue() -> Optional[Queue]:
    """Get billing queue"""
    return queue_manager.get_billing_queue()
