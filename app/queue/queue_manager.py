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
        # New service worker queues
        self._whisperx_service_queue = None
        self._fish_speech_service_queue = None
    
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
    
    def get_whisperx_service_queue(self) -> Optional[Queue]:
        """Get WhisperX service queue with lazy initialization"""
        if self._whisperx_service_queue is None:
            redis_client = self._get_redis_client()
            if redis_client:
                self._whisperx_service_queue = Queue("whisperx_service_queue", connection=redis_client)
                logger.info("✅ WhisperX service queue initialized")
        
        return self._whisperx_service_queue
    
    def get_fish_speech_service_queue(self) -> Optional[Queue]:
        """Get Fish Speech service queue with lazy initialization"""
        if self._fish_speech_service_queue is None:
            redis_client = self._get_redis_client()
            if redis_client:
                self._fish_speech_service_queue = Queue("fish_speech_service_queue", connection=redis_client)
                logger.info("✅ Fish Speech service queue initialized")
        
        return self._fish_speech_service_queue
    
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
                job_timeout=3600
            )
            
            logger.info(f"✅ Enqueued separation task: {job_id} (RQ job: {job.id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to enqueue separation task {job_id}: {e}")
            return False
    
    def enqueue_dub_task(self, request_dict: dict, user_id: str) -> bool:
        try:
            queue = self.get_dub_queue()
            if not queue:
                logger.error("❌ Dub queue not available")
                return False
            
            from app.workers.dub_worker import enqueue_dub_task
            
            job = queue.enqueue(
                enqueue_dub_task,
                request_dict, user_id,
                job_timeout=7200
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
                job_timeout='300s'  # Increased timeout to handle first-time imports/API latency
            )
            
            logger.info(f"✅ Enqueued billing task: {operation} (RQ job: {job.id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to enqueue billing task: {e}")
            return False
    
    def enqueue_whisperx_service_task(self, request_data: dict) -> bool:
        """Enqueue WhisperX service task with error handling"""
        try:
            queue = self.get_whisperx_service_queue()
            if not queue:
                logger.error("❌ WhisperX service queue not available")
                return False
            
            from app.config.pipeline_settings import pipeline_settings
            
            job = queue.enqueue(
                'app.workers.whisperx_service_worker.process_whisperx_request',
                request_data,
                job_timeout=pipeline_settings.SERVICE_WORKER_TIMEOUT
            )
            
            logger.info(f"✅ Enqueued WhisperX service task: {request_data.get('request_id')} (RQ job: {job.id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to enqueue WhisperX service task: {e}")
            return False
    
    def enqueue_fish_speech_service_task(self, request_data: dict) -> bool:
        """Enqueue Fish Speech service task with error handling"""
        try:
            queue = self.get_fish_speech_service_queue()
            if not queue:
                logger.error("❌ Fish Speech service queue not available")
                return False
                
            from app.config.pipeline_settings import pipeline_settings
            
            job = queue.enqueue(
                'app.workers.fish_speech_service_worker.process_fish_speech_request',
                request_data,
                job_timeout=pipeline_settings.SERVICE_WORKER_TIMEOUT
            )
            
            logger.info(f"✅ Enqueued Fish Speech service task: {request_data.get('request_id')} (RQ job: {job.id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to enqueue Fish Speech service task: {e}")
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
            whisperx_service_queue = self.get_whisperx_service_queue()
            fish_speech_service_queue = self.get_fish_speech_service_queue()
            
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
                },
                "whisperx_service_queue": {
                    "length": len(whisperx_service_queue) if whisperx_service_queue else 0,
                    "workers": Worker.count(queue=whisperx_service_queue) if whisperx_service_queue else 0
                },
                "fish_speech_service_queue": {
                    "length": len(fish_speech_service_queue) if fish_speech_service_queue else 0,
                    "workers": Worker.count(queue=fish_speech_service_queue) if fish_speech_service_queue else 0
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
            whisperx_service_queue = self.get_whisperx_service_queue()
            fish_speech_service_queue = self.get_fish_speech_service_queue()
            
            return all([
                dub_queue is not None, 
                separation_queue is not None, 
                billing_queue is not None,
                whisperx_service_queue is not None,
                fish_speech_service_queue is not None
            ])
            
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


def get_whisperx_service_queue() -> Optional[Queue]:
    """Get WhisperX service queue"""
    return queue_manager.get_whisperx_service_queue()


def get_fish_speech_service_queue() -> Optional[Queue]:
    """Get Fish Speech service queue"""
    return queue_manager.get_fish_speech_service_queue()
