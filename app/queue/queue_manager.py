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
        self._whisperx_service_queue = None
        self._fish_speech_service_queue = None
        self._video_processing_queue = None
    
    def _get_redis_client(self) -> Optional[Redis]:
        """Get Redis client with connection validation and retry"""
        if self._redis_client is None or not self._test_redis_connection():
            try:
                redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
                self._redis_client = Redis.from_url(
                    redis_url,
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    health_check_interval=30,
                    retry_on_timeout=True,
                    retry_on_error=[ConnectionError],
                    max_connections=50
                )
                
                # Test connection with retry
                for attempt in range(3):
                    try:
                        self._redis_client.ping()
                        logger.info(f"✅ Redis connected: {redis_url}")
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                        import time
                        time.sleep(1)
                
            except ConnectionError as e:
                logger.error(f"❌ Redis connection failed: {e}")
                self._redis_client = None
                
        return self._redis_client
    
    def _test_redis_connection(self) -> bool:
        """Test if Redis connection is still alive"""
        if self._redis_client is None:
            return False
        try:
            self._redis_client.ping()
            return True
        except:
            return False
    
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
    
    def get_video_processing_queue(self) -> Optional[Queue]:
        """Get video processing queue with lazy initialization"""
        if self._video_processing_queue is None:
            redis_client = self._get_redis_client()
            if redis_client:
                self._video_processing_queue = Queue("video_processing_queue", connection=redis_client)
                logger.info("✅ Video processing queue initialized")

        return self._video_processing_queue

    
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
                job_timeout=3600,
                result_ttl=1800    # Keep result for 30 minutes after completion
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
                job_timeout=7200,
                result_ttl=3600    # Keep result for 1 hour after completion
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
    
    def enqueue_video_processing_task(self, task_data: dict) -> bool:
        """Enqueue video processing task with error handling"""
        try:
            queue = self.get_video_processing_queue()
            if not queue:
                logger.error("❌ Video processing queue not available")
                return False
            
            job = queue.enqueue(
                'app.workers.video_processing_worker.process_video_task',
                task_data,
                job_timeout=7200,  # 2 hours timeout for video processing
                result_ttl=3600    # Keep result for 1 hour after completion
            )
            
            logger.info(f"✅ Enqueued video processing task: {task_data.get('job_id')} (RQ job: {job.id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to enqueue video processing task: {e}")
            return False

    def enqueue_with_load_balance(self, request_data: dict, service_type: str) -> bool:
        """Smart load balancing: GPU first, CPU fallback"""
        try:
            from app.config.pipeline_settings import pipeline_settings

            if service_type == "whisperx":
                gpu_queue = self.get_whisperx_service_queue()
                gpu_function = 'app.workers.whisperx_service_worker.process_whisperx_request'
                cpu_function = 'app.workers.cpu_whisperx_service_worker.process_cpu_whisperx_request'
            elif service_type == "fish_speech":
                gpu_queue = self.get_fish_speech_service_queue()
                gpu_function = 'app.workers.fish_speech_service_worker.process_fish_speech_request'
                cpu_function = 'app.workers.cpu_fish_speech_service_worker.process_cpu_fish_speech_request'
            else:
                logger.error(f"❌ Unknown service type: {service_type}")
                return False

            # Check GPU capacity - simple load balancing
            gpu_busy = len(gpu_queue) >= 1 if gpu_queue else True

            if not gpu_busy:
                # GPU has capacity - use GPU worker
                job = gpu_queue.enqueue(
                    gpu_function,
                    request_data,
                    job_timeout=pipeline_settings.SERVICE_WORKER_TIMEOUT
                )
                logger.info(f"🎯 GPU {service_type}: {request_data.get('request_id')} (Fast processing)")
                return True
            else:
                # GPU busy - try CPU queue if available, otherwise fallback to GPU
                try:
                    # Try to create CPU queue dynamically
                    from rq import Queue
                    redis_client = self._get_redis_client()
                    if redis_client:
                        cpu_queue_name = f"cpu_{service_type}_service_queue"
                        cpu_queue = Queue(cpu_queue_name, connection=redis_client)

                        job = cpu_queue.enqueue(
                            cpu_function,
                            request_data,
                            job_timeout=pipeline_settings.SERVICE_WORKER_TIMEOUT
                        )
                        logger.info(f"🐌 CPU {service_type}: {request_data.get('request_id')} (Overflow handling)")
                        return True
                except Exception as cpu_error:
                    logger.warning(f"CPU queue failed, falling back to GPU: {cpu_error}")

                # Fallback to GPU if CPU fails
                job = gpu_queue.enqueue(
                    gpu_function,
                    request_data,
                    job_timeout=pipeline_settings.SERVICE_WORKER_TIMEOUT
                )
                logger.info(f"⚠️ GPU Fallback {service_type}: {request_data.get('request_id')} (GPU was busy)")
                return True

        except Exception as e:
            logger.error(f"❌ Load balancing failed for {service_type}: {e}")
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


def get_video_processing_queue() -> Optional[Queue]:
    """Get video processing queue"""
    return queue_manager.get_video_processing_queue()


