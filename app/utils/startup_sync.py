import asyncio
import logging
import time
from typing import Optional
import os

logger = logging.getLogger(__name__)

class StartupSync:
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
        self._redis = None
        
    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.Redis.from_url(self.redis_url)
                await self._redis.ping()
            except Exception:
                return None
        return self._redis
    
    async def acquire_startup_lock(self, task_name: str, timeout: int = 30) -> bool:
        redis_client = await self._get_redis()
        if not redis_client:
            return self._acquire_file_lock(task_name, timeout)
            
        lock_key = f"startup_lock:{task_name}"
        
        try:
            acquired = await redis_client.set(
                lock_key, 
                f"worker_{os.getpid()}_{int(time.time())}", 
                nx=True,
                ex=timeout
            )
            
            if acquired:
                logger.info(f"Acquired startup lock: {task_name}")
                return True
            else:
                logger.info(f"Task '{task_name}' being handled by another worker")
                return False
                
        except Exception:
            return True
    
    async def release_startup_lock(self, task_name: str):
        redis_client = await self._get_redis()
        if not redis_client:
            self._release_file_lock(task_name)
            return
            
        lock_key = f"startup_lock:{task_name}"
        try:
            await redis_client.delete(lock_key)
            logger.info(f"Released startup lock: {task_name}")
        except Exception:
            pass
    
    async def wait_for_task_completion(self, task_name: str, max_wait: int = 60):
        redis_client = await self._get_redis()
        if not redis_client:
            await asyncio.sleep(2)
            return
            
        completion_key = f"startup_complete:{task_name}"
        
        for i in range(max_wait):
            try:
                completed = await redis_client.get(completion_key)
                if completed:
                    logger.info(f"Task '{task_name}' completed by another worker")
                    return
                await asyncio.sleep(1)
            except Exception:
                break
    
    async def mark_task_complete(self, task_name: str, ttl: int = 300):
        redis_client = await self._get_redis()
        if not redis_client:
            return
            
        completion_key = f"startup_complete:{task_name}"
        try:
            await redis_client.set(completion_key, "1", ex=ttl)
            logger.info(f"Marked task '{task_name}' complete")
        except Exception:
            pass
    
    def _acquire_file_lock(self, task_name: str, timeout: int) -> bool:
        lock_file = f"/tmp/startup_lock_{task_name}.lock"
        
        try:
            if os.path.exists(lock_file):
                lock_age = time.time() - os.path.getmtime(lock_file)
                if lock_age > timeout:
                    os.remove(lock_file)
                else:
                    return False
            
            with open(lock_file, 'w') as f:
                f.write(f"{os.getpid()}_{int(time.time())}")
            
            return True
            
        except Exception:
            return True
    
    def _release_file_lock(self, task_name: str):
        lock_file = f"/tmp/startup_lock_{task_name}.lock"
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception:
            pass


startup_sync = StartupSync()
