import redis
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from app.config.settings import settings

logger = logging.getLogger(__name__)


class RedisStatusService:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        self._test_connection()

    def _test_connection(self):
        try:
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def set_status(self, job_id: str, job_type: str, status_data: Dict[str, Any]) -> bool:
        try:
            key = f"status:{job_type}:{job_id}"
            data = {
                **status_data,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "job_type": job_type
            }
            # Ensure created_at is set once
            try:
                existing_created = self.redis_client.hget(key, "created_at")
            except Exception:
                existing_created = None
            if not existing_created:
                data["created_at"] = datetime.now(timezone.utc).isoformat()
            else:
                data["created_at"] = existing_created
            
            # Convert dict values to JSON strings for Redis storage
            redis_data = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    redis_data[k] = json.dumps(v)
                else:
                    redis_data[k] = str(v) if v is not None else ""
            
            pipe = self.redis_client.pipeline()
            pipe.hset(key, mapping=redis_data)
            pipe.expire(key, 86400)
            pipe.execute()
            
            return True
        except Exception as e:
            logger.error(f"Redis set_status failed for {job_id}: {e}")
            return False

    def get_status(self, job_id: str, job_type: str) -> Optional[Dict[str, Any]]:
        try:
            key = f"status:{job_type}:{job_id}"
            data = self.redis_client.hgetall(key)
            
            if not data:
                return None
                
            if data.get("progress"):
                try:
                    data["progress"] = int(data["progress"])
                except Exception:
                    data["progress"] = 0
                
            if data.get("details"):
                try:
                    data["details"] = json.loads(data["details"])
                except (json.JSONDecodeError, TypeError):
                    data["details"] = {}
            
            return data
        except Exception as e:
            logger.error(f"Redis get_status failed for {job_id}: {e}")
            return None

    def update_progress(self, job_id: str, job_type: str, progress: int, details: Dict[str, Any] = None) -> bool:
        try:
            key = f"status:{job_type}:{job_id}"
            
            update_data = {
                "progress": progress,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            if details:
                update_data["details"] = json.dumps(details)
            
            self.redis_client.hset(key, mapping=update_data)
            return True
        except Exception as e:
            logger.error(f"Redis update_progress failed for {job_id}: {e}")
            return False

    def delete_status(self, job_id: str, job_type: str) -> bool:
        try:
            key = f"status:{job_type}:{job_id}"
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete_status failed for {job_id}: {e}")
            return False

    def is_available(self) -> bool:
        try:
            self.redis_client.ping()
            return True
        except:
            return False

    def get_all_jobs(self, job_type: str) -> list:
        try:
            pattern = f"status:{job_type}:*"
            jobs = []
            for key in self.redis_client.scan_iter(match=pattern):
                data = self.redis_client.hgetall(key)
                if data:
                    job_id = key.split(":")[-1]
                    data["job_id"] = job_id
                    if data.get("progress"):
                        try:
                            data["progress"] = int(data["progress"])
                        except Exception:
                            data["progress"] = 0
                    
                    # Deserialize details if present
                    if data.get("details"):
                        try:
                            data["details"] = json.loads(data["details"])
                        except (json.JSONDecodeError, TypeError):
                            data["details"] = {}
                    
                    jobs.append(data)
            
            return jobs
        except Exception as e:
            logger.error(f"Redis get_all_jobs failed for {job_type}: {e}")
            return []


redis_status_service = RedisStatusService()