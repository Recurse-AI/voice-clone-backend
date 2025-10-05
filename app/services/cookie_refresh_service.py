import os
import logging
import redis
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class CookieRefreshService:
    def __init__(self):
        self.cookie_file = Path("youtube_cookies.txt")
        self.redis_client = None
        self._init_redis()

    def _init_redis(self):
        try:
            redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
    
    def get_cookie_age_days(self) -> int:
        if not self.cookie_file.exists():
            return -1
        return (datetime.now() - datetime.fromtimestamp(self.cookie_file.stat().st_mtime)).days
    
    async def refresh_cookies(self) -> dict:
        lock_key = "cookie_refresh_lock"
        
        if self.redis_client:
            try:
                lock_acquired = self.redis_client.set(lock_key, "1", nx=True, ex=120)
                if not lock_acquired:
                    return {"success": False, "error": "Refresh in progress"}
                
                try:
                    from app.services.youtube_cookie_fetcher import youtube_cookie_fetcher
                    return await youtube_cookie_fetcher.fetch_cookies(force_refresh=False)
                finally:
                    self.redis_client.delete(lock_key)
            except Exception as e:
                logger.error(f"Cookie refresh failed: {e}")
                self.redis_client.delete(lock_key)
                return {"success": False, "error": str(e)}
        else:
            try:
                from app.services.youtube_cookie_fetcher import youtube_cookie_fetcher
                return await youtube_cookie_fetcher.fetch_cookies()
            except Exception as e:
                logger.error(f"Cookie refresh failed: {e}")
                return {"success": False, "error": str(e)}


cookie_refresh_service = CookieRefreshService()