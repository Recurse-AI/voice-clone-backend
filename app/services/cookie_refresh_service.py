import os
import logging
import redis
import asyncio
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class CookieRefreshService:
    def __init__(self):
        self.cookie_file = Path("youtube_cookies.txt")
        self.max_age_days = int(os.getenv("COOKIE_MAX_AGE_DAYS", "7"))
        self.max_downloads_before_refresh = int(os.getenv("MAX_DOWNLOADS_BEFORE_REFRESH", "20"))
        self.min_delay_between_downloads = int(os.getenv("YT_MIN_DELAY_SECONDS", "3"))
        self.download_count_key = "yt_cookie_download_count"
        self.rate_limit_key = "yt_download_rate_limit"
        self.redis_client = None
        self._init_redis()
    
    def _init_redis(self):
        try:
            redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis not available for cookie locking: {e}")
            self.redis_client = None
    
    def is_cookie_expired(self) -> bool:
        if not self.cookie_file.exists():
            return True
        age_days = (datetime.now() - datetime.fromtimestamp(self.cookie_file.stat().st_mtime)).days
        return age_days >= self.max_age_days
    
    def get_cookie_age_days(self) -> int:
        if not self.cookie_file.exists():
            return -1
        return (datetime.now() - datetime.fromtimestamp(self.cookie_file.stat().st_mtime)).days
    
    def should_refresh_cookies(self) -> bool:
        if not self.cookie_file.exists():
            return True
        if self.redis_client:
            try:
                count = self.redis_client.get(self.download_count_key)
                if count and int(count) >= self.max_downloads_before_refresh:
                    logger.info(f"Cookie refresh needed: {count} downloads reached")
                    return True
            except Exception as e:
                logger.warning(f"Could not check download count: {e}")
        return False
    
    def record_download(self):
        if self.redis_client:
            try:
                self.redis_client.incr(self.download_count_key)
            except Exception as e:
                logger.warning(f"Could not record download: {e}")
    
    async def apply_rate_limit(self):
        if self.redis_client:
            try:
                last_download = self.redis_client.get(self.rate_limit_key)
                if last_download:
                    elapsed = (datetime.now() - datetime.fromisoformat(last_download)).total_seconds()
                    if elapsed < self.min_delay_between_downloads:
                        delay = self.min_delay_between_downloads - elapsed
                        logger.info(f"Rate limiting: waiting {delay:.1f}s")
                        await asyncio.sleep(delay)
                self.redis_client.set(self.rate_limit_key, datetime.now().isoformat(), ex=60)
            except Exception as e:
                logger.warning(f"Rate limit check failed: {e}")
    
    async def validate_and_refresh_if_needed(self) -> bool:
        await self.apply_rate_limit()
        if self.should_refresh_cookies():
            logger.info("Cookie refresh required due to download limit")
            from app.services.youtube_cookie_fetcher import youtube_cookie_fetcher
            result = await youtube_cookie_fetcher.fetch_cookies(force_refresh=True)
            if result.get("success"):
                if self.redis_client:
                    self.redis_client.delete(self.download_count_key)
                return True
            return False
        return True
    
    async def refresh_cookies(self) -> dict:
        lock_key = "cookie_refresh_lock"
        lock_timeout = 120
        
        if self.redis_client:
            try:
                lock_acquired = self.redis_client.set(lock_key, "1", nx=True, ex=lock_timeout)
                if not lock_acquired:
                    logger.info("Cookie refresh already in progress by another worker")
                    return {"success": False, "error": "Refresh in progress"}
                
                try:
                    from app.services.youtube_cookie_fetcher import youtube_cookie_fetcher
                    result = await youtube_cookie_fetcher.fetch_cookies(force_refresh=False)
                    if result.get("success") and self.redis_client:
                        self.redis_client.delete(self.download_count_key)
                    return result
                finally:
                    self.redis_client.delete(lock_key)
            except Exception as e:
                logger.error(f"Cookie refresh failed: {e}")
                self.redis_client.delete(lock_key)
                return {"success": False, "error": str(e)}
        else:
            try:
                from app.services.youtube_cookie_fetcher import youtube_cookie_fetcher
                result = await youtube_cookie_fetcher.fetch_cookies()
                return result
            except Exception as e:
                logger.error(f"Cookie refresh failed: {e}")
                return {"success": False, "error": str(e)}


cookie_refresh_service = CookieRefreshService()
