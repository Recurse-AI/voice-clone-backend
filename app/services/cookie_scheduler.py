import asyncio
import logging
import redis
import os
from app.services.cookie_refresh_service import cookie_refresh_service

logger = logging.getLogger(__name__)


class CookieScheduler:
    def __init__(self):
        self.check_interval_hours = 12
        self.running = False
        self.redis_client = None
        self._init_redis()

    def _init_redis(self):
        try:
            redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis not available for scheduler locking: {e}")
            self.redis_client = None

    async def start(self):
        scheduler_lock_key = "cookie_scheduler_active"
        
        if self.redis_client:
            try:
                is_leader = self.redis_client.set(scheduler_lock_key, "1", nx=True, ex=self.check_interval_hours * 3600 + 300)
                if not is_leader:
                    logger.info("Cookie scheduler already running in another worker")
                    return
            except Exception as e:
                logger.warning(f"Could not acquire scheduler lock: {e}")
                return

        self.running = True
        while self.running:
            try:
                age_days = cookie_refresh_service.get_cookie_age_days()

                if age_days == -1:
                    logger.info("⚠️ Cookie not found. Attempting auto-fetch...")
                    result = await cookie_refresh_service.refresh_cookies()
                    if result["success"]:
                        logger.info("✅ Cookies auto-fetched successfully")
                    elif result.get("error") != "Refresh in progress":
                        logger.warning(f"❌ Auto-fetch failed: {result.get('error')}")
                        
                elif cookie_refresh_service.is_cookie_expired():
                    logger.info(f"⚠️ Cookie expired ({age_days} days). Refreshing...")
                    result = await cookie_refresh_service.refresh_cookies()
                    if result["success"]:
                        logger.info("✅ Cookies refreshed successfully")
                    elif result.get("error") != "Refresh in progress":
                        logger.warning(f"❌ Refresh failed: {result.get('error')}")
                else:
                    logger.info(f"✅ Cookie valid ({age_days}/{cookie_refresh_service.max_age_days} days)")

                if self.redis_client:
                    self.redis_client.expire(scheduler_lock_key, self.check_interval_hours * 3600 + 300)

                await asyncio.sleep(self.check_interval_hours * 3600)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(3600)


cookie_scheduler = CookieScheduler()

