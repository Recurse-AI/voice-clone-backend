import asyncio
import logging
import redis
import os
from datetime import datetime, time
from app.services.cookie_refresh_service import cookie_refresh_service

logger = logging.getLogger(__name__)


class CookieScheduler:
    def __init__(self):
        self.target_hour = 3
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

    def _seconds_until_3am(self) -> int:
        now = datetime.now()
        target = datetime.combine(now.date(), time(self.target_hour, 0))
        if now >= target:
            target = datetime.combine(now.date(), time(self.target_hour, 0))
            target = target.replace(day=target.day + 1)
        return int((target - now).total_seconds())

    async def start(self):
        scheduler_lock_key = "cookie_scheduler_active"
        daily_refresh_key = "cookie_refreshed_today"
        
        if self.redis_client:
            try:
                is_leader = self.redis_client.set(scheduler_lock_key, "1", nx=True, ex=86400)
                if not is_leader:
                    logger.info("Cookie scheduler already running in another worker")
                    return
            except Exception as e:
                logger.warning(f"Could not acquire scheduler lock: {e}")
                return

        self.running = True
        while self.running:
            try:
                now = datetime.now()
                current_date = now.strftime("%Y-%m-%d")
                
                if self.redis_client:
                    last_refresh_date = self.redis_client.get(daily_refresh_key)
                    already_refreshed_today = last_refresh_date == current_date
                else:
                    already_refreshed_today = False

                if now.hour == self.target_hour and not already_refreshed_today:
                    logger.info("🕒 Running daily 3 AM cookie refresh...")
                    result = await cookie_refresh_service.refresh_cookies()
                    
                    if result.get("success"):
                        logger.info("✅ Daily cookie refresh successful")
                        if self.redis_client:
                            self.redis_client.set(daily_refresh_key, current_date, ex=86400)
                    elif result.get("error") != "Refresh in progress":
                        logger.warning(f"❌ Daily refresh failed: {result.get('error')}")
                    
                    await asyncio.sleep(3600)
                else:
                    age_days = cookie_refresh_service.get_cookie_age_days()
                    if age_days == -1:
                        logger.info("⚠️ Cookie not found. Attempting auto-fetch...")
                        result = await cookie_refresh_service.refresh_cookies()
                        if result["success"]:
                            logger.info("✅ Cookies auto-fetched successfully")
                    
                    sleep_seconds = min(self._seconds_until_3am(), 3600)
                    await asyncio.sleep(sleep_seconds)

                if self.redis_client:
                    self.redis_client.expire(scheduler_lock_key, 86400)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(3600)


cookie_scheduler = CookieScheduler()

