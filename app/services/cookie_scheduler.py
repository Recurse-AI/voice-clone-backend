import asyncio
import logging
from app.services.cookie_refresh_service import cookie_refresh_service

logger = logging.getLogger(__name__)


class CookieScheduler:
    def __init__(self):
        self.check_interval_hours = 24
        self.running = False

    async def start(self):
        self.running = True
        while self.running:
            try:
                age_days = cookie_refresh_service.get_cookie_age_days()

                if age_days == -1:
                    logger.info("⚠️ Cookie not found. Attempting auto-fetch...")
                    result = await cookie_refresh_service.refresh_cookies()
                    if result["success"]:
                        logger.info("✅ Cookies auto-fetched successfully")
                    else:
                        logger.warning(f"❌ Auto-fetch failed: {result.get('error')}")
                        
                elif cookie_refresh_service.is_cookie_expired():
                    logger.info(f"⚠️ Cookie expired ({age_days} days). Refreshing...")
                    result = await cookie_refresh_service.refresh_cookies()
                    if result["success"]:
                        logger.info("✅ Cookies refreshed successfully")
                    else:
                        logger.warning(f"❌ Refresh failed: {result.get('error')}")
                else:
                    logger.info(f"✅ Cookie valid ({age_days}/{cookie_refresh_service.max_age_days} days)")

                await asyncio.sleep(self.check_interval_hours * 3600)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(3600)


cookie_scheduler = CookieScheduler()

