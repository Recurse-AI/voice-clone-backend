import os
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class CookieRefreshService:
    def __init__(self):
        self.cookie_file = Path("youtube_cookies.txt")
        self.max_age_days = int(os.getenv("COOKIE_MAX_AGE_DAYS", "14"))
    
    def is_cookie_expired(self) -> bool:
        if not self.cookie_file.exists():
            return True
        age_days = (datetime.now() - datetime.fromtimestamp(self.cookie_file.stat().st_mtime)).days
        return age_days >= self.max_age_days
    
    def get_cookie_age_days(self) -> int:
        if not self.cookie_file.exists():
            return -1
        return (datetime.now() - datetime.fromtimestamp(self.cookie_file.stat().st_mtime)).days
    
    async def refresh_cookies(self) -> dict:
        try:
            from app.services.youtube_cookie_fetcher import youtube_cookie_fetcher
            result = await youtube_cookie_fetcher.fetch_cookies()
            return result
        except Exception as e:
            logger.error(f"Cookie refresh failed: {e}")
            return {"success": False, "error": str(e)}


cookie_refresh_service = CookieRefreshService()
