import httpx
import logging
from typing import Optional
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class AnalyticsService:
    _executor = ThreadPoolExecutor(max_workers=2)
    
    @staticmethod
    async def track_api_call(
        user_id: str,
        provider: str,
        tokens: int = 0,
        chars: int = 0,
        cost: float = 0.0,
        success: bool = True
    ):
        try:
            from app.config.settings import settings
            
            if not settings.GA_API_SECRET:
                return
            
            payload = {
                "client_id": user_id,
                "events": [{
                    "name": "api_usage",
                    "params": {
                        "provider": provider,
                        "tokens": tokens,
                        "chars": chars,
                        "cost": cost,
                        "success": success,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }]
            }
            
            async with httpx.AsyncClient() as client:
                await client.post(
                    "https://www.google-analytics.com/mp/collect",
                    params={
                        "measurement_id": settings.GA_MEASUREMENT_ID,
                        "api_secret": settings.GA_API_SECRET
                    },
                    json=payload,
                    timeout=5.0
                )
        except Exception as e:
            logger.error(f"Analytics tracking failed: {e}")
    
    @staticmethod
    def track_api_call_sync(user_id: str, provider: str, tokens: int = 0, chars: int = 0, cost: float = 0.0, success: bool = True):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(AnalyticsService.track_api_call(user_id, provider, tokens, chars, cost, success))
            else:
                loop.run_until_complete(AnalyticsService.track_api_call(user_id, provider, tokens, chars, cost, success))
        except:
            pass

