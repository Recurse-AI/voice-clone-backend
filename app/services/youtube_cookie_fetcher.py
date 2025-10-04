import logging
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from app.config.settings import settings

logger = logging.getLogger(__name__)


class YouTubeCookieFetcher:
    def __init__(self):
        self.cookie_file = Path("youtube_cookies.txt")
        self.email = settings.YOUTUBE_EMAIL
        self.password = settings.YOUTUBE_PASSWORD
        self.profile_dir = Path("tmp/browser_profile")
    
    async def fetch_cookies(self, manual_mode: bool = False) -> dict:
        if not self.email or not self.password:
            return {"success": False, "error": "YouTube credentials not set in .env"}
        
        try:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            
            async with async_playwright() as p:
                context = await p.chromium.launch_persistent_context(
                    str(self.profile_dir),
                    headless=not manual_mode,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-features=IsolateOrigins,site-per-process'
                    ],
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                page = context.pages[0] if context.pages else await context.new_page()
                
                logger.info("Opening YouTube...")
                await page.goto("https://www.youtube.com")
                await asyncio.sleep(3)
                
                page_content = await page.content()
                is_logged_in = 'ucbcb' in page_content or await page.locator('button[aria-label*="Google Account"]').count() > 0
                
                if not is_logged_in:
                    if manual_mode:
                        logger.warning("⚠️ Not logged in. Browser will stay open for MANUAL login.")
                        logger.warning("Please login in the browser window, then press Enter here...")
                        input("Press Enter after logging in manually...")
                    else:
                        logger.info("Not logged in. Starting auto-login...")
                        
                        await page.goto("https://accounts.google.com/ServiceLogin?service=youtube", wait_until="networkidle")
                        await asyncio.sleep(3)
                        
                        logger.info(f"Entering email: {self.email}")
                        await page.fill('input[type="email"]', self.email)
                        await asyncio.sleep(0.5)
                        await page.press('input[type="email"]', 'Enter')
                        await asyncio.sleep(5)
                        
                        logger.info("Entering password...")
                        await page.wait_for_selector('input[type="password"]', timeout=10000)
                        await asyncio.sleep(1)
                        await page.fill('input[type="password"]', self.password)
                        await asyncio.sleep(0.5)
                        await page.press('input[type="password"]', 'Enter')
                        
                        logger.warning("⚠️ If you get security notification on your device, approve it now...")
                        await asyncio.sleep(15)
                        
                        if "challenge" in page.url or "verification" in page.url:
                            logger.warning("Security challenge detected. Waiting 30 seconds for approval...")
                            await asyncio.sleep(30)
                        
                        logger.info("✅ Login process completed")
                
                logger.info("Extracting cookies...")
                cookies = await context.cookies()
                youtube_cookies = [c for c in cookies if 'youtube.com' in c.get('domain', '')]
                
                if not youtube_cookies:
                    await context.close()
                    return {"success": False, "error": "No YouTube cookies found - login may have failed"}
                
                with open(self.cookie_file, 'w') as f:
                    f.write("# Netscape HTTP Cookie File\n")
                    for cookie in youtube_cookies:
                        f.write(f"{cookie['domain']}\tTRUE\t{cookie['path']}\t"
                              f"{'TRUE' if cookie['secure'] else 'FALSE'}\t"
                              f"{int(cookie.get('expires', 0))}\t{cookie['name']}\t{cookie['value']}\n")
                
                await context.close()
                logger.info(f"✅ Cookies saved: {len(youtube_cookies)} YouTube cookies")
                return {"success": True, "cookie_count": len(youtube_cookies)}
                
        except Exception as e:
            logger.error(f"Cookie fetch error: {e}")
            return {"success": False, "error": str(e)}


youtube_cookie_fetcher = YouTubeCookieFetcher()

