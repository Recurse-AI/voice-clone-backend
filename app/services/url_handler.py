import os
import re
import requests
from typing import Dict, Any, Optional
from app.config.settings import settings
from urllib.parse import urlparse
import yt_dlp

class URLHandler:
    def __init__(self):
        self.supported_platforms = ['youtube', 'youtu.be', 'instagram', 'facebook', 'tiktok', 'twitter', 'x.com']
    
    def detect_platform(self, url: str) -> str:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace('www.', '')
        
        if 'youtube.com' in domain or 'youtu.be' in domain:
            return 'youtube'
        elif 'instagram.com' in domain:
            return 'instagram'
        elif 'facebook.com' in domain or 'fb.watch' in domain:
            return 'facebook'
        elif 'tiktok.com' in domain:
            return 'tiktok'
        elif 'twitter.com' in domain or 'x.com' in domain:
            return 'twitter'
        elif any(x in domain for x in ['r2.dev', 'cloudflare', 's3', 'amazonaws']):
            return 'cloud_storage'
        return 'direct'
    
    def download_from_url(self, url: str, output_path: str) -> Dict[str, Any]:
        platform = self.detect_platform(url)
        
        if platform == 'cloud_storage' or platform == 'direct':
            return self._download_direct(url, output_path)
        else:
            return self._download_social_media(url, output_path, platform)
    
    def _download_direct(self, url: str, output_path: str) -> Dict[str, Any]:
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return {
                'success': True,
                'platform': 'direct',
                'local_path': output_path,
                'metadata': {}
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _download_social_media(self, url: str, output_path: str, platform: str) -> Dict[str, Any]:
        def _run_dl(fmt: str):
            opts = {
                'format': fmt,
                'outtmpl': output_path,
                'merge_output_format': 'mp4',
                'postprocessors': [
                    { 'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4' }
                ],
                'noplaylist': True,
                'retries': 10,
                'fragment_retries': 10,
                'concurrent_fragment_downloads': 5,
                'quiet': True,
                'no_warnings': True,
                'hls_prefer_native': True,
                'geo_bypass': True,
                'geo_bypass_country': 'US',
                'socket_timeout': 30,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.youtube.com/'
                },
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android', 'web'],
                        'player_skip': ['js', 'configs']
                    }
                }
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=True)
        
        max_h = settings.YTDLP_MAX_HEIGHT
        heights = []
        if isinstance(max_h, int) and max_h > 0:
            heights.append(max_h)
        for h in [720, 480]:
            if h not in heights and (not heights or h <= max(heights)):
                heights.append(h)
        # Build a small, ordered fallback chain across a few qualities
        format_chain = []
        for h in heights:
            format_chain.extend([
                f'bv*[height<={h}][vcodec^=avc1][ext=mp4]+ba[acodec^=mp4a]/b[height<={h}][vcodec^=avc1][ext=mp4]',
                f'bv*[height<={h}]+ba/b[height<={h}]',
            ])
        format_chain.append('bv*+ba/best')
        try:
            last_err = None
            info = None
            for fmt in format_chain:
                try:
                    info = _run_dl(fmt)
                    break
                except Exception as e:
                    last_err = e
                    continue
            if info is None:
                raise last_err or RuntimeError('Download failed')
            
            metadata = {
                'title': info.get('title'),
                'duration': info.get('duration'),
                'description': info.get('description'),
                'uploader': info.get('uploader'),
                'view_count': info.get('view_count'),
            }
            return {
                'success': True,
                'platform': platform,
                'local_path': output_path,
                'metadata': metadata
            }
        except Exception as e:
            return { 'success': False, 'error': str(e) }
