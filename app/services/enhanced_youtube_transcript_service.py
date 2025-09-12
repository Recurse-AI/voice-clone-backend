import logging
import os
import tempfile
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter

logger = logging.getLogger(__name__)

class EnhancedYouTubeTranscriptService:
    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 1
        self.proxy_list = self._load_proxy_list()
        
    def _load_proxy_list(self) -> List[str]:
        proxy_env = os.environ.get('YOUTUBE_PROXY_LIST', '')
        if proxy_env:
            return [p.strip() for p in proxy_env.split(',') if p.strip()]
        return []
    
    def _get_random_proxy(self) -> Optional[str]:
        if not self.proxy_list:
            return None
        return random.choice(self.proxy_list)
    
    def _extract_video_id(self, url_or_id: str) -> str:
        if 'youtube.com' in url_or_id or 'youtu.be' in url_or_id:
            import re
            patterns = [
                r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
                r'(?:embed\/)([0-9A-Za-z_-]{11})',
                r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
            ]
            for pattern in patterns:
                match = re.search(pattern, url_or_id)
                if match:
                    return match.group(1)
        return url_or_id
    
    def _try_original_api(self, video_id: str, languages: List[str]) -> Optional[str]:
        proxy = self._get_random_proxy()
        try:
            if proxy:
                logger.info(f"Using proxy: {proxy}")
                import os
                os.environ['HTTP_PROXY'] = proxy
                os.environ['HTTPS_PROXY'] = proxy
            
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id, languages=languages)
            formatter = SRTFormatter()
            return formatter.format_transcript(transcript)
            
        except Exception as e:
            logger.warning(f"YouTube API failed: {str(e)}")
            return None
        finally:
            if proxy:
                import os
                os.environ.pop('HTTP_PROXY', None)
                os.environ.pop('HTTPS_PROXY', None)
    
    def _extract_embedded_subtitles(self, video_url: str) -> Optional[str]:
        try:
            logger.info("Attempting to extract embedded subtitles")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts = {
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en', 'en-US', 'en-GB'],
                    'subtitlesformat': 'srt',
                    'skip_download': True,
                    'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                    'geo_bypass': True,
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['android', 'web'],
                            'player_skip': ['js']
                        }
                    }
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                
                for srt_file in Path(temp_dir).glob('*.srt'):
                    with open(srt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            logger.info("Successfully extracted embedded subtitles")
                            return content
                            
        except Exception as e:
            logger.warning(f"Embedded subtitle extraction failed: {str(e)}")
        
        return None
    
    def _transcribe_with_whisperx(self, video_url: str, target_language: str = 'en') -> Optional[str]:
        try:
            logger.info("Downloading audio and transcribing with WhisperX")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = os.path.join(temp_dir, 'audio.wav')
                
                ydl_opts = {
                    'format': 'bestaudio[ext=m4a]/bestaudio/best',
                    'outtmpl': audio_path.replace('.wav', '.%(ext)s'),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'quiet': True,
                    'no_warnings': True,
                    'geo_bypass': True,
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['android', 'web'],
                            'player_skip': ['js']
                        }
                    }
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                
                if os.path.exists(audio_path):
                    from app.services.dub.whisperx_transcription import WhisperXTranscriptionService
                    transcription_service = WhisperXTranscriptionService()
                    
                    result = transcription_service.transcribe_audio_file(
                        audio_path=audio_path,
                        language=target_language
                    )
                    
                    if result.get('success'):
                        return self._convert_whisperx_to_srt(result['segments'])
                        
        except Exception as e:
            logger.warning(f"WhisperX transcription failed: {str(e)}")
        
        return None
    
    def _convert_whisperx_to_srt(self, segments: List[Dict]) -> str:
        srt_content = []
        for i, segment in enumerate(segments, 1):
            start_time = self._seconds_to_srt_time(segment['start'])
            end_time = self._seconds_to_srt_time(segment['end'])
            text = segment['text'].strip()
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")
        
        return "\n".join(srt_content)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def get_transcript_srt(self, video_url_or_id: str, languages: List[str] = None, force_transcribe: bool = False) -> str:
        if not languages:
            languages = ['en']
        
        video_id = self._extract_video_id(video_url_or_id)
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        logger.info(f"Smart transcript processing: {video_id}")
        
        if not force_transcribe:
            for attempt in range(self.max_retries):
                result = self._try_original_api(video_id, languages)
                if result:
                    logger.info("✅ YouTube API success")
                    return result
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
            
            result = self._extract_embedded_subtitles(video_url)
            if result:
                logger.info("✅ Subtitle extraction success")
                return result
        
        result = self._transcribe_with_whisperx(video_url, languages[0])
        if result:
            logger.info("✅ Audio transcription success")
            return result
        
        raise Exception("All transcript methods failed")
    
    def get_available_transcripts(self, video_url_or_id: str) -> Dict[str, Any]:
        video_id = self._extract_video_id(video_url_or_id)
        
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            
            available_transcripts = []
            for transcript in transcript_list:
                available_transcripts.append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                })
            
            return {
                'video_id': video_id,
                'available_transcripts': available_transcripts,
                'total_count': len(available_transcripts),
                'fallback_available': True
            }
            
        except Exception as e:
            logger.warning(f"Could not get transcript list: {str(e)}")
            return {
                'video_id': video_id,
                'available_transcripts': [],
                'total_count': 0,
                'fallback_available': True,
                'note': 'Direct API unavailable, but fallback transcription available'
            }

enhanced_youtube_transcript_service = EnhancedYouTubeTranscriptService()
