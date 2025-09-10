from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
from typing import Dict, Any
import logging
import time

logger = logging.getLogger(__name__)

class YouTubeTranscriptService:
    def __init__(self):
        self.api = YouTubeTranscriptApi()
        self.max_retries = 3
        self.retry_delay = 1

    def get_transcript_srt(self, video_id: str, languages: list = None) -> str:
        if not languages:
            languages = ['en']

        logger.info(f"Fetching transcript for video ID: {video_id}")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                transcript = self.api.fetch(video_id, languages=languages)
                formatter = SRTFormatter()
                srt_content = formatter.format_transcript(transcript)

                logger.info(f"Successfully fetched SRT transcript for video: {video_id}")
                return srt_content

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for video {video_id}: {str(e)}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All {self.max_retries} attempts failed for video {video_id}: {str(e)}")

        raise Exception(f"Failed to fetch YouTube transcript after {self.max_retries} attempts: {str(last_error)}")

    def get_available_transcripts(self, video_id: str) -> Dict[str, Any]:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                transcript_list = self.api.list(video_id)

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
                    'total_count': len(available_transcripts)
                }

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for transcript info {video_id}: {str(e)}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All {self.max_retries} attempts failed for transcript info {video_id}: {str(e)}")

        raise Exception(f"Failed to get available transcripts after {self.max_retries} attempts: {str(last_error)}")

youtube_transcript_service = YouTubeTranscriptService()
