from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class YouTubeTranscriptService:
    def __init__(self):
        from app.services.enhanced_youtube_transcript_service import enhanced_youtube_transcript_service
        self.enhanced_service = enhanced_youtube_transcript_service

    def get_transcript_srt(self, video_id: str, languages: list = None) -> str:
        if not languages:
            languages = ['en']
        
        logger.info(f"Smart transcript fetch for video: {video_id}")
        return self.enhanced_service.get_transcript_srt(video_id, languages)

    def get_available_transcripts(self, video_id: str) -> Dict[str, Any]:
        logger.info(f"Smart transcript info for video: {video_id}")
        return self.enhanced_service.get_available_transcripts(video_id)

youtube_transcript_service = YouTubeTranscriptService()
