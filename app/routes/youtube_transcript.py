from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from typing import Optional, List
import logging

from app.services.youtube_transcript_service import youtube_transcript_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/youtube-transcript/{video_id}")
async def get_youtube_transcript_srt(
    video_id: str,
    languages: Optional[List[str]] = Query(None, description="Preferred languages")
):
    try:
        lang_list = None
        if languages:
            lang_list = [lang.strip() for lang in languages if lang.strip()]

        logger.info(f"Requesting SRT transcript for video: {video_id}")

        srt_content = youtube_transcript_service.get_transcript_srt(
            video_id=video_id,
            languages=lang_list
        )

        return Response(
            content=srt_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=\"{video_id}_transcript.srt\"",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )

    except Exception as e:
        logger.error(f"Failed to get YouTube transcript: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch YouTube transcript: {str(e)}"
        )

@router.get("/youtube-transcript/{video_id}/info")
async def get_youtube_transcript_info(video_id: str):
    try:
        logger.info(f"Requesting transcript info for video: {video_id}")

        info = youtube_transcript_service.get_available_transcripts(video_id)

        return {
            "success": True,
            "video_id": info["video_id"],
            "available_transcripts": info["available_transcripts"],
            "total_count": info["total_count"]
        }

    except Exception as e:
        logger.error(f"Failed to get YouTube transcript info: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to get transcript info: {str(e)}"
        )
