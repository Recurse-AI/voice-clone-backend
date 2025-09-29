from fastapi import APIRouter, Query, HTTPException
import httpx
from typing import List, Optional
from app.config.settings import settings

router = APIRouter(prefix="/api/fish", tags=["fish-audio"])


@router.get("/models")
async def list_fish_models(
    page_size: int = Query(20, ge=1, le=100),
    page_number: int = Query(1, ge=1),
    title: Optional[str] = None,
    tag: Optional[List[str]] = Query(None),
    self_only: Optional[bool] = Query(None, alias="self"),
    author_id: Optional[str] = None,
    language: Optional[List[str]] = Query(None),
    title_language: Optional[str] = None,
):
    if not settings.FISH_AUDIO_API_KEY:
        raise HTTPException(status_code=500, detail="Fish Audio API key not configured")

    params: dict = {
        "page_size": page_size,
        "page_number": page_number,
    }
    if title:
        params["title"] = title
    if tag:
        for t in tag:
            params.setdefault("tag", []).append(t)
    if self_only is not None:
        params["self"] = str(self_only).lower()
    if author_id:
        params["author_id"] = author_id
    if language:
        for lang in language:
            params.setdefault("language", []).append(lang)
    if title_language:
        params["title_language"] = title_language

    headers = {"Authorization": f"Bearer {settings.FISH_AUDIO_API_KEY}"}

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get("https://api.fish.audio/model", headers=headers, params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {str(exc)}")

    return resp.json()


