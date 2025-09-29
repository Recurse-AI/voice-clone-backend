from fastapi import APIRouter, Query, HTTPException, UploadFile, File, Form, Header
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
    # Default sorting by score (supported by upstream API)
    params["sort_by"] = "score"
    params["order"] = "desc"

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


@router.post("/models")
async def create_fish_model(
    visibility: str = Form(...),
    type: str = Form(...),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    train_mode: str = Form("fast"),
    texts: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    voices: Optional[List[str]] = Form(None),
    enhance_audio_quality: Optional[bool] = Form(False),
    cover_image: Optional[UploadFile] = File(None),
    x_fish_audio_key: Optional[str] = Header(None, convert_underscores=False),
):
    api_key = x_fish_audio_key or settings.FISH_AUDIO_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="Fish Audio API key not configured")

    headers = {"Authorization": f"Bearer {api_key}"}

    form_data = {
        "visibility": visibility,
        "type": type,
        "title": title,
        "train_mode": train_mode,
        "enhance_audio_quality": str(bool(enhance_audio_quality)).lower(),
    }
    if description is not None:
        form_data["description"] = description
    if texts is not None:
        form_data["texts"] = texts
    if tags is not None:
        form_data["tags"] = tags
    if voices is not None:
        for v in voices:
            form_data.setdefault("voices", []).append(v)

    files = None
    if cover_image is not None:
        content = await cover_image.read()
        files = {"cover_image": (cover_image.filename, content, cover_image.content_type or "application/octet-stream")}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                "https://api.fish.audio/model",
                headers=headers,
                data=form_data,
                files=files,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {str(exc)}")

    return resp.json()

