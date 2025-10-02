from fastapi import APIRouter, Query, HTTPException, UploadFile, File, Form, Header
import os
import tempfile
import httpx
from typing import List, Optional
from app.config.settings import settings
from app.services.r2_service import R2Service
from starlette.concurrency import run_in_threadpool

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

    async with httpx.AsyncClient(timeout=60.0) as client:
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
    texts: Optional[List[str]] = Form(None),
    tags: Optional[List[str]] = Form(None),
    voices: List[UploadFile] = File(...),
    enhance_audio_quality: Optional[bool] = Form(False),
    cover_image: Optional[UploadFile] = File(None),
    x_fish_audio_key: Optional[str] = Header(None, convert_underscores=False),
):
    api_key = x_fish_audio_key or settings.FISH_AUDIO_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="Fish Audio API key not configured")

    headers = {"Authorization": f"Bearer {api_key}"}

    # If model is requested as public but no cover image is provided, make it private
    normalized_visibility = (visibility or "").lower()
    if normalized_visibility == "public" and cover_image is None:
        normalized_visibility = "private"

    # Normalize list fields
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(tags, str):
        tags = [tags]

    # Build one multipart list including form fields and files
    multipart_parts: List[tuple] = []
    multipart_parts.append(("visibility", (None, normalized_visibility)))
    multipart_parts.append(("type", (None, type)))
    multipart_parts.append(("title", (None, title)))
    multipart_parts.append(("train_mode", (None, train_mode)))
    multipart_parts.append(("enhance_audio_quality", (None, str(bool(enhance_audio_quality)).lower())))
    if description is not None:
        multipart_parts.append(("description", (None, description)))
    if texts:
        for t in texts:
            multipart_parts.append(("texts", (None, t)))
    if tags:
        for tg in tags:
            multipart_parts.append(("tags", (None, tg)))

    if not voices or len(voices) == 0:
        raise HTTPException(status_code=422, detail="At least one voice file is required")

    files = []
    r2_service = R2Service()
    r2_job_id = r2_service.generate_job_id()
    for v in voices:
        content = await v.read()
        multipart_parts.append(("voices", (v.filename, content, v.content_type or "audio/wav")))
        try:
            suffix = os.path.splitext(v.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            sanitized_filename = r2_service._sanitize_filename(v.filename)
            r2_key = r2_service.generate_file_path(r2_job_id, "", sanitized_filename)
            r2_service.upload_file(tmp_path, r2_key, v.content_type or "audio/wav")
        finally:
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    if cover_image is not None:
        content = await cover_image.read()
        multipart_parts.append(("cover_image", (cover_image.filename, content, cover_image.content_type or "application/octet-stream")))
    if not multipart_parts:
        multipart_parts = None

    def _do_post():
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "https://api.fish.audio/model",
                headers=headers,
                files=multipart_parts,
            )
            response.raise_for_status()
            return response

    try:
        resp = await run_in_threadpool(_do_post)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {str(exc)}")

    return resp.json()

