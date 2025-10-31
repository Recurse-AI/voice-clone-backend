from fastapi import APIRouter, Query, HTTPException, UploadFile, File, Form, Header
import httpx
from typing import List, Optional
from app.config.settings import settings

router = APIRouter(prefix="/api/fish", tags=["fish-audio"])


@router.get("/models")
async def list_models(
    model_type: str = Query("fish_speech", regex="^(fish_speech|elevenlabs)$"),
    page_size: int = Query(20, ge=1, le=100),
    page_number: int = Query(1, ge=1),
    title: Optional[str] = None,
    tag: Optional[List[str]] = Query(None),
    self_only: Optional[bool] = Query(None, alias="self"),
    author_id: Optional[str] = None,
    language: Optional[List[str]] = Query(None),
    title_language: Optional[str] = None,
):
    if model_type == "elevenlabs":
        if not settings.ELEVENLABS_API_KEY:
            raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")
        
        from app.services.dub.elevenlabs_service import get_elevenlabs_service
        elevenlabs_service = get_elevenlabs_service()
        result = elevenlabs_service.get_all_voices()
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to fetch voices"))
        
        voices = result["items"]
        
        # Apply filters
        if title:
            voices = [v for v in voices if title.lower() in v.get("title", "").lower()]
        if self_only:
            voices = [v for v in voices if v.get("type") in ["cloned", "generated"]]
        
        # Pagination
        start = (page_number - 1) * page_size
        end = start + page_size
        paginated = voices[start:end]
        
        return {
            "items": paginated,
            "total": len(voices),
            "page_size": page_size,
            "page_number": page_number
        }
    
    # Fish Audio (default)
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


@router.get("/models/{reference_id}")
async def get_model_details(
    reference_id: str,
    model_type: str = Query("fish_speech", regex="^(fish_speech|elevenlabs)$")
):
    if model_type == "elevenlabs":
        if not settings.ELEVENLABS_API_KEY:
            raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")
        
        from app.services.dub.elevenlabs_service import get_elevenlabs_service
        elevenlabs_service = get_elevenlabs_service()
        result = elevenlabs_service.get_voice_details(reference_id)
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "Voice not found"))
        
        return result["data"]
    
    # Fish Audio (default)
    if not settings.FISH_AUDIO_API_KEY:
        raise HTTPException(status_code=500, detail="Fish Audio API key not configured")

    headers = {"Authorization": f"Bearer {settings.FISH_AUDIO_API_KEY}"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(f"https://api.fish.audio/model/{reference_id}", headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {str(exc)}")

    return resp.json()


@router.post("/models")
async def create_model(
    model_type: str = Form("fish_speech", regex="^(fish_speech|elevenlabs)$"),
    visibility: str = Form("private"),
    type: str = Form("tts"),
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
    if model_type == "elevenlabs":
        if not settings.ELEVENLABS_API_KEY:
            raise HTTPException(status_code=500, detail="ElevenLabs API key not configured")
        
        if not voices or len(voices) == 0:
            raise HTTPException(status_code=422, detail="At least one voice file is required")
        
        from app.services.dub.elevenlabs_service import get_elevenlabs_service
        elevenlabs_service = get_elevenlabs_service()
        
        voice_file = voices[0]
        audio_bytes = await voice_file.read()
        
        result = elevenlabs_service.create_voice_reference(audio_bytes, title)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to create voice"))
        
        from app.services.language_service import LanguageService
        
        return {
            "_id": result["voice_id"],
            "title": title,
            "cover_image": None,
            "languages": sorted(list(LanguageService.ELEVENLABS_V3_LANGUAGES)),
            "tags": tags or ["cloned", "custom"],
            "samples": [],
            "description": description or "",
            "like_count": 0,
            "task_count": 0,
            "author": {"nickname": "You"},
            "visibility": "private",
            "train_mode": "instant",
            "model_type": "elevenlabs"
        }
    
    if not voices or len(voices) == 0:
        raise HTTPException(status_code=422, detail="At least one voice file is required")
    
    from app.services.dub.fish_audio_api_service import get_fish_audio_api_service
    fish_service = get_fish_audio_api_service()
    
    voice_file = voices[0]
    audio_bytes = await voice_file.read()
    
    result = fish_service.create_voice_reference(audio_bytes, title)
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to create voice"))
    
    return {
        "_id": result["reference_id"],
        "title": title,
        "cover_image": None,
        "languages": ['en', 'zh', 'ja'],
        "tags": tags or ["cloned", "custom"],
        "samples": [],
        "description": description or "",
        "like_count": 0,
        "task_count": 0,
        "author": {"nickname": "You"},
        "visibility": "private",
        "train_mode": "fast",
        "model_type": "fish_speech"
    }

