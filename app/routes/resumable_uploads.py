from fastapi import APIRouter, HTTPException, UploadFile, Header, Response, Query
from pydantic import BaseModel
import os
import hashlib
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class CreateResp(BaseModel):
    upload_id: str
    size: int | None = None


def _root_dir() -> str:
    from app.config.settings import settings
    base = os.path.join(settings.TEMP_DIR, "resumable")
    os.makedirs(base, exist_ok=True)
    return base


def _meta_path(upload_id: str) -> str:
    return os.path.join(_root_dir(), upload_id, ".meta")


def _file_path(upload_id: str) -> str:
    return os.path.join(_root_dir(), upload_id, "data.bin")


def _read_offset(upload_id: str) -> int:
    try:
        with open(_meta_path(upload_id), "r") as f:
            content = f.read().strip()
            return int(content) if content else 0
    except FileNotFoundError:
        return 0


def _write_offset(upload_id: str, offset: int) -> None:
    os.makedirs(os.path.join(_root_dir(), upload_id), exist_ok=True)
    with open(_meta_path(upload_id), "w") as f:
        f.write(str(offset))


@router.post("/api/resumable/uploads", response_model=CreateResp)
async def create_upload(upload_id: str = Query(...), size: int | None = Query(default=None)):
    os.makedirs(os.path.join(_root_dir(), upload_id), exist_ok=True)
    if not os.path.exists(_file_path(upload_id)):
        open(_file_path(upload_id), "wb").close()
    _write_offset(upload_id, 0)
    return CreateResp(upload_id=upload_id, size=size)


@router.head("/api/resumable/uploads/{upload_id}")
async def head_upload(upload_id: str):
    offset = _read_offset(upload_id)
    resp = Response(status_code=200)
    resp.headers["Upload-Offset"] = str(offset)
    return resp


@router.patch("/api/resumable/uploads/{upload_id}")
async def patch_upload(
    upload_id: str,
    chunk: UploadFile,
    upload_offset: int = Header(..., alias="Upload-Offset"),
):
    current = _read_offset(upload_id)
    if upload_offset != current:
        raise HTTPException(status_code=409, detail=f"Expected offset {current}", headers={"Upload-Offset": str(current)})
    data = await chunk.read()
    # Simple max size guard (optional): rely on FE or create settings if needed
    with open(_file_path(upload_id), "ab") as f:
        f.write(data)
    new_offset = current + len(data)
    _write_offset(upload_id, new_offset)
    resp = Response(status_code=204)
    resp.headers["Upload-Offset"] = str(new_offset)
    return resp


@router.post("/api/resumable/uploads/{upload_id}/finalize")
async def finalize_upload(upload_id: str, sha256: str | None = Query(default=None), original_filename: str | None = Query(default=None), associate_dub_job: bool = Query(default=False)):
    path = _file_path(upload_id)
    if not os.path.exists(path):
        raise HTTPException(404, "upload not found")

    if sha256:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != sha256:
            raise HTTPException(400, "checksum mismatch")

    final_name = original_filename or f"{upload_id}.bin"
    final_dir = os.path.join(_root_dir(), upload_id)
    final_path = os.path.join(final_dir, final_name)
    if os.path.abspath(path) != os.path.abspath(final_path):
        try:
            os.replace(path, final_path)
        except Exception:
            # fallback copy if cross-device
            with open(path, "rb") as src, open(final_path, "wb") as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    dst.write(chunk)

    video_url = None
    if associate_dub_job and final_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        try:
            from app.services.dub_service import dub_service
            from app.services.r2_service import R2Service
            r2_service = R2Service()
            sanitized_final_name = r2_service._sanitize_filename(final_name)
            r2_key = f"videos/{upload_id}/{sanitized_final_name}"
            upload_result = r2_service.upload_file(final_path, r2_key, "video/mp4")
            video_url = upload_result.get("url") if upload_result.get("success") else None
            dub_service.associate_video(upload_id, final_path, video_url)
        except Exception as e:
            logger.error(f"Failed to associate video for dub job {upload_id}: {e}")

    return {"success": True, "file_path": final_path, "video_url": video_url}


