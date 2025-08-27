from fastapi import APIRouter, HTTPException, Depends
import logging
import os
from app.schemas import (
    SegmentsResponse,
    SaveEditsRequest,
    RegenerateSegmentRequest,
    RegenerateSegmentResponse,
)
from app.dependencies.share_token_auth import get_video_dub_user
from app.services.dub_job_service import dub_job_service
from app.utils.unified_status_manager import ProcessingStatus

router = APIRouter()

logger = logging.getLogger(__name__)

from app.services.dub.manifest_service import load_manifest as _load_manifest_json, ensure_job_dir as _ensure_job_dir
from app.services.dub.manifest_service import write_json as _write_temp_json

@router.get("/video-dub/{job_id}/segments", response_model=SegmentsResponse)
async def get_segments(job_id: str, current_user = Depends(get_video_dub_user)):
    job = await dub_job_service.get_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check job status first
    if job.status != "awaiting_review" and job.status != "reviewing":
        raise HTTPException(status_code=400, detail=f"Job is in {job.status} status. Segments are only available for review jobs.")
    
    manifest_url = job.segments_manifest_url or (job.details or {}).get("segments_manifest_url")
    if not manifest_url:
        raise HTTPException(
            status_code=400, 
            detail=f"No manifest available for job {job_id}. Status: {job.status}, Manifest URL: {job.segments_manifest_url}"
        )
    
    try:
        manifest = _load_manifest_json(manifest_url)
        return SegmentsResponse(
            job_id=job_id, 
            segments=manifest.get("segments", []), 
            manifestUrl=manifest_url, 
            version=manifest.get("version"),
            target_language=manifest.get("target_language")
        )
    except Exception as e:
        logger.error(f"Failed to load manifest from {manifest_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load manifest: {str(e)}")

@router.put("/video-dub/{job_id}/segments", response_model=SegmentsResponse)
async def save_segment_edits(job_id: str, request_body: SaveEditsRequest, current_user = Depends(get_video_dub_user)):
    job = await dub_job_service.get_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    manifest_url = job.segments_manifest_url or (job.details or {}).get("segments_manifest_url")
    manifest_key = job.segments_manifest_key or (job.details or {}).get("segments_manifest_key")
    if not manifest_url:
        raise HTTPException(status_code=400, detail="No manifest available for this job")
    manifest = _load_manifest_json(manifest_url)
    id_to_edit = {e.id: e for e in request_body.segments}
    for seg in manifest.get("segments", []):
        if seg["id"] in id_to_edit:
            edit = id_to_edit[seg["id"]]
            if edit.dubbed_text is not None:
                seg["dubbed_text"] = edit.dubbed_text
            if edit.start is not None:
                seg["start"] = edit.start
            if edit.end is not None:
                seg["end"] = edit.end
            seg["duration_ms"] = max(0, seg["end"] - seg["start"])
    manifest["version"] = int(manifest.get("version", 1)) + 1

    # Write and upload manifest back to R2
    job_dir = _ensure_job_dir(job_id)
    manifest_path = os.path.join(job_dir, f"dubbing_manifest_{job_id}.json")
    _write_temp_json(manifest, manifest_path)
    from app.services.r2_service import get_r2_service
    r2 = get_r2_service()
    if manifest_key:
        up_res = r2.upload_file(manifest_path, manifest_key, content_type="application/json")
        manifest_url_out = (up_res or {}).get("url") or manifest_url
    else:
        # If key not known, upload new copy
        r2_key = r2.generate_file_path(job_id, "", os.path.basename(manifest_path))
        res = r2.upload_file(manifest_path, r2_key, content_type="application/json")
        manifest_url_out = res.get("url") if res.get("success") else manifest_url
    # Update DB details and status reviewing
    await dub_job_service.update_job_status(job_id, ProcessingStatus.REVIEWING.value, 79, details={
        "review_required": True,
        "review_status": "in_progress",
        "segments_manifest_url": manifest_url_out,
        "edited_segments_version": (job.edited_segments_version or 0) + 1,
    })
    return SegmentsResponse(
        job_id=job_id, 
        segments=manifest.get("segments", []), 
        manifestUrl=manifest_url_out, 
        version=manifest.get("version"),
        target_language=manifest.get("target_language")
    )

@router.post("/video-dub/{job_id}/segments/{segment_id}/regenerate", response_model=RegenerateSegmentResponse)
async def regenerate_segment(job_id: str, segment_id: str, request_body: RegenerateSegmentRequest, current_user = Depends(get_video_dub_user)):
    job = await dub_job_service.get_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    manifest_url = job.segments_manifest_url or (job.details or {}).get("segments_manifest_url")
    if not manifest_url:
        raise HTTPException(status_code=400, detail="No manifest available for this job")
    manifest = _load_manifest_json(manifest_url)
    seg = next((s for s in manifest.get("segments", []) if s.get("id") == segment_id), None)
    if not seg:
        raise HTTPException(status_code=404, detail="Segment not found")

    # Update text with OpenAI if prompt is provided, otherwise use existing logic
    dubbed_text = request_body.dubbed_text if request_body.dubbed_text is not None else seg.get("dubbed_text")
    
    # If custom prompt is provided, use OpenAI to regenerate text
    if request_body.prompt and dubbed_text:
        from app.services.openai_service import get_openai_service
        service = get_openai_service()
        
        target_lang = request_body.target_language or manifest.get("target_language", "Bengali")
        regenerated_text = service.regenerate_text_with_prompt(dubbed_text, target_lang, request_body.prompt)
        
        # Use regenerated text if successful, otherwise fallback
        if not regenerated_text.startswith("[Error]"):
            dubbed_text = regenerated_text
        else:
            dubbed_text = f"[{request_body.prompt}] " + dubbed_text
    
    # Apply tone if provided
    if request_body.tone and dubbed_text:
        dubbed_text = f"({request_body.tone}) " + dubbed_text
    
    seg["dubbed_text"] = dubbed_text
    
    # Store prompt separately for future reference
    if request_body.prompt:
        seg["custom_prompt"] = request_body.prompt
    
    # Store tone separately for future reference  
    if request_body.tone:
        seg["tone"] = request_body.tone

    # If target_language provided, update manifest target_language (optional)
    if request_body.target_language:
        manifest["target_language"] = request_body.target_language

    # Persist manifest (version +1)
    manifest["version"] = int(manifest.get("version", 1)) + 1
    job_dir = _ensure_job_dir(job_id)
    manifest_path = os.path.join(job_dir, f"dubbing_manifest_{job_id}.json")
    _write_temp_json(manifest, manifest_path)
    from app.services.r2_service import get_r2_service
    r2 = get_r2_service()
    # Try to overwrite existing manifest key if present
    manifest_key = job.segments_manifest_key or (job.details or {}).get("segments_manifest_key")
    if manifest_key:
        up = r2.upload_file(manifest_path, manifest_key, content_type="application/json")
        if up.get("success"):
            manifest_url = up.get("url")
    else:
        r2_key = r2.generate_file_path(job_id, "", os.path.basename(manifest_path))
        up = r2.upload_file(manifest_path, r2_key, content_type="application/json")
        if up.get("success"):
            manifest_url = up.get("url")

    # Return updated segment + manifest info
    return RegenerateSegmentResponse(
    success=True,
    message="Segment text updated for re-dub",
    job_id=job_id,
    segment_id=segment_id,
    manifestUrl=manifest_url,
    version=manifest.get("version", 1),
    segment=seg
)