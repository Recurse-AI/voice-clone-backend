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
from app.services.simple_status_service import JobStatus

router = APIRouter()

logger = logging.getLogger(__name__)

async def regenerate_text_with_openai(original_text: str, target_language: str, custom_prompt: str) -> str:
    """Regenerate text using OpenAI with custom prompt"""
    try:
        from openai import OpenAI
        from app.config.settings import settings
        from app.services.language_service import language_service
        
        # Normalize target language to proper code
        target_lang_code = language_service.normalize_language_input(target_language)
        
        # Initialize OpenAI client
        if not settings.OPENAI_API_KEY:
            return f"[Error] OpenAI API key not configured"
        
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        system_prompt = f"""You are a professional dubbing script writer. 
Rewrite the given text in {target_lang_code} according to the specific instructions provided. 
Keep the meaning accurate but adapt the style based on the prompt. 
Return only the rewritten text, nothing else."""

        user_prompt = f"""Instructions: {custom_prompt}
Original text: {original_text}
Target language: {target_lang_code}
Rewrite this text following the instructions:"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        if result:
            return result
        else:
            return f"[Error] Empty response from OpenAI"
            
    except Exception as e:
        logger.error(f"OpenAI regeneration failed: {e}")
        return f"[Error] {str(e)}"

from app.services.dub.manifest_manager import manifest_manager
from app.services.dub.manifest_service import ensure_job_dir as _ensure_job_dir, write_json as _write_temp_json

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
        manifest = manifest_manager.load_manifest(manifest_url)
        normalized_manifest = manifest_manager._normalize_manifest(manifest)
        
        return SegmentsResponse(
            job_id=job_id, 
            segments=normalized_manifest.get("segments", []), 
            manifestUrl=manifest_url, 
            version=normalized_manifest.get("version"),
            target_language=normalized_manifest.get("target_language"),
            reference_ids=normalized_manifest.get("reference_ids", []),
            model_type=normalized_manifest.get("model_type", "normal")
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
    manifest = manifest_manager.load_manifest(manifest_url)
    id_to_edit = {e.id: e for e in request_body.segments}
    for seg in manifest.get("segments", []):
        if seg["id"] in id_to_edit:
            edit = id_to_edit[seg["id"]]
            if edit.dubbed_text is not None:
                dubbed_text = edit.dubbed_text.strip()
                if not dubbed_text and not seg.get("original_text", "").strip():
                    raise HTTPException(status_code=400, detail=f"Segment {seg['id']} cannot have empty text")
                seg["dubbed_text"] = dubbed_text if dubbed_text else seg.get("original_text", "")
            if edit.start is not None:
                seg["start"] = edit.start
            if edit.end is not None:
                seg["end"] = edit.end
            if edit.reference_id is not None:
                seg["reference_id"] = edit.reference_id
            seg["duration_ms"] = max(0, seg["end"] - seg["start"])
    manifest["version"] = int(manifest.get("version", 1)) + 1

    # Write and upload manifest back to R2
    job_dir = _ensure_job_dir(job_id)
    manifest_path = os.path.join(job_dir, f"manifest_{job_id}.json")
    _write_temp_json(manifest, manifest_path)
    from app.services.r2_service import R2Service
    r2 = R2Service()
    manifest_key_out = manifest_key
    if manifest_key:
        up_res = r2.upload_file(manifest_path, manifest_key, content_type="application/json")
        manifest_url_out = (up_res or {}).get("url") or manifest_url
    else:
        # If key not known, upload new copy and persist key for future overwrites
        manifest_filename = r2._sanitize_filename(os.path.basename(manifest_path))
        r2_key = r2.generate_file_path(job_id, "", manifest_filename)
        res = r2.upload_file(manifest_path, r2_key, content_type="application/json")
        manifest_url_out = res.get("url") if res.get("success") else manifest_url
        if res.get("success"):
            manifest_key_out = r2_key
    # Only update job details (no status/progress change during edits)
    current_details = (job.details or {}).copy()
    current_details.update({
        "review_required": True,
        "review_status": "in_progress",
        "segments_manifest_url": manifest_url_out,
        "segments_manifest_key": manifest_key_out,
        "edited_segments_version": (job.edited_segments_version or 0) + 1,
    })
    await dub_job_service.update_details(job_id, current_details)
    
    normalized_manifest = manifest_manager._normalize_manifest(manifest)
    
    return SegmentsResponse(
        job_id=job_id, 
        segments=normalized_manifest.get("segments", []), 
        manifestUrl=manifest_url_out, 
        version=normalized_manifest.get("version"),
        target_language=normalized_manifest.get("target_language")
    )

@router.post("/video-dub/{job_id}/segments/{segment_id}/regenerate", response_model=RegenerateSegmentResponse)
async def regenerate_segment(job_id: str, segment_id: str, request_body: RegenerateSegmentRequest, current_user = Depends(get_video_dub_user)):
    job = await dub_job_service.get_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    manifest_url = job.segments_manifest_url or (job.details or {}).get("segments_manifest_url")
    if not manifest_url:
        raise HTTPException(status_code=400, detail="No manifest available for this job")
    manifest = manifest_manager.load_manifest(manifest_url)
    seg = next((s for s in manifest.get("segments", []) if s.get("id") == segment_id), None)
    if not seg:
        raise HTTPException(status_code=404, detail="Segment not found")

    # Update text with OpenAI if prompt is provided, otherwise use existing logic
    dubbed_text = request_body.dubbed_text if request_body.dubbed_text is not None else seg.get("dubbed_text")
    
    # If custom prompt is provided, use OpenAI to regenerate text
    if request_body.prompt and dubbed_text:
        target_lang = request_body.target_language or manifest.get("target_language", "English")
        regenerated_text = await regenerate_text_with_openai(dubbed_text, target_lang, request_body.prompt)
        
        # Use regenerated text if successful, otherwise keep original
        if regenerated_text and not regenerated_text.startswith("[Error]"):
            dubbed_text = regenerated_text
    
    # Apply tone if provided
    if request_body.tone and dubbed_text:
        dubbed_text = f"({request_body.tone}) " + dubbed_text
    
    seg["dubbed_text"] = dubbed_text
    
    # Optionally update timings
    if request_body.start is not None:
        seg["start"] = request_body.start
    if request_body.end is not None:
        seg["end"] = request_body.end
    if (request_body.start is not None) or (request_body.end is not None):
        seg["duration_ms"] = max(0, seg["end"] - seg["start"])
    
    # Update reference_id if provided
    if request_body.reference_id is not None:
        seg["reference_id"] = request_body.reference_id
    
    # Store prompt and tone separately for future reference
    if request_body.prompt:
        seg["custom_prompt"] = request_body.prompt
        
    if request_body.tone:
        seg["tone"] = request_body.tone

    # If target_language provided, update manifest target_language (optional)
    if request_body.target_language:
        manifest["target_language"] = request_body.target_language

    # Persist manifest (version +1)
    manifest["version"] = int(manifest.get("version", 1)) + 1
    job_dir = _ensure_job_dir(job_id)
    manifest_path = os.path.join(job_dir, f"manifest_{job_id}.json")
    _write_temp_json(manifest, manifest_path)
    from app.services.r2_service import R2Service
    r2 = R2Service()
    # Try to overwrite existing manifest key if present
    manifest_key = job.segments_manifest_key or (job.details or {}).get("segments_manifest_key")
    if manifest_key:
        up = r2.upload_file(manifest_path, manifest_key, content_type="application/json")
        if up.get("success"):
            manifest_url = up.get("url")
    else:
        manifest_filename = r2._sanitize_filename(os.path.basename(manifest_path))
        r2_key = r2.generate_file_path(job_id, "", manifest_filename)
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