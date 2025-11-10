from fastapi import APIRouter, HTTPException, Depends
import logging
import os
from typing import List
from app.schemas import (
    SegmentsResponse,
    SaveEditsRequest,
    RegenerateSegmentRequest,
    RegenerateSegmentResponse,
    SegmentItem,
)
from app.dependencies.share_token_auth import get_video_dub_user
from app.services.dub_job_service import dub_job_service
from app.services.simple_status_service import JobStatus

router = APIRouter()

logger = logging.getLogger(__name__)

async def translate_with_context(manifest: dict, current_seg: dict, edited_original_text: str, target_language: str) -> str:
    """Translate edited original text using contextual translation with surrounding segments"""
    try:
        from app.services.dub.ai_segmentation_service import get_ai_segmentation_service
        
        segments = manifest.get("segments", [])
        current_index = next((i for i, s in enumerate(segments) if s.get("id") == current_seg.get("id")), -1)
        
        if current_index == -1:
            logger.warning("Current segment not found in manifest segments")
            return current_seg.get("dubbed_text", "")
        
        # Get surrounding segments (upper 2 + lower 2)
        start_idx = max(0, current_index - 2)
        end_idx = min(len(segments), current_index + 3)
        context_segments = segments[start_idx:end_idx]
        
        # Create segments list with original_text
        context_texts = []
        target_segment_index = current_index - start_idx  # Index within context segments
        
        for seg in context_segments:
            if seg.get("id") == current_seg.get("id"):
                context_texts.append(edited_original_text)  # Use edited text
            else:
                context_texts.append(seg.get("original_text", ""))
        
        # Use contextual translation
        ai_service = get_ai_segmentation_service()
        translated = ai_service.translate_contextual_segment(
            segments=context_texts,
            target_segment_index=target_segment_index,
            target_language=target_language,
            source_language="auto"
        )
        
        logger.info(f"Contextual translation completed for segment {current_seg.get('id')}")
        return translated
        
    except Exception as e:
        logger.error(f"Contextual translation failed: {str(e)}")
        return current_seg.get("dubbed_text", "")

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
            vocal_url=normalized_manifest.get("vocal_audio_url"),
            instrument_url=normalized_manifest.get("instrument_audio_url"),
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
    
    # Create mapping of request segments by ID
    request_segments_by_id = {e.id: e for e in request_body.segments}
    request_segment_ids = set(request_segments_by_id.keys())
    
    # Process segments: keep only those in request, update their data
    updated_segments = []
    for seg in manifest.get("segments", []):
        if seg["id"] in request_segment_ids:
            # Segment exists in request - update it
            edit = request_segments_by_id[seg["id"]]
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
            if edit.speaker is not None:
                seg["speaker"] = edit.speaker
            if edit.original_text is not None:
                logger.info(f"Updating original_text for segment {seg['id']}: '{edit.original_text}'")
                seg["original_text"] = edit.original_text
            seg["duration_ms"] = max(0, seg["end"] - seg["start"])
            updated_segments.append(seg)
        # Segments not in request_segment_ids are effectively deleted
    
    # Reassign sequential IDs based on array index
    for idx, seg in enumerate(updated_segments):
        seg["id"] = f"seg_{idx+1:03d}"
        seg["segment_index"] = idx
    
    # Replace manifest segments with updated ones
    manifest["segments"] = updated_segments
    manifest["version"] = int(manifest.get("version", 1)) + 1
    logger.info(f"Saving manifest with {len(updated_segments)} segments")
    logger.info(f"Manifest: {manifest}")
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

@router.put("/video-dub/{job_id}/segments/update", response_model=SegmentsResponse)
async def update_segments(job_id: str, segments: List[SegmentItem], current_user = Depends(get_video_dub_user)):
    """Update manifest with new segments data and upload to R2"""
    job = await dub_job_service.get_job(job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    manifest_url = job.segments_manifest_url or (job.details or {}).get("segments_manifest_url")
    manifest_key = job.segments_manifest_key or (job.details or {}).get("segments_manifest_key")
    
    if not manifest_url:
        raise HTTPException(status_code=400, detail="No manifest available for this job")
    
    # Load existing manifest
    manifest = manifest_manager.load_manifest(manifest_url)
    
    # Convert SegmentItem list to manifest format
    updated_segments = []
    for seg in segments:
        segment_data = {
            "id": seg.id,
            "segment_index": seg.segment_index,
            "start": seg.start,
            "end": seg.end,
            "duration_ms": seg.duration_ms,
            "original_text": seg.original_text,
            "dubbed_text": seg.dubbed_text,
            "voice_cloned": False,
            "original_audio_file": seg.original_audio_file,
            "cloned_audio_file": None,
            "speaker": seg.speaker,
            "reference_id": seg.reference_id
        }
        updated_segments.append(segment_data)
    
    # Update manifest with new segments
    manifest["segments"] = updated_segments
    manifest["version"] = int(manifest.get("version", 1)) + 1
    
    logger.info(f"Updating manifest with {len(updated_segments)} segments")
    
    # Write and upload manifest to R2
    job_dir = _ensure_job_dir(job_id)
    manifest_path = os.path.join(job_dir, f"manifest_{job_id}.json")
    _write_temp_json(manifest, manifest_path)
    
    from app.services.r2_service import R2Service
    r2 = R2Service()
    
    if manifest_key:
        up_res = r2.upload_file(manifest_path, manifest_key, content_type="application/json")
        manifest_url_out = (up_res or {}).get("url") or manifest_url
    else:
        # Upload new copy and persist key
        manifest_filename = r2._sanitize_filename(os.path.basename(manifest_path))
        r2_key = r2.generate_file_path(job_id, "", manifest_filename)
        res = r2.upload_file(manifest_path, r2_key, content_type="application/json")
        manifest_url_out = res.get("url") if res.get("success") else manifest_url
        if res.get("success"):
            manifest_key = r2_key
    
    # Update job details with new manifest info
    current_details = (job.details or {}).copy()
    current_details.update({
        "segments_manifest_url": manifest_url_out,
        "segments_manifest_key": manifest_key,
        "updated_segments_version": (job.edited_segments_version or 0) + 1,
    })
    await dub_job_service.update_details(job_id, current_details)
    
    normalized_manifest = manifest_manager._normalize_manifest(manifest)
    
    return SegmentsResponse(
        job_id=job_id,
        segments=normalized_manifest.get("segments", []),
        manifestUrl=manifest_url_out,
        version=normalized_manifest.get("version"),
        target_language=normalized_manifest.get("target_language"),
        reference_ids=normalized_manifest.get("reference_ids", []),
        vocal_url=normalized_manifest.get("vocal_audio_url"),
        instrument_url=normalized_manifest.get("instrument_audio_url")
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
    
    # If original_text is provided (user edited), use contextual translation
    if request_body.original_text and request_body.original_text != seg.get("original_text"):
        target_lang = request_body.target_language or manifest.get("target_language", "English")
        dubbed_text = await translate_with_context(manifest, seg, request_body.original_text, target_lang)
    
    # Apply tone if provided
    if request_body.tone and dubbed_text:
        dubbed_text = f"({request_body.tone}) " + dubbed_text
    
    # Update original_text if provided
    if request_body.original_text:
        seg["original_text"] = request_body.original_text
    
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

    if request_body.speaker is not None:
        seg["speaker"] = request_body.speaker
    
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