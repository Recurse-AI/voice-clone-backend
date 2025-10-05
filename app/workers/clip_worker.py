import os
import tempfile
import asyncio
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.clip_service import ClipService
from app.repositories.clip_repository import ClipRepository
from app.config.settings import settings

logger = logging.getLogger(__name__)

def _process_single_segment(args):
    """Process a single segment with subtitle rendering (parallel worker)"""
    i, seg, base_clip, words, temp_dir, video_duration, job, service, user_id, job_id = args
    
    try:
        seg_start = max(0.0, float(seg["start"]))
        seg_end = min(video_duration, max(seg_start, float(seg["end"])))
        
        seg_clip = os.path.join(temp_dir, f"seg_{i+1}.mp4")
        service.cut_segment(base_clip, seg_start, seg_end, seg_clip)
        
        seg_words = [w for w in words if seg_start * 1000 <= w["start"] <= seg_end * 1000]
        seg_words_copy = []
        for w in seg_words:
            word_copy = w.copy()
            word_copy["start"] = max(0, w["start"] - int(seg_start * 1000))
            word_copy["end"] = max(0, w["end"] - int(seg_start * 1000))
            seg_words_copy.append(word_copy)
        
        seg_text = " ".join(w["text"] for w in seg_words)
        seg["text"] = seg_text
        seg["words"] = seg_words_copy
        
        subtitle_style = job.get("subtitle_style")
        preset = job.get("subtitle_preset", "reels")
        
        if seg_words_copy and subtitle_style and subtitle_style.lower() not in ["none", ""]:
            final_clip = os.path.join(temp_dir, f"final_{i+1}.mp4")
            service.render_subtitles(
                seg_clip, seg_words_copy, final_clip,
                style=subtitle_style,
                preset=preset,
                font=job.get("subtitle_font"),
                font_size=job.get("subtitle_font_size"),
                wpl=job.get("subtitle_wpl")
            )
            upload_path = final_clip
        else:
            resized_clip = os.path.join(temp_dir, f"resized_{i+1}.mp4")
            service.resize_video(seg_clip, resized_clip, preset=preset)
            upload_path = resized_clip
        
        r2_key = f"clips/{user_id}/{job_id}/seg_{i+1}.mp4"
        result = service.r2.upload_file(upload_path, r2_key, "video/mp4")
        if not result["success"]:
            raise RuntimeError(f"Upload failed: {result.get('error')}")
        
        seg["clip_url"] = result["url"]
        return (i, seg, True, None)
    
    except Exception as e:
        logger.error(f"Segment {i+1} processing failed: {e}")
        return (i, seg, False, str(e))

def _send_clip_completion_email(job_id: str, user_id: str):
    """Send completion email notification for clip jobs"""
    try:
        from app.utils.db_sync_operations import get_user_sync
        from app.utils.email_helper import send_email, create_job_completion_template
        from app.config.settings import settings

        user = get_user_sync(user_id)
        if not user:
            logger.warning(f"User {user_id} not found, skipping email")
            return

        logger.info(f"Sending completion email to user {user_id} ({user.get('email')}) for clip job {job_id}")

        download_urls = {"clips_url": f"{settings.FRONTEND_URL}/workspace/clips/results/{job_id}"}
        
        html_body = create_job_completion_template(
            user.get('name', 'User'), "clip", job_id, download_urls
        )

        subject = "✂️ Your Video Clips are Ready - ClearVocals"

        if not settings.EMAIL_HOST_USER or not settings.EMAIL_HOST_PASSWORD:
            logger.warning(f"⚠️ Email credentials not configured - skipping email for clip job {job_id}")
            return

        email_sent = send_email(
            sender_email=settings.EMAIL_HOST_USER,
            receiver_email=user.get('email'),
            subject=subject,
            body=html_body,
            password=settings.EMAIL_HOST_PASSWORD,
            is_html=True,
            raise_on_error=False
        )
        
        if email_sent:
            logger.info(f"✅ Completion email sent for clip job {job_id}")
        else:
            logger.error(f"❌ Email failed for clip job {job_id}")

    except Exception as e:
        logger.error(f"❌ Failed to send completion email for clip job {job_id}: {e}")

def process_clip_job(job_id: str, user_id: str):
    try:
        asyncio.run(_process_clip_job_async(job_id, user_id))
    except Exception as e:
        logger.error(f"Clip job {job_id} failed: {e}")
        asyncio.run(_update_failed(job_id, str(e)))

async def _update_failed(job_id: str, error: str):
    repo = ClipRepository()
    await repo.update(job_id, {"status": "failed", "error_message": error})
    
    # Refund credits on failure
    try:
        job = await repo.get_by_id(job_id)
        if job:
            from app.utils.job_utils import job_utils
            job_utils.refund_job_credits_sync(job_id, "clip", "job_failed")
    except Exception as e:
        logger.error(f"Failed to refund credits for clip job {job_id}: {e}")

async def _process_clip_job_async(job_id: str, user_id: str):
    repo = ClipRepository()
    service = ClipService()
    
    job = await repo.get_by_id(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        await repo.update_status(job_id, "downloading", 5)
        video_local = os.path.join(temp_dir, "source.mp4")
        download_result = service.download_video(job["video_url"], video_local)
        
        if not download_result.get("success"):
            raise RuntimeError(f"Download failed: {download_result.get('error')}")
        
        await repo.update_status(job_id, "trimming", 10)
        base_clip = os.path.join(temp_dir, "base.mp4")
        service.trim_video(video_local, job["start_time"], job["end_time"], base_clip)
        
        words = []
        if job.get("srt_url"):
            srt_local = os.path.join(temp_dir, "sub.srt")
            srt_download = service.download_video(job["srt_url"], srt_local)
            if not srt_download.get("success"):
                raise RuntimeError(f"SRT download failed: {srt_download.get('error')}")
            sentences = service.parse_srt(srt_local)
            words = service.srt_to_words(sentences)
            transcript_text = " ".join(s["text"] for s in sentences)
        else:
            await repo.update_status(job_id, "transcribing", 20)
            audio_local = os.path.join(temp_dir, "audio.wav")
            service.extract_audio(base_clip, audio_local)
            result = service.transcribe_assemblyai(audio_local)
            words = result.get("words", [])
            transcript_text = result.get("text", "")
        
        await repo.update(job_id, {"transcript": transcript_text})
        
        await repo.update_status(job_id, "segmenting", 40)
        video_duration = job["end_time"] - job["start_time"]
        seg_resp = service.segment_openai(transcript_text, job["expected_duration"], video_duration)
        segments = seg_resp.get("segments", [])
        overall = seg_resp.get("overall", {})
        
        valid_segments = []
        for s in segments:
            start, end = float(s["start"]), float(s["end"])
            if start >= video_duration or end > video_duration:
                logger.warning(f"Skipping segment [{start}-{end}s] - exceeds video duration {video_duration}s")
                continue
            if end - start < 1.0:
                logger.warning(f"Skipping segment [{start}-{end}s] - too short (< 1s)")
                continue
            valid_segments.append(s)
        
        if not valid_segments and segments:
            logger.warning("All segments filtered out, creating fallback segment from available content")
            valid_segments = [{
                "start": 0.0,
                "end": min(video_duration, max(s.get("end", video_duration) for s in segments)),
                "reason": "Fallback segment - original segments filtered",
                "ratings": segments[0].get("ratings", {})
            }]
        elif not valid_segments:
            logger.warning("No segments available, creating single segment from entire video")
            valid_segments = [{
                "start": 0.0,
                "end": video_duration,
                "reason": "Full video segment - no valid segments found",
                "ratings": {}
            }]
        
        segments = valid_segments
        
        await repo.update(job_id, {"overall_rating": overall})
        
        await repo.update_status(job_id, "rendering", 50)
        
        total_segs = len(segments)
        max_workers = settings.SUBTITLE_PARALLEL_WORKERS
        logger.info(f"🚀 Processing {total_segs} segments in parallel (max {max_workers} at a time)")
        
        # Prepare arguments for parallel processing
        tasks = [
            (i, seg, base_clip, words, temp_dir, video_duration, job, service, user_id, job_id)
            for i, seg in enumerate(segments)
        ]
        
        # Process segments in parallel
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_seg = {executor.submit(_process_single_segment, task): task[0] for task in tasks}
            
            for future in as_completed(future_to_seg):
                seg_idx = future_to_seg[future]
                try:
                    i, seg, success, error = future.result()
                    
                    if not success:
                        raise RuntimeError(f"Segment {i+1} failed: {error}")
                    
                    await repo.add_segment(job_id, seg)
                    completed_count += 1
                    
                    progress = 50 + int(completed_count / total_segs * 45)
                    await repo.update_status(job_id, "rendering", progress)
                    logger.info(f"✅ Segment {i+1}/{total_segs} completed")
                    
                except Exception as e:
                    logger.error(f"❌ Segment {seg_idx+1} failed: {e}")
                    raise
        
        await repo.update(job_id, {"status": "completed", "progress": 100, "completed_at": datetime.now(timezone.utc)})
        
        # Complete credit billing
        try:
            from app.utils.job_utils import job_utils
            job_utils.complete_job_billing_sync(job_id, "clip", user_id, 1.0)
        except Exception as e:
            logger.error(f"Failed to complete billing for clip job {job_id}: {e}")
        
        # Send completion email
        try:
            _send_clip_completion_email(job_id, user_id)
        except Exception as e:
            logger.error(f"Failed to send completion email for clip job {job_id}: {e}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
