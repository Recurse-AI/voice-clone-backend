import os
import tempfile
import asyncio
import logging
from datetime import datetime, timezone
from app.services.clip_service import ClipService
from app.repositories.clip_repository import ClipRepository

logger = logging.getLogger(__name__)

def process_clip_job(job_id: str, user_id: str):
    try:
        asyncio.run(_process_clip_job_async(job_id, user_id))
    except Exception as e:
        logger.error(f"Clip job {job_id} failed: {e}")
        asyncio.run(_update_failed(job_id, str(e)))

async def _update_failed(job_id: str, error: str):
    repo = ClipRepository()
    await repo.update(job_id, {"status": "failed", "error_message": error})

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
        segments = valid_segments
        
        await repo.update(job_id, {"overall_rating": overall})
        
        await repo.update_status(job_id, "rendering", 50)
        
        total_segs = len(segments)
        for i, seg in enumerate(segments):
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
            await repo.add_segment(job_id, seg)
            
            progress = 50 + int((i + 1) / total_segs * 45)
            await repo.update_status(job_id, "rendering", progress)
        
        await repo.update(job_id, {"status": "completed", "progress": 100, "completed_at": datetime.now(timezone.utc)})
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
