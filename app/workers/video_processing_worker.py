"""
Video Processing Worker - Background task processing for video operations
Handles complete video processing workflow asynchronously
"""
import logging
import json
import subprocess
import requests
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from app.services.simple_status_service import status_service, JobStatus
from app.schemas import TimelineAudioSegment, VideoProcessingOptions
from app.config.settings import settings

logger = logging.getLogger(__name__)


def process_video_task(task_data: dict):
    """Background worker function for video processing"""
    job_id = task_data.get("job_id")
    
    try:
        logger.info(f"🎬 Starting video processing job {job_id} - GPU: {settings.FFMPEG_USE_GPU}")
        
        # Update status to processing
        status_service.update_status(
            job_id, "video_processing", JobStatus.PROCESSING, 10,
            {"message": "Starting video processing"}
        )
        
        # Extract parameters from task_data
        video_url = task_data.get("video_url")
        dubbed_audio_url = task_data.get("dubbed_audio_url")
        instrument_audio_url = task_data.get("instrument_audio_url")
        timeline_audio = task_data.get("timeline_audio")
        subtitle_url = task_data.get("subtitle_url")
        options_str = task_data.get("options", "{}")
        
        # Parse options
        options_dict = json.loads(options_str) if options_str != "{}" else {}
        processing_options = VideoProcessingOptions(**options_dict)
        
        # Parse timeline audio if provided
        timeline_segments: List[TimelineAudioSegment] = []
        if timeline_audio:
            try:
                timeline_data = json.loads(timeline_audio)
                timeline_segments = [TimelineAudioSegment(**seg) for seg in timeline_data]
                logger.info(f"📋 Parsed {len(timeline_segments)} timeline segments")
            except Exception as e:
                _fail_job(job_id, f"Timeline parsing failed: {str(e)}", "INVALID_TIMELINE")
                return
        
        # Validate: At least one input required
        has_input = any([
            video_url,
            dubbed_audio_url,
            instrument_audio_url, 
            len(timeline_segments) > 0
        ])
        
        if not has_input:
            _fail_job(job_id, "At least one input required", "NO_INPUT")
            return
        
        # Update progress
        status_service.update_status(
            job_id, "video_processing", JobStatus.PROCESSING, 20,
            {"message": "Processing inputs"}
        )
        
        # Create permanent directory for processing (instead of temp)
        output_dir = Path("tmp") / "processed" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Using output directory: {output_dir}")
        
        # Process the video using URL
        result = _process_video_complete(
            output_dir, job_id, video_url, dubbed_audio_url, 
            instrument_audio_url, timeline_segments, subtitle_url, 
            processing_options
        )
            
        if result["success"]:
            # Schedule auto-cleanup after 5 minutes
            _schedule_cleanup(job_id, output_dir)
            
            # 🆕 ADD R2 UPLOAD HERE - Before completion status
            try:
                from app.services.r2_service import R2Service
                r2_service = R2Service()
                
                # Generate R2 key for processed video/audio
                r2_key = r2_service.generate_file_path(job_id, "processed", result["output_filename"])
                
                # Upload final processed file to R2
                upload_result = r2_service.upload_file(
                    str(output_dir / result["output_filename"]),  
                    r2_key,
                    r2_service._get_content_type(result["output_filename"])
                )
                
                if upload_result["success"]:
                    # Use R2 URL instead of local download URL
                    r2_download_url = upload_result["url"]
                    logger.info(f"✅ File uploaded to R2: {r2_key}")
                else:
                    # Fallback to local URL if R2 upload fails
                    r2_download_url = result["download_url"]
                    logger.error(f"❌ R2 upload failed: {upload_result.get('error')}")
                    
            except Exception as e:
                # Fallback to local URL if anything goes wrong
                r2_download_url = result["download_url"]
                logger.error(f"❌ R2 integration failed: {e}")
            
            # Complete the job with R2 URL
            status_service.update_status(
                job_id, "video_processing", JobStatus.COMPLETED, 100,
                {
                    "message": "Video processing completed successfully",
                    "download_url": r2_download_url,  # ← Now points to R2
                    "output_filename": result["output_filename"],
                    "output_type": result["output_type"],
                    "file_size_mb": result["file_size_mb"],
                    "duration_seconds": result["duration_seconds"]
                }
            )
            logger.info(f"✅ Video processing completed: {job_id}")
        else:
            _fail_job(job_id, result["error"], result["error_code"])
                
    except subprocess.TimeoutExpired:
        _fail_job(job_id, "Processing exceeded 2-hour limit", "TIMEOUT")
    except Exception as e:
        logger.error(f"💥 Video processing job {job_id} failed: {str(e)}")
        _fail_job(job_id, str(e), "UNEXPECTED_ERROR")


def _process_video_complete(output_path: Path, job_id: str, video_url: Optional[str], 
                           dubbed_audio_url: Optional[str], instrument_audio_url: Optional[str],
                           timeline_segments: List[TimelineAudioSegment], subtitle_url: Optional[str],
                           processing_options: VideoProcessingOptions) -> Dict[str, Any]:
    try:
        
        status_service.update_status(job_id, "video_processing", JobStatus.PROCESSING, 30, {"message": "Downloading input files"})
        
        video_path = None
        dubbed_audio_path = None
        instrument_audio_path = None
        subtitle_path = None
        
        if video_url:
            # Download video from URL
            video_path = output_path / "video_source.mp4"
            _download_file(video_url, video_path)
        
        if dubbed_audio_url:
            dubbed_audio_path = output_path / "dubbed_audio.mp3"
            _download_file(dubbed_audio_url, dubbed_audio_path)
        
        if instrument_audio_url:
            instrument_audio_path = output_path / "instrument_audio.mp3"
            _download_file(instrument_audio_url, instrument_audio_path)
        
        if subtitle_url:
            subtitle_path = output_path / "subtitles.srt"
            _download_file(subtitle_url, subtitle_path)
        
        status_service.update_status(job_id, "video_processing", JobStatus.PROCESSING, 50, {"message": "Processing audio"})
        
        final_audio_path = None
        
        if len(timeline_segments) > 0:
            reconstructed_audio = _reconstruct_timeline_audio(timeline_segments, output_path, job_id)
            if reconstructed_audio:
                final_audio_path = _mix_audio_files(reconstructed_audio, instrument_audio_path, output_path, processing_options.instrument_volume, job_id)
        elif dubbed_audio_path or instrument_audio_path:
            final_audio_path = _mix_audio_files(dubbed_audio_path, instrument_audio_path, output_path, processing_options.instrument_volume, job_id)
        
        status_service.update_status(job_id, "video_processing", JobStatus.PROCESSING, 70, {"message": "Processing video"})
        
        final_output_path = None
        output_type = "audio" if processing_options.audio_only else "video"
        
        if processing_options.audio_only:
            if final_audio_path:
                audio_format = processing_options.audio_format or "mp3"
                final_output_path = output_path / f"final_audio_{job_id}.{audio_format}"
                if audio_format != "mp3":
                    _convert_audio_format(final_audio_path, final_output_path, audio_format)
                else:
                    import shutil
                    shutil.copy2(final_audio_path, final_output_path)
        elif video_path:
            final_output_path = _process_video_with_audio(video_path, final_audio_path, subtitle_path, output_path, processing_options, job_id)
        
        if not final_output_path:
            return {"success": False, "error": "No output generated", "error_code": "NO_OUTPUT"}
        
        status_service.update_status(job_id, "video_processing", JobStatus.PROCESSING, 90, {"message": "Finalizing result"})
        
        output_filename = final_output_path.name
        file_size_mb = round(final_output_path.stat().st_size / (1024 * 1024), 2)
        duration_seconds = _get_media_duration(final_output_path)
        
        # Create direct download URL
        download_url = f"/api/video/download/{job_id}/{output_filename}"
        
        return {
            "success": True,
            "download_url": download_url,
            "output_filename": output_filename,
            "output_type": output_type,
            "file_size_mb": file_size_mb,
            "duration_seconds": duration_seconds
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "error_code": "PROCESSING_ERROR"}


def _download_file(url: str, file_path: Path) -> None:
    import requests
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)


def _reconstruct_timeline_audio(segments: List[TimelineAudioSegment], temp_path: Path, job_id: str) -> Optional[Path]:
    try:
        segment_files = []
        for i, segment in enumerate(segments):
            segment_path = temp_path / f"segment_{i}.mp3"
            _download_file(segment.audio_url, segment_path)
            segment_files.append((segment_path, segment.start, segment.end))
        
        total_duration = max(seg.end for seg in segments)
        output_path = temp_path / f"timeline_audio_{job_id}.mp3"
        
        cmd = ["ffmpeg", "-y"]
        if settings.FFMPEG_USE_GPU:
            cmd.extend(["-hwaccel", "cuda"])
        cmd.extend(["-f", "lavfi", "-i", f"anullsrc=duration={total_duration/1000}:sample_rate=44100"])
        
        input_count = 1
        delay_filters = []
        mix_inputs = ["[0:a]"]
        
        for segment_path, start_ms, end_ms in segment_files:
            cmd.extend(["-i", str(segment_path)])
            delay_ms = start_ms
            delay_filter = f"[{input_count}:a]volume=1.0,adelay={delay_ms}|{delay_ms}[delayed{input_count}]"
            delay_filters.append(delay_filter)
            mix_inputs.append(f"[delayed{input_count}]")
            input_count += 1
        
        if len(delay_filters) > 0:
            all_filters = ";".join(delay_filters)
            mix_filter = f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest[out]"
            filter_complex = f"{all_filters};{mix_filter}"
            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", "[out]"])
        else:
            cmd.extend(["-map", "0:a"])
        
        cmd.extend(["-c:a", "mp3", str(output_path)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return None
        
        return output_path
    except Exception:
        return None


def _mix_audio_files(dubbed_path: Optional[Path], instrument_path: Optional[Path], temp_path: Path, instrument_volume: float, job_id: str) -> Optional[Path]:
    try:
        output_path = temp_path / f"mixed_audio_{job_id}.mp3"
        
        if dubbed_path and instrument_path:
            cmd = ["ffmpeg", "-y"]
            if settings.FFMPEG_USE_GPU:
                cmd.extend(["-hwaccel", "cuda"])
            cmd.extend([
                "-i", str(dubbed_path),
                "-i", str(instrument_path),
                "-filter_complex", f"[0:a]volume=2.0[dub];[1:a]volume={instrument_volume}[inst];[dub][inst]amix=inputs=2:duration=longest[out]",
                "-map", "[out]",
                str(output_path)
            ])
        elif dubbed_path:
            cmd = ["ffmpeg", "-y"]
            if settings.FFMPEG_USE_GPU:
                cmd.extend(["-hwaccel", "cuda"])
            cmd.extend(["-i", str(dubbed_path), "-filter:a", "volume=2.0", "-c:a", "mp3", str(output_path)])
        elif instrument_path:
            cmd = ["ffmpeg", "-y"]
            if settings.FFMPEG_USE_GPU:
                cmd.extend(["-hwaccel", "cuda"])
            cmd.extend(["-i", str(instrument_path), "-filter:a", f"volume={instrument_volume}", "-c:a", "mp3", str(output_path)])
        else:
            return None
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return None
        
        return output_path
    except Exception:
        return None


def _process_video_with_audio(video_path: Path, audio_path: Optional[Path], subtitle_path: Optional[Path], temp_path: Path, options: VideoProcessingOptions, job_id: str) -> Optional[Path]:
    try:
        video_ext = video_path.suffix
        output_path = temp_path / f"final_video_{job_id}{video_ext}"
        
        # GPU acceleration for video encoding
        video_codec = 'h264_nvenc' if settings.FFMPEG_USE_GPU else 'libx264'
        preset = 'fast' if settings.FFMPEG_USE_GPU else 'medium'
        
        cmd = ["ffmpeg", "-y"]
        if settings.FFMPEG_USE_GPU:
            cmd.extend(["-hwaccel", "cuda"])
        cmd.extend(["-i", str(video_path)])
        
        if audio_path:
            cmd.extend(["-i", str(audio_path)])
        
        filters = []
        
        if subtitle_path and options.include_subtitles:
            subtitle_str = str(subtitle_path).replace('\\', '/').replace(':', '\\:')
            filters.append(f"subtitles='{subtitle_str}'")
        
        if options.resolution == "720p":
            filters.append("scale=1280:720")
        elif options.resolution == "4k":
            filters.append("scale=3840:2160")
        
        if filters:
            cmd.extend(["-vf", ",".join(filters)])
        
        if audio_path:
            cmd.extend(["-c:a", "aac", "-map", "0:v", "-map", "1:a"])
        else:
            cmd.extend(["-c:a", "copy"])
        
        if options.quality == "high":
            cmd.extend(["-c:v", video_codec, "-crf", "20", "-preset", preset])
        else:
            cmd.extend(["-c:v", video_codec, "-crf", "23", "-preset", preset])
        
        if options.format == "webm":
            # For WebM, use VP9 codec
            webm_codec = "libvpx-vp9" if not settings.FFMPEG_USE_GPU else "libvpx-vp9"
            cmd.extend(["-f", "webm", "-c:v", webm_codec])
        elif options.format == "mov":
            cmd.extend(["-f", "mov"])
        
        cmd.append(str(output_path))
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            return None
        
        return output_path
    except Exception:
        return None


def _convert_audio_format(input_path: Path, output_path: Path, audio_format: str) -> None:
    codec_map = {"mp3": "mp3", "wav": "pcm_s16le", "aac": "aac"}
    codec = codec_map.get(audio_format.lower(), "mp3")
    
    cmd = ["ffmpeg", "-y"]
    if settings.FFMPEG_USE_GPU:
        cmd.extend(["-hwaccel", "cuda"])
    cmd.extend(["-i", str(input_path), "-c:a", codec, str(output_path)])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        import shutil
        shutil.copy2(input_path, output_path)


def _get_media_duration(file_path: Path) -> Optional[float]:
    try:
        cmd = ["ffprobe", "-v", "quiet"]
        if settings.FFMPEG_USE_GPU:
            cmd.extend(["-hwaccel", "cuda"])
        cmd.extend(["-show_entries", "format=duration", "-of", "csv=p=0", str(file_path)])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def _schedule_cleanup(job_id: str, output_dir: Path):
    from app.utils.cleanup_utils import cleanup_utils
    cleanup_utils.schedule_auto_cleanup(job_id, 30)
    logger.info(f"⏰ Scheduled cleanup for job {job_id} in 30 minutes")


def _fail_job(job_id: str, error: str, error_code: str):
    """Helper function to mark job as failed"""
    status_service.update_status(
        job_id, "video_processing", JobStatus.FAILED, 0,
        {
            "message": "Video processing failed",
            "error": error,
            "error_code": error_code
        }
    )
    logger.error(f"❌ Video processing job {job_id} failed: {error}")
