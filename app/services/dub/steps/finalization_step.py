import os
import json
import time
import logging
import subprocess
import requests
from pathlib import Path
from typing import Dict, Any
from app.services.dub.context import DubbingContext
from app.services.dub.utils.progress_tracker import ProgressTracker
from app.services.dub.handlers.audio_handler import AudioHandler
from app.services.dub.handlers.manifest_handler import ManifestHandler
from app.services.dub.manifest_service import upload_process_dir_to_r2
from app.services.r2_service import R2Service
from app.services.dub.video_processor import VideoProcessor
from app.config.settings import settings
from app.config.constants import INSTRUMENT_DEFAULT_VOLUME

logger = logging.getLogger(__name__)

class FinalizationStep:
    @staticmethod
    def execute(context: DubbingContext) -> Dict[str, Any]:
        ProgressTracker.update_phase_progress(
            context.job_id, "final_processing", 0.0, "Reconstructing final audio..."
        )
        
        final_audio_path = AudioHandler.reconstruct_final_audio(context)
        
        ProgressTracker.update_phase_progress(
            context.job_id, "final_processing", 0.5, "Finalizing output files..."
        )
        
        subtitle_path = FinalizationStep._generate_srt_file(context)
        FinalizationStep._create_process_summary(context, final_audio_path, subtitle_path)
        
        try:
            ManifestHandler.build_and_save(context)
        except Exception as e:
            logger.error(f"Failed to save manifest for job {context.job_id}: {e}")
        
        video_result = FinalizationStep._create_video_if_available(context, final_audio_path)
        
        return FinalizationStep._upload_and_finalize(context, final_audio_path, video_result)
    
    @staticmethod
    def _generate_srt_file(context: DubbingContext) -> str:
        processor = VideoProcessor(temp_dir=context.process_temp_dir)
        subtitle_data = []
        
        for seg in context.segments:
            text = seg["dubbed_text"]
            start = seg["start"] / 1000.0
            end = seg["end"] / 1000.0
            subtitle_data.append({"start": start, "end": end, "text": text})
        
        srt_path = os.path.join(context.process_temp_dir, f"subtitles_{context.job_id}.srt")
        processor.create_srt_file(subtitle_data, srt_path)
        
        ass_path = os.path.join(context.process_temp_dir, f"subtitles_{context.job_id}.ass")
        processor.create_ass_file(subtitle_data, ass_path)
        
        return ass_path
    
    @staticmethod
    def _create_process_summary(context: DubbingContext, final_audio_path: str, subtitle_path: str):
        instrument_file = f"instrument_{context.job_id}.wav"
        vocal_file = f"vocal_{context.job_id}.wav"
        
        if not os.path.exists(os.path.join(context.process_temp_dir, instrument_file)):
            instrument_file = None
        
        if not os.path.exists(os.path.join(context.process_temp_dir, vocal_file)):
            vocal_file = None
        
        process_summary = {
            "success": True,
            "job_id": context.job_id,
            "segments_count": len(context.segments),
            "target_language": context.target_language,
            "final_audio_file": os.path.basename(final_audio_path) if final_audio_path else None,
            "subtitle_file": os.path.basename(subtitle_path) if subtitle_path else None,
            "instrument_file": instrument_file,
            "vocal_file": vocal_file,
            "final_video_file": None,
            "processing_timestamp": int(time.time()),
            "segments": context.segments,
            "transcript_id": context.transcript_id,
        }
        
        summary_filename = f"process_summary_{context.job_id}.json"
        summary_path = os.path.join(context.process_temp_dir, summary_filename).replace('\\', '/')
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(process_summary, f, ensure_ascii=False, indent=2)
            logger.info(f"Process summary saved: {summary_filename}")
        except Exception as e:
            logger.error(f"Failed to save process summary: {e}")
    
    @staticmethod
    def _create_video_if_available(context: DubbingContext, final_audio_path: str) -> dict:
        try:
            from app.utils.db_sync_operations import get_dub_job_sync
            
            job_data = get_dub_job_sync(context.job_id)
            if not job_data:
                return {"success": False, "error": "Job not found"}
            
            video_url = job_data.get("video_url")
            local_video_path = job_data.get("local_video_path")
            
            if not video_url and not local_video_path:
                logger.info(f"No video source available for job {context.job_id}, skipping video creation")
                return {"success": True, "skipped": True}
            
            ProgressTracker.update_phase_progress(
                context.job_id, "final_processing", 0.7, "Creating video result..."
            )
            
            return FinalizationStep._process_video(
                context, video_url, local_video_path, final_audio_path
            )
        except Exception as e:
            logger.error(f"Failed to create video for job {context.job_id}: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def _get_duration(file_path: str) -> float:
        try:
            from app.utils.ffmpeg_helper import get_ffmpeg_path
            ffmpeg = get_ffmpeg_path()
            cmd = [ffmpeg, "-i", file_path, "-hide_banner", "-f", "null", "-"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            for line in result.stderr.split('\n'):
                if 'Duration:' in line:
                    time_str = line.split('Duration:')[1].split(',')[0].strip()
                    h, m, s = time_str.split(':')
                    return float(h) * 3600 + float(m) * 60 + float(s)
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to get duration: {e}")
            return 0.0
    
    @staticmethod
    def _process_video(context: DubbingContext, video_url: str, local_video_path: str, audio_path: str) -> dict:
        try:
            output_dir_path = Path(context.process_temp_dir)
            downloaded_video_path = output_dir_path / "source_video.mp4"
            
            if not downloaded_video_path.exists():
                video_found = False
                for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v']:
                    original_path = output_dir_path / f"original{ext}"
                    if original_path.exists():
                        import shutil
                        shutil.copy2(original_path, downloaded_video_path)
                        logger.info(f"Using original video file: {original_path}")
                        video_found = True
                        break
                
                if not video_found:
                    if local_video_path and os.path.exists(local_video_path):
                        import shutil
                        shutil.copy2(local_video_path, downloaded_video_path)
                        logger.info(f"Copied local video file: {local_video_path}")
                    elif video_url:
                        response = requests.get(video_url)
                        response.raise_for_status()
                        with open(downloaded_video_path, "wb") as f:
                            f.write(response.content)
                        logger.info(f"Downloaded video from URL")
                    else:
                        return {"success": False, "error": "No valid video source available"}
            
            video_duration = FinalizationStep._get_duration(str(downloaded_video_path))
            audio_duration = FinalizationStep._get_duration(audio_path)
            
            duration_mode = "shortest" if audio_duration > video_duration else "longest"
            logger.info(f"Video: {video_duration:.2f}s, Audio: {audio_duration:.2f}s → Using '{duration_mode}' mode")
            
            final_video_path = output_dir_path / f"final_video_{context.job_id}.mp4"
            
            subtitle_path = None
            if context.add_subtitle_to_video:
                subtitle_file = f"subtitles_{context.job_id}.ass"
                potential_subtitle_path = output_dir_path / subtitle_file
                if potential_subtitle_path.exists():
                    subtitle_path = str(potential_subtitle_path)
            
            cmd = ["ffmpeg", "-y"]
            if settings.FFMPEG_USE_GPU:
                cmd.extend(["-hwaccel", "cuda"])
            
            cmd.extend(["-i", str(downloaded_video_path), "-i", audio_path])
            
            if context.instrument_url:
                instrument_path = output_dir_path / "instrument_audio.mp3"
                response = requests.get(context.instrument_url)
                response.raise_for_status()
                with open(instrument_path, "wb") as f:
                    f.write(response.content)
                
                cmd.extend(["-i", str(instrument_path)])
                cmd.extend([
                    "-filter_complex", f"[1:a]volume=2.0[dub];[2:a]volume={INSTRUMENT_DEFAULT_VOLUME}[inst];[dub][inst]amix=inputs=2:duration={duration_mode}[out]",
                    "-map", "0:v", "-map", "[out]"
                ])
            else:
                cmd.extend(["-map", "0:v", "-map", "1:a", "-filter:a", "volume=2.0"])
            
            if subtitle_path:
                video_codec = 'h264_nvenc' if settings.FFMPEG_USE_GPU else 'libx264'
                preset = 'fast' if settings.FFMPEG_USE_GPU else 'veryfast'
                escaped_path = subtitle_path.replace(chr(92), '/').replace(':', r'\:')
                subtitle_filter = f"ass='{escaped_path}'"
                cmd.extend(["-vf", subtitle_filter, "-c:v", video_codec, "-preset", preset, "-crf", "23"])
            else:
                cmd.extend(["-c:v", "copy"])
            
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])
            
            if duration_mode == "shortest":
                cmd.append("-shortest")
            
            cmd.extend(["-movflags", "+faststart", str(final_video_path)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return {"success": False, "error": "Video processing failed"}
            
            r2_storage = R2Service()
            video_r2_key = r2_storage.generate_file_path(context.job_id, "processed", f"final_video_{context.job_id}.mp4")
            upload_result = r2_storage.upload_file(str(final_video_path), video_r2_key, "video/mp4")
            
            if upload_result["success"]:
                return {
                    "success": True,
                    "video_url": upload_result["url"],
                    "video_filename": f"final_video_{context.job_id}.mp4",
                    "local_path": str(final_video_path),
                    "subtitles_added": bool(subtitle_path)
                }
            else:
                return {"success": False, "error": "Failed to upload video to R2"}
        
        except Exception as e:
            logger.error(f"Video processing failed for job {context.job_id}: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def _upload_and_finalize(context: DubbingContext, final_audio_path: str, video_result: dict = None) -> dict:
        ProgressTracker.update_phase_progress(
            context.job_id, "upload", 0.0, "Uploading and finalizing..."
        )
        
        exclude_files = []
        for filename in os.listdir(context.process_temp_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
                exclude_files.append(filename)
            elif filename.startswith('segment_') and filename.endswith('.wav'):
                exclude_files.append(filename)
        
        if exclude_files:
            logger.info(f"Skipping files during folder upload: {exclude_files}")
        
        r2_storage = R2Service()
        folder_upload_result, manifest_url, manifest_key = upload_process_dir_to_r2(
            context.job_id, context.process_temp_dir, r2_storage, exclude_files=exclude_files
        )
        
        logger.info("Dubbed processing completed successfully")
        
        result_url = None
        video_upload = None
        video_error = None
        
        if video_result:
            if video_result.get("success"):
                result_url = video_result.get("video_url")
                video_upload = {
                    "success": True,
                    "url": video_result.get("video_url"),
                    "filename": video_result.get("video_filename")
                }
                logger.info(f"Video result included for job {context.job_id}: {result_url}")
            elif not video_result.get("skipped"):
                video_error = video_result.get("error")
                logger.warning(f"Video creation failed for job {context.job_id}: {video_error}")
        
        return {
            "success": True,
            "job_id": context.job_id,
            "result_url": result_url,
            "result_urls": {},
            "folder_upload": folder_upload_result,
            "manifest_url": manifest_url,
            "manifest_key": manifest_key,
            "video_upload": video_upload,
            "video_error": video_error
        }

