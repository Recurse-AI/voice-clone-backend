import os
import time
import logging
from app.services.dub.context import DubbingContext
from app.services.dub.utils.progress_tracker import ProgressTracker
from app.services.dub.whisperx_transcription import get_whisperx_transcription_service
from app.utils.audio import AudioUtils

logger = logging.getLogger(__name__)

class TranscriptionStep:
    @staticmethod
    def execute(context: DubbingContext):
        if context.manifest:
            TranscriptionStep._load_from_manifest(context)
        elif context.video_subtitle:
            TranscriptionStep._load_from_srt(context)
        else:
            TranscriptionStep._transcribe_audio(context)
    
    @staticmethod
    def _load_from_manifest(context: DubbingContext):
        logger.info("Using manifest override data")
        manifest_segments = context.manifest.get("segments", [])
        
        context.transcription_result = {
            "success": True,
            "segments": manifest_segments,
            "language": context.manifest.get("language", "auto")
        }
        context.transcript_id = context.manifest.get("transcript_id")
        
        logger.info(f"Loaded {len(manifest_segments)} segments from manifest")
    
    @staticmethod
    def _load_from_srt(context: DubbingContext):
        ProgressTracker.update_phase_progress(
            context.job_id, "transcription", 0.0, "Using provided SRT file"
        )
        
        srt_file_path = os.path.join(context.process_temp_dir, f"{context.job_id}.srt")
        if not os.path.exists(srt_file_path):
            raise Exception(f"SRT file not found at {srt_file_path}")
        
        from app.utils.srt_parser import parse_srt_to_whisperx_format
        from app.utils.db_sync_operations import get_dub_job_sync
        
        job_data = get_dub_job_sync(context.job_id)
        duration = job_data.get("duration") if job_data else None
        
        context.transcription_result = parse_srt_to_whisperx_format(srt_file_path, duration)
        
        if not context.transcription_result["success"]:
            raise Exception(context.transcription_result.get("error", "SRT parsing failed"))
        
        context.transcript_id = f"srt_{int(time.time())}"
        ProgressTracker.update_phase_progress(context.job_id, "transcription", 1.0, "SRT loaded")
    
    @staticmethod
    def _transcribe_audio(context: DubbingContext):
        ProgressTracker.update_phase_progress(
            context.job_id, "transcription", 0.0, "Starting audio transcription"
        )
        
        vocal_file_path = os.path.join(context.process_temp_dir, f"vocal_{context.job_id}.wav")
        
        if not os.path.exists(vocal_file_path):
            raise Exception(f"Vocal file not found at {vocal_file_path}")
        
        transcription_service = get_whisperx_transcription_service()
        context.transcription_result = transcription_service._transcribe_via_service_worker(
            vocal_file_path, context.source_video_language, context.job_id
        )
        
        if not context.transcription_result["success"]:
            AudioUtils.remove_temp_dir(folder_path=context.process_temp_dir)
            raise Exception(context.transcription_result.get("error", "Transcription failed"))
        
        context.transcript_id = f"whisperx_{int(time.time())}"
        ProgressTracker.update_phase_progress(
            context.job_id, "transcription", 1.0, "Transcription completed"
        )

