import logging
from app.services.simple_status_service import status_service, JobStatus

logger = logging.getLogger(__name__)

class ProgressTracker:
    PROGRESS_PHASES = {
        "initialization": {"start": 0, "end": 10},
        "separation": {"start": 25, "end": 45},
        "transcription": {"start": 45, "end": 60},
        "dubbing": {"start": 60, "end": 75},
        "review_prep": {"start": 75, "end": 80},
        "reviewing": {"start": 80, "end": 81},
        "voice_cloning": {"start": 81, "end": 91},
        "final_processing": {"start": 91, "end": 96},
        "upload": {"start": 96, "end": 100}
    }
    
    @staticmethod
    def update_status(job_id: str, status: JobStatus, progress: int, details: dict):
        try:
            status_service.update_status(job_id, "dub", status, progress, details)
        except Exception as e:
            logger.error(f"Failed to update status for {job_id}: {e}")
    
    @staticmethod
    def update_phase_progress(job_id: str, phase: str, sub_progress: float, message: str):
        if phase not in ProgressTracker.PROGRESS_PHASES:
            logger.warning(f"Unknown phase '{phase}' for job {job_id}")
            return
        
        sub_progress = min(1.0, max(0.0, sub_progress))
        start = ProgressTracker.PROGRESS_PHASES[phase]["start"]
        end = ProgressTracker.PROGRESS_PHASES[phase]["end"]
        progress = min(100, max(0, start + int((end - start) * sub_progress)))
        
        phase_status = {
            "separation": JobStatus.SEPARATING,
            "transcription": JobStatus.TRANSCRIBING,
            "reviewing": JobStatus.REVIEWING,
            "upload": JobStatus.UPLOADING,
        }.get(phase, JobStatus.PROCESSING)
        
        ProgressTracker.update_status(
            job_id, phase_status, progress,
            {"message": message, "phase": phase, "sub_progress": sub_progress}
        )

