"""
Job Response Service - Clean response formatting for user jobs
"""
from typing import List, Optional, Dict, Any
from app.models.dub_job import DubJob
from app.schemas import UserDubJob, FileInfo


class JobResponseService:
    """Service for formatting job responses cleanly"""
    
    @staticmethod
    def _get_file_type(filename: str) -> str:
        """Determine file type category based on filename"""
        filename_lower = filename.lower()
        if filename_lower.endswith('.mp4'):
            return 'video'
        elif filename_lower.endswith('.wav'):
            return 'audio'
        elif filename_lower.endswith('.srt'):
            return 'subtitle'
        elif filename_lower.endswith('.json'):
            if 'summary' in filename_lower:
                return 'summary'
            else:
                return 'metadata'
        else:
            return 'other'
    
    @staticmethod
    def _extract_result_url(job: DubJob) -> Optional[str]:
        """Extract actual result URL from job details"""
        result_url = job.result_url
        if job.details and isinstance(job.details, dict) and job.details.get("result_url"):
            result_url = job.details["result_url"]
        return result_url
    
    @staticmethod
    def _extract_files_info(job: DubJob) -> Optional[List[FileInfo]]:
        """Extract files information from job details"""
        if not job.details or not isinstance(job.details, dict):
            return None
        
        # Get folder upload data directly
        folder_upload = job.details.get("folder_upload")
        if not folder_upload or not isinstance(folder_upload, dict):
            return None
        
        files_info = []
        for filename, upload_data in folder_upload.items():
            if isinstance(upload_data, dict) and upload_data.get("success"):
                file_info = FileInfo(
                    filename=filename,
                    url=upload_data.get("url"),
                    size=upload_data.get("file_size"),
                    type=JobResponseService._get_file_type(filename)
                )
                files_info.append(file_info)
        
        # Sort files by type and name for better organization
        if files_info:
            files_info.sort(key=lambda x: (x.type, x.filename))
            return files_info
            
        return None
    
    @staticmethod
    def format_dub_job(job: DubJob) -> UserDubJob:
        """Format a single dub job for user response"""
        return UserDubJob(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            original_filename=job.original_filename,
            target_language=job.target_language,
            source_video_language=job.source_video_language,
            expected_speaker=job.expected_speaker,
            subtitle=job.subtitle,
            instrument=job.instrument,
            result_url=JobResponseService._extract_result_url(job),
            files=JobResponseService._extract_files_info(job),
            error=job.error,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
            completed_at=job.completed_at.isoformat() if job.completed_at else None
        )
    
    @staticmethod
    def format_dub_jobs(jobs: List[DubJob]) -> List[UserDubJob]:
        """Format multiple dub jobs for user response"""
        return [JobResponseService.format_dub_job(job) for job in jobs]


# Global service instance
job_response_service = JobResponseService()