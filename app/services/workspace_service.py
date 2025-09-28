import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.config.database import dub_jobs_collection, separation_jobs_collection
from app.schemas import WorkspaceStats, JobSummary

logger = logging.getLogger(__name__)


class WorkspaceService:
    """Optimized service for workspace status and summary data"""
    
    async def get_workspace_status(self, user_id: str, recent_limit: int = 5) -> Dict[str, Any]:
        """
        Get comprehensive workspace status with minimal database queries
        """
        try:
            # Get all stats and recent jobs in parallel
            stats_data = await self._get_user_statistics(user_id)
            recent_dubs = await self._get_recent_jobs(dub_jobs_collection, user_id, recent_limit)
            recent_separations = await self._get_recent_jobs(separation_jobs_collection, user_id, recent_limit)
            
            return {
                "stats": stats_data,
                "recent_dubs": recent_dubs,
                "recent_separations": recent_separations
            }
            
        except Exception as e:
            logger.error(f"Failed to get workspace status for user {user_id}: {e}")
            return {
                "stats": self._empty_stats(),
                "recent_dubs": [],
                "recent_separations": []
            }
    
    async def _get_user_statistics(self, user_id: str) -> WorkspaceStats:
        """Get user statistics with aggregation pipeline for efficiency"""
        try:
            # Use aggregation pipeline for both collections
            dub_stats = await self._get_collection_stats(dub_jobs_collection, user_id)
            separation_stats = await self._get_collection_stats(separation_jobs_collection, user_id)
            
            return WorkspaceStats(
                total_dubs=dub_stats["total"],
                total_separations=separation_stats["total"],
                total_completed_dubs=dub_stats["completed"],
                total_completed_separations=separation_stats["completed"],
                total_processing_dubs=dub_stats["processing"],
                total_processing_separations=separation_stats["processing"]
            )
            
        except Exception as e:
            logger.error(f"Failed to get user statistics: {e}")
            return self._empty_stats()
    
    async def _get_collection_stats(self, collection, user_id: str) -> Dict[str, int]:
        """Get statistics for a specific collection using aggregation"""
        try:
            pipeline = [
                {"$match": {"user_id": user_id}},
                {
                    "$group": {
                        "_id": None,
                        "total": {"$sum": 1},
                        "completed": {
                            "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                        },
                        "processing": {
                            "$sum": {
                                "$cond": [
                                    {
                                        "$in": [
                                            "$status", 
                                            ["pending", "processing", "downloading", "separating", "transcribing", "uploading", "awaiting_review", "reviewing"]
                                        ]
                                    }, 
                                    1, 
                                    0
                                ]
                            }
                        }
                    }
                }
            ]
            
            result = await collection.aggregate(pipeline).to_list(1)
            
            if result:
                stats = result[0]
                return {
                    "total": stats.get("total", 0),
                    "completed": stats.get("completed", 0),
                    "processing": stats.get("processing", 0)
                }
            else:
                return {"total": 0, "completed": 0, "processing": 0}
                
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"total": 0, "completed": 0, "processing": 0}
    
    async def _get_recent_jobs(self, collection, user_id: str, limit: int) -> List[JobSummary]:
        """Get recent jobs with complete data"""
        try:
            # Check if this is separation collection
            is_separation = collection.name == 'separation_jobs'
            
            if is_separation:
                projection = {
                    "job_id": 1,
                    "status": 1,
                    "progress": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "completed_at": 1,
                    "original_filename": 1,
                    "audio_url": 1,
                    "vocal_url": 1,
                    "instrument_url": 1,
                    "error": 1,
                    "_id": 0
                }
            else:
                # Dub jobs projection (existing)
                projection = {
                    "job_id": 1,
                    "status": 1,
                    "progress": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "completed_at": 1,
                    "original_filename": 1,
                    "target_language": 1,
                    "source_video_language": 1,
                    "result_url": 1,
                    "details": 1,
                    "error": 1,
                    "_id": 0
                }

            cursor = collection.find(
                {"user_id": user_id},
                projection
            ).sort("created_at", -1).limit(limit)
            
            jobs = []
            async for job_data in cursor:
                if is_separation:
                    # Separation job data
                    job_summary_data = {
                        "job_id": job_data["job_id"],
                        "status": job_data["status"],
                        "progress": job_data.get("progress", 0),
                        "original_filename": job_data.get("original_filename"),
                        "audio_url": job_data.get("audio_url"),
                        "vocal_url": job_data.get("vocal_url"),
                        "instrument_url": job_data.get("instrument_url"),
                        "error": job_data.get("error"),
                        "created_at": job_data["created_at"].isoformat(),
                        "updated_at": job_data["updated_at"].isoformat() if job_data.get("updated_at") else None,
                        "completed_at": job_data["completed_at"].isoformat() if job_data.get("completed_at") else None,
                        # Dub fields (None for separation)
                        "target_language": None,
                        "source_video_language": None,
                        "result_url": None,
                        "files": None
                    }
                else:
                    # Dub job data (existing logic)
                    files_info = None
                    if job_data.get("details") and isinstance(job_data["details"], dict):
                        folder_upload = job_data["details"].get("folder_upload")
                        if folder_upload and isinstance(folder_upload, dict):
                            files_info = []
                            for filename, upload_data in folder_upload.items():
                                if isinstance(upload_data, dict) and upload_data.get("success"):
                                    file_info = {
                                        "filename": filename,
                                        "url": upload_data.get("url"),
                                        "size": upload_data.get("file_size") or upload_data.get("size"),
                                        "type": self._get_file_type(filename)
                                    }
                                    files_info.append(file_info)
                    
                    result_url = job_data.get("result_url")
                    if not result_url and job_data.get("details", {}).get("result_url"):
                        result_url = job_data["details"]["result_url"]

                    job_summary_data = {
                        "job_id": job_data["job_id"],
                        "status": job_data["status"],
                        "progress": job_data.get("progress", 0),
                        "original_filename": job_data.get("original_filename"),
                        "target_language": job_data.get("target_language"),
                        "source_video_language": job_data.get("source_video_language"),
                        "result_url": result_url,
                        "files": files_info,
                        "error": job_data.get("error"),
                        "created_at": job_data["created_at"].isoformat(),
                        "updated_at": job_data["updated_at"].isoformat() if job_data.get("updated_at") else None,
                        "completed_at": job_data["completed_at"].isoformat() if job_data.get("completed_at") else None,
                        # Separation fields (None for dub)
                        "audio_url": None,
                        "vocal_url": None,
                        "instrument_url": None
                    }

                job_summary = JobSummary(**job_summary_data)
                jobs.append(job_summary)
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to get recent jobs: {e}")
            return []
    
    def _empty_stats(self) -> WorkspaceStats:
        """Return empty stats object"""
        return WorkspaceStats(
            total_dubs=0,
            total_separations=0,
            total_completed_dubs=0,
            total_completed_separations=0,
            total_processing_dubs=0,
            total_processing_separations=0
        )
    
    def _get_file_type(self, filename: str) -> str:
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


# Global service instance
workspace_service = WorkspaceService()
