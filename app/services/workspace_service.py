import logging
from typing import List, Dict, Any
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
        """Get recent jobs with minimal data"""
        try:
            projection = {
                "job_id": 1,
                "status": 1,
                "progress": 1,
                "created_at": 1,
                "completed_at": 1,
                "original_filename": 1,
                "_id": 0
            }

            cursor = collection.find(
                {"user_id": user_id},
                projection
            ).sort("created_at", -1).limit(limit)
            
            jobs = []
            async for job_data in cursor:
                # Create base data dictionary
                job_summary_data = {
                    "job_id": job_data["job_id"],
                    "status": job_data["status"],
                    "progress": job_data.get("progress", 0),
                    "created_at": job_data["created_at"].isoformat(),
                    "completed_at": job_data["completed_at"].isoformat() if job_data.get("completed_at") else None
                }

                if "original_filename" in job_data and job_data["original_filename"]:
                    job_summary_data["original_filename"] = job_data["original_filename"]

                # Create JobSummary from dictionary
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


# Global service instance
workspace_service = WorkspaceService()
