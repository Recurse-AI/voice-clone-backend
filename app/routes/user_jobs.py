from fastapi import APIRouter, HTTPException, Depends, Request, Query
import logging
from typing import Optional
from app.schemas import (
    UserSeparationListResponse, UserDubListResponse, 
    SeparationJobDetailResponse, DubJobDetailResponse,
    UserSeparationJob, UserDubJob
)
from app.services.separation_job_service import separation_job_service
from app.services.dub_job_service import dub_job_service
from app.services.job_response_service import job_response_service
from app.dependencies.auth import get_current_user
from app.config.constants import DEFAULT_QUERY_LIMIT, MAX_QUEUE_POSITION_CHECKS
from app.utils.runpod_service import runpod_service
from app.services.simple_status_service import status_service

router = APIRouter()
logger = logging.getLogger(__name__)


def get_job_queue_position(job) -> Optional[int]:
    """Get queue position for a job from RunPod service"""
    if not job.runpod_request_id or job.status not in ['pending', 'processing']:
        return None
    
    try:
        runpod_status = runpod_service.get_separation_status(job.runpod_request_id)
        if runpod_status:
            queue_position = runpod_status.get("queue_position")
            delay_time = runpod_status.get("delay_time")
            
            # Log RunPod response for debugging (only for pending jobs)
            if job.status == 'pending' and (queue_position is not None or delay_time is not None):
                logger.info(f"Job {job.job_id}: queue_position={queue_position}, delay_time={delay_time}ms")
            
            return queue_position
        else:

            return None
    except Exception as e:
        logger.warning(f"Failed to get RunPod queue position for job {job.job_id}: {e}")
        return None


# Separation Job APIs
@router.get("/separations", response_model=UserSeparationListResponse)
async def get_user_separations(
    request: Request,
    page: int = 1,
    limit: int = None,
    current_user = Depends(get_current_user)
):
    """Get paginated separation jobs for current user"""
    try:
        user_id = current_user.id
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found")
        
        # Validate pagination parameters
        if page < 1:
            raise HTTPException(status_code=400, detail="Page must be greater than 0")
        
        # Get jobs from database
        actual_limit = limit if limit else DEFAULT_QUERY_LIMIT
        jobs, total_count = await separation_job_service.get_user_jobs(str(user_id), page, actual_limit)
        
        # Get job statistics
        statistics = await separation_job_service.get_user_job_statistics(str(user_id))
        
        # Build response with current status
        user_jobs = []
        for job in jobs:
            # Get current status from simple service
            status_data = status_service.get_status(job.job_id, "separation")
            
            current_status = status_data["status"] if status_data else job.status
            current_progress = status_data["progress"] if status_data else job.progress
            
            user_job = UserSeparationJob(
                job_id=job.job_id,
                status=current_status,
                progress=current_progress,
                audio_url=job.audio_url,
                vocal_url=job.vocal_url,
                instrument_url=job.instrument_url,
                error=job.error,
                queuePosition=None,  # Simple implementation
                created_at=job.created_at.isoformat(),
                updated_at=status_data["updated_at"].isoformat() if status_data and status_data.get("updated_at") else job.updated_at.isoformat(),
                completed_at=job.completed_at.isoformat() if job.completed_at else None
            )
            user_jobs.append(user_job)
        
        # Calculate pagination metadata
        total_pages = (total_count + actual_limit - 1) // actual_limit  # Ceiling division
        
        return UserSeparationListResponse(
            success=True,
            message=f"Found {len(user_jobs)} separation jobs (page {page})",
            jobs=user_jobs,
            total=total_count,
            page=page,
            limit=actual_limit,
            total_pages=total_pages,
            total_completed=statistics["completed"],
            total_processing=statistics["processing"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user separations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get separation jobs")

@router.get("/separation/{job_id}", response_model=SeparationJobDetailResponse)
async def get_separation_job_detail(
    request: Request,
    job_id: str,
    current_user = Depends(get_current_user)
):
    """Get specific separation job details"""
    try:
        user_id = current_user.id
        
        # Get status from simple service
        status_data = status_service.get_status(job_id, "separation")
        
        if not status_data:
            raise HTTPException(status_code=404, detail="Separation job not found")
        
        # Get job details from database
        job = await separation_job_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job details not found")
        
        # Verify user ownership
        if job.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Use current status if available, otherwise use database status
        current_status = status_data["status"] if status_data else job.status
        current_progress = status_data["progress"] if status_data else job.progress
        
        user_job = UserSeparationJob(
            job_id=job.job_id,
            status=current_status,
            progress=current_progress,
            audio_url=job.audio_url,
            vocal_url=job.vocal_url,
            instrument_url=job.instrument_url,
            error=job.error,
            queuePosition=None,  # Simple implementation
            created_at=job.created_at.isoformat(),
            updated_at=status_data["updated_at"].isoformat() if status_data and status_data.get("updated_at") else job.updated_at.isoformat(),
            completed_at=job.completed_at.isoformat() if job.completed_at else None
        )
        
        return SeparationJobDetailResponse(success=True, job=user_job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get separation job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job details")


# Dub Job APIs
@router.get("/dubs", response_model=UserDubListResponse)
async def get_user_dubs(
    request: Request,
    page: int = 1,
    limit: int = None,
    current_user = Depends(get_current_user)
):
    """Get paginated dub jobs for current user"""
    try:
        user_id = current_user.id
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found")
        
        # Validate pagination parameters
        if page < 1:
            raise HTTPException(status_code=400, detail="Page must be greater than 0")
        
        # Get jobs from database
        actual_limit = limit if limit else DEFAULT_QUERY_LIMIT
        jobs, total_count = await dub_job_service.get_user_jobs(str(user_id), page, actual_limit)
        
        # Get job statistics
        statistics = await dub_job_service.get_user_job_statistics(str(user_id))
        
        # Build response with current status
        user_jobs = []
        for job in jobs:
            # Get current status from simple service
            status_data = status_service.get_status(job.job_id, "dub")
            
            # Use job_response_service for consistent formatting
            formatted_job = job_response_service.format_dub_job(job)
            
            # Override with current status data
            if status_data:
                formatted_job.status = status_data["status"]
                formatted_job.progress = status_data["progress"]
                formatted_job.updated_at = status_data["updated_at"].isoformat() if status_data.get("updated_at") else formatted_job.updated_at
            
            formatted_job.queuePosition = None  # Simple implementation
            user_jobs.append(formatted_job)
        
        # Calculate pagination metadata
        total_pages = (total_count + actual_limit - 1) // actual_limit  # Ceiling division
        
        return UserDubListResponse(
            success=True,
            message=f"Found {len(user_jobs)} dub jobs (page {page})",
            jobs=user_jobs,
            total=total_count,
            page=page,
            limit=actual_limit,
            total_pages=total_pages,
            total_completed=statistics["completed"],
            total_processing=statistics["processing"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user dubs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dub jobs")

@router.get("/dub/{job_id}", response_model=DubJobDetailResponse)
async def get_dub_job_detail(
    request: Request,
    job_id: str,
    current_user = Depends(get_current_user)
):
    """Get specific dub job details"""
    try:
        user_id = current_user.id
        
        # Get job details from database
        job = await dub_job_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job details not found")
        
        # Verify user ownership
        if job.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get current status from simple service
        status_data = status_service.get_status(job_id, "dub")
        
        # Format job using service
        user_job = job_response_service.format_dub_job(job)
        
        # Override with current status data
        if status_data:
            user_job.status = status_data["status"]
            user_job.progress = status_data["progress"]
            user_job.updated_at = status_data["updated_at"].isoformat() if status_data.get("updated_at") else user_job.updated_at
        
        user_job.queuePosition = None  # Simple implementation
        
        return DubJobDetailResponse(success=True, job=user_job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dub job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job details")



# Delete Job APIs
@router.delete("/separation/{job_id}")
async def delete_separation_job(
    job_id: str,
    current_user = Depends(get_current_user)
):
    """Delete separation job from database"""
    try:
        user_id = current_user.id
        
        # Get job details first
        job = await separation_job_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Separation job not found")
        
        if job.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete from database
        deleted = await separation_job_service.delete_job(job_id, user_id)
        if not deleted:
            raise HTTPException(status_code=500, detail="Failed to delete job")
        
        return {
            "success": True,
            "message": "Job deleted successfully",
            "job_id": job_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete separation job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete job")

@router.delete("/dub/{job_id}")
async def delete_dub_job(
    job_id: str,
    current_user = Depends(get_current_user)
):
    """Delete dub job from database"""
    try:
        user_id = current_user.id
        
        # Get job details first
        job = await dub_job_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Dub job not found")
            
        if job.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete from database
        deleted = await dub_job_service.delete_job(job_id, user_id)
        if not deleted:
            raise HTTPException(status_code=500, detail="Failed to delete job")
        
        return {
            "success": True,
            "message": "Job deleted successfully",
            "job_id": job_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dub job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete job")