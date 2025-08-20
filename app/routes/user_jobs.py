from fastapi import APIRouter, HTTPException, Depends, Request
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
            logger.debug(f"No RunPod status returned for job {job.job_id}")
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
        
        jobs, total_count = await separation_job_service.get_user_jobs(str(user_id), page, limit)
        
        # Get job statistics
        statistics = await separation_job_service.get_user_job_statistics(str(user_id))
        
        # Convert to response format
        user_jobs = []
        pending_jobs_count = 0
        
        for job in jobs:
            # Get queue position for pending/processing jobs (limit to avoid performance issues)
            queue_position = None
            if job.status in ['pending', 'processing'] and pending_jobs_count < MAX_QUEUE_POSITION_CHECKS:
                pending_jobs_count += 1
                queue_position = get_job_queue_position(job)
            
            user_job = UserSeparationJob(
                job_id=job.job_id,
                status=job.status,
                progress=job.progress,
                audio_url=job.audio_url,
                vocal_url=job.vocal_url,
                instrument_url=job.instrument_url,
                error=job.error,
                queuePosition=queue_position,
                created_at=job.created_at.isoformat(),
                updated_at=job.updated_at.isoformat(),
                completed_at=job.completed_at.isoformat() if job.completed_at else None
            )
            user_jobs.append(user_job)
        
        # Calculate pagination metadata
        actual_limit = limit if limit else DEFAULT_QUERY_LIMIT
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
        job = await separation_job_service.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Separation job not found")
        
                # Check if job belongs to current user
        if job.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get queue position for pending/processing jobs
        queue_position = get_job_queue_position(job)
        
        user_job = UserSeparationJob(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            audio_url=job.audio_url,
            vocal_url=job.vocal_url,
            instrument_url=job.instrument_url,
            error=job.error,
            queuePosition=queue_position,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
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
        
        jobs, total_count = await dub_job_service.get_user_jobs(str(user_id), page, limit)
        
        # Get job statistics
        statistics = await dub_job_service.get_user_job_statistics(str(user_id))
        
        # Format jobs using service
        user_jobs = job_response_service.format_dub_jobs(jobs)
        
        # Calculate pagination metadata
        actual_limit = limit if limit else DEFAULT_QUERY_LIMIT
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
        job = await dub_job_service.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Dub job not found")
        
        # Check if job belongs to current user
        if job.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Format job using service
        user_job = job_response_service.format_dub_job(job)
        
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
    """Delete a specific separation job"""
    try:
        user_id = current_user.id
        
        # Delete the job using service
        success = await separation_job_service.delete_job(job_id, user_id)
        
        if not success:
            # Check if job exists but doesn't belong to user
            job = await separation_job_service.get_job(job_id)
            if job and job.user_id != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
            else:
                raise HTTPException(status_code=404, detail="Separation job not found")
        
        return {
            "success": True,
            "message": "Separation job deleted successfully",
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
    """Delete a specific dub job"""
    try:
        user_id = current_user.id
        
        # Delete the job using service
        success = await dub_job_service.delete_job(job_id, user_id)
        
        if not success:
            # Check if job exists but doesn't belong to user
            job = await dub_job_service.get_job(job_id)
            if job and job.user_id != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
            else:
                raise HTTPException(status_code=404, detail="Dub job not found")
        
        return {
            "success": True,
            "message": "Dub job deleted successfully",
            "job_id": job_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dub job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete job")