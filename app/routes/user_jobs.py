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
from app.config.constants import DEFAULT_QUERY_LIMIT

router = APIRouter()
logger = logging.getLogger(__name__)



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
        
        # Convert to response format
        user_jobs = []
        for job in jobs:
            user_job = UserSeparationJob(
                job_id=job.job_id,
                status=job.status,
                progress=job.progress,
                audio_url=job.audio_url,
                vocal_url=job.vocal_url,
                instrument_url=job.instrument_url,
                error=job.error,
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
            total_pages=total_pages
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
        
        user_job = UserSeparationJob(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            audio_url=job.audio_url,
            vocal_url=job.vocal_url,
            instrument_url=job.instrument_url,
            error=job.error,
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
            total_pages=total_pages
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