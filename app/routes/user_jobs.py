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
        
        from app.utils.unified_status_manager import get_unified_status_manager, JobType
        
        # Get enriched status data from unified manager
        manager = get_unified_status_manager()
        actual_limit = limit if limit else DEFAULT_QUERY_LIMIT
        status_data_list = await manager.get_user_jobs_status(str(user_id), JobType.SEPARATION, actual_limit, page)
        
        # Get additional job details from database
        jobs, total_count = await separation_job_service.get_user_jobs(str(user_id), page, actual_limit)
        
        # Get job statistics
        statistics = await separation_job_service.get_user_job_statistics(str(user_id))
        
        # Create job lookup for additional details
        job_lookup = {job.job_id: job for job in jobs}
        
        # Build response with enriched status data
        user_jobs = []
        for status_data in status_data_list:
            job = job_lookup.get(status_data.job_id)
            if not job:
                continue  # Skip if job not in current page
            
            user_job = UserSeparationJob(
                job_id=status_data.job_id,
                status=status_data.status.value,
                progress=status_data.progress,
                audio_url=job.audio_url,
                vocal_url=job.vocal_url,
                instrument_url=job.instrument_url,
                error=job.error,
                queuePosition=status_data.queue_position,
                created_at=job.created_at.isoformat(),
                updated_at=status_data.updated_at.isoformat(),
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
        
        from app.utils.unified_status_manager import get_unified_status_manager, JobType
        
        # Get enriched status from unified manager
        manager = get_unified_status_manager()
        status_data = await manager.get_status(job_id, JobType.SEPARATION)
        
        if not status_data:
            raise HTTPException(status_code=404, detail="Separation job not found")
        
        # Verify user ownership
        if status_data.user_id != user_id:
            # Fallback: check database job
            job = await separation_job_service.get_job(job_id)
            if not job or job.user_id != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Get additional job details from database
        job = await separation_job_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job details not found")
        
        user_job = UserSeparationJob(
            job_id=status_data.job_id,
            status=status_data.status.value,
            progress=status_data.progress,
            audio_url=job.audio_url,
            vocal_url=job.vocal_url,
            instrument_url=job.instrument_url,
            error=job.error,
            queuePosition=status_data.queue_position,
            created_at=job.created_at.isoformat(),
            updated_at=status_data.updated_at.isoformat(),
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
        
        from app.utils.unified_status_manager import get_unified_status_manager, JobType
        
        # Get enriched status data from unified manager
        manager = get_unified_status_manager()
        actual_limit = limit if limit else DEFAULT_QUERY_LIMIT
        status_data_list = await manager.get_user_jobs_status(str(user_id), JobType.DUB, actual_limit, page)
        
        # Get additional job details from database
        jobs, total_count = await dub_job_service.get_user_jobs(str(user_id), page, actual_limit)
        
        # Get job statistics
        statistics = await dub_job_service.get_user_job_statistics(str(user_id))
        
        # Create job lookup for additional details
        job_lookup = {job.job_id: job for job in jobs}
        
        # Enhanced formatting with status data
        user_jobs = []
        for status_data in status_data_list:
            job = job_lookup.get(status_data.job_id)
            if not job:
                continue  # Skip if job not in current page
            
            # Use job_response_service for consistent formatting but override status data
            formatted_job = job_response_service.format_dub_job(job)
            
            # Override with fresh status data
            formatted_job.status = status_data.status.value
            formatted_job.progress = status_data.progress
            formatted_job.queuePosition = status_data.queue_position
            formatted_job.updated_at = status_data.updated_at.isoformat()
            
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
        
        from app.utils.unified_status_manager import get_unified_status_manager, JobType
        
        # Get enriched status from unified manager
        manager = get_unified_status_manager()
        status_data = await manager.get_status(job_id, JobType.DUB)
        
        if not status_data:
            raise HTTPException(status_code=404, detail="Dub job not found")
        
        # Verify user ownership
        if status_data.user_id != user_id:
            # Fallback: check database job
            job = await dub_job_service.get_job(job_id)
            if not job or job.user_id != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Get additional job details from database
        job = await dub_job_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job details not found")
        
        # Format job using service but override with fresh status
        user_job = job_response_service.format_dub_job(job)
        
        # Override with unified status data
        user_job.status = status_data.status.value
        user_job.progress = status_data.progress
        user_job.queuePosition = status_data.queue_position
        user_job.updated_at = status_data.updated_at.isoformat()
        
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
        deleted = await separation_job_service.delete_job(job_id)
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
        deleted = await dub_job_service.delete_job(job_id)
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