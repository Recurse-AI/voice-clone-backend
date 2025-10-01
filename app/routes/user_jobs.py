from fastapi import APIRouter, HTTPException, Depends, Query, Request
import logging
from app.schemas import (
    UserSeparationListResponse, UserDubListResponse,
    SeparationJobDetailResponse, DubJobDetailResponse,
    UserSeparationJob, UserDubJob, WorkspaceStatusResponse,
    ClipJobListResponse, ClipJobDetailResponse
)
from app.models import ClipJob
from app.services.separation_job_service import separation_job_service
from app.services.dub_job_service import dub_job_service
from app.services.job_response_service import job_response_service
from app.services.workspace_service import workspace_service
from app.dependencies.auth import get_current_user
from app.config.constants import DEFAULT_QUERY_LIMIT
from app.services.status_api_service import api_status_service

router = APIRouter()
logger = logging.getLogger(__name__)


def safe_isoformat(value):
    """Safely convert datetime or string to ISO format string"""
    if value is None:
        return None
    if isinstance(value, str):
        return value  # Already a string, assume it's in ISO format
    if hasattr(value, 'isoformat'):
        return value.isoformat()  # DateTime object
    return str(value)  # Fallback to string conversion


# Workspace Status API
@router.get("/workspace/status", 
           response_model=WorkspaceStatusResponse,
           summary="Get Workspace Status",
           description="Get lightweight workspace status with summary statistics and recent jobs")
async def get_workspace_status(
    recent_limit: int = Query(5, ge=1, le=20, description="Number of recent jobs to return (1-20)"),
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive workspace status including:
    - Total statistics (dubs, separations, completed jobs)
    - Recent dubs and separations
    - Processing job counts
    
    **Parameters:**
    - recent_limit: Number of recent jobs to return (1-20, default: 5)
    
    **Returns:**
    - Workspace statistics
    - Recent job summaries
    """
    try:
        user_id = current_user.id
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found")
        
        # Validate recent_limit
        if recent_limit < 1 or recent_limit > 20:
            raise HTTPException(status_code=400, detail="recent_limit must be between 1 and 20")
        
        # Get workspace status data
        workspace_data = await workspace_service.get_workspace_status(str(user_id), recent_limit)
        
        return WorkspaceStatusResponse(
            success=True,
            message=f"Workspace status retrieved successfully",
            stats=workspace_data["stats"],
            recent_dubs=workspace_data["recent_dubs"],
            recent_separations=workspace_data["recent_separations"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workspace status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get workspace status")




# Separation Job APIs
@router.get("/separations", response_model=UserSeparationListResponse)
async def get_user_separations(
    page: int = 1,
    limit: int = None,
    current_user = Depends(get_current_user)
):
    """Get paginated separation jobs for current user"""
    try:
        user_id = str(current_user.id)
        page = max(1, page)
        actual_limit = limit or DEFAULT_QUERY_LIMIT
        
        jobs, total_count = await separation_job_service.get_user_jobs(user_id, page, actual_limit)
        
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
                queuePosition=None,
                created_at=job.created_at.isoformat(),
                updated_at=job.updated_at.isoformat(),
                completed_at=job.completed_at.isoformat() if job.completed_at else None
            )
            user_jobs.append(user_job)
        
        return UserSeparationListResponse(
            success=True,
            message=f"Found {len(user_jobs)} separation jobs",
            jobs=user_jobs,
            total=total_count,
            page=page,
            limit=actual_limit,
            total_pages=(total_count + actual_limit - 1) // actual_limit
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
        
        # Get latest status from API service
        status_data = api_status_service.get_job_status(job_id, "separation")
        
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
            updated_at=safe_isoformat(status_data.get("updated_at")) if status_data and status_data.get("updated_at") else job.updated_at.isoformat(),
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
    page: int = 1,
    limit: int = None,
    current_user = Depends(get_current_user)
):
    """Get paginated dub jobs for current user"""
    try:
        user_id = str(current_user.id)
        page = max(1, page)
        actual_limit = limit or DEFAULT_QUERY_LIMIT
        
        jobs, total_count = await dub_job_service.get_user_jobs(user_id, page, actual_limit)
        
        user_jobs = []
        for job in jobs:
            formatted_job = job_response_service.format_dub_job(job)
            formatted_job.queuePosition = None
            user_jobs.append(formatted_job)
        
        return UserDubListResponse(
            success=True,
            message=f"Found {len(user_jobs)} dub jobs",
            jobs=user_jobs,
            total=total_count,
            page=page,
            limit=actual_limit,
            total_pages=(total_count + actual_limit - 1) // actual_limit
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user dubs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dub jobs")

@router.get("/clips", response_model=ClipJobListResponse)
async def get_user_clips(
    page: int = 1,
    limit: int = None,
    current_user = Depends(get_current_user)
):
    """Get paginated clip jobs for current user"""
    try:
        from app.repositories.clip_repository import ClipRepository
        from app.config.constants import DEFAULT_QUERY_LIMIT
        
        user_id = str(current_user.id)
        page = max(1, page)
        actual_limit = limit or DEFAULT_QUERY_LIMIT
        
        repo = ClipRepository()
        jobs, total_count = await repo.get_user_jobs(user_id, page, actual_limit)
        
        clip_jobs = [ClipJob(**job) for job in jobs]
        
        return ClipJobListResponse(
            success=True,
            message=f"Found {len(clip_jobs)} clip jobs",
            jobs=clip_jobs,
            total=total_count,
            page=page,
            limit=actual_limit,
            total_pages=(total_count + actual_limit - 1) // actual_limit
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user clips: {e}")
        raise HTTPException(status_code=500, detail="Failed to get clip jobs")

@router.get("/clip/{job_id}", response_model=ClipJobDetailResponse)
async def get_clip_job_detail(
    job_id: str,
    current_user = Depends(get_current_user)
):
    """Get detailed clip job information"""
    try:
        from app.repositories.clip_repository import ClipRepository
        
        user_id = str(current_user.id)
        repo = ClipRepository()
        job = await repo.get_by_id(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Clip job not found")
        
        if job["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        clip_job = ClipJob(**job)
        return ClipJobDetailResponse(success=True, job=clip_job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get clip job detail: {e}")
        raise HTTPException(status_code=500, detail="Failed to get clip job details")

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
        
        # Get latest status from API service
        status_data = api_status_service.get_job_status(job_id, "dub")
        
        # Format job using service
        user_job = job_response_service.format_dub_job(job)
        
        # Override with current status data
        if status_data:
            user_job.status = status_data["status"]
            user_job.progress = status_data["progress"]
            user_job.updated_at = safe_isoformat(status_data.get("updated_at")) if status_data.get("updated_at") else user_job.updated_at
        
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