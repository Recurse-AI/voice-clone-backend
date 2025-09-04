"""
Unified Status Manager - Complete solution for status, progress, and position tracking
Consolidates all status management into one clean, reliable system
"""
import logging
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from datetime import datetime, timezone
from collections import defaultdict
import threading


from app.services.dub_job_service import dub_job_service
from app.services.separation_job_service import separation_job_service
from app.utils.db_sync_operations import update_job_status_sync
from app.utils.runpod_service import runpod_service

logger = logging.getLogger(__name__)


class JobType(Enum):
    """Job type enumeration"""
    DUB = "dub"
    SEPARATION = "separation"


class ProcessingStatus(Enum):
    """Processing status enum with clear hierarchy"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    SEPARATING = "separating"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_REVIEW = "awaiting_review"
    REVIEWING = "reviewing"


class StatusData:
    """Structured status data container"""
    
    def __init__(self, job_id: str, job_type: JobType, status: ProcessingStatus, 
                 progress: int = 0, details: Optional[Dict[str, Any]] = None,
                 queue_position: Optional[int] = None, user_id: Optional[str] = None):
        self.job_id = job_id
        self.job_type = job_type
        self.status = status
        self.progress = max(0, min(100, progress))  # Clamp 0-100
        self.details = details or {}
        self.queue_position = queue_position
        self.user_id = user_id
        self.updated_at = datetime.now(timezone.utc)
        
        # Auto-generate message
        self.message = self._get_status_message()
    
    def _get_status_message(self) -> str:
        """Generate descriptive status message based on status and progress"""
        base_messages = {
            ProcessingStatus.PENDING: "Job queued for processing",
            ProcessingStatus.DOWNLOADING: "Downloading video...",
            ProcessingStatus.SEPARATING: "Separating audio tracks...",
            ProcessingStatus.TRANSCRIBING: "Transcribing audio...",
            ProcessingStatus.UPLOADING: "Uploading results...",
            ProcessingStatus.COMPLETED: "Processing completed successfully",
            ProcessingStatus.FAILED: "Processing failed",
            ProcessingStatus.AWAITING_REVIEW: "Awaiting human review - Please review dubbed text",
            ProcessingStatus.REVIEWING: "Applying human edits and continuing dubbing"
        }
        
        if self.status == ProcessingStatus.PROCESSING:
            return self._get_processing_message()
        
        return base_messages.get(self.status, f"Job is {self.status.value}")
    
    def _get_processing_message(self) -> str:
        """Get detailed processing message based on progress - FIXED to match actual phases"""
        # Check if phase is provided in details (more accurate)
        phase = self.details.get("phase")
        if phase:
            phase_messages = {
                "initialization": "Initializing dubbing process...",
                "separation": "Separating audio tracks...",
                "transcription": "Transcribing audio with AI...",
                "dubbing": "Dubbing text with AI translation...",
                "review_prep": "Preparing for review...",
                "reviewing": "Applying your edits...",
                "voice_cloning": "Voice cloning with AI...",
                "final_processing": "Generating final audio...",
                "upload": "Uploading results...",
                "queued": "Job queued for processing"
            }
            return phase_messages.get(phase, f"Processing ({phase})...")
        
        # Fallback to progress-based ranges (matching SimpleDubbedAPI.PROGRESS_PHASES)
        if self.progress <= 10:
            return "Initializing dubbing process..."
        elif self.progress <= 25:
            return "Queued - Waiting to start processing..."
        elif self.progress <= 45:
            return "Separating audio tracks..."
        elif self.progress <= 60:
            return "Transcribing audio with AI..."
        elif self.progress <= 75:
            return "Processing translation and dubbing with AI..."
        elif self.progress <= 80:
            return "Preparing for review..."
        elif self.progress <= 81:
            return "Applying your edits..."
        elif self.progress < 96:
            return "Voice cloning with AI..."
        elif self.progress < 100:
            return "Uploading results..."
        else:
            return "Processing completed successfully"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "details": self.details,
            "queue_position": self.queue_position,
            "user_id": self.user_id,
            "updated_at": self.updated_at.isoformat()
        }


class UnifiedStatusManager:
    """
    Unified Status Manager - The single source of truth for all job status tracking
    
    Features:
    - Hybrid caching (fast reads) + persistent storage (reliability)
    - Monotonic progress protection (never go backwards)
    - Queue position tracking for both job types
    - Thread-safe operations
    - Automatic cache invalidation
    - Consistent API responses
    """
    
    def __init__(self):
        self._cache: Dict[str, StatusData] = {}
        self._cache_lock = threading.RLock()
        
        # Status categories for smart caching
        self._processing_states = {
            ProcessingStatus.PENDING, ProcessingStatus.DOWNLOADING,
            ProcessingStatus.SEPARATING, ProcessingStatus.TRANSCRIBING,
            ProcessingStatus.PROCESSING, ProcessingStatus.UPLOADING,
            ProcessingStatus.AWAITING_REVIEW, ProcessingStatus.REVIEWING
        }
        
        self._final_states = {
            ProcessingStatus.COMPLETED, ProcessingStatus.FAILED
        }
        
        # Progress validation rules - Minimum progress for each status (aligned with phases)
        self._progress_floors = {
            ProcessingStatus.PENDING: 0,
            ProcessingStatus.SEPARATING: 25,
            ProcessingStatus.TRANSCRIBING: 45,
            ProcessingStatus.PROCESSING: 60,
            ProcessingStatus.UPLOADING: 96,
            ProcessingStatus.COMPLETED: 100,
            ProcessingStatus.AWAITING_REVIEW: 80,
            ProcessingStatus.REVIEWING: 80,
            ProcessingStatus.FAILED: 0,
        }
        

    
    def _ensure_timezone_aware(self, dt: datetime) -> datetime:
        """Ensure datetime is timezone-aware (UTC)"""
        if dt and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    
    def _job_to_dict(self, job) -> Dict[str, Any]:
        """Convert job object to dictionary with timezone-aware handling"""
        updated_at = self._ensure_timezone_aware(
            getattr(job, 'updated_at', datetime.now(timezone.utc))
        )
        created_at = self._ensure_timezone_aware(
            getattr(job, 'created_at', datetime.now(timezone.utc))
        )
        
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "details": getattr(job, 'details', None) or {},
            "user_id": job.user_id,
            "updated_at": updated_at,
            "created_at": created_at,
            "error": getattr(job, 'error', None)
        }
    
    async def _get_jobs_from_service(self, job_type: JobType, user_id: str, 
                                   page: int = 1, limit: int = 50) -> Tuple[List, int]:
        """Get jobs from appropriate service based on job type"""
        if job_type == JobType.DUB:
            from app.services.dub_job_service import dub_job_service
            return await dub_job_service.get_user_jobs(user_id, page=page, limit=limit)
        elif job_type == JobType.SEPARATION:
            from app.services.separation_job_service import separation_job_service
            return await separation_job_service.get_user_jobs(user_id, page=page, limit=limit)
        else:
            return [], 0
    
    async def update_status(self, job_id: str, job_type: JobType, status: ProcessingStatus,
                           progress: Optional[int] = None, details: Optional[Dict[str, Any]] = None,
                           user_id: Optional[str] = None) -> bool:
        """
        Update job status with smart caching, validation and duplicate prevention
        
        Args:
            job_id: Job identifier
            job_type: Type of job (dub/separation)
            status: New status
            progress: Progress percentage (0-100)
            details: Additional details dictionary
            user_id: User ID for ownership validation
            
        Returns:
            True if update successful
        """
        try:
            with self._cache_lock:
                # Get current status for validation
                current_data = self._cache.get(job_id)
                
                # Validate progress (monotonic + status-based floors)
                validated_progress = self._validate_progress(
                    status, progress, current_data
                )
                
                # 🛡️ Prevent unnecessary updates (same status + progress)
                if current_data:
                    if (current_data.status == status and 
                        current_data.progress == validated_progress and
                        status in self._processing_states):
                        # Skip if same status and progress for processing states

                        return True
                
                # 🛡️ Smart progress grouping to reduce updates
                if current_data and status in self._processing_states:
                    progress_diff = abs(validated_progress - current_data.progress)
                    # Only update if significant progress change (min 5% or status change)
                    if progress_diff < 5 and current_data.status == status:

                        return True
                
                # Get queue position if applicable
                queue_position = await self._get_queue_position(job_id, job_type, status)
                
                # Merge details with existing to avoid losing prior context (e.g., runpod_urls)
                merged_details: Dict[str, Any] = {}
                if current_data and current_data.details:
                    merged_details.update(current_data.details)
                if details:
                    merged_details.update(details)

                # Create new status data
                status_data = StatusData(
                    job_id=job_id,
                    job_type=job_type,
                    status=status,
                    progress=validated_progress,
                    details=merged_details,
                    queue_position=queue_position,
                    user_id=user_id
                )
                
                # Update cache immediately (fast UI updates)
                self._cache[job_id] = status_data
                
                # Always persist to database for reliability
                await self._persist_status_to_database(status_data)
                
                # For final states, clean cache after short delay (will auto-expire)
                if status in self._final_states:
                    # Keep in cache briefly for immediate reads, will auto-expire
                    pass
                
                self._log_status_update(job_id, status.value, validated_progress)
                return True
                
        except Exception as e:
            logger.error(f"Failed to update status for {job_id}: {e}")
            return False
    
    def update_status_sync(self, job_id: str, job_type: JobType, status: ProcessingStatus,
                          progress: Optional[int] = None, details: Optional[Dict[str, Any]] = None,
                          user_id: Optional[str] = None) -> bool:
        """
        Synchronous version of update_status to avoid event loop conflicts
        Safe to call from any thread without async context
        """
        try:
            with self._cache_lock:
                # Get current status for validation
                current_data = self._cache.get(job_id)
                
                # Validate progress (monotonic + status-based floors)
                validated_progress = self._validate_progress(
                    status, progress, current_data
                )
                
                # Prevent unnecessary updates (same status + progress)
                if current_data:
                    if (current_data.status == status and 
                        current_data.progress == validated_progress and
                        status in self._processing_states):

                        return True
                
                # Smart progress grouping to reduce updates
                if current_data and status in self._processing_states:
                    progress_diff = abs(validated_progress - current_data.progress)
                    if progress_diff < 5 and current_data.status == status:

                        return True
                
                # Merge details with existing to avoid losing prior context (e.g., runpod_urls)
                merged_details_sync: Dict[str, Any] = {}
                if current_data and current_data.details:
                    merged_details_sync.update(current_data.details)
                if details:
                    merged_details_sync.update(details)

                # Create new status data
                status_data = StatusData(
                    job_id=job_id,
                    job_type=job_type,
                    status=status,
                    progress=validated_progress,
                    details=merged_details_sync,
                    queue_position=None,  # Skip queue position for sync version
                    user_id=user_id
                )
                
                # Update cache immediately (fast UI updates)
                self._cache[job_id] = status_data
                
                # Persist to database synchronously
                self._persist_status_to_database_sync(status_data)
                
                self._log_status_update(job_id, status.value, validated_progress, "(sync)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update status (sync) for {job_id}: {e}")
            return False
    
    def _log_status_update(self, job_id: str, status: str, progress: int, suffix: str = ""):
        """Centralized status update logging to reduce code duplication"""
        logger.info(f"Status updated{suffix}: {job_id} -> {status} ({progress}%)")

    def _persist_status_to_database_sync(self, status_data: StatusData) -> bool:
        """Synchronous database persistence using sync operations"""
        try:

            
            return update_job_status_sync(
                job_id=status_data.job_id,
                job_type=status_data.job_type.value,
                status=status_data.status.value,
                progress=status_data.progress,
                details=status_data.details
            )
        except Exception as e:
            logger.error(f"Failed to persist status (sync) for {status_data.job_id}: {e}")
            return False

    async def get_status(self, job_id: str, job_type: Optional[JobType] = None) -> Optional[StatusData]:
        """
        Get current job status with smart caching
        
        Args:
            job_id: Job identifier
            job_type: Job type hint for optimization
            
        Returns:
            StatusData object or None if not found
        """
        try:
            with self._cache_lock:
                # Check cache first (fastest)
                cached_data = self._cache.get(job_id)
                if cached_data:
                    # Refresh queue position for active jobs
                    if cached_data.status in self._processing_states:
                        cached_data.queue_position = await self._get_queue_position(
                            job_id, cached_data.job_type, cached_data.status
                        )
                    return cached_data
            
            # Cache miss - check database
            if job_type:
                return await self._load_from_database(job_id, job_type)
            
            # Try both job types if type not specified
            for jtype in JobType:
                data = await self._load_from_database(job_id, jtype)
                if data:
                    return data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get status for {job_id}: {e}")
            return None
    
    def get_status_sync(self, job_id: str, job_type: Optional[JobType] = None) -> Optional[StatusData]:
        """
        Synchronous version of get_status for non-async contexts
        
        Args:
            job_id: Job identifier
            job_type: Job type hint for optimization
            
        Returns:
            StatusData object or None if not found
        """
        try:
            with self._cache_lock:
                # Check cache first (fastest)
                cached_data = self._cache.get(job_id)
                if cached_data:
                    return cached_data
            
            # Cache miss - need to check database synchronously
            # Note: This simplified version doesn't update queue positions
            # For full features, use the async version when possible
            return None
            
        except Exception as e:
            logger.error(f"Failed to get status sync for {job_id}: {e}")
            return None
    
    async def get_user_jobs_status(self, user_id: str, job_type: JobType, 
                                  limit: int = 50, page: int = 1) -> List[StatusData]:
        """
        Get status for all user jobs with batch optimization
        
        Args:
            user_id: User identifier
            job_type: Type of jobs to fetch
            limit: Maximum jobs to return per page
            page: Page number (1-based)
            
        Returns:
            List of StatusData objects
        """
        try:
            # Get jobs from database with pagination using reusable method
            jobs, _ = await self._get_jobs_from_service(job_type, user_id, page, limit)
            if not jobs:
                return []
            
            # Convert jobs to dict format using reusable method
            jobs_data = [self._job_to_dict(job) for job in jobs]
            
            # Enrich with cache data and queue positions
            enriched_jobs = []
            for job_data in jobs_data:
                job_id = job_data.get("job_id")
                
                # Check if we have fresher cache data
                with self._cache_lock:
                    cached = self._cache.get(job_id)
                    if cached:
                        # Ensure timezone-aware comparison using reusable method
                        db_updated_at = self._ensure_timezone_aware(
                            job_data.get("updated_at", datetime.min.replace(tzinfo=timezone.utc))
                        )
                        
                        if cached.updated_at > db_updated_at:
                            enriched_jobs.append(cached)
                            continue
                
                # Create StatusData from DB data
                status_data = self._status_data_from_db(job_data, job_type)
                
                # Add queue position for active jobs
                if status_data.status in self._processing_states:
                    status_data.queue_position = await self._get_queue_position(
                        job_id, job_type, status_data.status
                    )
                
                enriched_jobs.append(status_data)
            
            return enriched_jobs
            
        except Exception as e:
            logger.error(f"Failed to get user jobs status: {e}")
            return []
    
    def _validate_progress(self, status: ProcessingStatus, progress: Optional[int], 
                          current_data: Optional[StatusData]) -> int:
        """Validate progress with monotonic protection and status floors"""
        if progress is None:
            # Use status-based floor if no progress provided
            return self._progress_floors.get(status, 0)
        
        # Clamp to 0-100 range
        progress = max(0, min(100, progress))
        
        # Special handling for reviewing→processing transition (resume after review)
        if (current_data and 
            current_data.status == ProcessingStatus.REVIEWING and 
            status == ProcessingStatus.PROCESSING and
            current_data.progress >= 80):
            # Maintain current progress when resuming processing after review
            progress = max(progress, current_data.progress)
        else:
            # Apply status-based floor
            status_floor = self._progress_floors.get(status, 0)
            progress = max(progress, status_floor)
        
        # Monotonic protection (never go backwards except for valid cases)
        if current_data and current_data.progress > progress:
            # Allow backward progress only for specific valid transitions
            if self._is_valid_backward_transition(current_data.status, status):
                logger.info(f"Valid transition: {current_data.status.value}({current_data.progress}%) → {status.value}({progress}%)")
                return progress
            
            # Special case: If current status is higher phase, don't go backwards
            if self._is_phase_regression(current_data.status, status):
                logger.warning(f"🛑 Phase regression blocked: {current_data.status.value}({current_data.progress}%) → {status.value}({progress}%)")
                return current_data.progress
            
            # Block invalid backward progress
            if True:
                logger.warning(f"🛑 Blocked backward transition: {current_data.status.value}({current_data.progress}%) → {status.value}({progress}%)")
                return current_data.progress
        
        return progress
    
    def _is_phase_regression(self, current_status: ProcessingStatus, new_status: ProcessingStatus) -> bool:
        """Check if this is a phase regression (going from higher phase to lower phase)"""
        # Define phase hierarchy
        phase_order = {
            ProcessingStatus.PENDING: 0,
            ProcessingStatus.DOWNLOADING: 1,
            ProcessingStatus.SEPARATING: 2,
            ProcessingStatus.TRANSCRIBING: 3,
            ProcessingStatus.PROCESSING: 4,  # Processing can be various phases
            ProcessingStatus.UPLOADING: 5,
            ProcessingStatus.AWAITING_REVIEW: 6,
            ProcessingStatus.REVIEWING: 7,
            ProcessingStatus.COMPLETED: 8,
        }
        
        current_order = phase_order.get(current_status, 4)  # Default to PROCESSING level
        new_order = phase_order.get(new_status, 4)
        
        # Block if trying to go from higher phase to lower phase
        # Exception: PROCESSING can transition to SEPARATING if it's actually separation phase
        if current_status == ProcessingStatus.PROCESSING and new_status == ProcessingStatus.SEPARATING:
            return False  # This is valid - PROCESSING can be separation phase
        
        return new_order < current_order and new_status not in [ProcessingStatus.FAILED]

    def _is_valid_backward_transition(self, current_status: ProcessingStatus, new_status: ProcessingStatus) -> bool:
        """Check if backward progress is allowed for this status transition"""
        # Allow transitions to terminal states from anywhere
        if new_status in {ProcessingStatus.FAILED}:
            return True
        
        # Allow PROCESSING → SEPARATING (separation callbacks during processing)
        if current_status == ProcessingStatus.PROCESSING and new_status == ProcessingStatus.SEPARATING:
            return True
        
        # Allow review workflow transitions
        valid_review_transitions = {
            (ProcessingStatus.PROCESSING, ProcessingStatus.AWAITING_REVIEW),
            (ProcessingStatus.REVIEWING, ProcessingStatus.PROCESSING)
        }
        
        return (current_status, new_status) in valid_review_transitions
    
    async def _get_queue_position(self, job_id: str, job_type: JobType, 
                                 status: ProcessingStatus) -> Optional[int]:
        """Get queue position based on job type and status"""
        if status not in {ProcessingStatus.PENDING, ProcessingStatus.PROCESSING}:
            return None
        
        try:
            if job_type == JobType.SEPARATION:
                return await self._get_runpod_queue_position(job_id)
            elif job_type == JobType.DUB:
                return await self._get_local_queue_position(job_id)
            
        except Exception as e:
            pass
        
        return None
    
    async def _get_runpod_queue_position(self, job_id: str) -> Optional[int]:
        """Get queue position from RunPod service"""
        try:

            
            # Get job to find runpod_request_id
            job = await separation_job_service.get_job(job_id)
            if not job or not job.runpod_request_id:
                return None
            
            runpod_status = runpod_service.get_separation_status(job.runpod_request_id)
            if runpod_status:
                return runpod_status.get("queue_position")
            
        except Exception as e:
            pass
        
        return None
    
    async def _get_local_queue_position(self, job_id: str) -> Optional[int]:
        """Get queue position from local dub queue manager"""
        try:
            from app.routes.video.dub_routes import get_dub_queue_position
            return get_dub_queue_position(job_id)
        except Exception:
            return None
    
    async def _persist_status_to_database(self, status_data: StatusData) -> bool:
        """Persist status to database for all status types"""
        try:
            if status_data.job_type == JobType.DUB:
                from app.services.dub_job_service import dub_job_service
                return await dub_job_service.update_job_status(
                    job_id=status_data.job_id,
                    status=status_data.status.value,
                    progress=status_data.progress,
                    details=status_data.details
                )
            elif status_data.job_type == JobType.SEPARATION:
                from app.services.separation_job_service import separation_job_service
                return await separation_job_service.update_job_status(
                    job_id=status_data.job_id,
                    status=status_data.status.value,
                    progress=status_data.progress,
                    details=status_data.details
                )
        except Exception as e:
            logger.error(f"Failed to persist status for {status_data.job_id}: {e}")
            return False
        
        return False
    
    async def _persist_final_status(self, status_data: StatusData) -> bool:
        """Persist final status to database - wrapper for backwards compatibility"""
        return await self._persist_status_to_database(status_data)
    
    async def _load_from_database(self, job_id: str, job_type: JobType) -> Optional[StatusData]:
        """Load status from database and update cache"""
        try:
            job = None
            
            if job_type == JobType.DUB:
                from app.services.dub_job_service import dub_job_service
                job = await dub_job_service.get_job(job_id)
            elif job_type == JobType.SEPARATION:
                from app.services.separation_job_service import separation_job_service
                job = await separation_job_service.get_job(job_id)
            
            if not job:
                return None
            
            # Convert to StatusData
            status_data = StatusData(
                job_id=job.job_id,
                job_type=job_type,
                status=ProcessingStatus(job.status),
                progress=job.progress,
                details=getattr(job, 'details', None) or {},
                user_id=job.user_id
            )
            
            # Add to cache for future fast access
            with self._cache_lock:
                self._cache[job_id] = status_data
            
            return status_data
            
        except Exception as e:
            logger.error(f"Failed to load from database {job_id}: {e}")
            return None
    
    async def _get_user_jobs_from_db(self, user_id: str, job_type: JobType, 
                                    limit: int) -> List[Dict[str, Any]]:
        """Get user jobs from database using reusable methods"""
        try:
            # Get jobs using reusable service method
            jobs, _ = await self._get_jobs_from_service(job_type, user_id, page=1, limit=limit)
            if not jobs:
                return []
            
            # Convert to dict format using reusable method
            return [self._job_to_dict(job) for job in jobs]
            
        except Exception as e:
            logger.error(f"Failed to get user jobs from DB: {e}")
            return []
    
    def _status_data_from_db(self, job_data: Dict[str, Any], job_type: JobType) -> StatusData:
        """Create StatusData from database job data"""
        status_data = StatusData(
            job_id=job_data["job_id"],
            job_type=job_type,
            status=ProcessingStatus(job_data["status"]),
            progress=job_data["progress"],
            details=job_data.get("details", {}),
            user_id=job_data["user_id"]
        )
        
        # Use actual database updated_at instead of current time
        db_updated_at = job_data.get("updated_at")
        if db_updated_at:
            # Ensure timezone-aware datetime using reusable method
            status_data.updated_at = self._ensure_timezone_aware(db_updated_at)
        
        return status_data
    
    def clear_cache(self, job_id: Optional[str] = None) -> None:
        """Clear cache - specific job or all"""
        with self._cache_lock:
            if job_id:
                self._cache.pop(job_id, None)
                logger.info(f"Cleared cache for job {job_id}")
            else:
                cleared_count = len(self._cache)
                self._cache.clear()
                logger.info(f"Cleared all cache ({cleared_count} entries)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        with self._cache_lock:
            return {
                "total_cached_jobs": len(self._cache),
                "cached_job_ids": list(self._cache.keys()),
                "processing_jobs": len([
                    j for j in self._cache.values() 
                    if j.status in self._processing_states
                ]),
                "completed_jobs": len([
                    j for j in self._cache.values() 
                    if j.status in self._final_states
                ])
            }


# Global instance - thread-safe singleton
_unified_status_manager = None
_manager_lock = threading.Lock()


def get_unified_status_manager() -> UnifiedStatusManager:
    """Get global unified status manager instance (thread-safe)"""
    global _unified_status_manager
    if _unified_status_manager is None:
        with _manager_lock:
            if _unified_status_manager is None:
                _unified_status_manager = UnifiedStatusManager()
    return _unified_status_manager


# Convenience functions for backward compatibility
async def update_job_status(job_id: str, job_type: str, status: str, 
                           progress: Optional[int] = None, 
                           details: Optional[Dict[str, Any]] = None,
                           user_id: Optional[str] = None) -> bool:
    """Convenience function for updating job status"""
    manager = get_unified_status_manager()
    return await manager.update_status(
        job_id=job_id,
        job_type=JobType(job_type),
        status=ProcessingStatus(status),
        progress=progress,
        details=details,
        user_id=user_id
    )


async def get_job_status(job_id: str, job_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Convenience function for getting job status"""
    manager = get_unified_status_manager()
    jtype = JobType(job_type) if job_type else None
    status_data = await manager.get_status(job_id, jtype)
    return status_data.to_dict() if status_data else None
