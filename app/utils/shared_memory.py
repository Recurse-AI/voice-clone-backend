"""
Production-safe shared memory with improved code quality.
Cleaner, more maintainable implementation with better separation of concerns.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from app.config.database import upload_status_collection

logger = logging.getLogger(__name__)


class CacheStatus(Enum):
    """Cache entry status"""
    VALID = "valid"
    EXPIRED = "expired"
    MISSING = "missing"


@dataclass
class CacheConfig:
    """Cache configuration"""
    timeout_seconds: int = 300  # 5 minutes
    cleanup_interval: int = 600  # 10 minutes


@dataclass 
class StatusEntry:
    """Status entry with metadata"""
    data: Dict[str, Any]
    timestamp: datetime
    job_id: str
    
    @classmethod
    def create(cls, job_id: str, data: Dict[str, Any]) -> 'StatusEntry':
        """Create new status entry with current timestamp"""
        return cls(
            data=data,
            timestamp=datetime.now(timezone.utc),
            job_id=job_id
        )
    
    def is_expired(self, timeout_seconds: int) -> bool:
        """Check if entry is expired"""
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > timeout_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            **self.data,
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat(),
            "updated_at": self.timestamp
        }


class DatabaseManager:
    """Handles database operations for status persistence"""
    
    async def save_status(self, entry: StatusEntry) -> bool:
        """Save status entry to database"""
        try:
            await upload_status_collection.replace_one(
                {"job_id": entry.job_id},
                entry.to_dict(),
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Database save failed for {entry.job_id}: {e}")
            return False
    
    async def load_status(self, job_id: str) -> Optional[StatusEntry]:
        """Load status entry from database"""
        try:
            result = await upload_status_collection.find_one({"job_id": job_id})
            if not result:
                return None
            
            # Remove MongoDB _id field
            result.pop("_id", None)
            
            # Extract metadata
            timestamp_str = result.pop("timestamp", None)
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
            else:
                timestamp = datetime.now(timezone.utc)
            
            return StatusEntry(
                data=result,
                timestamp=timestamp,
                job_id=job_id
            )
        except Exception as e:
            logger.error(f"Database load failed for {job_id}: {e}")
            return None
    
    async def delete_status(self, job_id: str) -> bool:
        """Delete status entry from database"""
        try:
            await upload_status_collection.delete_one({"job_id": job_id})
            return True
        except Exception as e:
            logger.error(f"Database delete failed for {job_id}: {e}")
            return False


class MemoryCache:
    """In-memory cache for fast access"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: Dict[str, StatusEntry] = {}
    
    def get(self, job_id: str) -> tuple[Optional[StatusEntry], CacheStatus]:
        """Get entry from cache with status"""
        if job_id not in self._cache:
            return None, CacheStatus.MISSING
        
        entry = self._cache[job_id]
        if entry.is_expired(self.config.timeout_seconds):
            del self._cache[job_id]
            return None, CacheStatus.EXPIRED
        
        return entry, CacheStatus.VALID
    
    def set(self, entry: StatusEntry) -> None:
        """Set entry in cache"""
        self._cache[entry.job_id] = entry
    
    def delete(self, job_id: str) -> bool:
        """Delete entry from cache"""
        if job_id in self._cache:
            del self._cache[job_id]
            return True
        return False
    
    def cleanup_expired(self) -> int:
        """Remove expired entries, return count removed"""
        expired_keys = [
            job_id for job_id, entry in self._cache.items()
            if entry.is_expired(self.config.timeout_seconds)
        ]
        
        for job_id in expired_keys:
            del self._cache[job_id]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)


class StatusManager:
    """
    Production-safe status manager with cache + database persistence.
    Clean separation of concerns with proper error handling.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.cache = MemoryCache(self.config)
        self.db = DatabaseManager()
        # Track cancelled jobs to signal background threads
        self.cancelled_jobs: set = set()
    
    async def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status with cache-first strategy"""
        # Check cache first
        entry, cache_status = self.cache.get(job_id)
        if cache_status == CacheStatus.VALID:
            return entry.data
        
        # Cache miss/expired - check database
        entry = await self.db.load_status(job_id)
        if entry:
            # Update cache with fresh data
            self.cache.set(entry)
            return entry.data
        
        return None
    
    async def set_status(self, job_id: str, data: Dict[str, Any]) -> bool:
        """Set status with cache + database persistence"""
        entry = StatusEntry.create(job_id, data)
        
        # Update cache immediately for fast access
        self.cache.set(entry)
        
        # Persist to database for cross-worker access
        success = await self.db.save_status(entry)
        if not success:
            logger.warning(f"Database persistence failed for {job_id}, cache-only")
        
        return True  # Always return True since cache was updated
    
    async def update_status(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing status with new data"""
        # Get current status
        current_data = await self.get_status(job_id) or {}
        
        # Merge with updates
        updated_data = {**current_data, **updates}
        
        return await self.set_status(job_id, updated_data)
    
    async def delete_status(self, job_id: str) -> bool:
        """Delete status from cache and database"""
        # Remove from cache
        cache_deleted = self.cache.delete(job_id)
        
        # Remove from cancelled jobs tracking
        self.cancelled_jobs.discard(job_id)
        
        # Remove from database
        db_deleted = await self.db.delete_status(job_id)
        
        return cache_deleted or db_deleted
    
    def exists(self, job_id: str) -> bool:
        """Check if job exists in cache (sync operation)"""
        entry, status = self.cache.get(job_id)
        return status == CacheStatus.VALID
    
    async def exists_async(self, job_id: str) -> bool:
        """Check if job exists with database fallback"""
        status = await self.get_status(job_id)
        return status is not None


# Global instance for application use
_status_manager = StatusManager()

# Legacy sync functions for backward compatibility
def get_upload_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Legacy sync function - cache only"""
    entry, status = _status_manager.cache.get(job_id)
    return entry.data if status == CacheStatus.VALID else None

def set_upload_status(job_id: str, status_data: Dict[str, Any]) -> None:
    """Legacy sync function - cache only"""
    entry = StatusEntry.create(job_id, status_data)
    _status_manager.cache.set(entry)

def update_upload_status(job_id: str, updates: Dict[str, Any]) -> None:
    """Legacy sync function - cache only"""
    current = get_upload_status(job_id) or {}
    updated = {**current, **updates}
    set_upload_status(job_id, updated)

def delete_upload_status(job_id: str) -> None:
    """Legacy sync function - cache only"""
    _status_manager.cache.delete(job_id)

def job_exists(job_id: str) -> bool:
    """Legacy sync function - cache only"""
    return _status_manager.exists(job_id)

# Modern async functions with full functionality
async def get_upload_status_async(job_id: str) -> Optional[Dict[str, Any]]:
    """Modern async function with database fallback"""
    return await _status_manager.get_status(job_id)

async def set_upload_status_async(job_id: str, status_data: Dict[str, Any]) -> bool:
    """Modern async function with persistence"""
    return await _status_manager.set_status(job_id, status_data)

async def update_upload_status_async(job_id: str, updates: Dict[str, Any]) -> bool:
    """Modern async function with persistence"""
    return await _status_manager.update_status(job_id, updates)

async def delete_upload_status_async(job_id: str) -> bool:
    """Modern async function with persistence"""
    return await _status_manager.delete_status(job_id)

async def job_exists_async(job_id: str) -> bool:
    """Modern async function with database fallback"""
    return await _status_manager.exists_async(job_id)

# Job Cancellation Management
def mark_job_cancelled(job_id: str) -> None:
    """Mark a job as cancelled to signal background threads"""
    _status_manager.cancelled_jobs.add(job_id)
    logger.info(f"🛑 Marked job {job_id} as cancelled | Total cancelled: {len(_status_manager.cancelled_jobs)} | List: {_status_manager.cancelled_jobs}")

def is_job_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled"""
    is_cancelled = job_id in _status_manager.cancelled_jobs
    logger.debug(f"🔍 Checking cancellation for {job_id}: {is_cancelled}")
    return is_cancelled

def unmark_job_cancelled(job_id: str) -> None:
    """Remove job from cancelled list"""
    _status_manager.cancelled_jobs.discard(job_id)
    logger.info(f"Unmarked job {job_id} as cancelled")