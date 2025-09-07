import logging
import threading
import time
from typing import Set, Dict, Any
from datetime import datetime, timezone
from pymongo import MongoClient
from app.config.settings import settings
from app.services.redis_status_service import redis_status_service

logger = logging.getLogger(__name__)


class StatusSyncService:
    def __init__(self):
        self.sync_interval = 30
        self.critical_statuses = {
            "processing", "awaiting_review", "completed", "failed"
        }
        self.pending_syncs: Set[str] = set()
        self.sync_lock = threading.Lock()
        self.running = False
        self.sync_thread = None
    
    def start(self):
        if self.running:
            return
        
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        logger.info("Status sync service started")
    
    def stop(self):
        self.running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
        logger.info("Status sync service stopped")
    
    def schedule_sync(self, job_id: str, job_type: str):
        with self.sync_lock:
            self.pending_syncs.add(f"{job_type}:{job_id}")
    
    def _sync_loop(self):
        while self.running:
            try:
                self._process_pending_syncs()
                time.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                time.sleep(5)
    
    def _process_pending_syncs(self):
        if not redis_status_service.is_available():
            return
        
        with self.sync_lock:
            current_syncs = self.pending_syncs.copy()
            self.pending_syncs.clear()
        
        if not current_syncs:
            return
        
        for sync_key in current_syncs:
            try:
                job_type, job_id = sync_key.split(":", 1)
                self._sync_job_to_mongodb(job_id, job_type)
            except Exception as e:
                logger.error(f"Failed to sync {sync_key}: {e}")
    
    def _sync_job_to_mongodb(self, job_id: str, job_type: str):
        try:
            redis_data = redis_status_service.get_status(job_id, job_type)
            
            if not redis_data:
                return
            
            status = redis_data.get("status")
            if status not in self.critical_statuses:
                return
            
            client = MongoClient(settings.MONGODB_URI)
            db = client[settings.DB_NAME]
            collection = db[f"{job_type}_jobs"]
            
            current_job = collection.find_one({"job_id": job_id}, {"details": 1})
            
            update_data = {
                "status": status,
                "progress": redis_data.get("progress", 0),
                "updated_at": datetime.now(timezone.utc)
            }
            
            if status == "processing":
                update_data["started_at"] = datetime.now(timezone.utc)
            elif status in ["completed", "failed"]:
                update_data["completed_at"] = datetime.now(timezone.utc)
            
            details = redis_data.get("details", {})
            if details and isinstance(details, dict):
                existing_details = {}
                if current_job and current_job.get("details"):
                    existing_details = current_job["details"]
                
                merged_details = {**existing_details, **details}
                update_data["details"] = merged_details
            
            result = collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            
            client.close()
            
            if result.modified_count > 0:
                logger.debug(f"Synced {job_id} to MongoDB: {status}")
            
        except Exception as e:
            logger.error(f"MongoDB sync failed for {job_id}: {e}")
    
    def force_sync_all(self, job_type: str = None):
        if not redis_status_service.is_available():
            return
        
        try:
            job_types = [job_type] if job_type else ["dub", "separation"]
            
            for jt in job_types:
                jobs = redis_status_service.get_all_jobs(jt)
                for job in jobs:
                    job_id = job.get("job_id")
                    if job_id:
                        self._sync_job_to_mongodb(job_id, jt)
            
            logger.info(f"Force sync completed for {job_types}")
        except Exception as e:
            logger.error(f"Force sync failed: {e}")


sync_service = StatusSyncService()
