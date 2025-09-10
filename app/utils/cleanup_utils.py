import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.config.settings import settings

logger = logging.getLogger(__name__)


class CleanupUtils:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._job_locks = {}
            self._lock = threading.Lock()
            self._scheduled_cleanups = {}

    # ===== FILE SYSTEM CLEANUP =====

    def cleanup_old_files(self, hours_old: int = 1) -> int:
        temp_dir = Path(settings.TEMP_DIR)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_old)
        cleaned_count = 0

        # Clean job-specific folders (dub_*, separation_*, etc.)
        for pattern in ["dub_*", "sep_*", "separation_*", "job_*", "temp_*"]:
            for path in temp_dir.glob(pattern):
                if path.is_dir() and self._should_cleanup_folder(path, cutoff_time):
                    shutil.rmtree(path, ignore_errors=True)
                    cleaned_count += 1
                    logger.info(f"Cleaned orphaned folder: {path.name}")

        # Clean old loose files in temp directory
        for path in temp_dir.iterdir():
            if path.is_file() and self._should_cleanup_file(path, cutoff_time):
                try:
                    path.unlink()
                    cleaned_count += 1
                    logger.info(f"Cleaned old file: {path.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove file {path.name}: {e}")

        return cleaned_count

    def cleanup_job(self, job_id: str, job_type: str = "dub") -> bool:
        with self._get_job_lock(job_id):
            try:
                self._cleanup_job_files(job_id, job_type)
                self._cleanup_job_dirs(job_id)
                return True
            except Exception as e:
                logger.error(f"Cleanup failed for {job_id}: {e}")
                return False
            finally:
                self._remove_job_lock(job_id)

    def cleanup_dirs(self, patterns: List[str]) -> int:
        temp_dir = Path(settings.TEMP_DIR)
        cleaned_count = 0

        for pattern in patterns:
            for path in temp_dir.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                    cleaned_count += 1

        return cleaned_count

    # ===== AUTO CLEANUP SCHEDULING =====

    def schedule_auto_cleanup(self, job_id: str, delay_minutes: int = 30) -> None:
        if job_id in self._scheduled_cleanups:
            return

        def delayed_cleanup():
            try:
                time.sleep(delay_minutes * 60)
                self.cleanup_job(job_id)
                logger.info(f"Auto cleanup completed for {job_id}")
            finally:
                self._scheduled_cleanups.pop(job_id, None)

        self._scheduled_cleanups[job_id] = True
        cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
        cleanup_thread.start()
        logger.info(f"Scheduled auto cleanup for {job_id} in {delay_minutes} minutes")

    def cancel_scheduled_cleanup(self, job_id: str) -> None:
        self._scheduled_cleanups.pop(job_id, None)

    # ===== CACHE & MEMORY CLEANUP =====

    def cleanup_expired_cache(self) -> int:
        # Simple status service doesn't use complex caching, so no cleanup needed
        logger.info("Cache cleanup skipped - using simple status service")
        return 0

    def cleanup_gpu_memory(self) -> bool:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cache cleared")
                return True
        except Exception as e:
            logger.warning(f"GPU cleanup failed: {e}")
        return False

    def cleanup_aggressive_gpu(self) -> bool:
        try:
            import torch
            import gc

            if not torch.cuda.is_available():
                return False

            # Smart cleanup - preserve loaded AI models
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            if current_memory < 1.0:  # Skip aggressive cleanup if memory usage is low
                logger.info(f"Skipping aggressive cleanup - memory usage is low ({current_memory:.2f}GB)")
                return True

            # Single cache clear instead of multiple - preserve model weights
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
            new_memory = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"Smart GPU cleanup: {current_memory:.2f}GB → {new_memory:.2f}GB")
            return True
        except Exception as e:
            logger.warning(f"Smart GPU cleanup failed: {e}")
        return False

    # ===== R2 STORAGE CLEANUP =====

    def delete_r2_file(self, r2_key: str) -> bool:
        try:
            from app.services.r2_service import R2Service
            r2_service = R2Service()
            result = r2_service.delete_file(r2_key)
            return result.get("success", False)
        except Exception as e:
            logger.error(f"R2 delete failed for {r2_key}: {e}")
            return False



    # ===== COMPREHENSIVE JOB CLEANUP =====

    def cleanup_job_comprehensive(self, job_id: str, job_type: str = "dub") -> Dict[str, bool]:
        results = {
            "files_cleaned": False,
            "dirs_cleaned": False
        }

        try:
            results["files_cleaned"] = self.cleanup_job(job_id, job_type)
            results["dirs_cleaned"] = self._cleanup_job_dirs(job_id)

            logger.info(f"Comprehensive cleanup completed for {job_type} job {job_id}")
            return results
        except Exception as e:
            logger.error(f"Comprehensive cleanup failed for {job_id}: {e}")
            return results

    # ===== BULK CLEANUP OPERATIONS =====

    def cleanup_all_expired(self) -> Dict[str, int]:
        results = {
            "old_files": 0,
            "expired_cache": 0,
            "gpu_memory": 0
        }

        try:
            results["old_files"] = self.cleanup_old_files()
            results["expired_cache"] = self.cleanup_expired_cache()
            results["gpu_memory"] = 1 if self.cleanup_gpu_memory() else 0

            logger.info(f"Bulk cleanup completed: {results}")
            return results
        except Exception as e:
            logger.error(f"Bulk cleanup failed: {e}")
            return results

    # ===== INTERNAL HELPER METHODS =====

    def _should_cleanup_folder(self, path: Path, cutoff_time: datetime) -> bool:
        try:
            dir_mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            if dir_mtime >= cutoff_time:
                return False

            for file_path in path.rglob("*"):
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)
                    if file_mtime >= cutoff_time:
                        return False

            return True
        except Exception:
            return True

    def _should_cleanup_file(self, path: Path, cutoff_time: datetime) -> bool:
        try:
            file_mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            return file_mtime < cutoff_time
        except Exception:
            return True

    def _cleanup_job_files(self, job_id: str, job_type: str = "separation"):
        from app.utils.db_sync_operations import SyncDBOperations

        client = SyncDBOperations._get_sync_client()
        try:
            db = client[settings.DB_NAME]
            collection_name = "separation_jobs" if job_type == "separation" else "dub_jobs"
            job_data = db[collection_name].find_one({"job_id": job_id})

            if job_data and job_data.get("details"):
                local_path = job_data["details"].get("local_audio_path")
                if local_path and os.path.exists(local_path):
                    os.remove(local_path)
        finally:
            client.close()

    def _cleanup_job_dirs(self, job_id: str) -> bool:
        temp_dir = Path(settings.TEMP_DIR)
        patterns = [
            job_id
        ]

        cleaned_any = False
        for pattern in patterns:
            folder_path = temp_dir / pattern
            if folder_path.exists():
                shutil.rmtree(folder_path, ignore_errors=True)
                cleaned_any = True
                logger.info(f"Removed temp dir: {folder_path}")

        return cleaned_any

    def _get_job_lock(self, job_id: str) -> threading.Lock:
        with self._lock:
            if job_id not in self._job_locks:
                self._job_locks[job_id] = threading.Lock()
            return self._job_locks[job_id]

    def _remove_job_lock(self, job_id: str):
        with self._lock:
            self._job_locks.pop(job_id, None)


cleanup_utils = CleanupUtils()



