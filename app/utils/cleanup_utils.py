import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, Any, List
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
            self._scheduled_cleanups = {}

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

    def cleanup_job(self, job_id: str) -> bool:
        try:
            deleted_files = []
            temp_paths = [
                Path(settings.TEMP_DIR) / job_id,
                Path("tmp") / job_id,
                Path("tmp") / "downloads" / job_id,
                Path("tmp") / "processed" / job_id
            ]
            
            for job_dir in temp_paths:
                if job_dir.exists():
                    if job_dir.is_dir():
                        for file_path in job_dir.rglob("*"):
                            if file_path.is_file():
                                try:
                                    file_path.unlink()
                                    deleted_files.append(str(file_path))
                                except Exception:
                                    pass
                        shutil.rmtree(job_dir, ignore_errors=True)
                    else:
                        try:
                            job_dir.unlink()
                            deleted_files.append(str(job_dir))
                        except Exception:
                            pass

            self._cleanup_empty_folders()
            
            if deleted_files:
                logger.info(f"Cleaned up {len(deleted_files)} files for job {job_id}")
            
            return True
        except Exception as e:
            logger.error(f"Cleanup failed for {job_id}: {e}")
            return False

    def delete_file_path(self, file_path: str) -> List[str]:
        deleted_files = []
        path = Path(file_path)
        
        try:
            if path.exists():
                if path.is_dir():
                    for file in path.rglob("*"):
                        if file.is_file():
                            try:
                                file.unlink()
                                deleted_files.append(str(file))
                            except Exception:
                                pass
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink()
                    deleted_files.append(str(path))
                    
            self._cleanup_empty_folders()
            
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {e}")
            
        return deleted_files

    def cleanup_old_files(self, hours_old: int = 0.5) -> int:
        temp_dir = Path(settings.TEMP_DIR)
        cleaned_count = 0
        
        if not temp_dir.exists():
            return 0
            
        for item in temp_dir.iterdir():
            try:
                if item.is_dir():
                    if self._is_old_directory(item, hours_old):
                        shutil.rmtree(item, ignore_errors=True)
                        cleaned_count += 1
                elif item.is_file():
                    if self._is_old_file(item, hours_old):
                        item.unlink()
                        cleaned_count += 1
            except Exception:
                pass
                
        self._cleanup_empty_folders()
        return cleaned_count

    def _is_old_directory(self, path: Path, hours_old: float = 0.5) -> bool:
        try:
            cutoff_time = time.time() - (hours_old * 3600)
            return path.stat().st_mtime < cutoff_time
        except Exception:
            return True

    def _is_old_file(self, path: Path, hours_old: float = 0.5) -> bool:
        try:
            cutoff_time = time.time() - (hours_old * 3600)
            return path.stat().st_mtime < cutoff_time
        except Exception:
            return True

    def _cleanup_empty_folders(self):
        temp_paths = [
            Path(settings.TEMP_DIR),
            Path("tmp"),
            Path("tmp") / "downloads",
            Path("tmp") / "processed"
        ]
        
        for base_path in temp_paths:
            if not base_path.exists():
                continue
                
            try:
                for root, dirs, files in os.walk(str(base_path), topdown=False):
                    root_path = Path(root)
                    if root_path != base_path and not files and not dirs:
                        try:
                            root_path.rmdir()
                            logger.info(f"Removed empty folder: {root_path}")
                        except Exception:
                            pass
            except Exception:
                pass

    def cleanup_gpu_memory(self) -> bool:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cleared")
                return True
        except Exception:
            pass
        return False

    def cleanup_job_comprehensive(self, job_id: str, job_type: str = "dub") -> Dict[str, bool]:
        results = {
            "files_cleaned": False,
            "dirs_cleaned": False
        }
        
        try:
            result = self.cleanup_job(job_id)
            results["files_cleaned"] = result
            results["dirs_cleaned"] = result
            logger.info(f"Comprehensive cleanup completed for {job_type} job {job_id}")
            return results
        except Exception as e:
            logger.error(f"Comprehensive cleanup failed for {job_id}: {e}")
            return results

    def cleanup_all_expired(self) -> Dict[str, int]:
        results = {
            "old_files": 0,
            "gpu_memory": 0
        }
        
        try:
            results["old_files"] = self.cleanup_old_files()
            results["gpu_memory"] = 1 if self.cleanup_gpu_memory() else 0
            logger.info(f"Bulk cleanup completed: {results}")
            return results
        except Exception as e:
            logger.error(f"Bulk cleanup failed: {e}")
            return results

cleanup_utils = CleanupUtils()