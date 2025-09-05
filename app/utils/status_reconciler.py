import threading
import time
from datetime import datetime, timezone, timedelta
from app.config.settings import settings
from app.utils.db_sync_operations import SyncDBOperations
from app.utils.runpod_service import runpod_service
from app.utils.job_utils import job_utils
from app.utils.cleanup_utils import cleanup_utils

class StatusReconciler:
    def __init__(self):
        self._running = False
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._reconciliation_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _reconciliation_loop(self):
        cleanup_counter = 0
        while not self._stop_event.is_set():
            try:
                self._reconcile_statuses()
                
                # Run cleanup every 6 cycles (1 hour)
                cleanup_counter += 1
                if cleanup_counter >= 6:
                    cleanup_counter = 0
                    self._run_periodic_cleanup()
                    
            except Exception:
                pass
            time.sleep(600)

    def _reconcile_statuses(self):
        client = SyncDBOperations._get_sync_client()
        try:
            db = client[settings.DB_NAME]
            collection = db.separation_jobs

            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)

            jobs = list(collection.find({
                "status": {"$in": ["processing", "pending"]},
                "runpod_request_id": {"$exists": True},
                "updated_at": {"$lt": cutoff_time}
            }).limit(5))

            for job in jobs:
                self._reconcile_job(job)

        finally:
            client.close()

    def _reconcile_job(self, job_data):
        job_id = job_data["job_id"]
        runpod_id = job_data["runpod_request_id"]

        try:
            runpod_status = runpod_service.get_separation_status(runpod_id)

            if runpod_status:
                runpod_state = runpod_status.get("status")
                db_state = job_data.get("status")

                if runpod_state != db_state:
                    self._update_job_status(job_id, runpod_state, runpod_status.get("progress", 0))

                    if runpod_state == "completed":
                        self._handle_completion(job_data, runpod_status)

        except Exception:
            pass

    def _update_job_status(self, job_id: str, status: str, progress: int):
        SyncDBOperations.update_separation_job_status(job_id, status, progress)

    def _handle_completion(self, job_data: dict, runpod_status: dict):
        job_id = job_data["job_id"]
        user_id = job_data.get("user_id")

        if runpod_status.get("result"):
            result = runpod_status["result"]
            vocal_url = result.get("vocal_audio") or result.get("vocals")
            instrument_url = result.get("instrument_audio") or result.get("instruments")

            self._update_job_status(job_id, "completed", 100)

            if user_id:
                job_utils.complete_job_billing_sync(job_id, "separation", user_id)

            cleanup_utils.cleanup_job_comprehensive(job_id, "separation")

    def _run_periodic_cleanup(self):
        """Run periodic cleanup of old tmp files"""
        try:
            import logging
            logger = logging.getLogger(__name__)
            
            cleaned_count = cleanup_utils.cleanup_old_files(hours_old=1)
            if cleaned_count > 0:
                logger.info(f"Periodic cleanup: removed {cleaned_count} old tmp folders")
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Periodic cleanup failed: {e}")

_reconciler = StatusReconciler()
