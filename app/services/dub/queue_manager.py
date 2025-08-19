from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Optional, Dict, Any, Callable


class DubQueueManager:
    """Reusable in-memory queue + scheduler for dub jobs.

    - Keeps at most `max_concurrency` jobs running concurrently
    - Remaining jobs wait in FIFO queue
    - Provides queue position lookup for clients
    - Runner function is injected per job to avoid circular imports
    """

    def __init__(self, max_concurrency: int = 10):
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = threading.Lock()
        self._queue: deque[Dict[str, Any]] = deque()
        self._queue_lock = threading.Lock()
        self._inflight_count: int = 0
        self._max_concurrency = max_concurrency

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=self._max_concurrency,
                        thread_name_prefix="dub_worker",
                    )
        return self._executor

    def enqueue(self, job_id: str, run_fn: Callable[[], None]) -> None:
        """Enqueue a job with a zero-argument runner callable."""
        with self._queue_lock:
            self._queue.append({"job_id": job_id, "run": run_fn})
            self._schedule_locked()

    def get_position(self, job_id: str) -> Optional[int]:
        with self._queue_lock:
            for idx, task in enumerate(self._queue):
                if task.get("job_id") == job_id:
                    return idx + 1
        return None

    def _schedule_locked(self) -> None:
        executor = self._get_executor()
        while self._inflight_count < self._max_concurrency and self._queue:
            task = self._queue.popleft()
            self._inflight_count += 1
            executor.submit(self._run_wrapper, task)

    def _run_wrapper(self, task: Dict[str, Any]) -> None:
        try:
            run_fn = task.get("run")
            if callable(run_fn):
                run_fn()
        except Exception:
            pass
        finally:
            with self._queue_lock:
                self._inflight_count = max(0, self._inflight_count - 1)
                self._schedule_locked()


_instance: Optional[DubQueueManager] = None
_instance_lock = threading.Lock()


def get_dub_queue_manager(max_concurrency: int = 10) -> DubQueueManager:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = DubQueueManager(max_concurrency=max_concurrency)
    return _instance


