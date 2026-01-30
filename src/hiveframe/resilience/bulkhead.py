"""
HiveFrame Resilience - Bulkhead
===============================
Bulkhead pattern for resource isolation.
"""

import threading
from typing import Any, Callable, Dict, TypeVar

from ..exceptions import WorkerExhausted

T = TypeVar("T")


class Bulkhead:
    """
    Bulkhead Pattern
    ----------------
    Isolates failures by limiting concurrent access to a resource.

    Like compartments in a ship's hull, if one area fails,
    others remain unaffected.

    Inspired by bee hive structure - separate cells for
    different functions prevent colony-wide failures.
    """

    def __init__(
        self, name: str, max_concurrent: int = 10, max_queue: int = 100, timeout: float = 30.0
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.timeout = timeout

        self._semaphore = threading.Semaphore(max_concurrent)
        self._queue_count = 0
        self._lock = threading.Lock()
        self._active_count = 0
        self._rejected_count = 0

    @property
    def current_concurrent(self) -> int:
        """Current number of concurrent executions."""
        with self._lock:
            return self._active_count

    def __enter__(self):
        """Context manager entry - acquire slot."""
        if not self.acquire():
            raise WorkerExhausted(
                f"Bulkhead '{self.name}' is full",
                active_workers=self._active_count,
                total_workers=self.max_concurrent,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release slot."""
        self.release()
        return False

    def acquire(self) -> bool:
        """
        Acquire slot in bulkhead.
        Returns False if would exceed limits.
        """
        with self._lock:
            if self._active_count >= self.max_concurrent:
                if self._queue_count >= self.max_queue:
                    self._rejected_count += 1
                    return False
                self._queue_count += 1

        acquired = self._semaphore.acquire(timeout=self.timeout)

        with self._lock:
            if self._queue_count > 0:
                self._queue_count -= 1
            if acquired:
                self._active_count += 1
            else:
                self._rejected_count += 1

        return acquired

    def release(self) -> None:
        """Release slot in bulkhead."""
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
        self._semaphore.release()

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function within bulkhead."""
        if not self.acquire():
            raise WorkerExhausted(
                f"Bulkhead '{self.name}' is full",
                active_workers=self._active_count,
                total_workers=self.max_concurrent,
            )

        try:
            return func(*args, **kwargs)
        finally:
            self.release()

    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        with self._lock:
            return {
                "name": self.name,
                "active": self._active_count,
                "queued": self._queue_count,
                "max_concurrent": self.max_concurrent,
                "max_queue": self.max_queue,
                "rejected_count": self._rejected_count,
                "utilization": self._active_count / self.max_concurrent,
            }
