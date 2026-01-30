"""
HiveFrame Dead Letter Queue
===========================
Storage and management of failed records for debugging and reprocessing.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .utils import Monitorable, ThreadSafeMixin

# Forward reference to avoid circular import
if TYPE_CHECKING:
    from .exceptions import HiveFrameError


@dataclass
class DeadLetterRecord:
    """
    Record that failed processing and was routed to dead letter queue.

    Contains full context for debugging and potential reprocessing.
    """

    original_data: Any
    error: "HiveFrameError"
    partition_id: str
    worker_id: str
    attempt_count: int
    first_failure: float
    last_failure: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "original_data": str(self.original_data)[:1000],  # Truncate large data
            "error": self.error.to_dict(),
            "partition_id": self.partition_id,
            "worker_id": self.worker_id,
            "attempt_count": self.attempt_count,
            "first_failure": self.first_failure,
            "last_failure": self.last_failure,
            "metadata": self.metadata,
        }


class DeadLetterQueue(ThreadSafeMixin, Monitorable):
    """
    Dead Letter Queue for failed records.

    Provides storage, monitoring, and potential reprocessing of failed items.
    Thread-safe implementation using ThreadSafeMixin.
    """

    def __init__(self, max_size: int = 10000):
        super().__init__()
        self.max_size = max_size
        self._queue: List[DeadLetterRecord] = []
        self._error_counts: Dict[str, int] = {}

    def push(self, record: DeadLetterRecord) -> bool:
        """Add failed record to DLQ. Returns False if full."""
        with self._lock:
            if len(self._queue) >= self.max_size:
                return False

            self._queue.append(record)

            # Track error type counts
            error_type = record.error.__class__.__name__
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

            return True

    def pop(self) -> Optional[DeadLetterRecord]:
        """Remove and return oldest record."""
        with self._lock:
            if self._queue:
                record = self._queue.pop(0)
                error_type = record.error.__class__.__name__
                self._error_counts[error_type] = max(0, self._error_counts.get(error_type, 0) - 1)
                return record
            return None

    def peek(self, n: int = 10) -> List[DeadLetterRecord]:
        """View oldest n records without removing."""
        with self._lock:
            return self._queue[:n]

    def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        with self._lock:
            return {
                "size": len(self._queue),
                "max_size": self.max_size,
                "utilization": len(self._queue) / self.max_size if self.max_size > 0 else 0,
                "error_distribution": dict(self._error_counts),
                "oldest_timestamp": self._queue[0].first_failure if self._queue else None,
                "newest_timestamp": self._queue[-1].last_failure if self._queue else None,
            }

    def clear(self) -> int:
        """Clear all records. Returns count of cleared records."""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            self._error_counts.clear()
            return count

    def get_by_error_type(self, error_type: str) -> List[DeadLetterRecord]:
        """Get all records with specific error type."""
        with self._lock:
            return [r for r in self._queue if r.error.__class__.__name__ == error_type]

    def __len__(self) -> int:
        """Return number of records in queue."""
        with self._lock:
            return len(self._queue)

    def __bool__(self) -> bool:
        """Return True if queue has records."""
        with self._lock:
            return len(self._queue) > 0
