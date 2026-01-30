"""
HiveFrame Streaming Delivery Guarantees
=======================================
Delivery guarantee support for exactly-once, at-least-once, and at-most-once semantics.
"""

import hashlib
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional

from .core import StreamRecord


class DeliveryGuarantee(Enum):
    """Message delivery guarantees."""

    AT_MOST_ONCE = auto()  # Fire and forget
    AT_LEAST_ONCE = auto()  # May duplicate on failure
    EXACTLY_ONCE = auto()  # Idempotent processing


@dataclass
class ProcessingContext:
    """Context for processing with delivery guarantees."""

    record: StreamRecord
    delivery_guarantee: DeliveryGuarantee
    checkpoint_id: Optional[str] = None
    attempt: int = 1
    idempotency_key: Optional[str] = None

    def generate_idempotency_key(self) -> str:
        """Generate unique key for exactly-once deduplication."""
        if self.idempotency_key:
            return self.idempotency_key

        key_data = f"{self.record.key}:{self.record.timestamp}:{self.record.partition}"
        return hashlib.md5(key_data.encode()).hexdigest()


class IdempotencyStore:
    """
    Store for tracking processed records (exactly-once semantics).

    In production, this would be backed by Redis or similar.
    """

    def __init__(self, ttl_seconds: float = 3600):
        self.ttl_seconds = ttl_seconds
        self._processed: Dict[str, float] = {}  # key -> timestamp
        self._lock = threading.Lock()

    def mark_processed(self, idempotency_key: str) -> None:
        """Mark a record as processed."""
        with self._lock:
            self._processed[idempotency_key] = time.time()
            self._cleanup()

    def is_duplicate(self, idempotency_key: str) -> bool:
        """Check if record was already processed."""
        with self._lock:
            return idempotency_key in self._processed

    def _cleanup(self) -> None:
        """Remove expired entries."""
        cutoff = time.time() - self.ttl_seconds
        to_remove = [k for k, v in self._processed.items() if v < cutoff]
        for k in to_remove:
            del self._processed[k]
