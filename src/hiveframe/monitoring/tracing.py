"""
HiveFrame Distributed Tracing
==============================
Distributed tracing for colony operations.

Tracks execution flow across workers and partitions.
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from contextlib import contextmanager


@dataclass
class TraceSpan:
    """A span in a distributed trace."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # 'ok', 'error'

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """
    Distributed tracing for colony operations.

    Tracks execution flow across workers and partitions.
    """

    def __init__(self):
        self._spans: Dict[str, List[TraceSpan]] = defaultdict(list)
        self._lock = threading.Lock()
        self._span_counter = 0

    def _generate_id(self) -> str:
        """Generate unique span ID."""
        with self._lock:
            self._span_counter += 1
            return f"{time.time_ns()}_{self._span_counter}"

    def start_trace(self, operation: str, **tags) -> TraceSpan:
        """Start a new trace."""
        trace_id = self._generate_id()
        span = TraceSpan(
            trace_id=trace_id,
            span_id=self._generate_id(),
            parent_span_id=None,
            operation=operation,
            start_time=time.time(),
            tags=tags,
        )

        with self._lock:
            self._spans[trace_id].append(span)

        return span

    def start_span(
        self, trace_id: str, operation: str, parent_span_id: Optional[str] = None, **tags
    ) -> TraceSpan:
        """Start a new span within a trace."""
        span = TraceSpan(
            trace_id=trace_id,
            span_id=self._generate_id(),
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time(),
            tags=tags,
        )

        with self._lock:
            self._spans[trace_id].append(span)

        return span

    def end_span(self, span: TraceSpan, status: str = "ok") -> None:
        """End a span."""
        span.end_time = time.time()
        span.status = status

    def log_to_span(self, span: TraceSpan, message: str, **fields) -> None:
        """Add log entry to span."""
        span.logs.append({"timestamp": time.time(), "message": message, **fields})

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        with self._lock:
            return list(self._spans.get(trace_id, []))

    @contextmanager
    def trace_operation(self, operation: str, **tags):
        """Context manager for tracing an operation."""
        span = self.start_trace(operation, **tags)
        try:
            yield span
            self.end_span(span, "ok")
        except Exception as e:
            self.log_to_span(span, f"Error: {e}")
            self.end_span(span, "error")
            raise


# Global tracer instance
_default_tracer = Tracer()


def get_tracer() -> Tracer:
    """Get the default tracer."""
    return _default_tracer
