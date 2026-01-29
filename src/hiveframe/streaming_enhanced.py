"""
HiveFrame Enhanced Streaming
============================
Production-grade streaming with windowing, watermarks, and delivery guarantees.

Builds on the base streaming module with:
- Windowing functions (tumbling, sliding, session)
- Watermark support for late data handling
- Exactly-once semantics (simulated)
- Checkpointing and state management
"""

import time
import threading
import json
import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, 
    Optional, Tuple, TypeVar, Union
)
from enum import Enum, auto
from queue import Queue, Empty
import math

from .core import ColonyState, BeeRole, WaggleDance, Pheromone
from .streaming import StreamRecord, StreamBuffer, StreamPartitioner
from .exceptions import HiveFrameError, TransientError, ProcessingError
from .resilience import RetryPolicy, CircuitBreaker, BackoffStrategy
from .monitoring import get_logger, get_registry, Counter, Gauge, Histogram


logger = get_logger("streaming.enhanced")
metrics = get_registry()

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


# ============================================================================
# Windowing
# ============================================================================

class WindowType(Enum):
    """Types of time windows."""
    TUMBLING = auto()   # Fixed, non-overlapping windows
    SLIDING = auto()    # Fixed, overlapping windows
    SESSION = auto()    # Gap-based dynamic windows


@dataclass
class Window:
    """Represents a time window."""
    start: float        # Window start timestamp
    end: float          # Window end timestamp
    window_type: WindowType
    
    @property
    def duration(self) -> float:
        return self.end - self.start
        
    def contains(self, timestamp: float) -> bool:
        return self.start <= timestamp < self.end
        
    def __hash__(self):
        return hash((self.start, self.end))
        
    def __eq__(self, other):
        if not isinstance(other, Window):
            return False
        return self.start == other.start and self.end == other.end


@dataclass 
class WindowedValue(Generic[K, V]):
    """A value associated with a window."""
    key: K
    value: V
    window: Window
    timestamp: float
    is_late: bool = False


class WindowAssigner(ABC):
    """Base class for window assignment strategies."""
    
    @abstractmethod
    def assign_windows(self, timestamp: float) -> List[Window]:
        """Assign windows for a given event timestamp."""
        pass
        
    @abstractmethod
    def get_next_window_end(self, current_time: float) -> float:
        """Get the end time of the next window to trigger."""
        pass


class TumblingWindowAssigner(WindowAssigner):
    """
    Tumbling Window Assigner
    
    Creates fixed-size, non-overlapping windows.
    
    Example: 5-minute tumbling windows
    [00:00-00:05], [00:05-00:10], [00:10-00:15], ...
    """
    
    def __init__(self, size_seconds: float):
        self.size_seconds = size_seconds
        
    def assign_windows(self, timestamp: float) -> List[Window]:
        window_start = math.floor(timestamp / self.size_seconds) * self.size_seconds
        return [Window(
            start=window_start,
            end=window_start + self.size_seconds,
            window_type=WindowType.TUMBLING
        )]
        
    def get_next_window_end(self, current_time: float) -> float:
        window_start = math.floor(current_time / self.size_seconds) * self.size_seconds
        return window_start + self.size_seconds


class SlidingWindowAssigner(WindowAssigner):
    """
    Sliding Window Assigner
    
    Creates overlapping windows with fixed size and slide interval.
    
    Example: 10-minute windows sliding every 5 minutes
    [00:00-00:10], [00:05-00:15], [00:10-00:20], ...
    """
    
    def __init__(self, size_seconds: float, slide_seconds: float):
        self.size_seconds = size_seconds
        self.slide_seconds = slide_seconds
        
    def assign_windows(self, timestamp: float) -> List[Window]:
        windows = []
        
        # Find all windows that contain this timestamp
        window_start = math.floor(timestamp / self.slide_seconds) * self.slide_seconds
        
        while window_start + self.size_seconds > timestamp:
            windows.append(Window(
                start=window_start,
                end=window_start + self.size_seconds,
                window_type=WindowType.SLIDING
            ))
            window_start -= self.slide_seconds
            
            # Safety limit
            if len(windows) > 100:
                break
                
        return windows
        
    def get_next_window_end(self, current_time: float) -> float:
        slot = math.floor(current_time / self.slide_seconds)
        return (slot + 1) * self.slide_seconds


class SessionWindowAssigner(WindowAssigner):
    """
    Session Window Assigner
    
    Creates windows based on activity gaps. A new window starts
    after a period of inactivity.
    
    Particularly useful for user session analysis.
    """
    
    def __init__(self, gap_seconds: float):
        self.gap_seconds = gap_seconds
        self._sessions: Dict[Any, Window] = {}  # key -> current session window
        self._lock = threading.Lock()
        
    def assign_windows(self, timestamp: float, key: Any = None) -> List[Window]:
        with self._lock:
            if key in self._sessions:
                session = self._sessions[key]
                
                # Check if within gap
                if timestamp < session.end + self.gap_seconds:
                    # Extend session
                    session.end = timestamp + self.gap_seconds
                    return [session]
                    
            # Start new session
            new_session = Window(
                start=timestamp,
                end=timestamp + self.gap_seconds,
                window_type=WindowType.SESSION
            )
            self._sessions[key] = new_session
            return [new_session]
            
    def get_next_window_end(self, current_time: float) -> float:
        # Sessions don't have fixed ends
        return current_time + self.gap_seconds


# ============================================================================
# Watermarks
# ============================================================================

@dataclass
class Watermark:
    """
    Watermark for tracking event-time progress.
    
    A watermark with timestamp T asserts that no more events
    with timestamp < T will arrive.
    """
    timestamp: float
    source: str = "default"
    
    def is_late(self, event_timestamp: float) -> bool:
        """Check if an event is late relative to this watermark."""
        return event_timestamp < self.timestamp


class WatermarkGenerator(ABC):
    """Base class for watermark generation strategies."""
    
    @abstractmethod
    def on_event(self, timestamp: float) -> Optional[Watermark]:
        """Called for each event. Returns watermark if one should be emitted."""
        pass
        
    @abstractmethod
    def on_periodic_emit(self) -> Optional[Watermark]:
        """Called periodically to emit watermarks even without events."""
        pass


class BoundedOutOfOrdernessWatermarkGenerator(WatermarkGenerator):
    """
    Generates watermarks allowing for bounded out-of-order events.
    
    The watermark is always (max_seen_timestamp - max_out_of_orderness).
    """
    
    def __init__(
        self,
        max_out_of_orderness_seconds: float = 5.0,
        emit_interval_seconds: float = 1.0
    ):
        self.max_out_of_orderness = max_out_of_orderness_seconds
        self.emit_interval = emit_interval_seconds
        self._max_timestamp = 0.0
        self._last_emit = 0.0
        self._lock = threading.Lock()
        
    def on_event(self, timestamp: float) -> Optional[Watermark]:
        with self._lock:
            self._max_timestamp = max(self._max_timestamp, timestamp)
            
            current_time = time.time()
            if current_time - self._last_emit >= self.emit_interval:
                self._last_emit = current_time
                return Watermark(
                    timestamp=self._max_timestamp - self.max_out_of_orderness
                )
        return None
        
    def on_periodic_emit(self) -> Optional[Watermark]:
        with self._lock:
            self._last_emit = time.time()
            return Watermark(
                timestamp=self._max_timestamp - self.max_out_of_orderness
            )


class PunctuatedWatermarkGenerator(WatermarkGenerator):
    """
    Generates watermarks based on special marker events.
    
    Some data sources include watermark markers in the stream itself.
    """
    
    def __init__(self, is_watermark_event: Callable[[Any], Optional[float]]):
        self.is_watermark_event = is_watermark_event
        self._current_watermark = 0.0
        
    def on_event(self, timestamp: float, event: Any = None) -> Optional[Watermark]:
        if event is not None:
            wm_timestamp = self.is_watermark_event(event)
            if wm_timestamp is not None:
                self._current_watermark = max(self._current_watermark, wm_timestamp)
                return Watermark(timestamp=wm_timestamp)
        return None
        
    def on_periodic_emit(self) -> Optional[Watermark]:
        return Watermark(timestamp=self._current_watermark)


# ============================================================================
# State Management & Checkpointing
# ============================================================================

@dataclass
class Checkpoint:
    """Represents a point-in-time snapshot of processing state."""
    checkpoint_id: str
    timestamp: float
    watermark: float
    offsets: Dict[int, int]  # partition -> offset
    window_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        return json.dumps({
            'checkpoint_id': self.checkpoint_id,
            'timestamp': self.timestamp,
            'watermark': self.watermark,
            'offsets': self.offsets,
            'window_state': {k: str(v)[:1000] for k, v in self.window_state.items()},
            'metadata': self.metadata
        })


class StateBackend(ABC):
    """Abstract state storage backend."""
    
    @abstractmethod
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint."""
        pass
        
    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a specific checkpoint."""
        pass
        
    @abstractmethod
    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get the most recent checkpoint."""
        pass


class InMemoryStateBackend(StateBackend):
    """In-memory state backend for testing."""
    
    def __init__(self, max_checkpoints: int = 10):
        self.max_checkpoints = max_checkpoints
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._ordered: deque = deque(maxlen=max_checkpoints)
        self._lock = threading.Lock()
        
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            self._checkpoints[checkpoint.checkpoint_id] = checkpoint
            self._ordered.append(checkpoint.checkpoint_id)
            
            # Cleanup old checkpoints
            while len(self._checkpoints) > self.max_checkpoints:
                old_id = self._ordered.popleft()
                if old_id in self._checkpoints:
                    del self._checkpoints[old_id]
                    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        with self._lock:
            return self._checkpoints.get(checkpoint_id)
            
    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        with self._lock:
            if self._ordered:
                return self._checkpoints.get(self._ordered[-1])
            return None


# ============================================================================
# Delivery Guarantees
# ============================================================================

class DeliveryGuarantee(Enum):
    """Message delivery guarantees."""
    AT_MOST_ONCE = auto()   # Fire and forget
    AT_LEAST_ONCE = auto()  # May duplicate on failure
    EXACTLY_ONCE = auto()   # Idempotent processing


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


# ============================================================================
# Enhanced Stream Processor
# ============================================================================

@dataclass
class WindowAggregation(Generic[K, V]):
    """Holds aggregation state for a window."""
    window: Window
    key: K
    accumulator: V
    record_count: int = 0
    first_timestamp: float = 0
    last_timestamp: float = 0


class EnhancedStreamProcessor:
    """
    Production-grade stream processor with:
    - Windowing (tumbling, sliding, session)
    - Watermarks and late data handling
    - Exactly-once processing (simulated)
    - Checkpointing
    - Colony-based adaptive scaling
    """
    
    def __init__(
        self,
        num_workers: int = 8,
        window_assigner: Optional[WindowAssigner] = None,
        watermark_generator: Optional[WatermarkGenerator] = None,
        delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE,
        checkpoint_interval_seconds: float = 60.0,
        late_data_handling: str = 'drop',  # 'drop', 'sideoutput', 'update'
        allowed_lateness_seconds: float = 0
    ):
        self.num_workers = num_workers
        self.window_assigner = window_assigner or TumblingWindowAssigner(60.0)
        self.watermark_generator = watermark_generator or BoundedOutOfOrdernessWatermarkGenerator()
        self.delivery_guarantee = delivery_guarantee
        self.checkpoint_interval = checkpoint_interval_seconds
        self.late_data_handling = late_data_handling
        self.allowed_lateness = allowed_lateness_seconds
        
        self._colony = ColonyState()
        self._buffer = StreamBuffer(max_size=10000)
        self._partitioner = StreamPartitioner(num_workers)
        
        # Window state
        self._window_state: Dict[Tuple[Any, Window], WindowAggregation] = {}
        self._window_lock = threading.Lock()
        
        # Watermark tracking
        self._current_watermark = Watermark(timestamp=0)
        
        # Late data side output
        self._late_data: deque = deque(maxlen=1000)
        
        # Checkpointing
        self._state_backend = InMemoryStateBackend()
        self._last_checkpoint = time.time()
        
        # Exactly-once
        self._idempotency = IdempotencyStore()
        
        # Metrics
        self._records_processed = metrics.counter("stream_records_processed", "Total records processed")
        self._late_records = metrics.counter("stream_late_records", "Late records received")
        self._windows_triggered = metrics.counter("stream_windows_triggered", "Windows triggered")
        self._processing_latency = metrics.histogram("stream_processing_latency", "Processing latency")
        
        # Control
        self._running = False
        self._workers: List[threading.Thread] = []
        
    def process_record(
        self,
        record: StreamRecord,
        aggregator: Callable[[V, Any], V],
        initial_value: V
    ) -> Optional[Any]:
        """
        Process a single record through the windowing pipeline.
        
        Returns result if window triggered, None otherwise.
        """
        start_time = time.time()
        
        # Check delivery guarantee
        if self.delivery_guarantee == DeliveryGuarantee.EXACTLY_ONCE:
            ctx = ProcessingContext(record, self.delivery_guarantee)
            idem_key = ctx.generate_idempotency_key()
            
            if self._idempotency.is_duplicate(idem_key):
                logger.debug("Duplicate record skipped", key=record.key)
                return None
                
        # Update watermark
        new_watermark = self.watermark_generator.on_event(record.timestamp)
        if new_watermark:
            self._advance_watermark(new_watermark)
            
        # Check for late data
        if self._current_watermark.is_late(record.timestamp):
            self._late_records.inc()
            
            if self.late_data_handling == 'drop':
                if record.timestamp < self._current_watermark.timestamp - self.allowed_lateness:
                    logger.debug("Dropping late record", key=record.key)
                    return None
            elif self.late_data_handling == 'sideoutput':
                self._late_data.append(record)
                return None
            # 'update' continues processing
            
        # Assign to windows
        windows = self.window_assigner.assign_windows(record.timestamp)
        
        results = []
        with self._window_lock:
            for window in windows:
                state_key = (record.key, window)
                
                if state_key not in self._window_state:
                    self._window_state[state_key] = WindowAggregation(
                        window=window,
                        key=record.key,
                        accumulator=initial_value,
                        first_timestamp=record.timestamp
                    )
                    
                state = self._window_state[state_key]
                state.accumulator = aggregator(state.accumulator, record.value)
                state.record_count += 1
                state.last_timestamp = record.timestamp
                
        # Mark as processed for exactly-once
        if self.delivery_guarantee == DeliveryGuarantee.EXACTLY_ONCE:
            self._idempotency.mark_processed(idem_key)
            
        # Update metrics
        self._records_processed.inc()
        self._processing_latency.observe(time.time() - start_time)
        
        # Check if we should checkpoint
        if time.time() - self._last_checkpoint >= self.checkpoint_interval:
            self._checkpoint()
            
        return None  # Window results come from trigger
        
    def _advance_watermark(self, watermark: Watermark) -> List[WindowAggregation]:
        """
        Advance watermark and trigger completed windows.
        
        Returns list of triggered window aggregations.
        """
        if watermark.timestamp <= self._current_watermark.timestamp:
            return []
            
        old_watermark = self._current_watermark
        self._current_watermark = watermark
        
        triggered = []
        
        with self._window_lock:
            # Find windows that should trigger
            to_remove = []
            
            for state_key, state in self._window_state.items():
                # Window ends before new watermark = trigger
                if state.window.end <= watermark.timestamp:
                    triggered.append(state)
                    to_remove.append(state_key)
                    self._windows_triggered.inc()
                    
            # Remove triggered windows
            for key in to_remove:
                del self._window_state[key]
                
        return triggered
        
    def get_late_data(self, limit: int = 100) -> List[StreamRecord]:
        """Get records routed to late data side output."""
        return list(self._late_data)[-limit:]
        
    def _checkpoint(self) -> Checkpoint:
        """Create a checkpoint of current state."""
        checkpoint_id = f"chk_{int(time.time() * 1000)}"
        
        with self._window_lock:
            window_state = {
                f"{k[0]}:{k[1].start}:{k[1].end}": {
                    'count': v.record_count,
                    'accumulator': str(v.accumulator)[:100]
                }
                for k, v in self._window_state.items()
            }
            
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            watermark=self._current_watermark.timestamp,
            offsets={},  # Would track actual offsets
            window_state=window_state
        )
        
        self._state_backend.save_checkpoint(checkpoint)
        self._last_checkpoint = time.time()
        
        logger.info("Checkpoint created", checkpoint_id=checkpoint_id)
        
        return checkpoint
        
    def restore_from_checkpoint(self, checkpoint: Optional[Checkpoint] = None) -> bool:
        """Restore state from checkpoint."""
        if checkpoint is None:
            checkpoint = self._state_backend.get_latest_checkpoint()
            
        if checkpoint is None:
            return False
            
        self._current_watermark = Watermark(timestamp=checkpoint.watermark)
        self._last_checkpoint = checkpoint.timestamp
        
        logger.info("Restored from checkpoint", checkpoint_id=checkpoint.checkpoint_id)
        
        return True
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current stream processor metrics."""
        return {
            'records_processed': self._records_processed.get(),
            'late_records': self._late_records.get(),
            'windows_triggered': self._windows_triggered.get(),
            'current_watermark': self._current_watermark.timestamp,
            'active_windows': len(self._window_state),
            'late_data_buffer': len(self._late_data),
            'colony_temperature': self._colony.get_colony_temperature()
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def tumbling_window(seconds: float) -> TumblingWindowAssigner:
    """Create a tumbling window assigner."""
    return TumblingWindowAssigner(seconds)


def sliding_window(size_seconds: float, slide_seconds: float) -> SlidingWindowAssigner:
    """Create a sliding window assigner."""
    return SlidingWindowAssigner(size_seconds, slide_seconds)


def session_window(gap_seconds: float) -> SessionWindowAssigner:
    """Create a session window assigner."""
    return SessionWindowAssigner(gap_seconds)


def bounded_watermark(
    max_out_of_orderness: float = 5.0
) -> BoundedOutOfOrdernessWatermarkGenerator:
    """Create a bounded out-of-orderness watermark generator."""
    return BoundedOutOfOrdernessWatermarkGenerator(max_out_of_orderness)


# Example aggregators

def count_aggregator(acc: int, value: Any) -> int:
    """Count aggregator."""
    return acc + 1


def sum_aggregator(acc: float, value: float) -> float:
    """Sum aggregator."""
    return acc + value


def avg_aggregator(acc: Tuple[float, int], value: float) -> Tuple[float, int]:
    """Average aggregator (sum, count)."""
    return (acc[0] + value, acc[1] + 1)


def max_aggregator(acc: float, value: float) -> float:
    """Max aggregator."""
    return max(acc, value)


def min_aggregator(acc: float, value: float) -> float:
    """Min aggregator."""
    return min(acc, value)


def collect_aggregator(acc: List, value: Any) -> List:
    """Collect all values into a list."""
    acc.append(value)
    return acc
