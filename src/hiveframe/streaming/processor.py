"""
HiveFrame Enhanced Stream Processor
===================================
Production-grade stream processor with windowing, watermarks,
and delivery guarantees.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, TYPE_CHECKING

from ..core import ColonyState

from .core import StreamRecord, StreamBuffer, StreamPartitioner
from .windows import Window, WindowAssigner, TumblingWindowAssigner
from .watermarks import Watermark, WatermarkGenerator, BoundedOutOfOrdernessWatermarkGenerator
from .state import Checkpoint, InMemoryStateBackend
from .delivery import DeliveryGuarantee, ProcessingContext, IdempotencyStore

# Optional monitoring - use dependency injection for better decoupling
if TYPE_CHECKING:
    from ..monitoring import Logger, MetricsRegistry


K = TypeVar("K")
V = TypeVar("V")


def _get_default_logger(name: str) -> "Logger":
    """Lazily get a logger instance."""
    from ..monitoring import get_logger

    return get_logger(name)


def _get_default_registry() -> "MetricsRegistry":
    """Lazily get a metrics registry instance."""
    from ..monitoring import get_registry

    return get_registry()


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

    Monitoring is optional and can be injected via constructor for better testability.
    """

    def __init__(
        self,
        num_workers: int = 8,
        window_assigner: Optional[WindowAssigner] = None,
        watermark_generator: Optional[WatermarkGenerator] = None,
        delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE,
        checkpoint_interval_seconds: float = 60.0,
        late_data_handling: str = "drop",  # 'drop', 'sideoutput', 'update'
        allowed_lateness_seconds: float = 0,
        *,
        logger: Optional["Logger"] = None,
        metrics_registry: Optional["MetricsRegistry"] = None,
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

        # Monitoring - use injected or default instances (lazy initialization)
        self._logger = logger
        self._metrics_registry = metrics_registry
        self._metrics_initialized = False

        # Control
        self._running = False
        self._workers: List[threading.Thread] = []

    def _ensure_metrics(self) -> None:
        """Lazily initialize metrics on first use."""
        if self._metrics_initialized:
            return

        # Get logger
        if self._logger is None:
            self._logger = _get_default_logger("streaming.enhanced")

        # Get metrics registry
        if self._metrics_registry is None:
            self._metrics_registry = _get_default_registry()

        # Create metrics
        self._records_processed = self._metrics_registry.counter(
            "stream_records_processed", "Total records processed"
        )
        self._late_records = self._metrics_registry.counter(
            "stream_late_records", "Late records received"
        )
        self._windows_triggered = self._metrics_registry.counter(
            "stream_windows_triggered", "Windows triggered"
        )
        self._processing_latency = self._metrics_registry.histogram(
            "stream_processing_latency", "Processing latency"
        )

        self._metrics_initialized = True

    def process_record(
        self,
        record: StreamRecord,
        aggregator: Callable[[V, Any], V] = None,
        initial_value: V = None,
        *,
        agg: Callable[[V, Any], V] = None,  # Alias for aggregator
    ) -> Optional[Any]:
        """
        Process a single record through the windowing pipeline.

        Returns result if window triggered, None otherwise.
        """
        # Ensure metrics are initialized
        self._ensure_metrics()

        # Support both 'aggregator' and 'agg' parameter names
        if aggregator is None and agg is not None:
            aggregator = agg
        if aggregator is None:
            raise ValueError("Must provide aggregator function")
        if initial_value is None:
            initial_value = 0  # Default initial value

        start_time = time.time()

        # Check delivery guarantee
        if self.delivery_guarantee == DeliveryGuarantee.EXACTLY_ONCE:
            ctx = ProcessingContext(record, self.delivery_guarantee)
            idem_key = ctx.generate_idempotency_key()

            if self._idempotency.is_duplicate(idem_key):
                self._logger.debug("Duplicate record skipped", key=record.key)
                return None

        # Update watermark
        new_watermark = self.watermark_generator.on_event(record.timestamp)
        if new_watermark:
            self._advance_watermark(new_watermark)

        # Check for late data
        if self._current_watermark.is_late(record.timestamp):
            self._late_records.inc()

            if self.late_data_handling == "drop":
                if record.timestamp < self._current_watermark.timestamp - self.allowed_lateness:
                    self._logger.debug("Dropping late record", key=record.key)
                    return None
            elif self.late_data_handling == "sideoutput":
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
                        first_timestamp=record.timestamp,
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
                    "count": v.record_count,
                    "accumulator": str(v.accumulator)[:100],
                }
                for k, v in self._window_state.items()
            }

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            watermark=self._current_watermark.timestamp,
            offsets={},  # Would track actual offsets
            window_state=window_state,
        )

        self._state_backend.save_checkpoint(checkpoint)
        self._last_checkpoint = time.time()

        self._logger.info("Checkpoint created", checkpoint_id=checkpoint_id)

        return checkpoint

    def restore_from_checkpoint(self, checkpoint: Optional[Checkpoint] = None) -> bool:
        """Restore state from checkpoint."""
        if checkpoint is None:
            checkpoint = self._state_backend.get_latest_checkpoint()

        if checkpoint is None:
            return False

        self._current_watermark = Watermark(timestamp=checkpoint.watermark)
        self._last_checkpoint = checkpoint.timestamp

        self._logger.info("Restored from checkpoint", checkpoint_id=checkpoint.checkpoint_id)

        return True

    def get_metrics(self) -> Dict[str, Any]:
        """Get current stream processor metrics."""
        self._ensure_metrics()  # Ensure metrics are initialized
        return {
            "records_processed": self._records_processed.get(),
            "late_records": self._late_records.get(),
            "windows_triggered": self._windows_triggered.get(),
            "current_watermark": self._current_watermark.timestamp,
            "active_windows": len(self._window_state),
            "late_data_buffer": len(self._late_data),
            "colony_temperature": self._colony.get_colony_temperature(),
        }
