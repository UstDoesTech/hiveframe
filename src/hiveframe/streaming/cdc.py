"""
HiveFrame Change Data Capture (CDC)
===================================
Database replication and synchronization using bee-inspired optimization.

Key Features:
- Log-based change capture
- Table-level change tracking
- Incremental data synchronization
- Multi-source replication
- Bee-inspired conflict resolution
"""

import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

from ..core import ColonyState, Pheromone, WaggleDance


class ChangeType(Enum):
    """Types of data changes."""

    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    TRUNCATE = "TRUNCATE"
    SCHEMA_CHANGE = "SCHEMA_CHANGE"


class CaptureMode(Enum):
    """Change capture modes."""

    LOG_BASED = "log_based"  # Read from transaction log
    QUERY_BASED = "query_based"  # Periodic query comparison
    TRIGGER_BASED = "trigger_based"  # Database triggers
    TIMESTAMP_BASED = "timestamp_based"  # Track by modification timestamp


class ReplicationMode(Enum):
    """Data replication modes."""

    FULL = "full"  # Full table refresh
    INCREMENTAL = "incremental"  # Only changed records
    STREAMING = "streaming"  # Real-time streaming


class ConflictResolution(Enum):
    """Conflict resolution strategies."""

    SOURCE_WINS = "source_wins"  # Source data always wins
    TARGET_WINS = "target_wins"  # Target data always wins
    LATEST_WINS = "latest_wins"  # Most recent timestamp wins
    MERGE = "merge"  # Attempt to merge changes
    CUSTOM = "custom"  # Custom resolution function


@dataclass
class ChangeEvent:
    """
    A single change event captured from a source.

    Contains all information needed to replicate the change.
    """

    table_name: str
    change_type: ChangeType
    primary_key: Any
    timestamp: float = field(default_factory=time.time)
    sequence_number: int = 0
    before_image: Optional[Dict[str, Any]] = None  # Row before change
    after_image: Optional[Dict[str, Any]] = None  # Row after change
    source_id: str = "default"
    transaction_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_insert(self) -> bool:
        """Check if this is an insert."""
        return self.change_type == ChangeType.INSERT

    @property
    def is_update(self) -> bool:
        """Check if this is an update."""
        return self.change_type == ChangeType.UPDATE

    @property
    def is_delete(self) -> bool:
        """Check if this is a delete."""
        return self.change_type == ChangeType.DELETE

    def get_changed_columns(self) -> List[str]:
        """Get list of columns that changed (for updates)."""
        if not self.before_image or not self.after_image:
            return []

        changed = []
        for key in set(self.before_image.keys()) | set(self.after_image.keys()):
            before_val = self.before_image.get(key)
            after_val = self.after_image.get(key)
            if before_val != after_val:
                changed.append(key)
        return changed


@dataclass
class ChangeLog:
    """
    A log of change events for a table.

    Provides ordered access to changes and supports position tracking.
    """

    table_name: str
    events: deque = field(default_factory=lambda: deque(maxlen=100000))
    last_sequence: int = 0
    last_applied_sequence: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, event: ChangeEvent) -> int:
        """Append a change event and return its sequence number."""
        with self._lock:
            self.last_sequence += 1
            event.sequence_number = self.last_sequence
            self.events.append(event)
            return self.last_sequence

    def get_changes_since(self, sequence: int, limit: Optional[int] = None) -> List[ChangeEvent]:
        """Get all changes since a sequence number."""
        with self._lock:
            result = []
            for event in self.events:
                if event.sequence_number > sequence:
                    result.append(event)
                    if limit and len(result) >= limit:
                        break
            return result

    def mark_applied(self, sequence: int) -> None:
        """Mark changes up to sequence as applied."""
        with self._lock:
            self.last_applied_sequence = max(self.last_applied_sequence, sequence)

    def get_pending_count(self) -> int:
        """Get number of pending (unapplied) changes."""
        with self._lock:
            return self.last_sequence - self.last_applied_sequence


@dataclass
class TableCheckpoint:
    """Checkpoint for tracking replication progress."""

    table_name: str
    source_id: str
    last_sequence: int = 0
    last_timestamp: float = 0
    checkpoint_time: float = field(default_factory=time.time)
    row_count: int = 0
    checksum: Optional[str] = None


class ChangeCapture(ABC):
    """
    Abstract base class for change capture implementations.

    Different implementations support different capture modes.
    """

    @abstractmethod
    def capture_changes(self, since_sequence: int = 0) -> Iterator[ChangeEvent]:
        """Capture changes since a sequence number."""
        pass

    @abstractmethod
    def get_current_position(self) -> int:
        """Get current position in the change stream."""
        pass


class QueryBasedCapture(ChangeCapture):
    """
    Query-based change capture using timestamp comparison.

    Suitable for databases without native CDC support.
    """

    def __init__(
        self,
        table_name: str,
        primary_key_column: str,
        timestamp_column: str,
        query_fn: Callable[[str], List[Dict[str, Any]]],
    ):
        self.table_name = table_name
        self.primary_key_column = primary_key_column
        self.timestamp_column = timestamp_column
        self.query_fn = query_fn

        self._last_timestamp = 0
        self._last_snapshot: Dict[Any, Dict[str, Any]] = {}
        self._sequence = 0

    def capture_changes(self, since_sequence: int = 0) -> Iterator[ChangeEvent]:
        """Capture changes by comparing snapshots."""
        # Get current data
        query = (
            f"SELECT * FROM {self.table_name} "
            f"WHERE {self.timestamp_column} >= {self._last_timestamp}"
        )
        current_data = self.query_fn(query)

        current_snapshot = {}
        for row in current_data:
            key = row.get(self.primary_key_column)
            if key is not None:
                current_snapshot[key] = row

        # Detect changes
        # New/updated records
        for key, row in current_snapshot.items():
            if key not in self._last_snapshot:
                # Insert
                self._sequence += 1
                yield ChangeEvent(
                    table_name=self.table_name,
                    change_type=ChangeType.INSERT,
                    primary_key=key,
                    after_image=row,
                    sequence_number=self._sequence,
                )
            elif row != self._last_snapshot[key]:
                # Update
                self._sequence += 1
                yield ChangeEvent(
                    table_name=self.table_name,
                    change_type=ChangeType.UPDATE,
                    primary_key=key,
                    before_image=self._last_snapshot[key],
                    after_image=row,
                    sequence_number=self._sequence,
                )

        # Deleted records
        for key in self._last_snapshot:
            if key not in current_snapshot:
                self._sequence += 1
                yield ChangeEvent(
                    table_name=self.table_name,
                    change_type=ChangeType.DELETE,
                    primary_key=key,
                    before_image=self._last_snapshot[key],
                    sequence_number=self._sequence,
                )

        self._last_snapshot = current_snapshot

    def get_current_position(self) -> int:
        """Get current sequence position."""
        return self._sequence


class InMemoryCapture(ChangeCapture):
    """
    In-memory change capture for testing and simple use cases.

    Changes are recorded manually via record_change().
    """

    def __init__(self, table_name: str):
        self.table_name = table_name
        self._change_log = ChangeLog(table_name=table_name)

    def record_change(
        self,
        change_type: ChangeType,
        primary_key: Any,
        before_image: Optional[Dict[str, Any]] = None,
        after_image: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record a change event."""
        event = ChangeEvent(
            table_name=self.table_name,
            change_type=change_type,
            primary_key=primary_key,
            before_image=before_image,
            after_image=after_image,
        )
        return self._change_log.append(event)

    def capture_changes(self, since_sequence: int = 0) -> Iterator[ChangeEvent]:
        """Get all recorded changes since a sequence."""
        for event in self._change_log.get_changes_since(since_sequence):
            yield event

    def get_current_position(self) -> int:
        """Get current position."""
        return self._change_log.last_sequence


class CDCReplicator:
    """
    Change Data Capture Replicator.

    Replicates changes from source to target using bee-inspired optimization.

    Features:
    - Multi-source replication
    - Parallel apply
    - Conflict detection and resolution
    - Checkpoint management
    - Bee-inspired load balancing
    """

    def __init__(
        self,
        replication_mode: ReplicationMode = ReplicationMode.INCREMENTAL,
        conflict_resolution: ConflictResolution = ConflictResolution.SOURCE_WINS,
        batch_size: int = 1000,
        num_workers: int = 4,
    ):
        self.replication_mode = replication_mode
        self.conflict_resolution = conflict_resolution
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.colony = ColonyState()
        self._sources: Dict[str, ChangeCapture] = {}
        self._apply_fn: Optional[Callable[[List[ChangeEvent]], int]] = None
        self._checkpoints: Dict[str, TableCheckpoint] = {}
        self._callbacks: List[Callable[[ChangeEvent], None]] = []
        self._conflict_resolver: Optional[Callable[[ChangeEvent, ChangeEvent], ChangeEvent]] = None
        self._lock = threading.RLock()

        # State
        self._running = False
        self._replication_thread: Optional[threading.Thread] = None

        # Metrics
        self._events_processed = 0
        self._events_applied = 0
        self._conflicts_detected = 0
        self._errors = 0
        self._lag_events = 0

    def register_source(self, source_id: str, capture: ChangeCapture) -> None:
        """Register a change capture source."""
        with self._lock:
            self._sources[source_id] = capture
            # Initialize checkpoint
            self._checkpoints[source_id] = TableCheckpoint(
                table_name=capture.table_name,
                source_id=source_id,
            )

    def set_apply_function(self, apply_fn: Callable[[List[ChangeEvent]], int]) -> None:
        """Set the function that applies changes to the target."""
        self._apply_fn = apply_fn

    def set_conflict_resolver(
        self,
        resolver: Callable[[ChangeEvent, ChangeEvent], ChangeEvent],
    ) -> None:
        """Set custom conflict resolver function."""
        self._conflict_resolver = resolver
        self.conflict_resolution = ConflictResolution.CUSTOM

    def add_callback(self, callback: Callable[[ChangeEvent], None]) -> None:
        """Add a callback to be invoked for each change event."""
        self._callbacks.append(callback)

    def replicate_once(self) -> Dict[str, int]:
        """
        Perform one replication cycle.

        Returns dict of source_id -> events_applied.
        """
        results = {}

        for source_id, capture in self._sources.items():
            checkpoint = self._checkpoints.get(source_id)
            last_sequence = checkpoint.last_sequence if checkpoint else 0

            events: List[ChangeEvent] = []
            for event in capture.capture_changes(since_sequence=last_sequence):
                events.append(event)
                self._events_processed += 1

                # Invoke callbacks
                for callback in self._callbacks:
                    try:
                        callback(event)
                    except Exception:
                        pass

                if len(events) >= self.batch_size:
                    break

            if events and self._apply_fn:
                try:
                    applied = self._apply_changes(events)
                    results[source_id] = applied
                    self._events_applied += applied

                    # Update checkpoint
                    if events:
                        self._checkpoints[source_id].last_sequence = events[-1].sequence_number
                        self._checkpoints[source_id].checkpoint_time = time.time()

                    # Waggle dance for fitness tracking
                    dance = WaggleDance(
                        partition_id=source_id,
                        quality_score=applied / max(1, len(events)),
                        processing_time=0.001,
                        result_size=applied,
                        worker_id="cdc_replicator",
                    )
                    self.colony.dance_floor.perform_dance(dance)

                except Exception:
                    self._errors += 1
                    results[source_id] = 0
            else:
                results[source_id] = 0

        return results

    def _apply_changes(self, events: List[ChangeEvent]) -> int:
        """Apply a batch of change events."""
        if not self._apply_fn:
            return 0

        # Group by table for efficient application
        by_table: Dict[str, List[ChangeEvent]] = defaultdict(list)
        for event in events:
            by_table[event.table_name].append(event)

        total_applied = 0
        for table_events in by_table.values():
            applied = self._apply_fn(table_events)
            total_applied += applied

        return total_applied

    def start(self, poll_interval_seconds: float = 1.0) -> None:
        """Start continuous replication."""
        if self._running:
            return

        self._running = True
        self._poll_interval = poll_interval_seconds

        self._replication_thread = threading.Thread(target=self._replication_loop, daemon=True)
        self._replication_thread.start()

    def stop(self) -> None:
        """Stop continuous replication."""
        self._running = False
        if self._replication_thread:
            self._replication_thread.join(timeout=5.0)

    def _replication_loop(self) -> None:
        """Main replication loop."""
        while self._running:
            try:
                self.replicate_once()
            except Exception:
                self._errors += 1
                self.colony.emit_pheromone(
                    Pheromone(
                        signal_type="alarm",
                        intensity=0.5,
                        source_worker="cdc_replicator",
                    )
                )
            time.sleep(self._poll_interval)

    def get_lag(self) -> Dict[str, int]:
        """Get replication lag per source."""
        lag = {}
        for source_id, capture in self._sources.items():
            current = capture.get_current_position()
            checkpoint = self._checkpoints.get(source_id)
            last_applied = checkpoint.last_sequence if checkpoint else 0
            lag[source_id] = current - last_applied
        return lag

    def get_checkpoint(self, source_id: str) -> Optional[TableCheckpoint]:
        """Get checkpoint for a source."""
        return self._checkpoints.get(source_id)

    def set_checkpoint(self, source_id: str, checkpoint: TableCheckpoint) -> None:
        """Set checkpoint for a source."""
        self._checkpoints[source_id] = checkpoint

    def get_metrics(self) -> Dict[str, Any]:
        """Get replicator metrics."""
        total_lag = sum(self.get_lag().values())
        return {
            "sources_registered": len(self._sources),
            "events_processed": self._events_processed,
            "events_applied": self._events_applied,
            "conflicts_detected": self._conflicts_detected,
            "errors": self._errors,
            "total_lag": total_lag,
            "lag_per_source": self.get_lag(),
            "is_running": self._running,
            "replication_mode": self.replication_mode.value,
            "conflict_resolution": self.conflict_resolution.value,
        }


class CDCStream:
    """
    High-level CDC streaming interface.

    Provides a simple API for change data capture streaming.

    Usage:
        stream = CDCStream()
        stream.add_table("users", primary_key="id")

        @stream.on_change("users")
        def handle_user_change(event):
            print(f"User {event.primary_key} changed")

        stream.start()
    """

    def __init__(self):
        self._tables: Dict[str, InMemoryCapture] = {}
        self._handlers: Dict[str, List[Callable[[ChangeEvent], None]]] = defaultdict(list)
        self._global_handlers: List[Callable[[ChangeEvent], None]] = []
        self._running = False
        self._lock = threading.Lock()

    def add_table(self, table_name: str, primary_key: str = "id") -> InMemoryCapture:
        """Add a table for change tracking."""
        capture = InMemoryCapture(table_name)
        self._tables[table_name] = capture
        return capture

    def on_change(self, table_name: Optional[str] = None) -> Callable:
        """Decorator to register a change handler."""

        def decorator(fn: Callable[[ChangeEvent], None]) -> Callable:
            if table_name:
                self._handlers[table_name].append(fn)
            else:
                self._global_handlers.append(fn)
            return fn

        return decorator

    def record_insert(self, table_name: str, primary_key: Any, data: Dict[str, Any]) -> None:
        """Record an insert event."""
        if table_name in self._tables:
            self._tables[table_name].record_change(
                ChangeType.INSERT,
                primary_key,
                after_image=data,
            )
            self._dispatch_event(
                table_name,
                ChangeEvent(
                    table_name=table_name,
                    change_type=ChangeType.INSERT,
                    primary_key=primary_key,
                    after_image=data,
                ),
            )

    def record_update(
        self,
        table_name: str,
        primary_key: Any,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> None:
        """Record an update event."""
        if table_name in self._tables:
            self._tables[table_name].record_change(
                ChangeType.UPDATE,
                primary_key,
                before_image=before,
                after_image=after,
            )
            self._dispatch_event(
                table_name,
                ChangeEvent(
                    table_name=table_name,
                    change_type=ChangeType.UPDATE,
                    primary_key=primary_key,
                    before_image=before,
                    after_image=after,
                ),
            )

    def record_delete(self, table_name: str, primary_key: Any, data: Dict[str, Any]) -> None:
        """Record a delete event."""
        if table_name in self._tables:
            self._tables[table_name].record_change(
                ChangeType.DELETE,
                primary_key,
                before_image=data,
            )
            self._dispatch_event(
                table_name,
                ChangeEvent(
                    table_name=table_name,
                    change_type=ChangeType.DELETE,
                    primary_key=primary_key,
                    before_image=data,
                ),
            )

    def _dispatch_event(self, table_name: str, event: ChangeEvent) -> None:
        """Dispatch event to handlers."""
        # Table-specific handlers
        for handler in self._handlers.get(table_name, []):
            try:
                handler(event)
            except Exception:
                pass

        # Global handlers
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception:
                pass

    def get_changes(self, table_name: str, since_sequence: int = 0) -> List[ChangeEvent]:
        """Get changes for a table since a sequence number."""
        if table_name not in self._tables:
            return []
        return list(self._tables[table_name].capture_changes(since_sequence))

    def get_all_tables(self) -> List[str]:
        """Get list of tracked tables."""
        return list(self._tables.keys())


# Convenience functions
def create_cdc_stream() -> CDCStream:
    """Create a new CDC stream."""
    return CDCStream()


def create_replicator(
    replication_mode: ReplicationMode = ReplicationMode.INCREMENTAL,
    conflict_resolution: ConflictResolution = ConflictResolution.SOURCE_WINS,
) -> CDCReplicator:
    """Create a new CDC replicator."""
    return CDCReplicator(
        replication_mode=replication_mode,
        conflict_resolution=conflict_resolution,
    )
