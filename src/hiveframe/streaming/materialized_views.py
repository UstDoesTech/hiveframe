"""
HiveFrame Materialized Views
============================
Automatically maintained aggregate tables using bee-inspired optimization.

Key Features:
- Incremental view maintenance
- Automatic refresh strategies
- Dependency tracking
- Query rewriting to use materialized views
- Bee-inspired cache invalidation
"""

import hashlib
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..core import ColonyState, WaggleDance


class RefreshStrategy(Enum):
    """Strategies for refreshing materialized views."""

    IMMEDIATE = "immediate"  # Refresh on every change
    ON_DEMAND = "on_demand"  # Manual refresh only
    PERIODIC = "periodic"  # Time-based refresh
    INCREMENTAL = "incremental"  # Delta-based incremental refresh
    DEFERRED = "deferred"  # Batch updates at transaction commit


class ViewState(Enum):
    """State of a materialized view."""

    FRESH = "fresh"  # View is up-to-date
    STALE = "stale"  # View needs refresh
    REFRESHING = "refreshing"  # Refresh in progress
    INVALID = "invalid"  # View is invalid and must be rebuilt
    DISABLED = "disabled"  # View is disabled


@dataclass
class ViewMetadata:
    """Metadata for a materialized view."""

    name: str
    query: str
    source_tables: List[str]
    refresh_strategy: RefreshStrategy
    created_at: float = field(default_factory=time.time)
    last_refresh: Optional[float] = None
    last_query_hash: Optional[str] = None
    row_count: int = 0
    size_bytes: int = 0
    state: ViewState = ViewState.STALE
    refresh_interval_seconds: Optional[float] = None
    partition_columns: List[str] = field(default_factory=list)
    cluster_columns: List[str] = field(default_factory=list)


@dataclass
class ViewChange:
    """Represents a change to a source table that affects a view."""

    table_name: str
    change_type: str  # insert, update, delete
    affected_keys: List[Any]
    timestamp: float = field(default_factory=time.time)
    change_data: Optional[Dict[str, Any]] = None


@dataclass
class IncrementalDelta:
    """Delta changes for incremental view refresh."""

    inserts: List[Dict[str, Any]] = field(default_factory=list)
    updates: List[Dict[str, Any]] = field(default_factory=list)
    deletes: List[Any] = field(default_factory=list)  # Keys to delete

    def is_empty(self) -> bool:
        """Check if delta is empty."""
        return not self.inserts and not self.updates and not self.deletes


class MaterializedView:
    """
    A single materialized view with automatic maintenance.

    Supports incremental updates and various refresh strategies.
    """

    def __init__(
        self,
        name: str,
        query: str,
        source_tables: List[str],
        refresh_strategy: RefreshStrategy = RefreshStrategy.INCREMENTAL,
        refresh_interval_seconds: Optional[float] = None,
    ):
        self.metadata = ViewMetadata(
            name=name,
            query=query,
            source_tables=source_tables,
            refresh_strategy=refresh_strategy,
            refresh_interval_seconds=refresh_interval_seconds,
        )

        self._data: List[Dict[str, Any]] = []
        self._index: Dict[Any, int] = {}  # key -> row index
        self._key_column: Optional[str] = None
        self._pending_changes: List[ViewChange] = []
        self._lock = threading.RLock()

        # Query hash for change detection
        self.metadata.last_query_hash = self._compute_query_hash(query)

    def _compute_query_hash(self, query: str) -> str:
        """Compute hash of query for change detection."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def set_key_column(self, column: str) -> None:
        """Set the key column for the view."""
        self._key_column = column
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the key index."""
        self._index.clear()
        if self._key_column:
            for i, row in enumerate(self._data):
                if self._key_column in row:
                    self._index[row[self._key_column]] = i

    def get_data(self) -> List[Dict[str, Any]]:
        """Get all data from the view."""
        with self._lock:
            return list(self._data)

    def query_view(
        self,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query the materialized view with optional filter."""
        with self._lock:
            result = self._data
            if filter_fn:
                result = [row for row in result if filter_fn(row)]
            if limit:
                result = result[:limit]
            return list(result)

    def record_change(self, change: ViewChange) -> None:
        """Record a change that affects this view."""
        with self._lock:
            if change.table_name in self.metadata.source_tables:
                self._pending_changes.append(change)
                if self.metadata.state == ViewState.FRESH:
                    self.metadata.state = ViewState.STALE

    def apply_delta(self, delta: IncrementalDelta) -> None:
        """Apply incremental delta to the view."""
        with self._lock:
            # Apply deletes
            if self._key_column:
                for key in delta.deletes:
                    if key in self._index:
                        idx = self._index.pop(key)
                        self._data[idx] = None  # Mark for cleanup

            # Apply updates
            for update in delta.updates:
                if self._key_column and self._key_column in update:
                    key = update[self._key_column]
                    if key in self._index:
                        self._data[self._index[key]] = update

            # Apply inserts
            for insert in delta.inserts:
                if self._key_column and self._key_column in insert:
                    key = insert[self._key_column]
                    if key in self._index:
                        # Update existing
                        self._data[self._index[key]] = insert
                    else:
                        # Insert new
                        self._index[key] = len(self._data)
                        self._data.append(insert)
                else:
                    self._data.append(insert)

            # Cleanup None entries
            self._data = [row for row in self._data if row is not None]
            self._rebuild_index()

            self.metadata.row_count = len(self._data)
            self.metadata.last_refresh = time.time()
            self.metadata.state = ViewState.FRESH
            self._pending_changes.clear()

    def full_refresh(self, data: List[Dict[str, Any]]) -> None:
        """Perform a full refresh of the view."""
        with self._lock:
            self._data = list(data)
            self._rebuild_index()
            self.metadata.row_count = len(self._data)
            self.metadata.last_refresh = time.time()
            self.metadata.state = ViewState.FRESH
            self._pending_changes.clear()

    def invalidate(self) -> None:
        """Mark the view as invalid."""
        with self._lock:
            self.metadata.state = ViewState.INVALID

    def needs_refresh(self) -> bool:
        """Check if the view needs to be refreshed."""
        if self.metadata.state in (ViewState.STALE, ViewState.INVALID):
            return True

        if (
            self.metadata.refresh_strategy == RefreshStrategy.PERIODIC
            and self.metadata.refresh_interval_seconds
        ):
            if self.metadata.last_refresh:
                elapsed = time.time() - self.metadata.last_refresh
                return elapsed >= self.metadata.refresh_interval_seconds
            return True

        return False

    def get_pending_changes(self) -> List[ViewChange]:
        """Get pending changes for incremental refresh."""
        with self._lock:
            return list(self._pending_changes)


class MaterializedViewManager:
    """
    Manager for materialized views with bee-inspired optimization.

    Handles:
    - View creation and management
    - Dependency tracking between views and tables
    - Automatic refresh scheduling
    - Query rewriting to use views
    - Bee-inspired resource allocation for refreshes
    """

    def __init__(
        self,
        enable_auto_refresh: bool = True,
        max_concurrent_refreshes: int = 4,
    ):
        self.enable_auto_refresh = enable_auto_refresh
        self.max_concurrent_refreshes = max_concurrent_refreshes

        self.colony = ColonyState()
        self._views: Dict[str, MaterializedView] = {}
        self._table_dependencies: Dict[str, Set[str]] = defaultdict(set)  # table -> views
        self._query_executor: Optional[Callable] = None
        self._lock = threading.RLock()

        # Refresh scheduling
        self._refresh_thread: Optional[threading.Thread] = None
        self._running = False

        # Metrics
        self._refresh_count = 0
        self._refresh_times: List[float] = []

    def set_query_executor(
        self, executor: Callable[[str], List[Dict[str, Any]]]
    ) -> None:
        """Set the query executor function for refreshing views."""
        self._query_executor = executor

    def create_view(
        self,
        name: str,
        query: str,
        source_tables: List[str],
        refresh_strategy: RefreshStrategy = RefreshStrategy.INCREMENTAL,
        refresh_interval_seconds: Optional[float] = None,
        key_column: Optional[str] = None,
    ) -> MaterializedView:
        """Create a new materialized view."""
        with self._lock:
            if name in self._views:
                raise ValueError(f"View '{name}' already exists")

            view = MaterializedView(
                name=name,
                query=query,
                source_tables=source_tables,
                refresh_strategy=refresh_strategy,
                refresh_interval_seconds=refresh_interval_seconds,
            )

            if key_column:
                view.set_key_column(key_column)

            self._views[name] = view

            # Track dependencies
            for table in source_tables:
                self._table_dependencies[table].add(name)

            return view

    def drop_view(self, name: str) -> bool:
        """Drop a materialized view."""
        with self._lock:
            if name not in self._views:
                return False

            view = self._views.pop(name)

            # Remove from dependencies
            for table in view.metadata.source_tables:
                self._table_dependencies[table].discard(name)

            return True

    def get_view(self, name: str) -> Optional[MaterializedView]:
        """Get a materialized view by name."""
        return self._views.get(name)

    def list_views(self) -> List[ViewMetadata]:
        """List all materialized views."""
        return [view.metadata for view in self._views.values()]

    def notify_table_change(self, change: ViewChange) -> None:
        """Notify the manager of a change to a source table."""
        with self._lock:
            affected_views = self._table_dependencies.get(change.table_name, set())
            for view_name in affected_views:
                view = self._views.get(view_name)
                if view:
                    view.record_change(change)

                    # Immediate refresh if configured
                    if view.metadata.refresh_strategy == RefreshStrategy.IMMEDIATE:
                        self._refresh_view(view)

    def refresh_view(self, name: str, force: bool = False) -> bool:
        """Manually refresh a view."""
        view = self._views.get(name)
        if not view:
            return False

        if force or view.needs_refresh():
            return self._refresh_view(view)
        return False

    def _refresh_view(self, view: MaterializedView) -> bool:
        """Internal method to refresh a view."""
        if not self._query_executor:
            return False

        try:
            view.metadata.state = ViewState.REFRESHING
            start_time = time.time()

            # Execute the view query
            data = self._query_executor(view.metadata.query)
            view.full_refresh(data)

            # Record metrics
            refresh_time = time.time() - start_time
            self._refresh_count += 1
            self._refresh_times.append(refresh_time)

            # Perform waggle dance
            dance = WaggleDance(
                partition_id=view.metadata.name,
                quality_score=1.0,
                processing_time=refresh_time,
                result_size=view.metadata.row_count,
                worker_id="view_manager",
            )
            self.colony.dance_floor.perform_dance(dance)

            return True
        except Exception:
            view.metadata.state = ViewState.INVALID
            return False

    def refresh_all(self, force: bool = False) -> Dict[str, bool]:
        """Refresh all views that need it."""
        results = {}
        for name in list(self._views.keys()):
            results[name] = self.refresh_view(name, force=force)
        return results

    def start_auto_refresh(self) -> None:
        """Start the auto-refresh background thread."""
        if self._running:
            return

        self._running = True
        self._refresh_thread = threading.Thread(
            target=self._auto_refresh_loop, daemon=True
        )
        self._refresh_thread.start()

    def stop_auto_refresh(self) -> None:
        """Stop the auto-refresh background thread."""
        self._running = False
        if self._refresh_thread:
            self._refresh_thread.join(timeout=5.0)

    def _auto_refresh_loop(self) -> None:
        """Background thread for auto-refreshing views."""
        while self._running:
            try:
                for view in list(self._views.values()):
                    if not self._running:
                        break
                    if view.needs_refresh():
                        self._refresh_view(view)
            except Exception:
                pass  # Don't crash the background thread
            time.sleep(1.0)  # Check every second

    def query_with_rewrite(
        self,
        query: str,
    ) -> Tuple[str, Optional[str]]:
        """
        Attempt to rewrite a query to use a materialized view.

        Returns:
            Tuple of (rewritten_query, view_name) or (original_query, None)
        """
        # Simple view matching - check if query matches any view query
        for view_name, view in self._views.items():
            if view.metadata.state == ViewState.FRESH:
                # Simple exact match
                if self._normalize_query(query) == self._normalize_query(
                    view.metadata.query
                ):
                    return f"SELECT * FROM _mv_{view_name}", view_name

        return query, None

    def _normalize_query(self, query: str) -> str:
        """Normalize query for comparison."""
        return " ".join(query.lower().split())

    def get_metrics(self) -> Dict[str, Any]:
        """Get manager metrics."""
        avg_refresh_time = (
            sum(self._refresh_times[-100:]) / len(self._refresh_times[-100:])
            if self._refresh_times
            else 0
        )

        views_by_state = defaultdict(int)
        for view in self._views.values():
            views_by_state[view.metadata.state.value] += 1

        return {
            "total_views": len(self._views),
            "total_refreshes": self._refresh_count,
            "avg_refresh_time_ms": avg_refresh_time * 1000,
            "views_by_state": dict(views_by_state),
            "auto_refresh_enabled": self.enable_auto_refresh,
            "is_running": self._running,
        }


# Convenience function
def create_materialized_view(
    name: str,
    query: str,
    source_tables: List[str],
    refresh_strategy: RefreshStrategy = RefreshStrategy.INCREMENTAL,
) -> MaterializedView:
    """Create a standalone materialized view."""
    return MaterializedView(
        name=name,
        query=query,
        source_tables=source_tables,
        refresh_strategy=refresh_strategy,
    )
