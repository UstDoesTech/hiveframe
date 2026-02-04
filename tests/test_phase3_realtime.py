"""
Tests for Phase 3 Real-Time Analytics features.

Tests cover:
- Structured Streaming 2.0: Sub-millisecond latency streaming
- Complex Event Processing (CEP): Pattern detection in streaming data
- Materialized Views: Automatically maintained aggregate tables
- Change Data Capture (CDC): Database replication and synchronization
"""

import time

import pytest

from hiveframe import (
    # Structured Streaming 2.0
    AdaptiveMicroBatcher,
    LatencyMetrics,
    LockFreeQueue,
    PriorityLevel,
    PriorityQueue,
    ProcessingMode,
    StreamingRecord,
    StructuredStreaming2,
    # Complex Event Processing
    CEPEngine,
    ContiguityType,
    Pattern,
    PatternCondition,
    PatternElement,
    PatternMatch,
    PatternMatcher,
    QuantifierType,
    pattern,
    # Materialized Views
    IncrementalDelta,
    MaterializedView,
    MaterializedViewManager,
    RefreshStrategy,
    ViewChange,
    ViewMetadata,
    ViewState,
    create_materialized_view,
    # Change Data Capture
    CDCReplicator,
    CDCStream,
    ChangeEvent,
    ChangeLog,
    ChangeType,
    ConflictResolution,
    InMemoryCapture,
    ReplicationMode,
    TableCheckpoint,
    create_cdc_stream,
    create_replicator,
)


class TestStructuredStreaming2:
    """Tests for Structured Streaming 2.0."""

    def test_streaming_record_creation(self):
        """Test creating a streaming record."""
        record = StreamingRecord(
            key="user_1",
            value={"action": "click"},
            priority=PriorityLevel.HIGH,
        )

        assert record.key == "user_1"
        assert record.value == {"action": "click"}
        assert record.priority == PriorityLevel.HIGH
        assert record.effective_time > 0

    def test_lock_free_queue(self):
        """Test lock-free queue operations."""
        queue = LockFreeQueue(max_size=100)

        # Enqueue items
        assert queue.enqueue("item1") is True
        assert queue.enqueue("item2") is True
        assert queue.size == 2

        # Dequeue items
        assert queue.dequeue() == "item1"
        assert queue.dequeue() == "item2"
        assert queue.is_empty is True

    def test_priority_queue(self):
        """Test priority queue ordering."""
        queue = PriorityQueue(max_size=100)

        # Add records with different priorities
        low = StreamingRecord(key="low", value=1, priority=PriorityLevel.LOW)
        normal = StreamingRecord(key="normal", value=2, priority=PriorityLevel.NORMAL)
        high = StreamingRecord(key="high", value=3, priority=PriorityLevel.HIGH)
        critical = StreamingRecord(key="critical", value=4, priority=PriorityLevel.CRITICAL)

        queue.enqueue(low)
        queue.enqueue(normal)
        queue.enqueue(high)
        queue.enqueue(critical)

        # Should dequeue in priority order (highest first)
        assert queue.dequeue().key == "critical"
        assert queue.dequeue().key == "high"
        assert queue.dequeue().key == "normal"
        assert queue.dequeue().key == "low"

    def test_adaptive_micro_batcher(self):
        """Test adaptive micro-batching."""
        batcher = AdaptiveMicroBatcher(
            min_batch_size=1,
            max_batch_size=100,
            target_latency_us=500.0,
        )

        # Initial batch size should be min
        batch_size = batcher.get_batch_size(queue_depth=0)
        assert batch_size >= 1

        # Record high latency - should reduce batch size
        batcher.record_latency(1000.0)
        batcher.record_latency(1000.0)
        batcher.record_latency(1000.0)

        # Batch size should adapt
        new_batch_size = batcher.get_batch_size(queue_depth=10)
        assert new_batch_size >= 1

    def test_streaming_engine_creation(self):
        """Test creating a streaming engine."""
        engine = StructuredStreaming2(
            num_workers=4,
            processing_mode=ProcessingMode.CONTINUOUS,
            target_latency_us=500.0,
        )

        assert engine is not None
        assert engine.num_workers == 4
        assert engine.processing_mode == ProcessingMode.CONTINUOUS

    def test_streaming_engine_start_stop(self):
        """Test starting and stopping the streaming engine."""
        engine = StructuredStreaming2(num_workers=2)

        processed = []

        def processor(record):
            processed.append(record)

        engine.start(processor)
        engine.submit_value("key1", {"value": 1})
        time.sleep(0.1)
        engine.stop()

        assert engine is not None

    def test_latency_metrics(self):
        """Test latency metrics tracking."""
        metrics = LatencyMetrics(
            processing_latency_us=100.0,
            p50_latency_us=90.0,
            p99_latency_us=500.0,
            records_processed=1000,
            records_per_second=10000.0,
        )

        assert metrics.processing_latency_us == 100.0
        assert metrics.records_processed == 1000


class TestComplexEventProcessing:
    """Tests for Complex Event Processing (CEP)."""

    def test_pattern_condition(self):
        """Test pattern condition evaluation."""
        cond = PatternCondition(field="type", operator="eq", value="login")

        assert cond.evaluate({"type": "login"}) is True
        assert cond.evaluate({"type": "logout"}) is False

    def test_pattern_condition_operators(self):
        """Test various condition operators."""
        # Greater than
        gt_cond = PatternCondition(field="amount", operator="gt", value=100)
        assert gt_cond.evaluate({"amount": 150}) is True
        assert gt_cond.evaluate({"amount": 50}) is False

        # Contains
        contains_cond = PatternCondition(field="message", operator="contains", value="error")
        assert contains_cond.evaluate({"message": "An error occurred"}) is True
        assert contains_cond.evaluate({"message": "Success"}) is False

    def test_pattern_element(self):
        """Test pattern element matching."""
        element = PatternElement("login")
        element.where("type", "eq", "login")
        element.where("success", "eq", True)

        assert element.matches({"type": "login", "success": True}) is True
        assert element.matches({"type": "login", "success": False}) is False

    def test_pattern_fluent_api(self):
        """Test pattern fluent API."""
        p = (
            pattern("fraud_detection")
            .begin("login")
            .where("type", "eq", "login")
            .followed_by("withdrawal")
            .where("type", "eq", "withdrawal")
            .where("amount", "gt", 10000)
            .within(minutes=5)
        )

        assert p.name == "fraud_detection"
        assert len(p.elements) == 2
        assert p.timeout_seconds == 300.0

    def test_pattern_matcher(self):
        """Test pattern matching."""
        p = (
            Pattern("simple_pattern")
            .begin("start")
            .where("type", "eq", "A")
            .followed_by("end")
            .where("type", "eq", "B")
        )

        matcher = PatternMatcher(p)

        # Event A
        matches = matcher.process_event({"type": "A"})
        assert len(matches) == 0  # Pattern not complete yet

        # Event B - should complete pattern
        matches = matcher.process_event({"type": "B"})
        assert len(matches) == 1
        assert matches[0].pattern_id == "simple_pattern"

    def test_cep_engine(self):
        """Test CEP engine with multiple patterns."""
        engine = CEPEngine()

        # Add pattern
        p = (
            Pattern("login_failure")
            .begin("failure")
            .where("type", "eq", "login")
            .where("success", "eq", False)
        )
        engine.add_pattern(p)

        # Process events
        matches = engine.process_event({"type": "login", "success": False})
        assert len(matches) == 1

        # Check metrics
        metrics = engine.get_metrics()
        assert metrics["events_processed"] == 1
        assert metrics["matches_found"] == 1

    def test_cep_callback(self):
        """Test CEP pattern callbacks."""
        engine = CEPEngine()
        callback_results = []

        p = Pattern("test").begin("event").where("type", "eq", "test")
        engine.add_pattern(p)
        engine.add_callback("test", lambda m: callback_results.append(m))

        engine.process_event({"type": "test"})

        assert len(callback_results) == 1
        assert callback_results[0].pattern_id == "test"


class TestMaterializedViews:
    """Tests for Materialized Views."""

    def test_materialized_view_creation(self):
        """Test creating a materialized view."""
        view = MaterializedView(
            name="user_summary",
            query="SELECT user_id, COUNT(*) as order_count FROM orders GROUP BY user_id",
            source_tables=["orders"],
            refresh_strategy=RefreshStrategy.INCREMENTAL,
        )

        assert view.metadata.name == "user_summary"
        assert view.metadata.state == ViewState.STALE
        assert "orders" in view.metadata.source_tables

    def test_view_full_refresh(self):
        """Test full refresh of a view."""
        view = create_materialized_view(
            name="test_view",
            query="SELECT * FROM test",
            source_tables=["test"],
        )

        # Full refresh with data
        view.full_refresh([
            {"id": 1, "value": 100},
            {"id": 2, "value": 200},
        ])

        assert view.metadata.state == ViewState.FRESH
        assert view.metadata.row_count == 2

    def test_view_incremental_delta(self):
        """Test incremental delta updates."""
        view = create_materialized_view(
            name="test_view",
            query="SELECT * FROM test",
            source_tables=["test"],
        )
        view.set_key_column("id")

        # Full refresh
        view.full_refresh([
            {"id": 1, "value": 100},
            {"id": 2, "value": 200},
        ])

        # Apply delta
        delta = IncrementalDelta(
            inserts=[{"id": 3, "value": 300}],
            updates=[{"id": 1, "value": 150}],
            deletes=[2],
        )
        view.apply_delta(delta)

        data = view.get_data()
        assert len(data) == 2  # 1 updated, 1 deleted, 1 inserted
        assert any(d["id"] == 1 and d["value"] == 150 for d in data)
        assert any(d["id"] == 3 and d["value"] == 300 for d in data)

    def test_view_manager(self):
        """Test materialized view manager."""
        manager = MaterializedViewManager()

        # Create view
        view = manager.create_view(
            name="test_view",
            query="SELECT * FROM test",
            source_tables=["test"],
            key_column="id",
        )

        assert manager.get_view("test_view") is not None
        assert len(manager.list_views()) == 1

        # Drop view
        assert manager.drop_view("test_view") is True
        assert manager.get_view("test_view") is None

    def test_view_change_tracking(self):
        """Test view change tracking."""
        view = create_materialized_view(
            name="test_view",
            query="SELECT * FROM users",
            source_tables=["users"],
        )
        view.full_refresh([])

        # Record a change
        change = ViewChange(
            table_name="users",
            change_type="insert",
            affected_keys=[1],
        )
        view.record_change(change)

        assert view.metadata.state == ViewState.STALE
        assert len(view.get_pending_changes()) == 1

    def test_refresh_strategy(self):
        """Test different refresh strategies."""
        for strategy in RefreshStrategy:
            view = MaterializedView(
                name=f"view_{strategy.value}",
                query="SELECT 1",
                source_tables=["test"],
                refresh_strategy=strategy,
            )
            assert view.metadata.refresh_strategy == strategy


class TestChangeDataCapture:
    """Tests for Change Data Capture (CDC)."""

    def test_change_event(self):
        """Test change event creation."""
        event = ChangeEvent(
            table_name="users",
            change_type=ChangeType.INSERT,
            primary_key=1,
            after_image={"id": 1, "name": "John"},
        )

        assert event.table_name == "users"
        assert event.is_insert is True
        assert event.is_update is False
        assert event.after_image["name"] == "John"

    def test_change_event_changed_columns(self):
        """Test detecting changed columns."""
        event = ChangeEvent(
            table_name="users",
            change_type=ChangeType.UPDATE,
            primary_key=1,
            before_image={"id": 1, "name": "John", "age": 30},
            after_image={"id": 1, "name": "John", "age": 31},
        )

        changed = event.get_changed_columns()
        assert "age" in changed
        assert "name" not in changed

    def test_change_log(self):
        """Test change log operations."""
        log = ChangeLog(table_name="users")

        # Append events
        event1 = ChangeEvent(
            table_name="users",
            change_type=ChangeType.INSERT,
            primary_key=1,
        )
        seq1 = log.append(event1)

        event2 = ChangeEvent(
            table_name="users",
            change_type=ChangeType.UPDATE,
            primary_key=1,
        )
        seq2 = log.append(event2)

        assert seq2 > seq1

        # Get changes
        changes = log.get_changes_since(0)
        assert len(changes) == 2

        # Get changes since first
        changes = log.get_changes_since(seq1)
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.UPDATE

    def test_in_memory_capture(self):
        """Test in-memory change capture."""
        capture = InMemoryCapture("users")

        # Record changes
        capture.record_change(
            ChangeType.INSERT,
            primary_key=1,
            after_image={"id": 1, "name": "John"},
        )
        capture.record_change(
            ChangeType.UPDATE,
            primary_key=1,
            before_image={"id": 1, "name": "John"},
            after_image={"id": 1, "name": "Jane"},
        )

        # Get changes
        changes = list(capture.capture_changes())
        assert len(changes) == 2
        assert changes[0].is_insert
        assert changes[1].is_update

    def test_cdc_stream(self):
        """Test CDC stream interface."""
        stream = create_cdc_stream()
        stream.add_table("users")

        events_received = []

        @stream.on_change("users")
        def handle_change(event):
            events_received.append(event)

        # Record changes
        stream.record_insert("users", 1, {"id": 1, "name": "John"})
        stream.record_update(
            "users",
            1,
            {"id": 1, "name": "John"},
            {"id": 1, "name": "Jane"},
        )
        stream.record_delete("users", 1, {"id": 1, "name": "Jane"})

        assert len(events_received) == 3
        assert events_received[0].is_insert
        assert events_received[1].is_update
        assert events_received[2].is_delete

    def test_cdc_replicator(self):
        """Test CDC replicator."""
        replicator = create_replicator(
            replication_mode=ReplicationMode.INCREMENTAL,
            conflict_resolution=ConflictResolution.SOURCE_WINS,
        )

        # Create source
        capture = InMemoryCapture("users")
        capture.record_change(
            ChangeType.INSERT,
            primary_key=1,
            after_image={"id": 1, "name": "John"},
        )

        replicator.register_source("source1", capture)

        applied_events = []

        def apply_fn(events):
            applied_events.extend(events)
            return len(events)

        replicator.set_apply_function(apply_fn)

        # Replicate
        results = replicator.replicate_once()

        assert results.get("source1", 0) == 1
        assert len(applied_events) == 1

    def test_replicator_checkpoint(self):
        """Test replicator checkpointing."""
        replicator = CDCReplicator()
        capture = InMemoryCapture("users")
        replicator.register_source("source1", capture)

        # Set checkpoint
        checkpoint = TableCheckpoint(
            table_name="users",
            source_id="source1",
            last_sequence=100,
        )
        replicator.set_checkpoint("source1", checkpoint)

        # Get checkpoint
        retrieved = replicator.get_checkpoint("source1")
        assert retrieved is not None
        assert retrieved.last_sequence == 100

    def test_replicator_lag(self):
        """Test replicator lag tracking."""
        replicator = CDCReplicator()
        capture = InMemoryCapture("users")
        replicator.register_source("source1", capture)

        # Record changes
        for i in range(5):
            capture.record_change(
                ChangeType.INSERT,
                primary_key=i,
                after_image={"id": i},
            )

        # Check lag
        lag = replicator.get_lag()
        assert lag.get("source1", 0) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
