"""
Tests for Phase 2 Advanced Query Engine Components.

Tests cover:
- Vectorized Execution
- Adaptive Query Execution (AQE)
"""

import time
from typing import Any, Dict, List

import pytest

from hiveframe.optimizer import (
    # Adaptive Query Execution
    AdaptiveQueryExecutor,
    AQEContext,
    JoinStrategy,
    JoinStrategySelector,
    ParallelVectorizedExecutor,
    PartitionCoalescer,
    RuntimeStatistics,
    SkewHandler,
    # Vectorized Execution
    VectorBatch,
    VectorizedAggregate,
    VectorizedFilter,
    VectorizedJoin,
    VectorizedLimit,
    VectorizedPipeline,
    VectorizedProject,
    VectorizedSort,
    VectorType,
    WaggleDanceFeedback,
)


class TestVectorBatch:
    """Test VectorBatch columnar data structure."""

    @pytest.fixture
    def sample_rows(self) -> List[Dict[str, Any]]:
        """Sample row-oriented data."""
        return [
            {"id": 1, "name": "Alice", "age": 30, "active": True},
            {"id": 2, "name": "Bob", "age": 25, "active": False},
            {"id": 3, "name": "Carol", "age": 35, "active": True},
        ]

    def test_from_rows(self, sample_rows):
        """Test creating batch from rows."""
        batch = VectorBatch.from_rows(sample_rows)

        assert batch.num_rows == 3
        assert "id" in batch.columns
        assert "name" in batch.columns

    def test_to_rows(self, sample_rows):
        """Test converting back to rows."""
        batch = VectorBatch.from_rows(sample_rows)
        rows = batch.to_rows()

        assert len(rows) == 3
        assert rows[0]["name"] == "Alice"

    def test_get_column(self, sample_rows):
        """Test getting a single column."""
        batch = VectorBatch.from_rows(sample_rows)

        ages = batch.get_column("age")

        assert ages == [30, 25, 35]

    def test_slice(self, sample_rows):
        """Test batch slicing."""
        batch = VectorBatch.from_rows(sample_rows)

        sliced = batch.slice(1, 3)

        assert sliced.num_rows == 2

    def test_select(self, sample_rows):
        """Test column selection."""
        batch = VectorBatch.from_rows(sample_rows)

        selected = batch.select(["id", "name"])

        assert "id" in selected.columns
        assert "name" in selected.columns
        assert "age" not in selected.columns

    def test_filter(self, sample_rows):
        """Test row filtering."""
        batch = VectorBatch.from_rows(sample_rows)

        mask = [True, False, True]
        filtered = batch.filter(mask)

        assert filtered.num_rows == 2

    def test_type_inference(self, sample_rows):
        """Test automatic type inference."""
        batch = VectorBatch.from_rows(sample_rows)

        assert batch.types["id"] == VectorType.INT64
        assert batch.types["name"] == VectorType.STRING
        assert batch.types["active"] == VectorType.BOOL


class TestVectorizedFilter:
    """Test VectorizedFilter operation."""

    @pytest.fixture
    def batch(self) -> VectorBatch:
        """Test batch."""
        return VectorBatch.from_rows(
            [
                {"id": 1, "value": 10},
                {"id": 2, "value": 20},
                {"id": 3, "value": 30},
                {"id": 4, "value": 40},
            ]
        )

    def test_equality_filter(self, batch):
        """Test equality filter."""
        filter_op = VectorizedFilter("value", "=", 20)

        result = filter_op.execute(batch)

        assert result.num_rows == 1

    def test_greater_than_filter(self, batch):
        """Test greater than filter."""
        filter_op = VectorizedFilter("value", ">", 20)

        result = filter_op.execute(batch)

        assert result.num_rows == 2

    def test_in_filter(self, batch):
        """Test IN filter."""
        filter_op = VectorizedFilter("value", "IN", [10, 30])

        result = filter_op.execute(batch)

        assert result.num_rows == 2


class TestVectorizedProject:
    """Test VectorizedProject operation."""

    @pytest.fixture
    def batch(self) -> VectorBatch:
        """Test batch."""
        return VectorBatch.from_rows(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
            ]
        )

    def test_column_selection(self, batch):
        """Test simple column selection."""
        project_op = VectorizedProject({"x": "a", "y": "b"})

        result = project_op.execute(batch)

        assert "x" in result.columns
        assert "y" in result.columns
        assert "c" not in result.columns

    def test_computed_column(self, batch):
        """Test computed column."""
        project_op = VectorizedProject({"a": "a", "sum": lambda row: row["a"] + row["b"]})

        result = project_op.execute(batch)

        sums = result.get_column("sum")
        assert sums == [3, 9]


class TestVectorizedAggregate:
    """Test VectorizedAggregate operation."""

    @pytest.fixture
    def batch(self) -> VectorBatch:
        """Test batch."""
        return VectorBatch.from_rows(
            [
                {"category": "A", "value": 10},
                {"category": "A", "value": 20},
                {"category": "B", "value": 30},
                {"category": "B", "value": 40},
                {"category": "B", "value": 50},
            ]
        )

    def test_sum_aggregate(self, batch):
        """Test SUM aggregation."""
        agg_op = VectorizedAggregate(
            group_by=["category"], aggregations={"total": ("value", "SUM")}
        )

        result = agg_op.execute(batch)

        assert result.num_rows == 2

    def test_count_aggregate(self, batch):
        """Test COUNT aggregation."""
        agg_op = VectorizedAggregate(
            group_by=["category"], aggregations={"cnt": ("value", "COUNT")}
        )

        result = agg_op.execute(batch)

        rows = result.to_rows()
        category_counts = {r["category"]: r["cnt"] for r in rows}

        assert category_counts["A"] == 2
        assert category_counts["B"] == 3

    def test_avg_aggregate(self, batch):
        """Test AVG aggregation."""
        agg_op = VectorizedAggregate(
            group_by=["category"], aggregations={"avg_val": ("value", "AVG")}
        )

        result = agg_op.execute(batch)

        rows = result.to_rows()
        category_avgs = {r["category"]: r["avg_val"] for r in rows}

        assert category_avgs["A"] == 15.0
        assert category_avgs["B"] == 40.0


class TestVectorizedJoin:
    """Test VectorizedJoin operation."""

    def test_inner_join(self):
        """Test inner join."""
        left = VectorBatch.from_rows(
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
                {"id": 3, "name": "Carol"},
            ]
        )

        right = VectorBatch.from_rows(
            [
                {"id": 1, "dept": "Sales"},
                {"id": 2, "dept": "Engineering"},
            ]
        )

        join_op = VectorizedJoin(right, "id", "id", "inner")

        result = join_op.execute(left)

        assert result.num_rows == 2

    def test_left_join(self):
        """Test left join."""
        left = VectorBatch.from_rows(
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
                {"id": 3, "name": "Carol"},
            ]
        )

        right = VectorBatch.from_rows(
            [
                {"id": 1, "dept": "Sales"},
            ]
        )

        join_op = VectorizedJoin(right, "id", "id", "left")

        result = join_op.execute(left)

        assert result.num_rows == 3


class TestVectorizedSort:
    """Test VectorizedSort operation."""

    def test_ascending_sort(self):
        """Test ascending sort."""
        batch = VectorBatch.from_rows(
            [
                {"id": 3, "name": "Carol"},
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        )

        sort_op = VectorizedSort([("id", True)])

        result = sort_op.execute(batch)
        rows = result.to_rows()

        assert rows[0]["id"] == 1
        assert rows[2]["id"] == 3

    def test_descending_sort(self):
        """Test descending sort."""
        batch = VectorBatch.from_rows(
            [
                {"id": 1, "name": "Alice"},
                {"id": 3, "name": "Carol"},
                {"id": 2, "name": "Bob"},
            ]
        )

        sort_op = VectorizedSort([("id", False)])

        result = sort_op.execute(batch)
        rows = result.to_rows()

        assert rows[0]["id"] == 3
        assert rows[2]["id"] == 1


class TestVectorizedLimit:
    """Test VectorizedLimit operation."""

    def test_limit(self):
        """Test limiting rows."""
        batch = VectorBatch.from_rows([{"id": i} for i in range(10)])

        limit_op = VectorizedLimit(5)

        result = limit_op.execute(batch)

        assert result.num_rows == 5

    def test_limit_with_offset(self):
        """Test limit with offset."""
        batch = VectorBatch.from_rows([{"id": i} for i in range(10)])

        limit_op = VectorizedLimit(3, offset=5)

        result = limit_op.execute(batch)

        assert result.num_rows == 3
        rows = result.to_rows()
        assert rows[0]["id"] == 5


class TestVectorizedPipeline:
    """Test VectorizedPipeline for chaining operations."""

    def test_filter_project_pipeline(self):
        """Test pipeline with filter and project."""
        data = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Carol", "age": 35},
        ]

        pipeline = VectorizedPipeline()
        pipeline.add(VectorizedFilter("age", ">", 25))
        pipeline.add(VectorizedProject({"name": "name"}))

        results = pipeline.execute(data)

        assert len(results) == 2
        assert "age" not in results[0]


class TestParallelVectorizedExecutor:
    """Test parallel vectorized execution."""

    def test_parallel_execution(self):
        """Test executing pipeline in parallel."""
        data = [{"id": i, "value": i * 10} for i in range(100)]

        pipeline = VectorizedPipeline(batch_size=25)
        pipeline.add(VectorizedFilter("value", ">", 500))

        executor = ParallelVectorizedExecutor(num_workers=4, batch_size=25)
        results = executor.execute(pipeline, data)

        assert len(results) == 49  # IDs 51-99 have value > 500


class TestRuntimeStatistics:
    """Test RuntimeStatistics for AQE."""

    def test_selectivity_calculation(self):
        """Test selectivity calculation."""
        stats = RuntimeStatistics(input_rows=1000, output_rows=100)

        assert stats.selectivity == 0.1

    def test_duration_calculation(self):
        """Test duration calculation."""
        stats = RuntimeStatistics()
        time.sleep(0.1)
        stats.end_time = time.time()

        assert stats.duration_ms >= 100

    def test_skew_detection(self):
        """Test skew detection."""
        stats = RuntimeStatistics(partition_sizes=[100, 100, 1000, 100])  # One large partition

        skew = stats.calculate_skew()

        assert skew > 2.0
        assert stats.has_skew


class TestWaggleDanceFeedback:
    """Test WaggleDanceFeedback for AQE."""

    def test_from_execution(self):
        """Test creating feedback from execution."""
        feedback = WaggleDanceFeedback.from_execution(
            stage_id="stage-1",
            operator_id="filter-1",
            input_rows=1000,
            output_rows=10,
            duration_ms=100.0,
        )

        assert feedback.stage_id == "stage-1"
        assert "highly_selective_filter" in feedback.suggestions

    def test_skew_suggestion(self):
        """Test skew detection in feedback."""
        feedback = WaggleDanceFeedback.from_execution(
            stage_id="stage-1",
            operator_id="join-1",
            input_rows=1000,
            output_rows=1000,
            duration_ms=100.0,
            partition_sizes=[100, 100, 5000, 100],
        )

        assert "data_skew_detected" in feedback.suggestions


class TestAQEContext:
    """Test AQEContext for adaptive execution."""

    def test_should_broadcast(self):
        """Test broadcast decision."""
        context = AQEContext(broadcast_threshold=10_000_000)

        assert context.should_broadcast(5_000_000)
        assert not context.should_broadcast(20_000_000)

    def test_calculate_partitions(self):
        """Test partition count calculation."""
        context = AQEContext(target_partition_size=64_000_000, min_partitions=1, max_partitions=100)

        # 256MB should give 4 partitions
        num_parts = context.calculate_num_partitions(256_000_000)

        assert num_parts == 4

    def test_record_feedback(self):
        """Test recording feedback."""
        context = AQEContext()

        feedback = WaggleDanceFeedback.from_execution(
            stage_id="stage-1",
            operator_id="filter-1",
            input_rows=1000,
            output_rows=100,
            duration_ms=50.0,
        )

        context.record_feedback(feedback)

        retrieved = context.get_feedback("stage-1")
        assert retrieved == feedback


class TestJoinStrategySelector:
    """Test JoinStrategySelector for AQE."""

    def test_select_broadcast_small_table(self):
        """Test broadcast selection for small table."""
        context = AQEContext(broadcast_threshold=10_000_000)
        selector = JoinStrategySelector(context)

        strategy = selector.select_strategy(
            left_size_bytes=100_000_000,
            right_size_bytes=5_000_000,
            left_rows=1_000_000,
            right_rows=50_000,
        )

        assert strategy == JoinStrategy.BROADCAST_HASH

    def test_select_sort_merge_large_tables(self):
        """Test sort-merge for large tables."""
        context = AQEContext(broadcast_threshold=10_000_000)
        selector = JoinStrategySelector(context)

        strategy = selector.select_strategy(
            left_size_bytes=1_000_000_000,
            right_size_bytes=500_000_000,
            left_rows=10_000_000,
            right_rows=5_000_000,
        )

        assert strategy == JoinStrategy.SORT_MERGE


class TestPartitionCoalescer:
    """Test PartitionCoalescer for AQE."""

    def test_should_coalesce(self):
        """Test coalescing decision."""
        context = AQEContext(target_partition_size=64_000_000)
        coalescer = PartitionCoalescer(context)

        # Small partitions should coalesce
        small_partitions = [1_000_000] * 10  # 1MB each
        assert coalescer.should_coalesce(small_partitions)

        # Large partitions should not
        large_partitions = [100_000_000] * 10  # 100MB each
        assert not coalescer.should_coalesce(large_partitions)

    def test_compute_target_partitions(self):
        """Test target partition calculation."""
        context = AQEContext(target_partition_size=64_000_000)
        coalescer = PartitionCoalescer(context)

        # 10 partitions of 1MB each = 10MB total
        partition_sizes = [1_000_000] * 10

        target = coalescer.compute_target_partitions(partition_sizes)

        assert target == 1  # Should coalesce to 1


class TestSkewHandler:
    """Test SkewHandler for AQE."""

    def test_detect_skewed_keys(self):
        """Test skewed key detection."""
        context = AQEContext(skew_threshold=2.0)
        handler = SkewHandler(context)

        key_counts = {
            "key1": 100,
            "key2": 100,
            "key3": 500,  # Skewed
            "key4": 100,
        }

        skewed = handler.detect_skewed_keys(key_counts)

        assert "key3" in skewed
        assert "key1" not in skewed

    def test_split_skewed_partition(self):
        """Test partition splitting plan."""
        context = AQEContext(target_partition_size=64_000_000)
        handler = SkewHandler(context)

        splits = handler.split_skewed_partition(key="skewed_key", partition_size=256_000_000)

        assert len(splits) >= 2
        assert all(s[1] > 0 for s in splits)


class TestAdaptiveQueryExecutor:
    """Test AdaptiveQueryExecutor main interface."""

    def test_record_stage_feedback(self):
        """Test recording stage feedback."""
        executor = AdaptiveQueryExecutor()

        feedback = executor.record_stage_feedback(
            stage_id="stage-1",
            operator_id="scan-1",
            input_rows=10000,
            output_rows=10000,
            duration_ms=500.0,
        )

        assert feedback is not None
        assert feedback.stage_id == "stage-1"

    def test_execution_summary(self):
        """Test getting execution summary."""
        executor = AdaptiveQueryExecutor()

        executor.record_stage_feedback(
            stage_id="stage-1",
            operator_id="scan-1",
            input_rows=10000,
            output_rows=10000,
            duration_ms=500.0,
        )

        summary = executor.get_execution_summary()

        assert summary["feedback_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
