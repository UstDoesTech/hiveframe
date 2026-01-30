"""
Adaptive Query Execution (AQE) - Phase 2
========================================

Real-time query plan modification based on waggle dance feedback.
Adjusts execution strategy during runtime based on actual data characteristics.

Key Features:
- Runtime statistics collection
- Dynamic plan modification
- Shuffle partition optimization
- Join strategy adaptation
- Skew handling
"""

import time
import threading
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
from collections import defaultdict


class AdaptationTrigger(Enum):
    """Triggers for plan adaptation."""

    PARTITION_SIZE = auto()  # Partitions too large/small
    SKEW_DETECTED = auto()  # Data skew in joins
    MEMORY_PRESSURE = auto()  # Memory limits reached
    SHUFFLE_SPILL = auto()  # Shuffle spilling to disk
    JOIN_EXPLOSION = auto()  # Join producing too many rows
    FILTER_SELECTIVITY = auto()  # Filter more/less selective than expected


class JoinStrategy(Enum):
    """Available join strategies."""

    BROADCAST_HASH = auto()  # Broadcast small table
    SHUFFLE_HASH = auto()  # Shuffle both tables
    SORT_MERGE = auto()  # Sort-merge join
    NESTED_LOOP = auto()  # Nested loop (for small datasets)


@dataclass
class RuntimeStatistics:
    """Statistics collected during query execution."""

    # Row counts
    input_rows: int = 0
    output_rows: int = 0

    # Size estimates
    input_bytes: int = 0
    output_bytes: int = 0

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Partition info
    num_partitions: int = 0
    partition_sizes: List[int] = field(default_factory=list)

    # Skew metrics
    skew_factor: float = 1.0  # Max partition / avg partition

    # Memory
    peak_memory_bytes: int = 0
    spill_bytes: int = 0

    @property
    def duration_ms(self) -> float:
        """Execution duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    @property
    def selectivity(self) -> float:
        """Filter selectivity (output/input ratio)."""
        if self.input_rows == 0:
            return 1.0
        return self.output_rows / self.input_rows

    @property
    def has_skew(self) -> bool:
        """Check if data has significant skew."""
        return self.skew_factor > 2.0

    def calculate_skew(self) -> float:
        """Calculate skew factor from partition sizes."""
        if not self.partition_sizes:
            return 1.0
        avg = sum(self.partition_sizes) / len(self.partition_sizes)
        if avg == 0:
            return 1.0
        max_size = max(self.partition_sizes)
        self.skew_factor = max_size / avg
        return self.skew_factor


@dataclass
class WaggleDanceFeedback:
    """
    Feedback signal from execution (waggle dance).

    Workers "dance" to communicate execution characteristics
    back to the optimizer for adaptive decisions.
    """

    stage_id: str
    operator_id: str
    statistics: RuntimeStatistics
    suggestions: List[str] = field(default_factory=list)
    fitness_score: float = 0.5  # 0.0 to 1.0

    @classmethod
    def from_execution(
        cls,
        stage_id: str,
        operator_id: str,
        input_rows: int,
        output_rows: int,
        duration_ms: float,
        partition_sizes: Optional[List[int]] = None,
    ) -> "WaggleDanceFeedback":
        """Create feedback from execution metrics."""
        stats = RuntimeStatistics(
            input_rows=input_rows, output_rows=output_rows, partition_sizes=partition_sizes or []
        )

        suggestions = []

        # Analyze and suggest adaptations
        if stats.selectivity < 0.1:
            suggestions.append("highly_selective_filter")
        elif stats.selectivity > 0.9:
            suggestions.append("low_selectivity_filter")

        if partition_sizes:
            stats.calculate_skew()
            if stats.has_skew:
                suggestions.append("data_skew_detected")

            avg_size = sum(partition_sizes) / len(partition_sizes) if partition_sizes else 0
            if avg_size > 100_000_000:  # 100MB
                suggestions.append("partitions_too_large")
            elif avg_size < 1_000_000:  # 1MB
                suggestions.append("partitions_too_small")

        # Calculate fitness score
        fitness = 1.0
        if duration_ms > 10000:  # Slow execution
            fitness *= 0.7
        if stats.has_skew:
            fitness *= 0.8
        if stats.selectivity < 0.01:  # Very selective
            fitness *= 1.2  # Bonus for good filtering

        return cls(
            stage_id=stage_id,
            operator_id=operator_id,
            statistics=stats,
            suggestions=suggestions,
            fitness_score=min(1.0, fitness),
        )


@dataclass
class AdaptationRule:
    """A rule for adapting query execution."""

    name: str
    trigger: AdaptationTrigger
    condition: Callable[[WaggleDanceFeedback], bool]
    action: Callable[[Any], Any]  # Takes plan node, returns modified node
    priority: int = 0


class AQEContext:
    """
    Context for Adaptive Query Execution.

    Maintains state and coordinates adaptations during query execution.
    """

    def __init__(
        self,
        broadcast_threshold: int = 10_000_000,  # 10MB
        target_partition_size: int = 64_000_000,  # 64MB
        skew_threshold: float = 2.0,
        min_partitions: int = 1,
        max_partitions: int = 200,
    ):
        self.broadcast_threshold = broadcast_threshold
        self.target_partition_size = target_partition_size
        self.skew_threshold = skew_threshold
        self.min_partitions = min_partitions
        self.max_partitions = max_partitions

        # Feedback collection
        self._feedback: Dict[str, WaggleDanceFeedback] = {}
        self._lock = threading.Lock()

        # Adaptation tracking
        self._adaptations: List[Dict[str, Any]] = []
        self._adaptation_count = 0

    def record_feedback(self, feedback: WaggleDanceFeedback) -> None:
        """Record feedback from execution stage."""
        with self._lock:
            self._feedback[feedback.stage_id] = feedback

    def get_feedback(self, stage_id: str) -> Optional[WaggleDanceFeedback]:
        """Get feedback for a stage."""
        with self._lock:
            return self._feedback.get(stage_id)

    def should_broadcast(self, table_size_bytes: int) -> bool:
        """Determine if a table should be broadcast."""
        return table_size_bytes <= self.broadcast_threshold

    def calculate_num_partitions(self, data_size_bytes: int) -> int:
        """Calculate optimal number of partitions."""
        if data_size_bytes == 0:
            return self.min_partitions

        num_partitions = math.ceil(data_size_bytes / self.target_partition_size)
        return max(self.min_partitions, min(self.max_partitions, num_partitions))

    def record_adaptation(
        self, adaptation_type: str, old_plan: Any, new_plan: Any, reason: str
    ) -> None:
        """Record an adaptation for audit/debugging."""
        with self._lock:
            self._adaptation_count += 1
            self._adaptations.append(
                {
                    "id": self._adaptation_count,
                    "type": adaptation_type,
                    "reason": reason,
                    "timestamp": time.time(),
                }
            )

    def get_adaptations(self) -> List[Dict[str, Any]]:
        """Get all adaptations made."""
        with self._lock:
            return self._adaptations.copy()


class PartitionCoalescer:
    """
    Coalesces small partitions to reduce overhead.

    After filtering, many partitions may be mostly empty.
    This combines them for efficient processing.
    """

    def __init__(self, context: AQEContext):
        self.context = context

    def should_coalesce(self, partition_sizes: List[int]) -> bool:
        """Check if coalescing would be beneficial."""
        if not partition_sizes:
            return False

        avg_size = sum(partition_sizes) / len(partition_sizes)
        return avg_size < self.context.target_partition_size / 4

    def compute_target_partitions(self, partition_sizes: List[int]) -> int:
        """Calculate target number of partitions after coalescing."""
        total_size = sum(partition_sizes)
        return self.context.calculate_num_partitions(total_size)

    def coalesce_plan(self, partition_sizes: List[int]) -> List[List[int]]:
        """
        Create a coalescing plan.

        Returns list of partition groups (indices to combine).
        """
        if not partition_sizes:
            return []

        target_partitions = self.compute_target_partitions(partition_sizes)

        # Greedy bin packing
        bins: List[List[int]] = [[] for _ in range(target_partitions)]
        bin_sizes = [0] * target_partitions

        # Sort partitions by size (largest first)
        sorted_partitions = sorted(enumerate(partition_sizes), key=lambda x: x[1], reverse=True)

        for part_idx, part_size in sorted_partitions:
            # Find bin with smallest size
            min_bin_idx = min(range(len(bins)), key=lambda i: bin_sizes[i])
            bins[min_bin_idx].append(part_idx)
            bin_sizes[min_bin_idx] += part_size

        return [b for b in bins if b]  # Remove empty bins


class SkewHandler:
    """
    Handles data skew in joins and aggregations.

    Detects skewed keys and splits them for parallel processing.
    """

    def __init__(self, context: AQEContext):
        self.context = context

    def detect_skewed_keys(
        self, key_counts: Dict[Any, int], threshold: Optional[float] = None
    ) -> Set[Any]:
        """
        Detect keys with skewed distribution.

        Returns set of skewed keys.
        """
        if not key_counts:
            return set()

        threshold = threshold or self.context.skew_threshold
        avg_count = sum(key_counts.values()) / len(key_counts)

        if avg_count == 0:
            return set()

        skewed = set()
        for key, count in key_counts.items():
            if count / avg_count > threshold:
                skewed.add(key)

        return skewed

    def split_skewed_partition(
        self, key: Any, partition_size: int, num_splits: Optional[int] = None
    ) -> List[Tuple[Any, int]]:
        """
        Plan how to split a skewed partition.

        Returns list of (sub_key, expected_size) tuples.
        """
        if num_splits is None:
            # Calculate based on target partition size
            num_splits = max(2, math.ceil(partition_size / self.context.target_partition_size))

        split_size = partition_size // num_splits

        return [(f"{key}_split_{i}", split_size) for i in range(num_splits)]


class JoinStrategySelector:
    """
    Selects optimal join strategy based on runtime statistics.
    """

    def __init__(self, context: AQEContext):
        self.context = context

    def select_strategy(
        self, left_size_bytes: int, right_size_bytes: int, left_rows: int, right_rows: int
    ) -> JoinStrategy:
        """
        Select best join strategy based on table sizes.
        """
        # Check for broadcast candidates
        if right_size_bytes <= self.context.broadcast_threshold:
            return JoinStrategy.BROADCAST_HASH
        if left_size_bytes <= self.context.broadcast_threshold:
            return JoinStrategy.BROADCAST_HASH

        # For larger tables, consider sort-merge vs shuffle-hash
        # Sort-merge is better for sorted data or when output is sorted
        if left_rows > 1_000_000 or right_rows > 1_000_000:
            return JoinStrategy.SORT_MERGE

        return JoinStrategy.SHUFFLE_HASH

    def should_switch_strategy(
        self, current_strategy: JoinStrategy, feedback: WaggleDanceFeedback
    ) -> Optional[JoinStrategy]:
        """
        Check if strategy should be switched based on feedback.
        """
        stats = feedback.statistics

        # Check for join explosion
        if stats.output_rows > stats.input_rows * 10:
            if current_strategy != JoinStrategy.SORT_MERGE:
                return JoinStrategy.SORT_MERGE

        # Check for skew
        if stats.has_skew:
            # Broadcast is bad with skew on the broadcast side
            if current_strategy == JoinStrategy.BROADCAST_HASH:
                return JoinStrategy.SHUFFLE_HASH

        return None


class AdaptiveQueryExecutor:
    """
    Main interface for Adaptive Query Execution.

    Coordinates runtime statistics collection and plan adaptation
    using waggle dance feedback from execution stages.

    Example:
        executor = AdaptiveQueryExecutor()

        # Execute with adaptation
        result = executor.execute(plan, data)

        # Check what adaptations were made
        adaptations = executor.context.get_adaptations()
    """

    def __init__(
        self,
        context: Optional[AQEContext] = None,
        enable_coalescing: bool = True,
        enable_skew_handling: bool = True,
        enable_join_optimization: bool = True,
    ):
        self.context = context or AQEContext()
        self.enable_coalescing = enable_coalescing
        self.enable_skew_handling = enable_skew_handling
        self.enable_join_optimization = enable_join_optimization

        self.coalescer = PartitionCoalescer(self.context)
        self.skew_handler = SkewHandler(self.context)
        self.join_selector = JoinStrategySelector(self.context)

        # Adaptation rules
        self.rules: List[AdaptationRule] = []
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Set up default adaptation rules."""
        # Coalescing rule
        if self.enable_coalescing:
            self.rules.append(
                AdaptationRule(
                    name="coalesce_small_partitions",
                    trigger=AdaptationTrigger.PARTITION_SIZE,
                    condition=lambda f: "partitions_too_small" in f.suggestions,
                    action=lambda plan: self._apply_coalescing(plan),
                    priority=10,
                )
            )

        # Skew handling rule
        if self.enable_skew_handling:
            self.rules.append(
                AdaptationRule(
                    name="handle_data_skew",
                    trigger=AdaptationTrigger.SKEW_DETECTED,
                    condition=lambda f: "data_skew_detected" in f.suggestions,
                    action=lambda plan: self._apply_skew_handling(plan),
                    priority=20,
                )
            )

    def _apply_coalescing(self, plan: Any) -> Any:
        """Apply partition coalescing to plan."""
        # In a real implementation, this would modify the physical plan
        return plan

    def _apply_skew_handling(self, plan: Any) -> Any:
        """Apply skew handling to plan."""
        # In a real implementation, this would add skew join optimization
        return plan

    def record_stage_feedback(
        self,
        stage_id: str,
        operator_id: str,
        input_rows: int,
        output_rows: int,
        duration_ms: float,
        partition_sizes: Optional[List[int]] = None,
    ) -> WaggleDanceFeedback:
        """
        Record feedback from an execution stage.

        Returns the feedback object with suggestions.
        """
        feedback = WaggleDanceFeedback.from_execution(
            stage_id=stage_id,
            operator_id=operator_id,
            input_rows=input_rows,
            output_rows=output_rows,
            duration_ms=duration_ms,
            partition_sizes=partition_sizes,
        )

        self.context.record_feedback(feedback)

        # Check adaptation rules
        self._check_adaptations(feedback)

        return feedback

    def _check_adaptations(self, feedback: WaggleDanceFeedback) -> None:
        """Check if any adaptations should be triggered."""
        for rule in sorted(self.rules, key=lambda r: -r.priority):
            if rule.condition(feedback):
                self.context.record_adaptation(
                    adaptation_type=rule.name,
                    old_plan=None,
                    new_plan=None,
                    reason=f"Triggered by {feedback.stage_id}: {feedback.suggestions}",
                )

    def get_optimal_join_strategy(
        self, left_stats: RuntimeStatistics, right_stats: RuntimeStatistics
    ) -> JoinStrategy:
        """Get optimal join strategy based on collected statistics."""
        return self.join_selector.select_strategy(
            left_size_bytes=left_stats.input_bytes,
            right_size_bytes=right_stats.input_bytes,
            left_rows=left_stats.input_rows,
            right_rows=right_stats.input_rows,
        )

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of adaptive execution."""
        adaptations = self.context.get_adaptations()

        return {
            "total_adaptations": len(adaptations),
            "adaptations_by_type": defaultdict(int, {a["type"]: 1 for a in adaptations}),
            "feedback_count": len(self.context._feedback),
            "stages": [
                {"stage_id": f.stage_id, "fitness": f.fitness_score, "suggestions": f.suggestions}
                for f in self.context._feedback.values()
            ],
        }
