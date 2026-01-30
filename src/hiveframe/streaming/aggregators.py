"""
HiveFrame Streaming Aggregators
===============================
Built-in aggregation functions for windowed stream processing.

These are convenience functions that return (accumulator_fn) compatible
with the streaming processor's window aggregation. They delegate to
the unified aggregation framework.

For more control, use the Aggregator classes from hiveframe.aggregations directly.
"""

from typing import Any, List, Tuple

# Re-export factory functions from unified aggregations module
from ..aggregations import (
    # Factory functions for streaming
    count_agg,
    sum_agg,
    avg_agg,
    min_agg,
    max_agg,
    collect_agg,
    # Aggregator classes for advanced use
    Aggregator,
    CountAggregator,
    SumAggregator,
    AvgAggregator,
    MinAggregator,
    MaxAggregator,
    CollectListAggregator,
    CollectSetAggregator,
    get_aggregator_registry,
)

# Legacy function signatures for backward compatibility
# These match the original streaming API (acc, value) -> acc


def count_aggregator(acc: int, value: Any) -> int:
    """Count aggregator (legacy streaming API)."""
    return acc + 1


def sum_aggregator(acc: float, value: float) -> float:
    """Sum aggregator (legacy streaming API)."""
    return acc + value


def avg_aggregator(acc: Tuple[float, int], value: float) -> Tuple[float, int]:
    """Average aggregator (sum, count) (legacy streaming API)."""
    return (acc[0] + value, acc[1] + 1)


def max_aggregator(acc: float, value: float) -> float:
    """Max aggregator (legacy streaming API)."""
    return max(acc, value)


def min_aggregator(acc: float, value: float) -> float:
    """Min aggregator (legacy streaming API)."""
    return min(acc, value)


def collect_aggregator(acc: List, value: Any) -> List:
    """Collect all values into a list (legacy streaming API)."""
    acc.append(value)
    return acc
