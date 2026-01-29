"""
HiveFrame Streaming Aggregators
===============================
Built-in aggregation functions for windowed stream processing.
"""

from typing import Any, List, Tuple


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
