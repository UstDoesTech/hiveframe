"""
HiveFrame Aggregations
======================
Unified aggregation framework for both batch DataFrame and streaming operations.

This module provides a common interface for aggregations that can be used
consistently across:
- DataFrame grouped operations (batch)
- Windowed stream processing (streaming)

The key abstraction is the Aggregator protocol which defines:
- initial_value(): Starting accumulator value
- accumulate(acc, value): Add a value to the accumulator
- extract(acc): Get the final result from the accumulator
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

# Type variables
T = TypeVar("T")  # Input value type
A = TypeVar("A")  # Accumulator type
R = TypeVar("R")  # Result type


# ============================================================================
# Aggregator Protocol
# ============================================================================


class Aggregator(ABC, Generic[T, A, R]):
    """
    Base class for all aggregation functions.

    An aggregator has three phases:
    1. Initialize: Create initial accumulator state
    2. Accumulate: Fold values into accumulator
    3. Extract: Get final result from accumulator

    This pattern supports both batch and streaming aggregations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the aggregation for display."""
        pass

    @abstractmethod
    def initial_value(self) -> A:
        """Return the initial accumulator value."""
        pass

    @abstractmethod
    def accumulate(self, acc: A, value: T) -> A:
        """Add a value to the accumulator."""
        pass

    @abstractmethod
    def extract(self, acc: A) -> R:
        """Extract the final result from the accumulator."""
        pass

    def aggregate(self, values: List[T]) -> R:
        """Convenience method to aggregate a list of values."""
        acc = self.initial_value()
        for value in values:
            if value is not None:
                acc = self.accumulate(acc, value)
        return self.extract(acc)


# ============================================================================
# Built-in Aggregators
# ============================================================================


class CountAggregator(Aggregator[Any, int, int]):
    """Count non-null values."""

    @property
    def name(self) -> str:
        return "count"

    def initial_value(self) -> int:
        return 0

    def accumulate(self, acc: int, value: Any) -> int:
        return acc + 1

    def extract(self, acc: int) -> int:
        return acc


class SumAggregator(Aggregator[float, float, float]):
    """Sum numeric values."""

    @property
    def name(self) -> str:
        return "sum"

    def initial_value(self) -> float:
        return 0.0

    def accumulate(self, acc: float, value: float) -> float:
        return acc + value

    def extract(self, acc: float) -> float:
        return acc


class AvgAggregator(Aggregator[float, Tuple[float, int], Optional[float]]):
    """Average numeric values."""

    @property
    def name(self) -> str:
        return "avg"

    def initial_value(self) -> Tuple[float, int]:
        return (0.0, 0)

    def accumulate(self, acc: Tuple[float, int], value: float) -> Tuple[float, int]:
        return (acc[0] + value, acc[1] + 1)

    def extract(self, acc: Tuple[float, int]) -> Optional[float]:
        total, count = acc
        return total / count if count > 0 else None


class MinAggregator(Aggregator[float, Optional[float], Optional[float]]):
    """Minimum value."""

    @property
    def name(self) -> str:
        return "min"

    def initial_value(self) -> Optional[float]:
        return None

    def accumulate(self, acc: Optional[float], value: float) -> float:
        if acc is None:
            return value
        return min(acc, value)

    def extract(self, acc: Optional[float]) -> Optional[float]:
        return acc


class MaxAggregator(Aggregator[float, Optional[float], Optional[float]]):
    """Maximum value."""

    @property
    def name(self) -> str:
        return "max"

    def initial_value(self) -> Optional[float]:
        return None

    def accumulate(self, acc: Optional[float], value: float) -> float:
        if acc is None:
            return value
        return max(acc, value)

    def extract(self, acc: Optional[float]) -> Optional[float]:
        return acc


class CollectListAggregator(Aggregator[T, List[T], List[T]]):
    """Collect all values into a list."""

    @property
    def name(self) -> str:
        return "collect_list"

    def initial_value(self) -> List[T]:
        return []

    def accumulate(self, acc: List[T], value: T) -> List[T]:
        acc.append(value)
        return acc

    def extract(self, acc: List[T]) -> List[T]:
        return acc


class CollectSetAggregator(Aggregator[T, set, List[T]]):
    """Collect unique values into a list (preserves set internally)."""

    @property
    def name(self) -> str:
        return "collect_set"

    def initial_value(self) -> set:
        return set()

    def accumulate(self, acc: set, value: T) -> set:
        acc.add(value)
        return acc

    def extract(self, acc: set) -> List[T]:
        return list(acc)


# ============================================================================
# Factory Functions (Streaming-Compatible)
# ============================================================================
# These functions return (initial_value, accumulator_fn) tuples for streaming


def count_agg() -> Tuple[int, Callable[[int, Any], int]]:
    """Count aggregator for streaming."""
    return (0, lambda acc, _: acc + 1)


def sum_agg() -> Tuple[float, Callable[[float, float], float]]:
    """Sum aggregator for streaming."""
    return (0.0, lambda acc, v: acc + v)


def avg_agg() -> Tuple[Tuple[float, int], Callable[[Tuple[float, int], float], Tuple[float, int]]]:
    """Average aggregator for streaming (returns (sum, count) tuple)."""
    return ((0.0, 0), lambda acc, v: (acc[0] + v, acc[1] + 1))


def min_agg() -> Tuple[float, Callable[[float, float], float]]:
    """Min aggregator for streaming."""
    return (float("inf"), lambda acc, v: min(acc, v))


def max_agg() -> Tuple[float, Callable[[float, float], float]]:
    """Max aggregator for streaming."""
    return (float("-inf"), lambda acc, v: max(acc, v))


def collect_agg() -> Tuple[List, Callable[[List, Any], List]]:
    """Collect aggregator for streaming."""

    def _accumulate(acc: List, v: Any) -> List:
        acc.append(v)
        return acc

    return ([], _accumulate)


# ============================================================================
# Aggregator Registry (Singleton Pattern)
# ============================================================================


class AggregatorRegistry:
    """Registry of available aggregators."""

    _instance: Optional["AggregatorRegistry"] = None
    _aggregators: Dict[str, type]

    def __new__(cls) -> "AggregatorRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._aggregators = {}
            cls._instance._register_defaults()
        return cls._instance

    def _register_defaults(self) -> None:
        """Register built-in aggregators."""
        self.register("count", CountAggregator)
        self.register("sum", SumAggregator)
        self.register("avg", AvgAggregator)
        self.register("min", MinAggregator)
        self.register("max", MaxAggregator)
        self.register("collect_list", CollectListAggregator)
        self.register("collect_set", CollectSetAggregator)

    def register(self, name: str, aggregator_cls: type) -> None:
        """Register an aggregator class."""
        self._aggregators[name] = aggregator_cls

    def get(self, name: str) -> Optional[type]:
        """Get an aggregator class by name."""
        return self._aggregators.get(name)

    def create(self, name: str) -> Aggregator:
        """Create an aggregator instance by name."""
        cls = self.get(name)
        if cls is None:
            raise ValueError(f"Unknown aggregator: {name}")
        return cls()

    def list_aggregators(self) -> List[str]:
        """List all registered aggregator names."""
        return list(self._aggregators.keys())


def get_aggregator_registry() -> AggregatorRegistry:
    """Get the global aggregator registry."""
    return AggregatorRegistry()
