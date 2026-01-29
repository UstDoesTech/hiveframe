"""
HiveFrame DataFrame Aggregations
================================
Aggregation functions for grouped DataFrame operations.

These functions wrap the unified aggregation framework for use with
DataFrame's Column-based API.
"""

from typing import Any, Callable, Dict, List

from .columns import Column
from ..aggregations import (
    Aggregator,
    CountAggregator,
    SumAggregator,
    AvgAggregator,
    MinAggregator,
    MaxAggregator,
    CollectListAggregator,
    CollectSetAggregator,
)


class AggFunc:
    """
    Aggregation function wrapper for DataFrame operations.
    
    Bridges the Column-based DataFrame API with the unified Aggregator framework.
    """
    def __init__(self, name: str, column: Column, aggregator: Aggregator):
        self.name = name
        self.column = column
        self._aggregator = aggregator
        
    def apply(self, rows: List[Dict]) -> Any:
        """Apply aggregation to rows, extracting column values."""
        values = [self.column.eval(row) for row in rows]
        # Filter None values and delegate to aggregator
        return self._aggregator.aggregate(values)


def sum_agg(column: Column) -> AggFunc:
    """Sum aggregation."""
    return AggFunc(f"sum({column.name})", column, SumAggregator())


def avg(column: Column) -> AggFunc:
    """Average aggregation."""
    return AggFunc(f"avg({column.name})", column, AvgAggregator())


def count(column: Column) -> AggFunc:
    """Count non-null values."""
    return AggFunc(f"count({column.name})", column, CountAggregator())


def count_all() -> AggFunc:
    """Count all rows."""
    # For count_all, we don't filter nulls - create a simple counter
    return AggFunc("count(*)", Column("*"), CountAggregator())


def min_agg(column: Column) -> AggFunc:
    """Minimum aggregation."""
    return AggFunc(f"min({column.name})", column, MinAggregator())


def max_agg(column: Column) -> AggFunc:
    """Maximum aggregation."""
    return AggFunc(f"max({column.name})", column, MaxAggregator())


def collect_list(column: Column) -> AggFunc:
    """Collect values into a list."""
    return AggFunc(f"collect_list({column.name})", column, CollectListAggregator())


def collect_set(column: Column) -> AggFunc:
    """Collect unique values into a list."""
    return AggFunc(f"collect_set({column.name})", column, CollectSetAggregator())
