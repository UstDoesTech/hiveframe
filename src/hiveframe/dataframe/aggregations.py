"""
HiveFrame DataFrame Aggregations
================================
Aggregation functions for grouped DataFrame operations.
"""

from typing import Any, Callable, Dict, List

from .columns import Column


class AggFunc:
    """Aggregation function wrapper."""
    def __init__(self, name: str, column: Column, fn: Callable[[List], Any]):
        self.name = name
        self.column = column
        self.fn = fn
        
    def apply(self, rows: List[Dict]) -> Any:
        values = [self.column.eval(row) for row in rows]
        return self.fn(values)


def sum_agg(column: Column) -> AggFunc:
    """Sum aggregation."""
    return AggFunc(f"sum({column.name})", column, lambda vals: sum(v for v in vals if v is not None))


def avg(column: Column) -> AggFunc:
    """Average aggregation."""
    return AggFunc(f"avg({column.name})", column, 
                   lambda vals: sum(v for v in vals if v is not None) / len([v for v in vals if v is not None]) if vals else None)


def count(column: Column) -> AggFunc:
    """Count non-null values."""
    return AggFunc(f"count({column.name})", column, lambda vals: len([v for v in vals if v is not None]))


def count_all() -> AggFunc:
    """Count all rows."""
    return AggFunc("count(*)", Column("*"), lambda vals: len(vals))


def min_agg(column: Column) -> AggFunc:
    """Minimum aggregation."""
    return AggFunc(f"min({column.name})", column, lambda vals: min((v for v in vals if v is not None), default=None))


def max_agg(column: Column) -> AggFunc:
    """Maximum aggregation."""
    return AggFunc(f"max({column.name})", column, lambda vals: max((v for v in vals if v is not None), default=None))


def collect_list(column: Column) -> AggFunc:
    """Collect values into a list."""
    return AggFunc(f"collect_list({column.name})", column, lambda vals: vals)


def collect_set(column: Column) -> AggFunc:
    """Collect unique values into a list."""
    return AggFunc(f"collect_set({column.name})", column, lambda vals: list(set(vals)))
