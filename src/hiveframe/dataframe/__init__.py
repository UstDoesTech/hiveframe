"""
HiveFrame DataFrame Package
===========================
A familiar DataFrame interface using bee-inspired processing underneath.

Provides Spark-like API:
- select, filter, groupBy, agg, join
- Lazy evaluation with optimization
- Type inference and schema support
"""

from .aggregations import (
    AggFunc,
    avg,
    collect_list,
    collect_set,
    count,
    count_all,
    max_agg,
    min_agg,
    sum_agg,
)
from .columns import (
    Column,
    DataType,
    col,
    lit,
)
from .frame import (
    HiveDataFrame,
    createDataFrame,
)
from .grouped import GroupedData
from .schema import Schema

__all__ = [
    # Column types and expressions
    "DataType",
    "Column",
    "col",
    "lit",
    # Aggregations
    "AggFunc",
    "sum_agg",
    "avg",
    "count",
    "count_all",
    "min_agg",
    "max_agg",
    "collect_list",
    "collect_set",
    # Schema
    "Schema",
    # Grouped data
    "GroupedData",
    # DataFrame
    "HiveDataFrame",
    "createDataFrame",
]
