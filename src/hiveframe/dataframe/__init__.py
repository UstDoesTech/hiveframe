"""
HiveFrame DataFrame Package
===========================
A familiar DataFrame interface using bee-inspired processing underneath.

Provides Spark-like API:
- select, filter, groupBy, agg, join
- Lazy evaluation with optimization
- Type inference and schema support
"""

from .columns import (
    DataType,
    Column,
    col,
    lit,
)

from .aggregations import (
    AggFunc,
    sum_agg,
    avg,
    count,
    count_all,
    min_agg,
    max_agg,
    collect_list,
    collect_set,
)

from .schema import Schema

from .grouped import GroupedData

from .frame import (
    HiveDataFrame,
    createDataFrame,
)

__all__ = [
    # Column types and expressions
    'DataType',
    'Column',
    'col',
    'lit',
    # Aggregations
    'AggFunc',
    'sum_agg',
    'avg',
    'count',
    'count_all',
    'min_agg',
    'max_agg',
    'collect_list',
    'collect_set',
    # Schema
    'Schema',
    # Grouped data
    'GroupedData',
    # DataFrame
    'HiveDataFrame',
    'createDataFrame',
]
