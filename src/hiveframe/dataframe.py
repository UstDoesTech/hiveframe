"""
HiveFrame DataFrame API
=======================
A familiar DataFrame interface using bee-inspired processing underneath.

Provides Spark-like API:
- select, filter, groupBy, agg, join
- Lazy evaluation with optimization
- Type inference and schema support
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, 
    TypeVar, Union, Sequence, Iterator
)
from enum import Enum, auto
import operator
from functools import reduce
import json
import csv
import io

from .core import HiveFrame, create_hive


T = TypeVar('T')


class DataType(Enum):
    """Supported data types."""
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    ARRAY = auto()
    MAP = auto()
    STRUCT = auto()
    NULL = auto()
    

@dataclass
class Column:
    """
    Column expression for DataFrame operations.
    Supports chained operations like Spark's Column API.
    """
    name: str
    dtype: DataType = DataType.STRING
    _expr: Optional[Callable[[Dict], Any]] = None
    
    def __post_init__(self):
        if self._expr is None:
            self._expr = lambda row: row.get(self.name)
            
    def eval(self, row: Dict) -> Any:
        """Evaluate column expression against a row."""
        return self._expr(row)
        
    # Comparison operators
    def __eq__(self, other) -> Column:
        if isinstance(other, Column):
            return Column(
                name=f"({self.name} == {other.name})",
                dtype=DataType.BOOLEAN,
                _expr=lambda row: self.eval(row) == other.eval(row)
            )
        return Column(
            name=f"({self.name} == {other})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row) == other
        )
        
    def __ne__(self, other) -> Column:
        if isinstance(other, Column):
            return Column(
                name=f"({self.name} != {other.name})",
                dtype=DataType.BOOLEAN,
                _expr=lambda row: self.eval(row) != other.eval(row)
            )
        return Column(
            name=f"({self.name} != {other})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row) != other
        )
        
    def __lt__(self, other) -> Column:
        return Column(
            name=f"({self.name} < {other})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row) < (other.eval(row) if isinstance(other, Column) else other)
        )
        
    def __le__(self, other) -> Column:
        return Column(
            name=f"({self.name} <= {other})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row) <= (other.eval(row) if isinstance(other, Column) else other)
        )
        
    def __gt__(self, other) -> Column:
        return Column(
            name=f"({self.name} > {other})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row) > (other.eval(row) if isinstance(other, Column) else other)
        )
        
    def __ge__(self, other) -> Column:
        return Column(
            name=f"({self.name} >= {other})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row) >= (other.eval(row) if isinstance(other, Column) else other)
        )
        
    # Logical operators
    def __and__(self, other: 'Column') -> 'Column':
        return Column(
            name=f"({self.name} AND {other.name})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: bool(self.eval(row)) and bool(other.eval(row))
        )
        
    def __or__(self, other: 'Column') -> 'Column':
        return Column(
            name=f"({self.name} OR {other.name})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: bool(self.eval(row)) or bool(other.eval(row))
        )
        
    def __invert__(self) -> 'Column':
        return Column(
            name=f"NOT({self.name})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: not bool(self.eval(row))
        )
        
    # Arithmetic operators
    def __add__(self, other) -> Column:
        return Column(
            name=f"({self.name} + {other})",
            dtype=DataType.FLOAT,
            _expr=lambda row: self.eval(row) + (other.eval(row) if isinstance(other, Column) else other)
        )
        
    def __sub__(self, other) -> Column:
        return Column(
            name=f"({self.name} - {other})",
            dtype=DataType.FLOAT,
            _expr=lambda row: self.eval(row) - (other.eval(row) if isinstance(other, Column) else other)
        )
        
    def __mul__(self, other) -> Column:
        return Column(
            name=f"({self.name} * {other})",
            dtype=DataType.FLOAT,
            _expr=lambda row: self.eval(row) * (other.eval(row) if isinstance(other, Column) else other)
        )
        
    def __truediv__(self, other) -> Column:
        return Column(
            name=f"({self.name} / {other})",
            dtype=DataType.FLOAT,
            _expr=lambda row: self.eval(row) / (other.eval(row) if isinstance(other, Column) else other)
        )
        
    # String operations
    def contains(self, substr: str) -> Column:
        return Column(
            name=f"{self.name}.contains({substr})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: substr in str(self.eval(row))
        )
        
    def startswith(self, prefix: str) -> Column:
        return Column(
            name=f"{self.name}.startswith({prefix})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: str(self.eval(row)).startswith(prefix)
        )
        
    def endswith(self, suffix: str) -> Column:
        return Column(
            name=f"{self.name}.endswith({suffix})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: str(self.eval(row)).endswith(suffix)
        )
        
    # Null handling
    def isNull(self) -> Column:
        return Column(
            name=f"{self.name}.isNull()",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row) is None
        )
        
    def isNotNull(self) -> Column:
        return Column(
            name=f"{self.name}.isNotNull()",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row) is not None
        )
        
    # Aliasing
    def alias(self, name: str) -> Column:
        return Column(
            name=name,
            dtype=self.dtype,
            _expr=self._expr
        )
        
    # Casting
    def cast(self, dtype: DataType) -> Column:
        type_map = {
            DataType.STRING: str,
            DataType.INTEGER: int,
            DataType.FLOAT: float,
            DataType.BOOLEAN: bool,
        }
        cast_fn = type_map.get(dtype, lambda x: x)
        return Column(
            name=self.name,
            dtype=dtype,
            _expr=lambda row: cast_fn(self.eval(row))
        )


def col(name: str) -> Column:
    """Create a column reference."""
    return Column(name=name)


def lit(value: Any) -> Column:
    """Create a literal column."""
    dtype = DataType.STRING
    if isinstance(value, int):
        dtype = DataType.INTEGER
    elif isinstance(value, float):
        dtype = DataType.FLOAT
    elif isinstance(value, bool):
        dtype = DataType.BOOLEAN
        
    return Column(
        name=str(value),
        dtype=dtype,
        _expr=lambda row: value
    )


# Aggregate functions
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
    return AggFunc(f"sum({column.name})", column, lambda vals: sum(v for v in vals if v is not None))

def avg(column: Column) -> AggFunc:
    return AggFunc(f"avg({column.name})", column, 
                   lambda vals: sum(v for v in vals if v is not None) / len([v for v in vals if v is not None]) if vals else None)

def count(column: Column) -> AggFunc:
    return AggFunc(f"count({column.name})", column, lambda vals: len([v for v in vals if v is not None]))

def count_all() -> AggFunc:
    return AggFunc("count(*)", Column("*"), lambda vals: len(vals))

def min_agg(column: Column) -> AggFunc:
    return AggFunc(f"min({column.name})", column, lambda vals: min((v for v in vals if v is not None), default=None))

def max_agg(column: Column) -> AggFunc:
    return AggFunc(f"max({column.name})", column, lambda vals: max((v for v in vals if v is not None), default=None))

def collect_list(column: Column) -> AggFunc:
    return AggFunc(f"collect_list({column.name})", column, lambda vals: vals)

def collect_set(column: Column) -> AggFunc:
    return AggFunc(f"collect_set({column.name})", column, lambda vals: list(set(vals)))


@dataclass
class Schema:
    """DataFrame schema."""
    fields: List[Tuple[str, DataType]]
    
    def __getitem__(self, name: str) -> DataType:
        for field_name, dtype in self.fields:
            if field_name == name:
                return dtype
        raise KeyError(f"Field {name} not found in schema")
        
    @property
    def names(self) -> List[str]:
        return [name for name, _ in self.fields]


class GroupedData:
    """Grouped DataFrame for aggregation operations."""
    
    def __init__(self, df: 'HiveDataFrame', group_cols: List[str]):
        self._df = df
        self._group_cols = group_cols
        
    def agg(self, *aggs: AggFunc) -> 'HiveDataFrame':
        """Apply aggregation functions to grouped data."""
        # Group rows by key columns
        groups: Dict[tuple, List[Dict]] = {}
        for row in self._df._data:
            key = tuple(row.get(col) for col in self._group_cols)
            if key not in groups:
                groups[key] = []
            groups[key].append(row)
            
        # Apply aggregations
        result_rows = []
        for key, rows in groups.items():
            result_row = {col: val for col, val in zip(self._group_cols, key)}
            for agg in aggs:
                result_row[agg.name] = agg.apply(rows)
            result_rows.append(result_row)
            
        return HiveDataFrame(result_rows, self._df._hive)
        
    def count(self) -> 'HiveDataFrame':
        """Count rows per group."""
        return self.agg(count_all())


class HiveDataFrame:
    """
    HiveDataFrame: Bee-Inspired DataFrame API
    =========================================
    
    A DataFrame implementation using HiveFrame's bee-inspired
    processing engine. Provides familiar Spark-like API.
    
    Usage:
        df = HiveDataFrame.from_json('data.json')
        result = df.filter(col('age') > 21).select('name', 'age')
    """
    
    def __init__(self, 
                 data: List[Dict[str, Any]], 
                 hive: Optional[HiveFrame] = None):
        self._data = data
        self._hive = hive or create_hive(num_workers=4)
        self._schema: Optional[Schema] = None
        
    @classmethod
    def from_json(cls, path: str, hive: Optional[HiveFrame] = None) -> 'HiveDataFrame':
        """Load DataFrame from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return cls(data, hive)
        
    @classmethod
    def from_csv(cls, 
                 path: str, 
                 header: bool = True,
                 hive: Optional[HiveFrame] = None) -> 'HiveDataFrame':
        """Load DataFrame from CSV file."""
        with open(path, 'r') as f:
            if header:
                reader = csv.DictReader(f)
                data = list(reader)
            else:
                reader = csv.reader(f)
                rows = list(reader)
                data = [{f"col_{i}": val for i, val in enumerate(row)} for row in rows]
        return cls(data, hive)
        
    @classmethod
    def from_records(cls, 
                     records: List[Dict[str, Any]],
                     hive: Optional[HiveFrame] = None) -> 'HiveDataFrame':
        """Create DataFrame from list of dictionaries."""
        return cls(records, hive)
        
    @property
    def schema(self) -> Schema:
        """Infer schema from data."""
        if self._schema is not None:
            return self._schema
            
        if not self._data:
            return Schema([])
            
        fields = []
        sample = self._data[0]
        for key, value in sample.items():
            if isinstance(value, bool):
                dtype = DataType.BOOLEAN
            elif isinstance(value, int):
                dtype = DataType.INTEGER
            elif isinstance(value, float):
                dtype = DataType.FLOAT
            elif isinstance(value, list):
                dtype = DataType.ARRAY
            elif isinstance(value, dict):
                dtype = DataType.MAP
            elif value is None:
                dtype = DataType.NULL
            else:
                dtype = DataType.STRING
            fields.append((key, dtype))
            
        self._schema = Schema(fields)
        return self._schema
        
    @property
    def columns(self) -> List[str]:
        """Get column names."""
        return self.schema.names
        
    def __len__(self) -> int:
        return len(self._data)
        
    def __iter__(self) -> Iterator[Dict]:
        return iter(self._data)
        
    # DataFrame operations
    def select(self, *cols: Union[str, Column]) -> 'HiveDataFrame':
        """Select columns."""
        def select_row(row: Dict) -> Dict:
            result = {}
            for c in cols:
                if isinstance(c, str):
                    result[c] = row.get(c)
                elif isinstance(c, Column):
                    result[c.name] = c.eval(row)
            return result
            
        # Use bee-inspired parallel processing
        new_data = self._hive.map(self._data, select_row)
        # Filter out None results
        new_data = [r for r in new_data if r is not None]
        return HiveDataFrame(new_data, self._hive)
        
    def filter(self, condition: Column) -> 'HiveDataFrame':
        """Filter rows by condition."""
        def filter_fn(row: Dict) -> bool:
            return condition.eval(row)
            
        new_data = self._hive.filter(self._data, filter_fn)
        return HiveDataFrame(new_data, self._hive)
        
    def where(self, condition: Column) -> 'HiveDataFrame':
        """Alias for filter."""
        return self.filter(condition)
        
    def withColumn(self, name: str, column: Column) -> 'HiveDataFrame':
        """Add or replace a column."""
        def add_col(row: Dict) -> Dict:
            new_row = dict(row)
            new_row[name] = column.eval(row)
            return new_row
            
        new_data = self._hive.map(self._data, add_col)
        # Filter out None results
        new_data = [r for r in new_data if r is not None]
        return HiveDataFrame(new_data, self._hive)
        
    def drop(self, *cols: str) -> 'HiveDataFrame':
        """Drop columns."""
        keep_cols = [c for c in self.columns if c not in cols]
        return self.select(*keep_cols)
        
    def distinct(self) -> 'HiveDataFrame':
        """Remove duplicate rows."""
        seen = set()
        result = []
        for row in self._data:
            key = tuple(sorted(row.items()))
            if key not in seen:
                seen.add(key)
                result.append(row)
        return HiveDataFrame(result, self._hive)
        
    def dropDuplicates(self, cols: Optional[List[str]] = None) -> 'HiveDataFrame':
        """Remove duplicates based on specific columns."""
        if cols is None:
            return self.distinct()
            
        seen = set()
        result = []
        for row in self._data:
            key = tuple(row.get(c) for c in cols)
            if key not in seen:
                seen.add(key)
                result.append(row)
        return HiveDataFrame(result, self._hive)
        
    def orderBy(self, *cols: Union[str, Column], ascending: bool = True) -> 'HiveDataFrame':
        """Sort by columns."""
        def get_sort_key(row: Dict) -> tuple:
            key = []
            for c in cols:
                if isinstance(c, str):
                    key.append(row.get(c))
                else:
                    key.append(c.eval(row))
            return tuple(key)
            
        sorted_data = sorted(self._data, key=get_sort_key, reverse=not ascending)
        return HiveDataFrame(sorted_data, self._hive)
        
    def sort(self, *cols: Union[str, Column], ascending: bool = True) -> 'HiveDataFrame':
        """Alias for orderBy."""
        return self.orderBy(*cols, ascending=ascending)
        
    def limit(self, n: int) -> 'HiveDataFrame':
        """Limit to first n rows."""
        return HiveDataFrame(self._data[:n], self._hive)
        
    def head(self, n: int = 5) -> List[Dict]:
        """Get first n rows as list."""
        return self._data[:n]
        
    def take(self, n: int) -> List[Dict]:
        """Take n rows."""
        return self._data[:n]
        
    def first(self) -> Optional[Dict]:
        """Get first row."""
        return self._data[0] if self._data else None
        
    def collect(self) -> List[Dict]:
        """Collect all rows as list."""
        return self._data
        
    def count(self) -> int:
        """Count rows."""
        return len(self._data)
        
    # Grouping
    def groupBy(self, *cols: str) -> GroupedData:
        """Group by columns."""
        return GroupedData(self, list(cols))
        
    # Joins
    def join(self, 
             other: 'HiveDataFrame', 
             on: Union[str, List[str]], 
             how: str = 'inner') -> 'HiveDataFrame':
        """Join with another DataFrame."""
        if isinstance(on, str):
            on = [on]
            
        # Build hash map from other
        other_map: Dict[tuple, List[Dict]] = {}
        for row in other._data:
            key = tuple(row.get(c) for c in on)
            if key not in other_map:
                other_map[key] = []
            other_map[key].append(row)
            
        result = []
        matched_keys = set()
        
        for left_row in self._data:
            key = tuple(left_row.get(c) for c in on)
            right_rows = other_map.get(key, [])
            
            if right_rows:
                matched_keys.add(key)
                for right_row in right_rows:
                    merged = dict(left_row)
                    for k, v in right_row.items():
                        if k not in merged:
                            merged[k] = v
                    result.append(merged)
            elif how in ('left', 'outer'):
                # Left outer - include left with nulls
                merged = dict(left_row)
                for k in other.columns:
                    if k not in merged:
                        merged[k] = None
                result.append(merged)
                
        if how in ('right', 'outer'):
            # Add unmatched right rows
            for key, right_rows in other_map.items():
                if key not in matched_keys:
                    for right_row in right_rows:
                        merged = dict(right_row)
                        for k in self.columns:
                            if k not in merged:
                                merged[k] = None
                        result.append(merged)
                        
        return HiveDataFrame(result, self._hive)
        
    def crossJoin(self, other: 'HiveDataFrame') -> 'HiveDataFrame':
        """Cartesian product."""
        result = []
        for left_row in self._data:
            for right_row in other._data:
                merged = dict(left_row)
                merged.update(right_row)
                result.append(merged)
        return HiveDataFrame(result, self._hive)
        
    # Union operations
    def union(self, other: 'HiveDataFrame') -> 'HiveDataFrame':
        """Union with another DataFrame."""
        return HiveDataFrame(self._data + other._data, self._hive)
        
    def unionByName(self, other: 'HiveDataFrame') -> 'HiveDataFrame':
        """Union by column name."""
        all_cols = set(self.columns) | set(other.columns)
        
        result = []
        for row in self._data:
            new_row = {c: row.get(c) for c in all_cols}
            result.append(new_row)
        for row in other._data:
            new_row = {c: row.get(c) for c in all_cols}
            result.append(new_row)
            
        return HiveDataFrame(result, self._hive)
        
    # Output
    def to_json(self, path: str) -> None:
        """Write to JSON file."""
        with open(path, 'w') as f:
            json.dump(self._data, f, indent=2)
            
    def to_csv(self, path: str) -> None:
        """Write to CSV file."""
        if not self._data:
            return
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self._data)
            
    def show(self, n: int = 20, truncate: int = 20) -> None:
        """Display DataFrame as table."""
        rows = self._data[:n]
        if not rows:
            print("Empty DataFrame")
            return
            
        cols = self.columns
        
        # Calculate column widths
        widths = {col: len(col) for col in cols}
        for row in rows:
            for col in cols:
                val = str(row.get(col, ''))[:truncate]
                widths[col] = max(widths[col], len(val))
                
        # Print header
        header = ' | '.join(col.ljust(widths[col]) for col in cols)
        separator = '-+-'.join('-' * widths[col] for col in cols)
        print(header)
        print(separator)
        
        # Print rows
        for row in rows:
            line = ' | '.join(
                str(row.get(col, ''))[:truncate].ljust(widths[col])
                for col in cols
            )
            print(line)
            
        if len(self._data) > n:
            print(f"... showing {n} of {len(self._data)} rows")
            
    def printSchema(self) -> None:
        """Print schema."""
        print("root")
        for name, dtype in self.schema.fields:
            print(f" |-- {name}: {dtype.name.lower()}")
            
    def describe(self) -> 'HiveDataFrame':
        """Compute statistics for numeric columns."""
        numeric_cols = [
            name for name, dtype in self.schema.fields
            if dtype in (DataType.INTEGER, DataType.FLOAT)
        ]
        
        if not numeric_cols:
            return HiveDataFrame([], self._hive)
            
        stats = ['count', 'mean', 'std', 'min', 'max']
        result = []
        
        for stat in stats:
            row = {'summary': stat}
            for col in numeric_cols:
                values = [r.get(col) for r in self._data if r.get(col) is not None]
                if not values:
                    row[col] = None
                elif stat == 'count':
                    row[col] = len(values)
                elif stat == 'mean':
                    row[col] = sum(values) / len(values)
                elif stat == 'std':
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    row[col] = variance ** 0.5
                elif stat == 'min':
                    row[col] = min(values)
                elif stat == 'max':
                    row[col] = max(values)
            result.append(row)
            
        return HiveDataFrame(result, self._hive)


# Convenience constructor
def createDataFrame(data: List[Dict[str, Any]], 
                    num_workers: int = 4) -> HiveDataFrame:
    """Create a HiveDataFrame from data."""
    hive = create_hive(num_workers=num_workers)
    return HiveDataFrame(data, hive)
