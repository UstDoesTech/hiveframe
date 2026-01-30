"""
HiveFrame DataFrame Columns
===========================
Column expressions and data types for DataFrame operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional


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
        return self._expr(row)  # type: ignore[misc]

    # Comparison operators
    def __eq__(self, other) -> Column:  # type: ignore[override]
        if isinstance(other, Column):
            return Column(
                name=f"({self.name} == {other.name})",
                dtype=DataType.BOOLEAN,
                _expr=lambda row: self.eval(row) == other.eval(row),
            )
        return Column(
            name=f"({self.name} == {other})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row) == other,
        )

    def __ne__(self, other) -> Column:  # type: ignore[override]
        if isinstance(other, Column):
            return Column(
                name=f"({self.name} != {other.name})",
                dtype=DataType.BOOLEAN,
                _expr=lambda row: self.eval(row) != other.eval(row),
            )
        return Column(
            name=f"({self.name} != {other})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row) != other,
        )

    def __lt__(self, other) -> Column:
        return Column(
            name=f"({self.name} < {other})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row)
            < (other.eval(row) if isinstance(other, Column) else other),
        )

    def __le__(self, other) -> Column:
        return Column(
            name=f"({self.name} <= {other})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row)
            <= (other.eval(row) if isinstance(other, Column) else other),
        )

    def __gt__(self, other) -> Column:
        return Column(
            name=f"({self.name} > {other})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row)
            > (other.eval(row) if isinstance(other, Column) else other),
        )

    def __ge__(self, other) -> Column:
        return Column(
            name=f"({self.name} >= {other})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row)
            >= (other.eval(row) if isinstance(other, Column) else other),
        )

    # Logical operators
    def __and__(self, other: "Column") -> "Column":
        return Column(
            name=f"({self.name} AND {other.name})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: bool(self.eval(row)) and bool(other.eval(row)),
        )

    def __or__(self, other: "Column") -> "Column":
        return Column(
            name=f"({self.name} OR {other.name})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: bool(self.eval(row)) or bool(other.eval(row)),
        )

    def __invert__(self) -> "Column":
        return Column(
            name=f"NOT({self.name})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: not bool(self.eval(row)),
        )

    # Arithmetic operators
    def __add__(self, other) -> Column:
        return Column(
            name=f"({self.name} + {other})",
            dtype=DataType.FLOAT,
            _expr=lambda row: self.eval(row)
            + (other.eval(row) if isinstance(other, Column) else other),
        )

    def __sub__(self, other) -> Column:
        return Column(
            name=f"({self.name} - {other})",
            dtype=DataType.FLOAT,
            _expr=lambda row: self.eval(row)
            - (other.eval(row) if isinstance(other, Column) else other),
        )

    def __mul__(self, other) -> Column:
        return Column(
            name=f"({self.name} * {other})",
            dtype=DataType.FLOAT,
            _expr=lambda row: self.eval(row)
            * (other.eval(row) if isinstance(other, Column) else other),
        )

    def __truediv__(self, other) -> Column:
        return Column(
            name=f"({self.name} / {other})",
            dtype=DataType.FLOAT,
            _expr=lambda row: self.eval(row)
            / (other.eval(row) if isinstance(other, Column) else other),
        )

    # String operations
    def contains(self, substr: str) -> Column:
        return Column(
            name=f"{self.name}.contains({substr})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: substr in str(self.eval(row)),
        )

    def startswith(self, prefix: str) -> Column:
        return Column(
            name=f"{self.name}.startswith({prefix})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: str(self.eval(row)).startswith(prefix),
        )

    def endswith(self, suffix: str) -> Column:
        return Column(
            name=f"{self.name}.endswith({suffix})",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: str(self.eval(row)).endswith(suffix),
        )

    # Null handling
    def isNull(self) -> Column:
        return Column(
            name=f"{self.name}.isNull()",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row) is None,
        )

    def isNotNull(self) -> Column:
        return Column(
            name=f"{self.name}.isNotNull()",
            dtype=DataType.BOOLEAN,
            _expr=lambda row: self.eval(row) is not None,
        )

    # Aliasing
    def alias(self, name: str) -> Column:
        return Column(name=name, dtype=self.dtype, _expr=self._expr)

    # Casting
    def cast(self, dtype: DataType) -> Column:
        type_map = {
            DataType.STRING: str,
            DataType.INTEGER: int,
            DataType.FLOAT: float,
            DataType.BOOLEAN: bool,
        }
        cast_fn = type_map.get(dtype, lambda x: x)
        return Column(name=self.name, dtype=dtype, _expr=lambda row: cast_fn(self.eval(row)))


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

    return Column(name=str(value), dtype=dtype, _expr=lambda row: value)
