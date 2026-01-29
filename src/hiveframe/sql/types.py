"""
SQL Type System
===============

Type definitions for SwarmQL SQL engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Type


class SQLType(ABC):
    """Base class for SQL types."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return type name."""
        pass
    
    @abstractmethod
    def python_type(self) -> Type:
        """Return corresponding Python type."""
        pass
    
    @abstractmethod
    def cast(self, value: Any) -> Any:
        """Cast value to this type."""
        pass
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SQLType):
            return False
        return self.name == other.name
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class IntegerType(SQLType):
    """SQL INTEGER type."""
    
    @property
    def name(self) -> str:
        return "INTEGER"
    
    def python_type(self) -> Type:
        return int
    
    def cast(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None


class FloatType(SQLType):
    """SQL FLOAT/DOUBLE type."""
    
    @property
    def name(self) -> str:
        return "FLOAT"
    
    def python_type(self) -> Type:
        return float
    
    def cast(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


class StringType(SQLType):
    """SQL STRING/VARCHAR type."""
    
    @property
    def name(self) -> str:
        return "STRING"
    
    def python_type(self) -> Type:
        return str
    
    def cast(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value)


class BooleanType(SQLType):
    """SQL BOOLEAN type."""
    
    @property
    def name(self) -> str:
        return "BOOLEAN"
    
    def python_type(self) -> Type:
        return bool
    
    def cast(self, value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes')
        return bool(value)


class DateType(SQLType):
    """SQL DATE type."""
    
    @property
    def name(self) -> str:
        return "DATE"
    
    def python_type(self) -> Type:
        from datetime import date
        return date
    
    def cast(self, value: Any) -> Any:
        if value is None:
            return None
        from datetime import date, datetime
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            return datetime.fromisoformat(value.split('T')[0]).date()
        return None


class TimestampType(SQLType):
    """SQL TIMESTAMP type."""
    
    @property
    def name(self) -> str:
        return "TIMESTAMP"
    
    def python_type(self) -> Type:
        from datetime import datetime
        return datetime
    
    def cast(self, value: Any) -> Any:
        if value is None:
            return None
        from datetime import datetime
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        return None


@dataclass
class ArrayType(SQLType):
    """SQL ARRAY type."""
    
    element_type: SQLType
    
    @property
    def name(self) -> str:
        return f"ARRAY<{self.element_type.name}>"
    
    def python_type(self) -> Type:
        return list
    
    def cast(self, value: Any) -> Optional[list]:
        if value is None:
            return None
        if isinstance(value, list):
            return [self.element_type.cast(v) for v in value]
        return None


@dataclass
class MapType(SQLType):
    """SQL MAP type."""
    
    key_type: SQLType
    value_type: SQLType
    
    @property
    def name(self) -> str:
        return f"MAP<{self.key_type.name}, {self.value_type.name}>"
    
    def python_type(self) -> Type:
        return dict
    
    def cast(self, value: Any) -> Optional[dict]:
        if value is None:
            return None
        if isinstance(value, dict):
            return {
                self.key_type.cast(k): self.value_type.cast(v)
                for k, v in value.items()
            }
        return None


def infer_sql_type(value: Any) -> SQLType:
    """Infer SQL type from Python value."""
    if value is None:
        return StringType()  # Default for null
    if isinstance(value, bool):
        return BooleanType()
    if isinstance(value, int):
        return IntegerType()
    if isinstance(value, float):
        return FloatType()
    if isinstance(value, list):
        if value:
            return ArrayType(infer_sql_type(value[0]))
        return ArrayType(StringType())
    if isinstance(value, dict):
        return MapType(StringType(), StringType())
    return StringType()
