"""
HiveFrame DataFrame Schema
==========================
Schema definition for DataFrames.
"""

from dataclasses import dataclass
from typing import List, Tuple

from .columns import DataType


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
