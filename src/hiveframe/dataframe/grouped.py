"""
HiveFrame DataFrame Grouped Data
================================
Grouped DataFrame for aggregation operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from .aggregations import AggFunc, count_all

if TYPE_CHECKING:
    from .frame import HiveDataFrame


class GroupedData:
    """Grouped DataFrame for aggregation operations."""

    def __init__(self, df: "HiveDataFrame", group_cols: List[str]):
        self._df = df
        self._group_cols = group_cols

    def agg(self, *aggs: AggFunc) -> "HiveDataFrame":
        """Apply aggregation functions to grouped data."""
        # Import here to avoid circular import
        from .frame import HiveDataFrame

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

    def count(self) -> "HiveDataFrame":
        """Count rows per group."""
        return self.agg(count_all())
