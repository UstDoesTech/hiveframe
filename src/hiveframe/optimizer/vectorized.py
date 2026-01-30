"""
Vectorized Execution Engine (Phase 2)
=====================================

SIMD-accelerated processing for numerical workloads.
Processes data in batches using columnar operations for maximum throughput.

Key Features:
- Batch-oriented execution model
- Columnar processing with array operations
- SIMD-friendly data layouts
- Efficient null handling
- Parallel batch processing
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class VectorType(Enum):
    """Supported vector data types."""

    INT32 = auto()
    INT64 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    STRING = auto()
    BOOL = auto()
    TIMESTAMP = auto()


@dataclass
class VectorBatch:
    """
    A columnar batch of data for vectorized processing.

    Data is stored in column-major format for efficient
    SIMD operations and cache utilization.
    """

    num_rows: int
    columns: Dict[str, List[Any]] = field(default_factory=dict)
    null_masks: Dict[str, List[bool]] = field(default_factory=dict)
    types: Dict[str, VectorType] = field(default_factory=dict)

    @classmethod
    def from_rows(
        cls, rows: List[Dict[str, Any]], schema: Optional[Dict[str, VectorType]] = None
    ) -> "VectorBatch":
        """Create a batch from row-oriented data."""
        if not rows:
            return cls(num_rows=0)

        # Collect column names
        col_names: set = set()
        for row in rows:
            col_names.update(row.keys())

        # Build columnar arrays
        columns: Dict[str, List[Any]] = {name: [] for name in col_names}
        null_masks: Dict[str, List[bool]] = {name: [] for name in col_names}

        for row in rows:
            for name in col_names:
                value = row.get(name)
                columns[name].append(value)
                null_masks[name].append(value is None)

        # Infer types if not provided
        types = schema or {}
        for name in col_names:
            if name not in types:
                types[name] = cls._infer_type(columns[name])

        return cls(num_rows=len(rows), columns=columns, null_masks=null_masks, types=types)

    @staticmethod
    def _infer_type(values: List[Any]) -> VectorType:
        """Infer vector type from values."""
        for val in values:
            if val is not None:
                if isinstance(val, bool):
                    return VectorType.BOOL
                elif isinstance(val, int):
                    return VectorType.INT64
                elif isinstance(val, float):
                    return VectorType.FLOAT64
                elif isinstance(val, str):
                    return VectorType.STRING
        return VectorType.STRING  # Default

    def to_rows(self) -> List[Dict[str, Any]]:
        """Convert back to row-oriented format."""
        rows = []
        for i in range(self.num_rows):
            row = {}
            for name, values in self.columns.items():
                if not self.null_masks.get(name, [False] * self.num_rows)[i]:
                    row[name] = values[i]
                else:
                    row[name] = None
            rows.append(row)
        return rows

    def get_column(self, name: str) -> List[Any]:
        """Get a column's values."""
        return self.columns.get(name, [])

    def slice(self, start: int, end: int) -> "VectorBatch":
        """Get a slice of the batch."""
        return VectorBatch(
            num_rows=end - start,
            columns={name: values[start:end] for name, values in self.columns.items()},
            null_masks={name: mask[start:end] for name, mask in self.null_masks.items()},
            types=self.types.copy(),
        )

    def select(self, columns: List[str]) -> "VectorBatch":
        """Select specific columns."""
        return VectorBatch(
            num_rows=self.num_rows,
            columns={name: self.columns[name] for name in columns if name in self.columns},
            null_masks={name: self.null_masks[name] for name in columns if name in self.null_masks},
            types={name: self.types[name] for name in columns if name in self.types},
        )

    def filter(self, mask: List[bool]) -> "VectorBatch":
        """Filter rows using a boolean mask."""
        indices = [i for i, m in enumerate(mask) if m]

        return VectorBatch(
            num_rows=len(indices),
            columns={name: [values[i] for i in indices] for name, values in self.columns.items()},
            null_masks={
                name: [mask_vals[i] for i in indices] for name, mask_vals in self.null_masks.items()
            },
            types=self.types.copy(),
        )


class VectorizedOp:
    """Base class for vectorized operations."""

    def execute(self, batch: VectorBatch) -> VectorBatch:
        """Execute operation on a batch."""
        raise NotImplementedError


class VectorizedFilter(VectorizedOp):
    """
    Vectorized filter operation.

    Evaluates predicates on entire columns at once.
    """

    def __init__(self, column: str, operator: str, value: Any):
        self.column = column
        self.operator = operator
        self.value = value

    def execute(self, batch: VectorBatch) -> VectorBatch:
        """Execute filter on batch."""
        values = batch.get_column(self.column)
        null_mask = batch.null_masks.get(self.column, [False] * batch.num_rows)

        # Vectorized comparison
        result_mask = self._compare_vector(values, null_mask)

        return batch.filter(result_mask)

    def _compare_vector(self, values: List[Any], null_mask: List[bool]) -> List[bool]:
        """Compare entire vector against value."""
        result = []

        for i, (val, is_null) in enumerate(zip(values, null_mask)):
            if is_null:
                result.append(False)
            else:
                result.append(self._compare_single(val))

        return result

    def _compare_single(self, val: Any) -> bool:
        """Compare single value."""
        try:
            if self.operator == "=":
                return bool(val == self.value)
            elif self.operator == "!=":
                return bool(val != self.value)
            elif self.operator == ">":
                return bool(val > self.value)
            elif self.operator == ">=":
                return bool(val >= self.value)
            elif self.operator == "<":
                return bool(val < self.value)
            elif self.operator == "<=":
                return bool(val <= self.value)
            elif self.operator == "LIKE":
                return self._like_match(str(val), str(self.value))
            elif self.operator == "IN":
                return bool(val in self.value)
            else:
                return False
        except (TypeError, ValueError):
            return False

    def _like_match(self, val: str, pattern: str) -> bool:
        """Simple LIKE pattern matching."""
        # Convert SQL LIKE pattern to regex-like matching
        import re

        regex_pattern = pattern.replace("%", ".*").replace("_", ".")
        return bool(re.match(f"^{regex_pattern}$", val, re.IGNORECASE))


class VectorizedProject(VectorizedOp):
    """
    Vectorized projection operation.

    Evaluates expressions on entire columns.
    """

    def __init__(self, expressions: Dict[str, Union[str, Callable]]):
        """
        Args:
            expressions: Mapping of output column name to either
                        source column name (string) or computation function.
        """
        self.expressions = expressions

    def execute(self, batch: VectorBatch) -> VectorBatch:
        """Execute projection on batch."""
        new_columns = {}
        new_null_masks = {}
        new_types = {}

        for out_name, expr in self.expressions.items():
            if isinstance(expr, str):
                # Simple column reference
                if expr in batch.columns:
                    new_columns[out_name] = batch.columns[expr].copy()
                    new_null_masks[out_name] = batch.null_masks.get(
                        expr, [False] * batch.num_rows
                    ).copy()
                    new_types[out_name] = batch.types.get(expr, VectorType.STRING)
            elif callable(expr):
                # Computed column
                values = []
                null_mask = []
                for i in range(batch.num_rows):
                    row = {name: vals[i] for name, vals in batch.columns.items()}
                    try:
                        result = expr(row)
                        values.append(result)
                        null_mask.append(result is None)
                    except Exception:
                        values.append(None)
                        null_mask.append(True)

                new_columns[out_name] = values
                new_null_masks[out_name] = null_mask
                new_types[out_name] = VectorBatch._infer_type(values)

        return VectorBatch(
            num_rows=batch.num_rows, columns=new_columns, null_masks=new_null_masks, types=new_types
        )


class VectorizedAggregate(VectorizedOp):
    """
    Vectorized aggregation operation.

    Computes aggregates over entire columns efficiently.
    """

    def __init__(
        self,
        group_by: List[str],
        aggregations: Dict[str, Tuple[str, str]],  # output_col -> (input_col, agg_func)
    ):
        self.group_by = group_by
        self.aggregations = aggregations

    def execute(self, batch: VectorBatch) -> VectorBatch:
        """Execute aggregation on batch."""
        # Group rows
        groups: Dict[tuple, List[int]] = defaultdict(list)

        for i in range(batch.num_rows):
            key = tuple(
                batch.columns[col][i] if col in batch.columns else None for col in self.group_by
            )
            groups[key].append(i)

        # Compute aggregates for each group
        result_rows = []

        for key, indices in groups.items():
            row = {}

            # Add group key columns
            for col, val in zip(self.group_by, key):
                row[col] = val

            # Compute aggregates
            for out_col, (in_col, agg_func) in self.aggregations.items():
                values = [
                    batch.columns[in_col][i]
                    for i in indices
                    if in_col in batch.columns
                    and not batch.null_masks.get(in_col, [False] * batch.num_rows)[i]
                ]

                row[out_col] = self._compute_aggregate(values, agg_func)

            result_rows.append(row)

        return VectorBatch.from_rows(result_rows)

    def _compute_aggregate(self, values: List[Any], func: str) -> Any:
        """Compute aggregate function."""
        if not values:
            return None

        if func == "COUNT":
            return len(values)
        elif func == "SUM":
            return sum(v for v in values if v is not None)
        elif func == "AVG":
            non_null = [v for v in values if v is not None]
            return sum(non_null) / len(non_null) if non_null else None
        elif func == "MIN":
            return min(v for v in values if v is not None)
        elif func == "MAX":
            return max(v for v in values if v is not None)
        elif func == "COUNT_DISTINCT":
            return len(set(v for v in values if v is not None))
        elif func == "COLLECT_LIST":
            return values
        else:
            return None


class VectorizedJoin(VectorizedOp):
    """
    Vectorized join operation.

    Implements hash join with batch processing.
    """

    def __init__(
        self, right_batch: VectorBatch, left_key: str, right_key: str, join_type: str = "inner"
    ):
        self.right_batch = right_batch
        self.left_key = left_key
        self.right_key = right_key
        self.join_type = join_type

        # Build hash table for right side
        self._hash_table: Dict[Any, List[int]] = defaultdict(list)
        for i in range(right_batch.num_rows):
            key = right_batch.columns.get(right_key, [None] * right_batch.num_rows)[i]
            if key is not None:
                self._hash_table[key].append(i)

    def execute(self, batch: VectorBatch) -> VectorBatch:
        """Execute join on batch."""
        result_rows = []

        for i in range(batch.num_rows):
            left_key_val = batch.columns.get(self.left_key, [None] * batch.num_rows)[i]

            # Find matching right rows
            right_indices = self._hash_table.get(left_key_val, [])

            if right_indices:
                for ri in right_indices:
                    row = {}
                    # Add left columns
                    for name, values in batch.columns.items():
                        row[f"left_{name}"] = values[i]
                    # Add right columns
                    for name, values in self.right_batch.columns.items():
                        row[f"right_{name}"] = values[ri]
                    result_rows.append(row)
            elif self.join_type == "left":
                row = {}
                for name, values in batch.columns.items():
                    row[f"left_{name}"] = values[i]
                for name in self.right_batch.columns:
                    row[f"right_{name}"] = None
                result_rows.append(row)

        return VectorBatch.from_rows(result_rows)


class VectorizedSort(VectorizedOp):
    """
    Vectorized sort operation.
    """

    def __init__(self, sort_keys: List[Tuple[str, bool]]):  # (column, ascending)
        self.sort_keys = sort_keys

    def execute(self, batch: VectorBatch) -> VectorBatch:
        """Execute sort on batch."""
        # Get sort indices
        indices = list(range(batch.num_rows))

        def sort_key(i):
            key = []
            for col, asc in self.sort_keys:
                val = batch.columns.get(col, [None] * batch.num_rows)[i]
                # Handle None values (sort to end)
                if val is None:
                    val = (1, None)  # Sort nulls last
                else:
                    val = (0, val if asc else -val if isinstance(val, (int, float)) else val)
                key.append(val)
            return key

        indices.sort(key=sort_key)

        # Reorder all columns
        return VectorBatch(
            num_rows=batch.num_rows,
            columns={name: [values[i] for i in indices] for name, values in batch.columns.items()},
            null_masks={
                name: [mask[i] for i in indices] for name, mask in batch.null_masks.items()
            },
            types=batch.types.copy(),
        )


class VectorizedLimit(VectorizedOp):
    """
    Vectorized limit operation.
    """

    def __init__(self, limit: int, offset: int = 0):
        self.limit = limit
        self.offset = offset

    def execute(self, batch: VectorBatch) -> VectorBatch:
        """Execute limit on batch."""
        return batch.slice(self.offset, min(self.offset + self.limit, batch.num_rows))


class VectorizedPipeline:
    """
    A pipeline of vectorized operations.

    Executes operations in sequence on batches of data
    for maximum throughput.
    """

    def __init__(self, batch_size: int = 1024):
        self.batch_size = batch_size
        self.operations: List[VectorizedOp] = []

    def add(self, op: VectorizedOp) -> "VectorizedPipeline":
        """Add an operation to the pipeline."""
        self.operations.append(op)
        return self

    def execute(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute pipeline on data.

        Processes data in batches for efficiency.
        """
        results = []

        # Process in batches
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i : i + self.batch_size]
            batch = VectorBatch.from_rows(batch_data)

            # Execute operations
            for op in self.operations:
                batch = op.execute(batch)

            # Collect results
            results.extend(batch.to_rows())

        return results

    def execute_batch(self, batch: VectorBatch) -> VectorBatch:
        """Execute pipeline on a single batch."""
        for op in self.operations:
            batch = op.execute(batch)
        return batch


class ParallelVectorizedExecutor:
    """
    Parallel executor for vectorized pipelines.

    Distributes batches across multiple threads for
    parallel processing.
    """

    def __init__(self, num_workers: int = 4, batch_size: int = 1024):
        self.num_workers = num_workers
        self.batch_size = batch_size

    def execute(
        self, pipeline: VectorizedPipeline, data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute pipeline in parallel."""
        # Split data into batches
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i : i + self.batch_size]
            batches.append(VectorBatch.from_rows(batch_data))

        # Process batches in parallel
        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(pipeline.execute_batch, batch): i for i, batch in enumerate(batches)
            }

            # Collect results in order
            result_batches: List[Optional[VectorBatch]] = [None] * len(batches)
            for future in as_completed(futures):
                idx = futures[future]
                result_batches[idx] = future.result()

        # Flatten results
        for batch in result_batches:
            if batch:
                results.extend(batch.to_rows())

        return results


def create_vectorized_filter(column: str, operator: str, value: Any) -> VectorizedFilter:
    """Create a vectorized filter operation."""
    return VectorizedFilter(column, operator, value)


def create_vectorized_project(columns: List[str]) -> VectorizedProject:
    """Create a vectorized projection with simple column selection."""
    return VectorizedProject({col: col for col in columns})


def create_vectorized_aggregate(
    group_by: List[str], **aggregations: Tuple[str, str]
) -> VectorizedAggregate:
    """Create a vectorized aggregation."""
    return VectorizedAggregate(group_by, dict(aggregations))
