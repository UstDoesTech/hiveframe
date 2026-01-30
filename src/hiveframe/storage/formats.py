"""
Storage Format Types
====================

Common types and utilities for storage formats.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class FileFormat(Enum):
    """Supported file formats."""

    PARQUET = auto()
    DELTA = auto()
    CSV = auto()
    JSON = auto()
    JSONL = auto()
    ORC = auto()
    AVRO = auto()


class CompressionCodec(Enum):
    """Supported compression codecs."""

    NONE = "none"
    SNAPPY = "snappy"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BROTLI = "brotli"


@dataclass
class PartitionSpec:
    """
    Partition Specification
    -----------------------
    Defines how data is partitioned for storage.
    """

    columns: List[str]
    transforms: Dict[str, str] = field(default_factory=dict)
    # Transform types: 'identity', 'year', 'month', 'day', 'hour', 'bucket', 'truncate'

    def partition_path(self, row: Dict[str, Any]) -> str:
        """Generate partition path for a row."""
        parts = []
        for col in self.columns:
            value = row.get(col)
            transform = self.transforms.get(col, "identity")

            if transform == "identity":
                part_value = str(value) if value is not None else "__null__"
            elif transform == "year":
                part_value = (
                    str(value.year)
                    if value is not None and hasattr(value, "year")
                    else str(value)[:4]
                )
            elif transform == "month":
                part_value = (
                    str(value.month)
                    if value is not None and hasattr(value, "month")
                    else str(value)[5:7]
                )
            elif transform == "day":
                part_value = (
                    str(value.day)
                    if value is not None and hasattr(value, "day")
                    else str(value)[8:10]
                )
            else:
                part_value = str(value)

            parts.append(f"{col}={part_value}")

        return "/".join(parts)


@dataclass
class StorageOptions:
    """
    Storage Options
    ---------------
    Configuration for reading/writing data.
    """

    compression: CompressionCodec = CompressionCodec.SNAPPY
    row_group_size: int = 128 * 1024 * 1024  # 128 MB
    page_size: int = 1024 * 1024  # 1 MB
    partition_spec: Optional[PartitionSpec] = None
    schema_evolution: bool = True
    merge_schema: bool = False
    overwrite: bool = False
    append: bool = False

    # Delta Lake specific
    enable_cdf: bool = False  # Change Data Feed
    optimize_write: bool = True
    auto_compact: bool = False

    # Performance options
    parallel_reads: int = 4
    predicate_pushdown: bool = True
    column_pruning: bool = True


@dataclass
class ColumnStats:
    """Statistics for a column."""

    null_count: int = 0
    distinct_count: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None


@dataclass
class FileMetadata:
    """
    File Metadata
    -------------
    Metadata about a data file.
    """

    path: str
    format: FileFormat
    size_bytes: int = 0
    row_count: int = 0
    column_count: int = 0
    compression: CompressionCodec = CompressionCodec.NONE
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    partition_values: Dict[str, Any] = field(default_factory=dict)
    column_stats: Dict[str, ColumnStats] = field(default_factory=dict)
