"""
HiveFrame Storage
=================

Storage layer supporting various file formats including
Parquet and Delta Lake for the HiveFrame data platform.

Key Features:
- Parquet file support (columnar storage)
- Delta Lake format support (ACID transactions)
- Schema evolution and type inference
- Partition pruning for optimized reads
"""

from .parquet import (
    ParquetReader,
    ParquetWriter,
    ParquetSchema,
    read_parquet,
    write_parquet,
)
from .delta import (
    DeltaTable,
    DeltaTransaction,
    DeltaLog,
    read_delta,
    write_delta,
)
from .formats import (
    FileFormat,
    StorageOptions,
    PartitionSpec,
    CompressionCodec,
)

__all__ = [
    # Parquet
    'ParquetReader',
    'ParquetWriter',
    'ParquetSchema',
    'read_parquet',
    'write_parquet',
    # Delta
    'DeltaTable',
    'DeltaTransaction',
    'DeltaLog',
    'read_delta',
    'write_delta',
    # Common
    'FileFormat',
    'StorageOptions',
    'PartitionSpec',
    'CompressionCodec',
]
