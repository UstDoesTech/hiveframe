"""
Parquet Support
===============

Parquet file format support for HiveFrame.
Provides columnar storage with efficient compression and encoding.

Note: This implementation provides a pure-Python fallback when
pyarrow is not available, with full functionality when pyarrow
is installed.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from ..dataframe import HiveDataFrame
from ..dataframe.schema import DataType, Schema
from .formats import (
    CompressionCodec,
    FileFormat,
    FileMetadata,
    StorageOptions,
)


@dataclass
class ParquetColumn:
    """Parquet column definition."""

    name: str
    type: str  # 'INT32', 'INT64', 'FLOAT', 'DOUBLE', 'BYTE_ARRAY', 'BOOLEAN'
    logical_type: Optional[str] = None  # 'STRING', 'DATE', 'TIMESTAMP', etc.
    nullable: bool = True
    encoding: str = "PLAIN"
    compression: str = "SNAPPY"


@dataclass
class ParquetSchema:
    """
    Parquet Schema
    --------------
    Schema definition for Parquet files.
    """

    columns: List[ParquetColumn]

    @classmethod
    def from_hive_schema(cls, schema: Schema) -> "ParquetSchema":
        """Convert HiveFrame schema to Parquet schema."""
        type_mapping = {
            DataType.INTEGER: ("INT64", None),
            DataType.FLOAT: ("DOUBLE", None),
            DataType.STRING: ("BYTE_ARRAY", "STRING"),
            DataType.BOOLEAN: ("BOOLEAN", None),
            DataType.ARRAY: ("BYTE_ARRAY", "JSON"),
            DataType.MAP: ("BYTE_ARRAY", "JSON"),
            DataType.NULL: ("BYTE_ARRAY", "STRING"),
        }

        columns = []
        for name, dtype in schema.fields:
            pq_type, logical = type_mapping.get(dtype, ("BYTE_ARRAY", "STRING"))
            columns.append(ParquetColumn(name=name, type=pq_type, logical_type=logical))

        return cls(columns)

    @classmethod
    def infer_from_data(cls, data: List[Dict[str, Any]]) -> "ParquetSchema":
        """Infer schema from data."""
        if not data:
            return cls([])

        sample = data[0]
        columns = []

        for name, value in sample.items():
            if isinstance(value, bool):
                pq_type, logical = "BOOLEAN", None
            elif isinstance(value, int):
                pq_type, logical = "INT64", None
            elif isinstance(value, float):
                pq_type, logical = "DOUBLE", None
            elif isinstance(value, (list, dict)):
                pq_type, logical = "BYTE_ARRAY", "JSON"
            else:
                pq_type, logical = "BYTE_ARRAY", "STRING"

            columns.append(ParquetColumn(name=name, type=pq_type, logical_type=logical))

        return cls(columns)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "columns": [
                {
                    "name": c.name,
                    "type": c.type,
                    "logical_type": c.logical_type,
                    "nullable": c.nullable,
                }
                for c in self.columns
            ]
        }


class ParquetWriter:
    """
    Parquet Writer
    --------------
    Writes data to Parquet format.

    Usage:
        writer = ParquetWriter('output.parquet')
        writer.write(data)
        # or
        with writer:
            writer.write_batch(batch1)
            writer.write_batch(batch2)
    """

    def __init__(
        self,
        path: str,
        schema: Optional[ParquetSchema] = None,
        options: Optional[StorageOptions] = None,
    ):
        """
        Initialize Parquet writer.

        Args:
            path: Output file path
            schema: Parquet schema (inferred if not provided)
            options: Storage options
        """
        self.path = path
        self.schema = schema
        self.options = options or StorageOptions()
        self._row_groups: List[List[Dict]] = []
        self._current_group: List[Dict] = []
        self._file_metadata: Optional[FileMetadata] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, data: Union[List[Dict], HiveDataFrame]) -> FileMetadata:
        """
        Write data to Parquet file.

        Args:
            data: List of dictionaries or HiveDataFrame

        Returns:
            File metadata
        """
        if isinstance(data, HiveDataFrame):
            records = data.collect()
        else:
            records = data

        # Infer schema if needed
        if self.schema is None:
            self.schema = ParquetSchema.infer_from_data(records)

        # Split into row groups
        row_group_size = 10000  # rows per group for simplicity
        for i in range(0, len(records), row_group_size):
            self._row_groups.append(records[i : i + row_group_size])

        return self._write_file()

    def write_batch(self, batch: List[Dict]) -> None:
        """Write a batch of records."""
        self._current_group.extend(batch)

        # Flush if group is large enough
        if len(self._current_group) >= 10000:
            self._row_groups.append(self._current_group)
            self._current_group = []

    def close(self) -> FileMetadata:
        """Close writer and finalize file."""
        if self._current_group:
            self._row_groups.append(self._current_group)
            self._current_group = []

        if self._row_groups:
            return self._write_file()
        return FileMetadata(path=self.path, format=FileFormat.PARQUET)

    def _write_file(self) -> FileMetadata:
        """Write Parquet file structure."""
        # Try to use pyarrow if available
        try:
            return self._write_with_pyarrow()
        except ImportError:
            return self._write_native()

    def _write_with_pyarrow(self) -> FileMetadata:
        """Write using pyarrow library."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Flatten row groups
        all_records = []
        for group in self._row_groups:
            all_records.extend(group)

        if not all_records:
            # Write empty file
            table = pa.table({})
            pq.write_table(table, self.path)
            return FileMetadata(path=self.path, format=FileFormat.PARQUET, row_count=0)

        # Convert to PyArrow table
        table = pa.Table.from_pylist(all_records)

        # Set compression
        compression_map = {
            CompressionCodec.NONE: None,
            CompressionCodec.SNAPPY: "snappy",
            CompressionCodec.GZIP: "gzip",
            CompressionCodec.LZ4: "lz4",
            CompressionCodec.ZSTD: "zstd",
            CompressionCodec.BROTLI: "brotli",
        }
        compression = compression_map.get(self.options.compression, "snappy")

        # Write file
        pq.write_table(
            table, self.path, compression=compression, row_group_size=self.options.row_group_size
        )

        # Get file metadata
        file_size = os.path.getsize(self.path)

        return FileMetadata(
            path=self.path,
            format=FileFormat.PARQUET,
            size_bytes=file_size,
            row_count=len(all_records),
            column_count=len(table.schema),
            compression=self.options.compression,
        )

    def _write_native(self) -> FileMetadata:
        """Write using native Python (simplified format)."""
        # Simplified Parquet-like format for when pyarrow is not available
        # This writes a JSON-based format that preserves structure

        all_records = []
        for group in self._row_groups:
            all_records.extend(group)

        output = {
            "format": "parquet_compat",
            "version": "1.0",
            "schema": self.schema.to_dict() if self.schema else {},
            "row_count": len(all_records),
            "row_groups": [{"data": group} for group in self._row_groups],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "compression": self.options.compression.value,
            },
        }

        # Write as JSON (for compatibility when pyarrow unavailable)
        path = Path(self.path)
        with open(path, "w") as f:
            json.dump(output, f)

        file_size = path.stat().st_size

        return FileMetadata(
            path=self.path,
            format=FileFormat.PARQUET,
            size_bytes=file_size,
            row_count=len(all_records),
            column_count=len(self.schema.columns) if self.schema else 0,
            compression=self.options.compression,
        )


class ParquetReader:
    """
    Parquet Reader
    --------------
    Reads data from Parquet format.

    Usage:
        reader = ParquetReader('data.parquet')
        df = reader.read()

        # Or with column selection
        df = reader.read(columns=['id', 'name'])
    """

    def __init__(self, path: str, options: Optional[StorageOptions] = None):
        """
        Initialize Parquet reader.

        Args:
            path: Input file path
            options: Storage options
        """
        self.path = path
        self.options = options or StorageOptions()
        self._metadata: Optional[FileMetadata] = None
        self._schema: Optional[ParquetSchema] = None

    @property
    def schema(self) -> Optional[ParquetSchema]:
        """Get file schema."""
        if self._schema is None:
            self._read_metadata()
        return self._schema

    @property
    def metadata(self) -> Optional[FileMetadata]:
        """Get file metadata."""
        if self._metadata is None:
            self._read_metadata()
        return self._metadata

    def _read_metadata(self) -> None:
        """Read file metadata without loading data."""
        try:
            self._read_metadata_pyarrow()
        except ImportError:
            self._read_metadata_native()

    def _read_metadata_pyarrow(self) -> None:
        """Read metadata using pyarrow."""
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(self.path)
        meta = parquet_file.metadata
        schema = parquet_file.schema_arrow

        # Convert schema
        columns = []
        for field in schema:
            columns.append(
                ParquetColumn(name=field.name, type=str(field.type), nullable=field.nullable)
            )
        self._schema = ParquetSchema(columns)

        self._metadata = FileMetadata(
            path=self.path,
            format=FileFormat.PARQUET,
            size_bytes=os.path.getsize(self.path),
            row_count=meta.num_rows,
            column_count=meta.num_columns,
        )

    def _read_metadata_native(self) -> None:
        """Read metadata from native format."""
        with open(self.path, "r") as f:
            data = json.load(f)

        schema_dict = data.get("schema", {})
        columns = [
            ParquetColumn(
                name=c["name"],
                type=c["type"],
                logical_type=c.get("logical_type"),
                nullable=c.get("nullable", True),
            )
            for c in schema_dict.get("columns", [])
        ]
        self._schema = ParquetSchema(columns)

        self._metadata = FileMetadata(
            path=self.path,
            format=FileFormat.PARQUET,
            size_bytes=os.path.getsize(self.path),
            row_count=data.get("row_count", 0),
            column_count=len(columns),
        )

    def read(
        self, columns: Optional[List[str]] = None, filter: Optional[Any] = None
    ) -> HiveDataFrame:
        """
        Read Parquet file to HiveDataFrame.

        Args:
            columns: Optional list of columns to read
            filter: Optional filter predicate

        Returns:
            HiveDataFrame with data
        """
        try:
            return self._read_with_pyarrow(columns, filter)
        except ImportError:
            return self._read_native(columns, filter)

    def _read_with_pyarrow(
        self, columns: Optional[List[str]], filter: Optional[Any]
    ) -> HiveDataFrame:
        """Read using pyarrow."""
        import pyarrow.parquet as pq

        table = pq.read_table(self.path, columns=columns, use_pandas_metadata=True)

        # Convert to list of dicts
        records = table.to_pylist()

        return HiveDataFrame(records)

    def _read_native(self, columns: Optional[List[str]], filter: Optional[Any]) -> HiveDataFrame:
        """Read using native format."""
        with open(self.path, "r") as f:
            data = json.load(f)

        records = []
        for group in data.get("row_groups", []):
            group_data = group.get("data", [])
            records.extend(group_data)

        # Apply column selection
        if columns:
            records = [{k: v for k, v in row.items() if k in columns} for row in records]

        return HiveDataFrame(records)

    def read_batches(self, batch_size: int = 10000) -> Iterator[List[Dict]]:
        """
        Read file in batches.

        Args:
            batch_size: Number of rows per batch

        Yields:
            Batches of records
        """
        try:
            yield from self._read_batches_pyarrow(batch_size)
        except ImportError:
            yield from self._read_batches_native(batch_size)

    def _read_batches_pyarrow(self, batch_size: int) -> Iterator[List[Dict]]:
        """Read batches using pyarrow."""
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(self.path)

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield batch.to_pylist()

    def _read_batches_native(self, batch_size: int) -> Iterator[List[Dict]]:
        """Read batches using native format."""
        with open(self.path, "r") as f:
            data = json.load(f)

        for group in data.get("row_groups", []):
            group_data = group.get("data", [])
            for i in range(0, len(group_data), batch_size):
                yield group_data[i : i + batch_size]


def read_parquet(
    path: str, columns: Optional[List[str]] = None, options: Optional[StorageOptions] = None
) -> HiveDataFrame:
    """
    Read Parquet file to HiveDataFrame.

    Args:
        path: File path
        columns: Optional column selection
        options: Storage options

    Returns:
        HiveDataFrame
    """
    reader = ParquetReader(path, options)
    return reader.read(columns)


def write_parquet(
    data: Union[List[Dict], HiveDataFrame],
    path: str,
    schema: Optional[ParquetSchema] = None,
    options: Optional[StorageOptions] = None,
) -> FileMetadata:
    """
    Write data to Parquet file.

    Args:
        data: Data to write
        path: Output path
        schema: Optional schema
        options: Storage options

    Returns:
        File metadata
    """
    writer = ParquetWriter(path, schema, options)
    return writer.write(data)
