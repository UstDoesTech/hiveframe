"""
HoneyStore: Native Columnar Storage Format (Phase 2)
====================================================

A bee-inspired columnar storage format optimized for swarm access patterns.
Designed for efficient parallel processing with adaptive compression.

Key Features:
- Columnar layout for efficient analytics
- Adaptive compression based on data patterns
- Swarm-optimized file structure for parallel reads
- Honeycomb blocks for balanced I/O
- Nectar encoding for efficient null handling
"""

import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, BinaryIO, Dict, List, Optional, Tuple


class EncodingType(Enum):
    """Supported column encoding types."""

    PLAIN = auto()  # No encoding
    DICTIONARY = auto()  # Dictionary encoding for low cardinality
    RLE = auto()  # Run-length encoding for repeated values
    DELTA = auto()  # Delta encoding for sorted/sequential data
    BITPACK = auto()  # Bit packing for small integers
    NECTAR = auto()  # HoneyStore's null-optimized encoding


class CompressionType(Enum):
    """Supported compression types."""

    NONE = auto()
    SNAPPY = auto()
    ZSTD = auto()
    LZ4 = auto()
    ADAPTIVE = auto()  # Automatically select best compression


@dataclass
class ColumnMetadata:
    """Metadata for a single column in HoneyStore."""

    name: str
    data_type: str  # 'int64', 'float64', 'string', 'bool', 'bytes'
    encoding: EncodingType = EncodingType.PLAIN
    compression: CompressionType = CompressionType.ADAPTIVE
    null_count: int = 0
    min_value: Any = None
    max_value: Any = None
    distinct_count: int = 0
    total_size_bytes: int = 0
    compressed_size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "encoding": self.encoding.name,
            "compression": self.compression.name,
            "null_count": self.null_count,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "distinct_count": self.distinct_count,
            "total_size_bytes": self.total_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnMetadata":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            data_type=data["data_type"],
            encoding=EncodingType[data.get("encoding", "PLAIN")],
            compression=CompressionType[data.get("compression", "ADAPTIVE")],
            null_count=data.get("null_count", 0),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            distinct_count=data.get("distinct_count", 0),
            total_size_bytes=data.get("total_size_bytes", 0),
            compressed_size_bytes=data.get("compressed_size_bytes", 0),
        )


@dataclass
class HoneycombBlock:
    """
    A honeycomb block is the basic unit of storage in HoneyStore.

    Like cells in a honeycomb, blocks are uniform in structure but
    can contain varying amounts of data (nectar).
    """

    block_id: str
    row_offset: int
    num_rows: int
    columns: Dict[str, bytes] = field(default_factory=dict)  # column_name -> encoded data
    metadata: Dict[str, ColumnMetadata] = field(default_factory=dict)
    checksum: str = ""
    created_at: float = field(default_factory=time.time)

    def compute_checksum(self) -> str:
        """Compute checksum for block integrity."""
        h = hashlib.sha256()
        for name in sorted(self.columns.keys()):
            h.update(name.encode())
            h.update(self.columns[name])
        self.checksum = h.hexdigest()[:16]
        return self.checksum

    @property
    def total_bytes(self) -> int:
        """Total bytes in block."""
        return sum(len(data) for data in self.columns.values())


@dataclass
class HoneyStoreMetadata:
    """File-level metadata for HoneyStore files."""

    version: str = "1.0"
    schema: List[ColumnMetadata] = field(default_factory=list)
    num_rows: int = 0
    num_blocks: int = 0
    block_size: int = 65536  # Default 64KB blocks
    created_at: float = field(default_factory=time.time)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "schema": [col.to_dict() for col in self.schema],
            "num_rows": self.num_rows,
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "created_at": self.created_at,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HoneyStoreMetadata":
        """Deserialize from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            schema=[ColumnMetadata.from_dict(c) for c in data.get("schema", [])],
            num_rows=data.get("num_rows", 0),
            num_blocks=data.get("num_blocks", 0),
            block_size=data.get("block_size", 65536),
            created_at=data.get("created_at", time.time()),
            properties=data.get("properties", {}),
        )


class NectarEncoder:
    """
    Nectar encoding for efficient null handling.

    Inspired by how bees store nectar in cells - empty cells are
    efficiently tracked without wasting space.
    """

    @staticmethod
    def encode(values: List[Any], data_type: str) -> Tuple[bytes, bytes]:
        """
        Encode values with null bitmap.

        Returns (null_bitmap, encoded_values).
        """
        # Create null bitmap
        null_bitmap = bytearray((len(values) + 7) // 8)
        non_null_values = []

        for i, val in enumerate(values):
            if val is not None:
                null_bitmap[i // 8] |= 1 << (i % 8)
                non_null_values.append(val)

        # Encode non-null values based on type
        if data_type == "int64":
            encoded = b"".join(struct.pack("<q", v) for v in non_null_values)
        elif data_type == "float64":
            encoded = b"".join(struct.pack("<d", v) for v in non_null_values)
        elif data_type == "bool":
            # Pack booleans into bits
            bool_bytes = bytearray((len(non_null_values) + 7) // 8)
            for i, v in enumerate(non_null_values):
                if v:
                    bool_bytes[i // 8] |= 1 << (i % 8)
            encoded = bytes(bool_bytes)
        elif data_type == "string":
            # Length-prefixed strings
            parts = []
            for v in non_null_values:
                v_bytes = str(v).encode("utf-8")
                parts.append(struct.pack("<I", len(v_bytes)))
                parts.append(v_bytes)
            encoded = b"".join(parts)
        else:
            # Default: JSON encoding
            encoded = json.dumps(non_null_values).encode("utf-8")

        return bytes(null_bitmap), encoded

    @staticmethod
    def decode(
        null_bitmap: bytes, encoded_values: bytes, data_type: str, num_values: int
    ) -> List[Any]:
        """Decode values with null bitmap."""
        # Decode non-null values
        if data_type == "int64":
            non_null = [
                struct.unpack("<q", encoded_values[i * 8 : (i + 1) * 8])[0]
                for i in range(len(encoded_values) // 8)
            ]
        elif data_type == "float64":
            non_null = [
                struct.unpack("<d", encoded_values[i * 8 : (i + 1) * 8])[0]
                for i in range(len(encoded_values) // 8)
            ]
        elif data_type == "bool":
            non_null = [
                bool(encoded_values[i // 8] & (1 << (i % 8)))
                for i in range(len(encoded_values) * 8)
            ]
        elif data_type == "string":
            non_null = []
            offset = 0
            while offset < len(encoded_values):
                length = struct.unpack("<I", encoded_values[offset : offset + 4])[0]
                offset += 4
                non_null.append(encoded_values[offset : offset + length].decode("utf-8"))
                offset += length
        else:
            non_null = json.loads(encoded_values.decode("utf-8"))

        # Reconstruct with nulls
        values = []
        non_null_idx = 0
        for i in range(num_values):
            if null_bitmap[i // 8] & (1 << (i % 8)):
                if non_null_idx < len(non_null):
                    values.append(non_null[non_null_idx])
                    non_null_idx += 1
                else:
                    values.append(None)
            else:
                values.append(None)

        return values


class DictionaryEncoder:
    """
    Dictionary encoding for low-cardinality columns.

    Replaces repeated values with compact indices.
    """

    @staticmethod
    def should_use(values: List[Any], threshold: float = 0.5) -> bool:
        """Determine if dictionary encoding is beneficial."""
        if not values:
            return False
        unique = len(set(v for v in values if v is not None))
        return unique / len(values) < threshold

    @staticmethod
    def encode(values: List[Any]) -> Tuple[bytes, bytes]:
        """
        Encode values using dictionary.

        Returns (dictionary, indices).
        """
        # Build dictionary
        unique_values = list(dict.fromkeys(v for v in values if v is not None))
        value_to_idx = {v: i for i, v in enumerate(unique_values)}
        value_to_idx[None] = len(unique_values)  # Special index for null

        # Encode dictionary
        dict_json = json.dumps(unique_values).encode("utf-8")

        # Encode indices (use smallest integer type that fits)
        max_idx = len(unique_values)
        if max_idx < 256:
            indices = bytes(value_to_idx.get(v, max_idx) for v in values)
        else:
            indices = b"".join(struct.pack("<H", value_to_idx.get(v, max_idx)) for v in values)

        return dict_json, indices

    @staticmethod
    def decode(dictionary: bytes, indices: bytes, use_short: bool = False) -> List[Any]:
        """Decode dictionary-encoded values."""
        unique_values = json.loads(dictionary.decode("utf-8"))
        len(unique_values)

        if use_short:
            idx_list = [
                struct.unpack("<H", indices[i * 2 : (i + 1) * 2])[0]
                for i in range(len(indices) // 2)
            ]
        else:
            idx_list = list(indices)

        return [unique_values[idx] if idx < len(unique_values) else None for idx in idx_list]


class RLEEncoder:
    """
    Run-length encoding for columns with repeated values.
    """

    @staticmethod
    def should_use(values: List[Any], threshold: float = 0.5) -> bool:
        """Determine if RLE is beneficial."""
        if len(values) < 2:
            return False

        runs = 1
        for i in range(1, len(values)):
            if values[i] != values[i - 1]:
                runs += 1

        return runs / len(values) < threshold

    @staticmethod
    def encode(values: List[Any], data_type: str) -> bytes:
        """
        Encode values using run-length encoding.

        Format: [(value, count), ...]
        """
        if not values:
            return b""

        runs = []
        current_val = values[0]
        current_count = 1

        for val in values[1:]:
            if val == current_val:
                current_count += 1
            else:
                runs.append((current_val, current_count))
                current_val = val
                current_count = 1
        runs.append((current_val, current_count))

        # Encode runs
        parts = []
        for val, count in runs:
            parts.append(struct.pack("<I", count))
            if data_type == "int64":
                parts.append(struct.pack("<q", val if val is not None else 0))
            elif data_type == "float64":
                parts.append(struct.pack("<d", val if val is not None else 0.0))
            else:
                val_bytes = json.dumps(val).encode("utf-8")
                parts.append(struct.pack("<I", len(val_bytes)))
                parts.append(val_bytes)

        return b"".join(parts)

    @staticmethod
    def decode(encoded: bytes, data_type: str) -> List[Any]:
        """Decode RLE-encoded values."""
        values = []
        offset = 0

        while offset < len(encoded):
            count = struct.unpack("<I", encoded[offset : offset + 4])[0]
            offset += 4

            if data_type == "int64":
                val = struct.unpack("<q", encoded[offset : offset + 8])[0]
                offset += 8
            elif data_type == "float64":
                val = struct.unpack("<d", encoded[offset : offset + 8])[0]
                offset += 8
            else:
                length = struct.unpack("<I", encoded[offset : offset + 4])[0]
                offset += 4
                val = json.loads(encoded[offset : offset + length].decode("utf-8"))
                offset += length

            values.extend([val] * count)

        return values


class HoneyStoreWriter:
    """
    Writer for HoneyStore format.

    Creates honeycomb-structured columnar files optimized for
    parallel swarm access.
    """

    MAGIC_BYTES = b"HONEY01"  # File magic number

    def __init__(
        self,
        path: str,
        schema: Optional[List[ColumnMetadata]] = None,
        block_size: int = 65536,
        compression: CompressionType = CompressionType.ADAPTIVE,
    ):
        self.path = path
        self.schema = schema or []
        self.block_size = block_size
        self.compression = compression

        self._buffer: List[Dict[str, Any]] = []
        self._blocks: List[HoneycombBlock] = []
        self._row_count = 0
        self._file: Optional[BinaryIO] = None

    def __enter__(self) -> "HoneyStoreWriter":
        """Open file for writing."""
        self._file = open(self.path, "wb")
        # Write magic bytes and placeholder for metadata offset
        self._file.write(self.MAGIC_BYTES)
        self._file.write(struct.pack("<Q", 0))  # Placeholder for metadata offset
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close file and write footer."""
        if self._file:
            # Flush any remaining buffer
            if self._buffer:
                self._flush_buffer()

            # Write metadata at end of file
            metadata_offset = self._file.tell()

            metadata = HoneyStoreMetadata(
                schema=self.schema,
                num_rows=self._row_count,
                num_blocks=len(self._blocks),
                block_size=self.block_size,
            )

            metadata_json = json.dumps(metadata.to_dict()).encode("utf-8")
            self._file.write(struct.pack("<I", len(metadata_json)))
            self._file.write(metadata_json)

            # Go back and write metadata offset
            self._file.seek(len(self.MAGIC_BYTES))
            self._file.write(struct.pack("<Q", metadata_offset))

            self._file.close()

    def write_row(self, row: Dict[str, Any]) -> None:
        """Write a single row."""
        self._buffer.append(row)

        if len(self._buffer) >= self.block_size // 100:  # Rough estimate
            self._flush_buffer()

    def write_batch(self, rows: List[Dict[str, Any]]) -> None:
        """Write a batch of rows."""
        for row in rows:
            self._buffer.append(row)

        if len(self._buffer) >= self.block_size // 100:
            self._flush_buffer()

    def _infer_schema(self, rows: List[Dict[str, Any]]) -> None:
        """Infer schema from data if not provided."""
        if self.schema:
            return

        # Collect all column names
        columns: Dict[str, str] = {}

        for row in rows:
            for key, value in row.items():
                if key not in columns and value is not None:
                    if isinstance(value, bool):
                        columns[key] = "bool"
                    elif isinstance(value, int):
                        columns[key] = "int64"
                    elif isinstance(value, float):
                        columns[key] = "float64"
                    elif isinstance(value, str):
                        columns[key] = "string"
                    else:
                        columns[key] = "string"  # Default

        self.schema = [
            ColumnMetadata(name=name, data_type=dtype) for name, dtype in columns.items()
        ]

    def _flush_buffer(self) -> None:
        """Flush buffer to a honeycomb block."""
        if not self._buffer:
            return

        # Infer schema if needed
        if not self.schema:
            self._infer_schema(self._buffer)

        # Create block
        block = HoneycombBlock(
            block_id=f"block_{len(self._blocks)}",
            row_offset=self._row_count,
            num_rows=len(self._buffer),
        )

        # Encode each column
        for col_meta in self.schema:
            values = [row.get(col_meta.name) for row in self._buffer]

            # Choose encoding
            if DictionaryEncoder.should_use(values):
                col_meta.encoding = EncodingType.DICTIONARY
                dict_data, indices = DictionaryEncoder.encode(values)
                # Combine dictionary and indices
                encoded = struct.pack("<I", len(dict_data)) + dict_data + indices
            elif RLEEncoder.should_use(values):
                col_meta.encoding = EncodingType.RLE
                encoded = RLEEncoder.encode(values, col_meta.data_type)
            else:
                col_meta.encoding = EncodingType.NECTAR
                null_bitmap, value_data = NectarEncoder.encode(values, col_meta.data_type)
                encoded = struct.pack("<I", len(null_bitmap)) + null_bitmap + value_data

            # Update column statistics
            non_null = [v for v in values if v is not None]
            col_meta.null_count += len(values) - len(non_null)
            if non_null:
                try:
                    if col_meta.min_value is None:
                        col_meta.min_value = min(non_null)
                    else:
                        col_meta.min_value = min(col_meta.min_value, min(non_null))
                    if col_meta.max_value is None:
                        col_meta.max_value = max(non_null)
                    else:
                        col_meta.max_value = max(col_meta.max_value, max(non_null))
                except TypeError:
                    pass  # Not comparable values

            col_meta.distinct_count = len(set(non_null))
            col_meta.total_size_bytes += len(encoded)

            block.columns[col_meta.name] = encoded
            block.metadata[col_meta.name] = col_meta

        block.compute_checksum()

        # Write block to file
        if self._file:
            # Write block header
            block_header = json.dumps(
                {
                    "block_id": block.block_id,
                    "row_offset": block.row_offset,
                    "num_rows": block.num_rows,
                    "checksum": block.checksum,
                    "columns": list(block.columns.keys()),
                }
            ).encode("utf-8")

            self._file.write(struct.pack("<I", len(block_header)))
            self._file.write(block_header)

            # Write column data
            for col_name, col_data in block.columns.items():
                self._file.write(struct.pack("<I", len(col_data)))
                self._file.write(col_data)

        self._blocks.append(block)
        self._row_count += len(self._buffer)
        self._buffer.clear()


class HoneyStoreReader:
    """
    Reader for HoneyStore format.

    Supports parallel block reading for swarm processing.
    """

    MAGIC_BYTES = b"HONEY01"

    def __init__(self, path: str):
        self.path = path
        self._metadata: Optional[HoneyStoreMetadata] = None
        self._block_offsets: List[int] = []

    @property
    def metadata(self) -> HoneyStoreMetadata:
        """Get file metadata."""
        if self._metadata is None:
            self._read_metadata()
        return self._metadata

    @property
    def schema(self) -> List[ColumnMetadata]:
        """Get schema."""
        return self.metadata.schema

    @property
    def num_rows(self) -> int:
        """Get total row count."""
        return self.metadata.num_rows

    def _read_metadata(self) -> None:
        """Read file metadata from footer."""
        with open(self.path, "rb") as f:
            # Verify magic bytes
            magic = f.read(len(self.MAGIC_BYTES))
            if magic != self.MAGIC_BYTES:
                raise ValueError("Invalid HoneyStore file: wrong magic bytes")

            # Read metadata offset
            metadata_offset = struct.unpack("<Q", f.read(8))[0]

            # Seek to metadata
            f.seek(metadata_offset)
            metadata_len = struct.unpack("<I", f.read(4))[0]
            metadata_json = f.read(metadata_len)

            self._metadata = HoneyStoreMetadata.from_dict(json.loads(metadata_json.decode("utf-8")))

    def read_all(self, columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Read all data from file.

        Args:
            columns: Optional list of columns to read (projection)

        Returns:
            List of row dictionaries
        """
        rows = []

        with open(self.path, "rb") as f:
            # Skip header
            f.seek(len(self.MAGIC_BYTES) + 8)

            # Read blocks until metadata

            while True:
                f.tell()

                # Try to read block header length
                header_len_bytes = f.read(4)
                if len(header_len_bytes) < 4:
                    break

                header_len = struct.unpack("<I", header_len_bytes)[0]

                # Check if this is metadata (detect by checking if we're near end)
                header_json = f.read(header_len)

                try:
                    header = json.loads(header_json.decode("utf-8"))
                except json.JSONDecodeError:
                    break

                if "block_id" not in header:
                    break

                num_rows = header["num_rows"]
                block_columns = header["columns"]

                # Read column data
                column_data: Dict[str, bytes] = {}
                for col_name in block_columns:
                    if columns is None or col_name in columns:
                        data_len = struct.unpack("<I", f.read(4))[0]
                        column_data[col_name] = f.read(data_len)
                    else:
                        # Skip column
                        data_len = struct.unpack("<I", f.read(4))[0]
                        f.seek(f.tell() + data_len)

                # Decode columns
                decoded_columns: Dict[str, List[Any]] = {}

                for col_name, encoded in column_data.items():
                    # Find column metadata
                    col_meta = next((c for c in self.schema if c.name == col_name), None)
                    if col_meta is None:
                        continue

                    # Decode based on encoding
                    if col_meta.encoding == EncodingType.DICTIONARY:
                        dict_len = struct.unpack("<I", encoded[:4])[0]
                        dict_data = encoded[4 : 4 + dict_len]
                        indices = encoded[4 + dict_len :]
                        decoded_columns[col_name] = DictionaryEncoder.decode(dict_data, indices)
                    elif col_meta.encoding == EncodingType.RLE:
                        decoded_columns[col_name] = RLEEncoder.decode(encoded, col_meta.data_type)
                    else:  # NECTAR or PLAIN
                        bitmap_len = struct.unpack("<I", encoded[:4])[0]
                        null_bitmap = encoded[4 : 4 + bitmap_len]
                        value_data = encoded[4 + bitmap_len :]
                        decoded_columns[col_name] = NectarEncoder.decode(
                            null_bitmap, value_data, col_meta.data_type, num_rows
                        )

                # Construct rows
                for i in range(num_rows):
                    row = {}
                    for col_name, values in decoded_columns.items():
                        if i < len(values):
                            row[col_name] = values[i]
                    rows.append(row)

        return rows

    def read_column(self, column_name: str) -> List[Any]:
        """Read a single column."""
        rows = self.read_all(columns=[column_name])
        return [row.get(column_name) for row in rows]


def write_honeystore(
    data: List[Dict[str, Any]], path: str, block_size: int = 65536
) -> HoneyStoreMetadata:
    """
    Convenience function to write data to HoneyStore format.

    Args:
        data: List of row dictionaries
        path: Output file path
        block_size: Size of honeycomb blocks

    Returns:
        File metadata
    """
    with HoneyStoreWriter(path, block_size=block_size) as writer:
        writer.write_batch(data)

    reader = HoneyStoreReader(path)
    return reader.metadata


def read_honeystore(path: str, columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to read data from HoneyStore format.

    Args:
        path: Input file path
        columns: Optional column projection

    Returns:
        List of row dictionaries
    """
    reader = HoneyStoreReader(path)
    return reader.read_all(columns=columns)
