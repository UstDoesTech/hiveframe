"""
Tests for Storage Module (Parquet & Delta Lake).

Tests cover:
- Parquet reading and writing
- Delta Lake operations
- Storage formats and options
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hiveframe import HiveDataFrame
from hiveframe.storage import (
    CompressionCodec,
    DeltaLog,
    DeltaTable,
    FileFormat,
    ParquetReader,
    ParquetSchema,
    ParquetWriter,
    StorageOptions,
    read_delta,
    read_parquet,
    write_delta,
    write_parquet,
)


class TestStorageOptions:
    """Test storage configuration options."""

    def test_default_options(self):
        """Test default storage options."""
        options = StorageOptions()

        assert options.compression == CompressionCodec.SNAPPY
        assert options.predicate_pushdown
        assert options.column_pruning

    def test_custom_options(self):
        """Test custom storage options."""
        options = StorageOptions(compression=CompressionCodec.GZIP, row_group_size=64 * 1024 * 1024)

        assert options.compression == CompressionCodec.GZIP
        assert options.row_group_size == 64 * 1024 * 1024


class TestParquetSchema:
    """Test Parquet schema handling."""

    def test_infer_from_data(self):
        """Test schema inference from data."""
        data = [
            {"id": 1, "name": "Alice", "active": True, "score": 95.5},
        ]

        schema = ParquetSchema.infer_from_data(data)

        assert len(schema.columns) == 4

    def test_empty_data_schema(self):
        """Test schema from empty data."""
        schema = ParquetSchema.infer_from_data([])

        assert len(schema.columns) == 0

    def test_schema_to_dict(self):
        """Test schema serialization."""
        data = [{"id": 1, "name": "test"}]
        schema = ParquetSchema.infer_from_data(data)

        schema_dict = schema.to_dict()

        assert "columns" in schema_dict
        assert len(schema_dict["columns"]) == 2


class TestParquetWriter:
    """Test Parquet writing."""

    @pytest.fixture
    def sample_data(self) -> List[Dict[str, Any]]:
        """Sample data for testing."""
        return [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Carol", "age": 35},
        ]

    def test_write_parquet_list(self, sample_data):
        """Test writing list of dicts to Parquet."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        try:
            metadata = write_parquet(sample_data, path)

            assert os.path.exists(path)
            assert metadata.row_count == 3
            assert metadata.format == FileFormat.PARQUET
        finally:
            os.unlink(path)

    def test_write_parquet_dataframe(self, sample_data):
        """Test writing HiveDataFrame to Parquet."""
        df = HiveDataFrame(sample_data)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        try:
            metadata = write_parquet(df, path)

            assert metadata.row_count == 3
        finally:
            os.unlink(path)

    def test_writer_context_manager(self, sample_data):
        """Test ParquetWriter as context manager."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        try:
            with ParquetWriter(path) as writer:
                writer.write_batch(sample_data[:2])
                writer.write_batch(sample_data[2:])

            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestParquetReader:
    """Test Parquet reading."""

    @pytest.fixture
    def parquet_file(self) -> str:
        """Create a test Parquet file."""
        data = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Carol", "age": 35},
        ]

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        write_parquet(data, path)
        yield path
        os.unlink(path)

    def test_read_parquet(self, parquet_file):
        """Test reading Parquet file."""
        df = read_parquet(parquet_file)

        assert df.count() == 3

    def test_read_parquet_columns(self, parquet_file):
        """Test reading specific columns."""
        df = read_parquet(parquet_file, columns=["id", "name"])
        collected = df.collect()

        # Should only have selected columns
        assert "id" in collected[0]
        assert "name" in collected[0]

    def test_reader_metadata(self, parquet_file):
        """Test reading file metadata."""
        reader = ParquetReader(parquet_file)

        assert reader.metadata is not None
        assert reader.metadata.row_count == 3


class TestDeltaLog:
    """Test Delta transaction log."""

    def test_initialize_log(self):
        """Test initializing Delta log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = DeltaLog(tmpdir)
            log.initialize()

            log_dir = Path(tmpdir) / "_delta_log"
            assert log_dir.exists()

    def test_get_latest_version_empty(self):
        """Test getting version from empty log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = DeltaLog(tmpdir)

            version = log.get_latest_version()

            assert version == -1

    def test_write_and_read_version(self):
        """Test writing and reading version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = DeltaLog(tmpdir)
            log.initialize()

            actions = [{"add": {"path": "test.parquet", "size": 100}}]
            log.write_version(0, actions)

            read_actions = log.read_version(0)

            assert len(read_actions) == 1
            assert "add" in read_actions[0]


class TestDeltaTable:
    """Test Delta table operations."""

    @pytest.fixture
    def sample_data(self) -> List[Dict[str, Any]]:
        """Sample data for testing."""
        return [
            {"id": 1, "name": "Alice", "value": 100},
            {"id": 2, "name": "Bob", "value": 200},
            {"id": 3, "name": "Carol", "value": 300},
        ]

    def test_create_table(self, sample_data):
        """Test creating a Delta table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            table = DeltaTable(tmpdir)
            table.write(sample_data)

            assert table.exists()

    def test_read_table(self, sample_data):
        """Test reading from Delta table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            table = DeltaTable(tmpdir)
            table.write(sample_data)

            df = table.to_dataframe()

            assert df.count() == 3

    def test_append_data(self, sample_data):
        """Test appending data to Delta table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            table = DeltaTable(tmpdir)
            table.write(sample_data)

            # Append more data
            new_data = [{"id": 4, "name": "David", "value": 400}]
            table.write(new_data, mode="append")

            df = table.to_dataframe()
            assert df.count() == 4

    def test_overwrite_data(self, sample_data):
        """Test overwriting Delta table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            table = DeltaTable(tmpdir)
            table.write(sample_data)

            # Overwrite
            new_data = [{"id": 10, "name": "New", "value": 1000}]
            table.write(new_data, mode="overwrite")

            df = table.to_dataframe()
            assert df.count() == 1

    def test_table_history(self, sample_data):
        """Test getting table history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            table = DeltaTable(tmpdir)
            table.write(sample_data)
            table.write([{"id": 4, "name": "D", "value": 4}], mode="append")

            history = table.history()

            assert len(history) >= 1


class TestDeltaConvenienceFunctions:
    """Test Delta convenience functions."""

    def test_write_delta(self):
        """Test write_delta function."""
        data = [{"id": 1, "value": 10}]

        with tempfile.TemporaryDirectory() as tmpdir:
            write_delta(data, tmpdir)

            # Verify table exists
            table = DeltaTable(tmpdir)
            assert table.exists()

    def test_read_delta(self):
        """Test read_delta function."""
        data = [{"id": 1, "value": 10}]

        with tempfile.TemporaryDirectory() as tmpdir:
            write_delta(data, tmpdir)
            df = read_delta(tmpdir)

            assert df.count() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
