"""
Tests for Phase 2 Storage Layer Components.

Tests cover:
- HoneyStore native columnar format
- Apache Iceberg support
- Caching Swarm
"""

import pytest
import tempfile
import os
import time
from typing import List, Dict, Any

from hiveframe.storage import (
    # HoneyStore
    HoneyStoreWriter,
    HoneyStoreReader,
    HoneyStoreMetadata,
    write_honeystore,
    read_honeystore,
    ColumnMetadata,
    EncodingType,
    NectarEncoder,
    DictionaryEncoder,
    RLEEncoder,
    # Iceberg
    IcebergTable,
    IcebergSchema,
    IcebergField,
    PartitionField,
    read_iceberg,
    write_iceberg,
    # Caching
    CachingSwarm,
    PheromoneCache,
    CacheEntry,
    PheromoneTrail,
    EvictionPolicy,
    SwarmPrefetcher,
)


class TestHoneyStoreWriter:
    """Test HoneyStore write operations."""

    @pytest.fixture
    def sample_data(self) -> List[Dict[str, Any]]:
        """Sample data for testing."""
        return [
            {"id": 1, "name": "Alice", "score": 95.5, "active": True},
            {"id": 2, "name": "Bob", "score": 87.0, "active": False},
            {"id": 3, "name": "Carol", "score": 91.5, "active": True},
        ]

    def test_write_honeystore(self, sample_data):
        """Test writing data to HoneyStore format."""
        with tempfile.NamedTemporaryFile(suffix=".honey", delete=False) as f:
            path = f.name

        try:
            metadata = write_honeystore(sample_data, path)

            assert os.path.exists(path)
            assert metadata.num_rows == 3
        finally:
            os.unlink(path)

    def test_write_with_context_manager(self, sample_data):
        """Test writing with context manager."""
        with tempfile.NamedTemporaryFile(suffix=".honey", delete=False) as f:
            path = f.name

        try:
            with HoneyStoreWriter(path) as writer:
                writer.write_batch(sample_data)

            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_write_incremental(self, sample_data):
        """Test incremental writing."""
        with tempfile.NamedTemporaryFile(suffix=".honey", delete=False) as f:
            path = f.name

        try:
            with HoneyStoreWriter(path) as writer:
                for row in sample_data:
                    writer.write_row(row)

            reader = HoneyStoreReader(path)
            assert reader.num_rows == 3
        finally:
            os.unlink(path)


class TestHoneyStoreReader:
    """Test HoneyStore read operations."""

    @pytest.fixture
    def honeystore_file(self) -> str:
        """Create a test HoneyStore file."""
        data = [
            {"id": 1, "name": "Alice", "score": 95.5},
            {"id": 2, "name": "Bob", "score": 87.0},
            {"id": 3, "name": "Carol", "score": 91.5},
        ]

        with tempfile.NamedTemporaryFile(suffix=".honey", delete=False) as f:
            path = f.name

        write_honeystore(data, path)
        yield path
        os.unlink(path)

    def test_read_all(self, honeystore_file):
        """Test reading all data."""
        rows = read_honeystore(honeystore_file)

        assert len(rows) == 3

    def test_read_metadata(self, honeystore_file):
        """Test reading metadata."""
        reader = HoneyStoreReader(honeystore_file)

        assert reader.num_rows == 3
        assert len(reader.schema) >= 1

    def test_read_specific_columns(self, honeystore_file):
        """Test reading specific columns (projection)."""
        rows = read_honeystore(honeystore_file, columns=["id", "name"])

        assert len(rows) == 3


class TestNectarEncoder:
    """Test Nectar encoding for null handling."""

    def test_encode_int64(self):
        """Test encoding int64 values with nulls."""
        values = [1, None, 3, 4, None]

        null_bitmap, encoded = NectarEncoder.encode(values, "int64")

        assert len(null_bitmap) == 1  # 5 values fit in 1 byte

    def test_decode_int64(self):
        """Test decoding int64 values."""
        values = [1, 2, 3]

        null_bitmap, encoded = NectarEncoder.encode(values, "int64")
        decoded = NectarEncoder.decode(null_bitmap, encoded, "int64", len(values))

        assert decoded == values

    def test_encode_string(self):
        """Test encoding string values."""
        values = ["hello", "world", None]

        null_bitmap, encoded = NectarEncoder.encode(values, "string")
        decoded = NectarEncoder.decode(null_bitmap, encoded, "string", len(values))

        assert decoded[0] == "hello"
        assert decoded[1] == "world"
        assert decoded[2] is None


class TestDictionaryEncoder:
    """Test dictionary encoding."""

    def test_should_use_low_cardinality(self):
        """Test dictionary encoding recommended for low cardinality."""
        values = ["A", "B", "A", "A", "B", "A"]

        assert DictionaryEncoder.should_use(values)

    def test_should_not_use_high_cardinality(self):
        """Test dictionary not recommended for high cardinality."""
        values = [f"value_{i}" for i in range(100)]

        assert not DictionaryEncoder.should_use(values)

    def test_encode_decode(self):
        """Test encoding and decoding."""
        values = ["cat", "dog", "cat", "bird", "dog", "cat"]

        dict_data, indices = DictionaryEncoder.encode(values)
        decoded = DictionaryEncoder.decode(dict_data, indices)

        assert decoded == values


class TestRLEEncoder:
    """Test run-length encoding."""

    def test_should_use_repeated(self):
        """Test RLE recommended for repeated values."""
        values = [1, 1, 1, 1, 2, 2, 2, 3, 3]

        assert RLEEncoder.should_use(values)

    def test_should_not_use_unique(self):
        """Test RLE not recommended for unique values."""
        values = list(range(100))

        assert not RLEEncoder.should_use(values)

    def test_encode_decode_int(self):
        """Test encoding/decoding integers."""
        values = [1, 1, 1, 2, 2, 3]

        encoded = RLEEncoder.encode(values, "int64")
        decoded = RLEEncoder.decode(encoded, "int64")

        assert decoded == values


class TestIcebergTable:
    """Test Iceberg table operations."""

    @pytest.fixture
    def iceberg_schema(self) -> List[IcebergField]:
        """Sample Iceberg schema."""
        return [
            IcebergField(1, "id", "long", required=True),
            IcebergField(2, "name", "string"),
            IcebergField(3, "value", "double"),
        ]

    def test_create_table(self, iceberg_schema):
        """Test creating an Iceberg table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            table = IcebergTable.create(path=tmpdir, schema=iceberg_schema)

            assert table.exists()
            assert len(table.schema.fields) == 3

    def test_schema_access(self, iceberg_schema):
        """Test accessing schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            table = IcebergTable.create(path=tmpdir, schema=iceberg_schema)

            assert table.schema.column_names == ["id", "name", "value"]

    def test_load_table(self, iceberg_schema):
        """Test loading existing table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            IcebergTable.create(path=tmpdir, schema=iceberg_schema)

            loaded = IcebergTable.load(tmpdir)

            assert loaded.exists()
            assert len(loaded.schema.fields) == 3

    def test_schema_evolution(self, iceberg_schema):
        """Test schema evolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            table = IcebergTable.create(path=tmpdir, schema=iceberg_schema)

            new_schema = table.evolve_schema().add_column("timestamp", "timestamp").apply()

            assert len(new_schema.fields) == 4


class TestIcebergField:
    """Test Iceberg field handling."""

    def test_field_serialization(self):
        """Test field to/from dict."""
        field = IcebergField(field_id=1, name="test", data_type="string", required=True)

        field_dict = field.to_dict()
        restored = IcebergField.from_dict(field_dict)

        assert restored.field_id == 1
        assert restored.name == "test"
        assert restored.required == True


class TestPheromoneCache:
    """Test PheromoneCache for intelligent caching."""

    def test_put_get(self):
        """Test basic put and get."""
        cache = PheromoneCache(max_size_bytes=1024)

        cache.put("key1", "value1")

        result = cache.get("key1")

        assert result == "value1"

    def test_cache_miss(self):
        """Test cache miss."""
        cache = PheromoneCache(max_size_bytes=1024)

        result = cache.get("nonexistent")

        assert result is None

    def test_eviction(self):
        """Test eviction when cache is full."""
        cache = PheromoneCache(max_size_bytes=100)

        # Fill cache
        cache.put("key1", "x" * 50)
        cache.put("key2", "y" * 50)

        # This should trigger eviction
        cache.put("key3", "z" * 50)

        # At least one key should be evicted
        assert cache.size <= 2

    def test_statistics(self):
        """Test statistics tracking."""
        cache = PheromoneCache(max_size_bytes=1024)

        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_statistics()

        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_invalidate(self):
        """Test cache invalidation."""
        cache = PheromoneCache(max_size_bytes=1024)

        cache.put("key1", "value1")
        cache.invalidate("key1")

        assert cache.get("key1") is None

    def test_ttl_expiry(self):
        """Test TTL-based expiry."""
        cache = PheromoneCache(max_size_bytes=1024)

        cache.put("key1", "value1", ttl=0.1)  # 100ms TTL

        time.sleep(0.2)  # Wait for expiry

        result = cache.get("key1")

        assert result is None


class TestCacheEntry:
    """Test CacheEntry data structure."""

    def test_fitness_calculation(self):
        """Test fitness score calculation."""
        entry = CacheEntry(key="test", value="value", size_bytes=100)

        # Access multiple times
        entry.access()
        entry.access()

        assert entry.fitness > 0
        assert entry.access_count == 3  # Initial + 2 accesses

    def test_is_expired(self):
        """Test TTL expiry check."""
        expired_entry = CacheEntry(
            key="expired", value="value", size_bytes=100, ttl=0.01  # 10ms TTL
        )

        # Fast forward time
        expired_entry.created_at = time.time() - 1

        assert expired_entry.is_expired

    def test_pheromone_decay(self):
        """Test pheromone decay."""
        entry = CacheEntry(key="test", value="value", size_bytes=100, pheromone_level=1.0)

        entry.decay_pheromone(rate=0.5)

        assert entry.pheromone_level < 1.0


class TestCachingSwarm:
    """Test distributed CachingSwarm."""

    def test_swarm_creation(self):
        """Test creating a caching swarm."""
        swarm = CachingSwarm(num_nodes=4)

        assert len(swarm.nodes) == 4

    def test_swarm_put_get(self):
        """Test distributed put/get."""
        swarm = CachingSwarm(num_nodes=4)
        swarm.start()

        try:
            swarm.put("key1", "value1")
            result = swarm.get("key1")

            assert result == "value1"
        finally:
            swarm.stop()

    def test_swarm_statistics(self):
        """Test aggregated statistics."""
        swarm = CachingSwarm(num_nodes=4)
        swarm.start()

        try:
            swarm.put("key1", "value1")
            swarm.get("key1")

            stats = swarm.get_statistics()

            assert stats["num_nodes"] == 4
            assert stats["total_entries"] >= 1
        finally:
            swarm.stop()


class TestPheromoneTrail:
    """Test PheromoneTrail for cache coordination."""

    def test_intensity_decay(self):
        """Test pheromone intensity decay over time."""
        trail = PheromoneTrail(
            key="test",
            intensity=1.0,
            source_node="node1",
            decay_rate=10.0,  # Fast decay for testing
        )

        # Intensity should decay
        time.sleep(0.1)

        assert trail.current_intensity() < 1.0

    def test_reinforce(self):
        """Test trail reinforcement."""
        trail = PheromoneTrail(key="test", intensity=1.0, source_node="node1")

        trail.reinforce(2.0)

        assert trail.intensity == 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
