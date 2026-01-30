"""
HiveFrame Storage
=================

Storage layer supporting various file formats including
Parquet, Delta Lake, HoneyStore, and Iceberg for the HiveFrame data platform.

Key Features (Phase 1):
- Parquet file support (columnar storage)
- Delta Lake format support (ACID transactions)
- Schema evolution and type inference
- Partition pruning for optimized reads

Phase 2 Additions:
- HoneyStore: Native columnar format optimized for swarm access
- Apache Iceberg: Open table format compatibility
- Caching Swarm: Distributed intelligent caching with pheromone trails
"""

# Phase 2: Caching Swarm
from .cache import (
    CacheEntry,
    CacheLevel,
    CacheStatistics,
    CachingSwarm,
    DistributedCacheNode,
    EvictionPolicy,
    PheromoneCache,
    PheromoneTrail,
    SwarmPrefetcher,
)
from .delta import (
    DeltaLog,
    DeltaTable,
    DeltaTransaction,
    read_delta,
    write_delta,
)
from .formats import (
    CompressionCodec,
    FileFormat,
    PartitionSpec,
    StorageOptions,
)

# Phase 2: HoneyStore native columnar format
from .honeystore import (
    ColumnMetadata,
    DictionaryEncoder,
    EncodingType,
    HoneycombBlock,
    HoneyStoreMetadata,
    HoneyStoreReader,
    HoneyStoreWriter,
    NectarEncoder,
    RLEEncoder,
    read_honeystore,
    write_honeystore,
)

# Phase 2: Apache Iceberg support
from .iceberg import (
    DataFile,
    IcebergDataType,
    IcebergField,
    IcebergSchema,
    IcebergTable,
    ManifestFile,
    PartitionField,
    PartitionTransform,
    SchemaEvolution,
    Snapshot,
    TableMetadata,
    read_iceberg,
    write_iceberg,
)
from .parquet import (
    ParquetReader,
    ParquetSchema,
    ParquetWriter,
    read_parquet,
    write_parquet,
)

__all__ = [
    # Parquet
    "ParquetReader",
    "ParquetWriter",
    "ParquetSchema",
    "read_parquet",
    "write_parquet",
    # Delta
    "DeltaTable",
    "DeltaTransaction",
    "DeltaLog",
    "read_delta",
    "write_delta",
    # Common
    "FileFormat",
    "StorageOptions",
    "PartitionSpec",
    "CompressionCodec",
    # Phase 2: HoneyStore
    "HoneyStoreWriter",
    "HoneyStoreReader",
    "HoneyStoreMetadata",
    "HoneycombBlock",
    "ColumnMetadata",
    "EncodingType",
    "NectarEncoder",
    "DictionaryEncoder",
    "RLEEncoder",
    "write_honeystore",
    "read_honeystore",
    # Phase 2: Iceberg
    "IcebergTable",
    "IcebergSchema",
    "IcebergField",
    "IcebergDataType",
    "PartitionField",
    "PartitionTransform",
    "Snapshot",
    "TableMetadata",
    "ManifestFile",
    "DataFile",
    "SchemaEvolution",
    "read_iceberg",
    "write_iceberg",
    # Phase 2: Caching Swarm
    "CachingSwarm",
    "PheromoneCache",
    "CacheEntry",
    "PheromoneTrail",
    "EvictionPolicy",
    "CacheLevel",
    "CacheStatistics",
    "SwarmPrefetcher",
    "DistributedCacheNode",
]
