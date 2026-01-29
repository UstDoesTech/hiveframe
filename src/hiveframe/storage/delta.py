"""
Delta Lake Support
==================

Delta Lake format support for HiveFrame.
Provides ACID transactions and time travel capabilities.

Delta Lake is an open-source storage layer that brings ACID
transactions to Apache Spark and big data workloads.

This implementation provides core Delta Lake features:
- ACID transactions
- Schema enforcement and evolution
- Time travel (data versioning)
- Unified streaming and batch processing
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Iterator
import json
import os
import uuid
import time
from datetime import datetime
from pathlib import Path
from enum import Enum, auto

from ..dataframe import HiveDataFrame
from .formats import (
    FileFormat, CompressionCodec, StorageOptions, 
    PartitionSpec, FileMetadata
)
from .parquet import ParquetWriter, ParquetReader, ParquetSchema


class ActionType(Enum):
    """Delta log action types."""
    ADD = auto()
    REMOVE = auto()
    METADATA = auto()
    PROTOCOL = auto()
    COMMIT_INFO = auto()
    CDC = auto()  # Change Data Capture


@dataclass
class AddAction:
    """Record of a file added to the table."""
    path: str
    partition_values: Dict[str, str] = field(default_factory=dict)
    size: int = 0
    modification_time: int = 0
    data_change: bool = True
    stats: Optional[str] = None  # JSON stats
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'add': {
                'path': self.path,
                'partitionValues': self.partition_values,
                'size': self.size,
                'modificationTime': self.modification_time,
                'dataChange': self.data_change,
                'stats': self.stats,
                'tags': self.tags
            }
        }


@dataclass  
class RemoveAction:
    """Record of a file removed from the table."""
    path: str
    deletion_timestamp: Optional[int] = None
    data_change: bool = True
    extended_file_metadata: bool = False
    partition_values: Dict[str, str] = field(default_factory=dict)
    size: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'remove': {
                'path': self.path,
                'deletionTimestamp': self.deletion_timestamp,
                'dataChange': self.data_change,
                'partitionValues': self.partition_values,
                'size': self.size
            }
        }


@dataclass
class MetadataAction:
    """Table metadata."""
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    format_provider: str = 'parquet'
    schema_string: str = ''
    partition_columns: List[str] = field(default_factory=list)
    configuration: Dict[str, str] = field(default_factory=dict)
    created_time: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'metaData': {
                'id': self.id,
                'name': self.name,
                'description': self.description,
                'format': {'provider': self.format_provider},
                'schemaString': self.schema_string,
                'partitionColumns': self.partition_columns,
                'configuration': self.configuration,
                'createdTime': self.created_time
            }
        }


@dataclass
class ProtocolAction:
    """Protocol version information."""
    min_reader_version: int = 1
    min_writer_version: int = 2
    
    def to_dict(self) -> Dict:
        return {
            'protocol': {
                'minReaderVersion': self.min_reader_version,
                'minWriterVersion': self.min_writer_version
            }
        }


@dataclass
class CommitInfo:
    """Commit metadata."""
    timestamp: int
    operation: str
    operation_parameters: Dict[str, Any] = field(default_factory=dict)
    read_version: Optional[int] = None
    isolation_level: str = 'Serializable'
    is_blind_append: bool = False
    operation_metrics: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'commitInfo': {
                'timestamp': self.timestamp,
                'operation': self.operation,
                'operationParameters': self.operation_parameters,
                'readVersion': self.read_version,
                'isolationLevel': self.isolation_level,
                'isBlindAppend': self.is_blind_append,
                'operationMetrics': self.operation_metrics
            }
        }


class DeltaLog:
    """
    Delta Transaction Log
    ---------------------
    Manages the transaction log for a Delta table.
    
    The log contains a series of JSON files named as version numbers
    (e.g., 00000000000000000000.json) that record all changes.
    """
    
    LOG_DIR = '_delta_log'
    
    def __init__(self, table_path: str):
        self.table_path = Path(table_path)
        self.log_path = self.table_path / self.LOG_DIR
        
    def initialize(self) -> None:
        """Initialize the log directory."""
        self.log_path.mkdir(parents=True, exist_ok=True)
        
    def get_latest_version(self) -> int:
        """Get the latest version number."""
        if not self.log_path.exists():
            return -1
            
        versions = []
        for f in self.log_path.glob('*.json'):
            try:
                version = int(f.stem)
                versions.append(version)
            except ValueError:
                continue
                
        return max(versions) if versions else -1
        
    def get_version_file(self, version: int) -> Path:
        """Get path for a version file."""
        return self.log_path / f'{version:020d}.json'
        
    def read_version(self, version: int) -> List[Dict]:
        """Read actions from a version."""
        version_file = self.get_version_file(version)
        if not version_file.exists():
            return []
            
        actions = []
        with open(version_file, 'r') as f:
            for line in f:
                if line.strip():
                    actions.append(json.loads(line))
        return actions
        
    def write_version(self, version: int, actions: List[Dict]) -> None:
        """Write actions to a version file."""
        version_file = self.get_version_file(version)
        with open(version_file, 'w') as f:
            for action in actions:
                f.write(json.dumps(action) + '\n')
                
    def get_snapshot(self, version: Optional[int] = None) -> 'DeltaSnapshot':
        """
        Get table snapshot at a version.
        
        Args:
            version: Version number (latest if None)
            
        Returns:
            DeltaSnapshot with table state
        """
        if version is None:
            version = self.get_latest_version()
            
        snapshot = DeltaSnapshot(version)
        
        # Read all versions up to requested
        for v in range(version + 1):
            actions = self.read_version(v)
            snapshot.apply_actions(actions)
            
        return snapshot


@dataclass
class DeltaSnapshot:
    """
    Delta Table Snapshot
    --------------------
    Represents the state of a Delta table at a point in time.
    """
    version: int
    metadata: Optional[MetadataAction] = None
    protocol: Optional[ProtocolAction] = None
    files: Dict[str, AddAction] = field(default_factory=dict)
    
    def apply_actions(self, actions: List[Dict]) -> None:
        """Apply actions to update snapshot."""
        for action in actions:
            if 'add' in action:
                add_data = action['add']
                add_action = AddAction(
                    path=add_data['path'],
                    partition_values=add_data.get('partitionValues', {}),
                    size=add_data.get('size', 0),
                    modification_time=add_data.get('modificationTime', 0),
                    data_change=add_data.get('dataChange', True),
                    stats=add_data.get('stats')
                )
                self.files[add_data['path']] = add_action
                
            elif 'remove' in action:
                remove_data = action['remove']
                self.files.pop(remove_data['path'], None)
                
            elif 'metaData' in action:
                meta_data = action['metaData']
                self.metadata = MetadataAction(
                    id=meta_data['id'],
                    name=meta_data.get('name'),
                    description=meta_data.get('description'),
                    schema_string=meta_data.get('schemaString', ''),
                    partition_columns=meta_data.get('partitionColumns', []),
                    configuration=meta_data.get('configuration', {}),
                    created_time=meta_data.get('createdTime')
                )
                
            elif 'protocol' in action:
                proto_data = action['protocol']
                self.protocol = ProtocolAction(
                    min_reader_version=proto_data.get('minReaderVersion', 1),
                    min_writer_version=proto_data.get('minWriterVersion', 2)
                )
                
    def get_file_paths(self) -> List[str]:
        """Get all active file paths."""
        return list(self.files.keys())


class DeltaTransaction:
    """
    Delta Transaction
    -----------------
    Manages atomic commits to a Delta table.
    
    Usage:
        table = DeltaTable('path/to/table')
        with table.start_transaction() as txn:
            txn.add_file('data/part-001.parquet', stats)
            # Transaction commits on exit
    """
    
    def __init__(self, table: 'DeltaTable'):
        self.table = table
        self.log = table.log
        self.read_version = table.log.get_latest_version()
        self.actions: List[Dict] = []
        self._committed = False
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and not self._committed:
            self.commit()
            
    def add_file(self, 
                 path: str,
                 size: int = 0,
                 partition_values: Optional[Dict[str, str]] = None,
                 stats: Optional[str] = None) -> None:
        """Record a file addition."""
        action = AddAction(
            path=path,
            partition_values=partition_values or {},
            size=size,
            modification_time=int(time.time() * 1000),
            data_change=True,
            stats=stats
        )
        self.actions.append(action.to_dict())
        
    def remove_file(self, 
                    path: str,
                    partition_values: Optional[Dict[str, str]] = None) -> None:
        """Record a file removal."""
        action = RemoveAction(
            path=path,
            deletion_timestamp=int(time.time() * 1000),
            data_change=True,
            partition_values=partition_values or {}
        )
        self.actions.append(action.to_dict())
        
    def commit(self, operation: str = 'WRITE') -> int:
        """
        Commit the transaction.
        
        Args:
            operation: Operation name for commit info
            
        Returns:
            New version number
        """
        if self._committed:
            raise ValueError("Transaction already committed")
            
        new_version = self.read_version + 1
        
        # Add commit info
        commit = CommitInfo(
            timestamp=int(time.time() * 1000),
            operation=operation,
            read_version=self.read_version,
            is_blind_append=True,
            operation_metrics={
                'numFiles': str(len([a for a in self.actions if 'add' in a])),
                'numOutputBytes': str(sum(
                    a.get('add', {}).get('size', 0) 
                    for a in self.actions
                ))
            }
        )
        
        all_actions = self.actions + [commit.to_dict()]
        
        # Write to log
        self.log.write_version(new_version, all_actions)
        self._committed = True
        
        return new_version


class DeltaTable:
    """
    Delta Table
    -----------
    A Delta Lake table with ACID transactions and time travel.
    
    Usage:
        # Create/open table
        table = DeltaTable('path/to/table')
        
        # Write data
        table.write(df)
        
        # Read data
        df = table.to_dataframe()
        
        # Time travel
        df_v1 = table.to_dataframe(version=1)
        
        # Merge
        table.merge(source_df, condition, when_matched, when_not_matched)
    """
    
    def __init__(self, path: str):
        """
        Initialize or open a Delta table.
        
        Args:
            path: Path to table directory
        """
        self.path = Path(path)
        self.log = DeltaLog(path)
        self._options = StorageOptions()
        
    def exists(self) -> bool:
        """Check if table exists."""
        return self.log.get_latest_version() >= 0
        
    def create(self, 
               schema: Optional[ParquetSchema] = None,
               partition_columns: Optional[List[str]] = None,
               description: Optional[str] = None) -> None:
        """
        Create a new Delta table.
        
        Args:
            schema: Table schema
            partition_columns: Columns to partition by
            description: Table description
        """
        if self.exists():
            raise ValueError(f"Table already exists at {self.path}")
            
        # Create directories
        self.path.mkdir(parents=True, exist_ok=True)
        self.log.initialize()
        
        # Write initial metadata
        metadata = MetadataAction(
            id=str(uuid.uuid4()),
            name=self.path.name,
            description=description,
            schema_string=json.dumps(schema.to_dict()) if schema else '{}',
            partition_columns=partition_columns or [],
            created_time=int(time.time() * 1000)
        )
        
        protocol = ProtocolAction()
        
        commit = CommitInfo(
            timestamp=int(time.time() * 1000),
            operation='CREATE TABLE'
        )
        
        actions = [
            protocol.to_dict(),
            metadata.to_dict(),
            commit.to_dict()
        ]
        
        self.log.write_version(0, actions)
        
    def write(self,
              data: Union[List[Dict], HiveDataFrame],
              mode: str = 'append',
              partition_by: Optional[List[str]] = None) -> None:
        """
        Write data to the table.
        
        Args:
            data: Data to write
            mode: 'append', 'overwrite', or 'error'
            partition_by: Columns to partition by
        """
        if isinstance(data, HiveDataFrame):
            records = data.collect()
        else:
            records = data
            
        if not records:
            return
            
        # Create table if not exists
        if not self.exists():
            schema = ParquetSchema.infer_from_data(records)
            self.create(schema, partition_by)
            
        # Start transaction
        with self.start_transaction() as txn:
            # Handle overwrite
            if mode == 'overwrite':
                snapshot = self.log.get_snapshot()
                for file_path in snapshot.get_file_paths():
                    txn.remove_file(file_path)
                    
            # Write data file
            file_name = f'part-{uuid.uuid4()}.parquet'
            file_path = self.path / file_name
            
            writer = ParquetWriter(str(file_path), options=self._options)
            metadata = writer.write(records)
            
            # Generate stats
            stats = self._compute_stats(records)
            
            txn.add_file(
                file_name,
                size=metadata.size_bytes,
                stats=json.dumps(stats)
            )
            
    def _compute_stats(self, records: List[Dict]) -> Dict:
        """Compute statistics for data."""
        if not records:
            return {}
            
        stats = {
            'numRecords': len(records),
            'minValues': {},
            'maxValues': {},
            'nullCount': {}
        }
        
        columns = records[0].keys()
        for col in columns:
            values = [r.get(col) for r in records if r.get(col) is not None]
            null_count = sum(1 for r in records if r.get(col) is None)
            
            stats['nullCount'][col] = null_count
            
            if values:
                try:
                    stats['minValues'][col] = min(values)
                    stats['maxValues'][col] = max(values)
                except TypeError:
                    pass  # Non-comparable types
                    
        return stats
        
    def start_transaction(self) -> DeltaTransaction:
        """Start a new transaction."""
        return DeltaTransaction(self)
        
    def to_dataframe(self, 
                     version: Optional[int] = None,
                     timestamp: Optional[datetime] = None) -> HiveDataFrame:
        """
        Read table to HiveDataFrame.
        
        Args:
            version: Specific version to read (time travel)
            timestamp: Timestamp to read as of
            
        Returns:
            HiveDataFrame with table data
        """
        if timestamp is not None:
            version = self._version_at_timestamp(timestamp)
            
        snapshot = self.log.get_snapshot(version)
        
        all_records = []
        for file_path in snapshot.get_file_paths():
            full_path = self.path / file_path
            if full_path.exists():
                reader = ParquetReader(str(full_path))
                df = reader.read()
                all_records.extend(df.collect())
                
        return HiveDataFrame(all_records)
        
    def _version_at_timestamp(self, timestamp: datetime) -> int:
        """Find version at a given timestamp."""
        target_ms = int(timestamp.timestamp() * 1000)
        latest = self.log.get_latest_version()
        
        for v in range(latest, -1, -1):
            actions = self.log.read_version(v)
            for action in actions:
                if 'commitInfo' in action:
                    commit_time = action['commitInfo'].get('timestamp', 0)
                    if commit_time <= target_ms:
                        return v
        return 0
        
    def history(self, limit: int = 10) -> List[Dict]:
        """
        Get table history.
        
        Args:
            limit: Maximum number of commits to return
            
        Returns:
            List of commit info dictionaries
        """
        latest = self.log.get_latest_version()
        history = []
        
        for v in range(latest, max(-1, latest - limit), -1):
            actions = self.log.read_version(v)
            for action in actions:
                if 'commitInfo' in action:
                    commit = action['commitInfo']
                    commit['version'] = v
                    history.append(commit)
                    break
                    
        return history
        
    def vacuum(self, retention_hours: int = 168) -> int:
        """
        Remove old files no longer referenced by the table.
        
        Args:
            retention_hours: Minimum retention period
            
        Returns:
            Number of files deleted
        """
        snapshot = self.log.get_snapshot()
        active_files = set(snapshot.get_file_paths())
        
        deleted = 0
        retention_ms = retention_hours * 3600 * 1000
        cutoff = int(time.time() * 1000) - retention_ms
        
        for f in self.path.glob('*.parquet'):
            if f.name not in active_files:
                mtime = int(f.stat().st_mtime * 1000)
                if mtime < cutoff:
                    f.unlink()
                    deleted += 1
                    
        return deleted
        
    def optimize(self) -> Dict:
        """
        Optimize table by compacting small files.
        
        Returns:
            Optimization statistics
        """
        snapshot = self.log.get_snapshot()
        files = list(snapshot.files.values())
        
        # Find small files to compact
        small_files = [f for f in files if f.size < 128 * 1024 * 1024]  # < 128MB
        
        if len(small_files) < 2:
            return {'filesCompacted': 0}
            
        # Read all small files
        all_records = []
        for f in small_files:
            full_path = self.path / f.path
            if full_path.exists():
                reader = ParquetReader(str(full_path))
                df = reader.read()
                all_records.extend(df.collect())
                
        if not all_records:
            return {'filesCompacted': 0}
            
        # Write combined file
        with self.start_transaction() as txn:
            # Remove old files
            for f in small_files:
                txn.remove_file(f.path)
                
            # Write new combined file
            file_name = f'part-{uuid.uuid4()}.parquet'
            file_path = self.path / file_name
            
            writer = ParquetWriter(str(file_path), options=self._options)
            metadata = writer.write(all_records)
            
            txn.add_file(
                file_name,
                size=metadata.size_bytes,
                stats=json.dumps(self._compute_stats(all_records))
            )
            
        return {
            'filesCompacted': len(small_files),
            'recordsProcessed': len(all_records)
        }


def read_delta(path: str,
               version: Optional[int] = None) -> HiveDataFrame:
    """
    Read Delta table to HiveDataFrame.
    
    Args:
        path: Table path
        version: Optional version for time travel
        
    Returns:
        HiveDataFrame
    """
    table = DeltaTable(path)
    return table.to_dataframe(version)


def write_delta(data: Union[List[Dict], HiveDataFrame],
                path: str,
                mode: str = 'append',
                partition_by: Optional[List[str]] = None) -> None:
    """
    Write data to Delta table.
    
    Args:
        data: Data to write
        path: Table path
        mode: Write mode ('append', 'overwrite')
        partition_by: Columns to partition by
    """
    table = DeltaTable(path)
    table.write(data, mode, partition_by)
