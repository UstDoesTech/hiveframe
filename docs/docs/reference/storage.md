---
sidebar_position: 6
---

# Storage Module

Parquet and Delta Lake storage backends.

```python
from hiveframe.storage import (
    ParquetReader, ParquetWriter,
    DeltaTable, DeltaTableBuilder
)
```

## Parquet Support

Apache Parquet columnar storage format.

### ParquetReader

```python
class ParquetReader:
    """
    Read Parquet files into DataFrames.
    """
    
    def __init__(
        self,
        colony: Optional[Colony] = None
    ) -> None:
        """
        Create Parquet reader.
        
        Args:
            colony: Colony for distributed reading
        """
```

#### Methods

```python
def read(
    self,
    path: Union[str, List[str]],
    columns: Optional[List[str]] = None,
    filter: Optional[str] = None,
    row_groups: Optional[List[int]] = None
) -> DataFrame:
    """
    Read Parquet file(s).
    
    Args:
        path: File path or list of paths
        columns: Columns to read (None = all)
        filter: Predicate pushdown filter
        row_groups: Specific row groups to read
        
    Returns:
        DataFrame with data
        
    Examples:
        # Single file
        df = reader.read("data.parquet")
        
        # Directory
        df = reader.read("data/")
        
        # With column pruning
        df = reader.read(
            "data.parquet",
            columns=["name", "age"]
        )
        
        # With predicate pushdown
        df = reader.read(
            "data.parquet",
            filter="age > 21"
        )
    """

def read_metadata(self, path: str) -> ParquetMetadata:
    """
    Read file metadata without loading data.
    
    Returns:
        ParquetMetadata with schema, row groups, etc.
    """

def read_schema(self, path: str) -> Schema:
    """
    Read schema without loading data.
    
    Returns:
        DataFrame schema
    """
```

### ParquetWriter

```python
class ParquetWriter:
    """
    Write DataFrames to Parquet files.
    """
```

#### Methods

```python
def write(
    self,
    df: DataFrame,
    path: str,
    mode: str = "error",
    partition_by: Optional[List[str]] = None,
    compression: str = "snappy",
    row_group_size: int = 1000000
) -> None:
    """
    Write DataFrame to Parquet.
    
    Args:
        df: DataFrame to write
        path: Output path
        mode: Write mode
            - "error": Fail if exists
            - "overwrite": Replace existing
            - "append": Add to existing
            - "ignore": Skip if exists
        partition_by: Partition columns
        compression: Compression codec
            - "snappy" (default)
            - "gzip"
            - "zstd"
            - "none"
        row_group_size: Rows per row group
        
    Examples:
        # Basic write
        writer.write(df, "output.parquet")
        
        # Overwrite
        writer.write(
            df, 
            "output.parquet",
            mode="overwrite"
        )
        
        # Partitioned
        writer.write(
            df,
            "output/",
            partition_by=["year", "month"]
        )
        
        # With compression
        writer.write(
            df,
            "output.parquet",
            compression="zstd"
        )
    """
```

### Convenience Methods

```python
import hiveframe as hf

# Read
df = hf.read.parquet("data.parquet")
df = hf.read.parquet("data/", recursive=True)

# Write
df.write.parquet("output.parquet")
df.write.mode("overwrite").parquet("output/")
df.write.partitionBy("date").parquet("output/")
```

---

## Delta Lake Support

Delta Lake provides ACID transactions, time travel, and schema evolution.

### DeltaTable

```python
class DeltaTable:
    """
    Delta Lake table with ACID transactions.
    """
    
    def __init__(
        self,
        path: str,
        colony: Optional[Colony] = None
    ) -> None:
        """
        Open existing Delta table.
        
        Args:
            path: Table path
            colony: Colony for distributed operations
            
        Raises:
            TableNotFoundError: Table doesn't exist
        """
```

#### Reading Methods

```python
def to_dataframe(self) -> DataFrame:
    """
    Read table as DataFrame.
    
    Example:
        table = DeltaTable("delta/users")
        df = table.to_dataframe()
    """

def as_of_version(self, version: int) -> "DeltaTable":
    """
    Time travel to specific version.
    
    Args:
        version: Version number
        
    Returns:
        DeltaTable at that version
        
    Example:
        # Read version 5
        old_df = table.as_of_version(5).to_dataframe()
    """

def as_of_timestamp(
    self, 
    timestamp: Union[str, datetime]
) -> "DeltaTable":
    """
    Time travel to specific timestamp.
    
    Args:
        timestamp: ISO string or datetime
        
    Returns:
        DeltaTable at that time
        
    Example:
        # Read as of yesterday
        old_df = table.as_of_timestamp(
            "2024-01-15T00:00:00Z"
        ).to_dataframe()
    """
```

#### Modification Methods

```python
def update(
    self,
    condition: str,
    set_values: Dict[str, Any]
) -> UpdateResult:
    """
    Update rows matching condition.
    
    Args:
        condition: SQL WHERE condition
        set_values: Column assignments
        
    Returns:
        UpdateResult with metrics
        
    Example:
        result = table.update(
            condition="status = 'pending'",
            set_values={
                "status": "'processed'",
                "updated_at": "current_timestamp()"
            }
        )
        print(f"Updated {result.rows_updated} rows")
    """

def delete(self, condition: str) -> DeleteResult:
    """
    Delete rows matching condition.
    
    Args:
        condition: SQL WHERE condition
        
    Returns:
        DeleteResult with metrics
        
    Example:
        result = table.delete(
            condition="created_at < '2023-01-01'"
        )
        print(f"Deleted {result.rows_deleted} rows")
    """

def merge(
    self,
    source: DataFrame,
    condition: str
) -> "MergeBuilder":
    """
    Merge source into table (upsert).
    
    Args:
        source: Source DataFrame
        condition: Join condition
        
    Returns:
        MergeBuilder for specifying operations
        
    Example:
        table.merge(
            source=updates_df,
            condition="target.id = source.id"
        ).when_matched_update_all() \
         .when_not_matched_insert_all() \
         .execute()
    """
```

#### Maintenance Methods

```python
def vacuum(self, retention_hours: int = 168) -> VacuumResult:
    """
    Remove old files.
    
    Args:
        retention_hours: Keep files newer than this
        
    Example:
        # Clean files older than 7 days
        table.vacuum(retention_hours=168)
    """

def optimize(
    self,
    partition_filter: Optional[str] = None,
    z_order_by: Optional[List[str]] = None
) -> OptimizeResult:
    """
    Compact small files.
    
    Args:
        partition_filter: Only optimize matching partitions
        z_order_by: Z-order columns for co-location
        
    Example:
        # Basic optimize
        table.optimize()
        
        # Z-order by query columns
        table.optimize(z_order_by=["user_id", "timestamp"])
    """

def history(self, limit: Optional[int] = None) -> DataFrame:
    """
    Get table history.
    
    Args:
        limit: Maximum entries to return
        
    Returns:
        DataFrame with version history
        
    Example:
        history_df = table.history(limit=10)
        history_df.show()
    """
```

#### Properties

```python
@property
def version(self) -> int:
    """Current table version."""

@property
def schema(self) -> Schema:
    """Table schema."""

@property
def partitioning(self) -> List[str]:
    """Partition columns."""
```

### DeltaTableBuilder

Create new Delta tables.

```python
class DeltaTableBuilder:
    """
    Builder for creating Delta tables.
    """
    
    @staticmethod
    def create(path: str) -> "DeltaTableBuilder":
        """Start building a new table."""
    
    @staticmethod
    def create_if_not_exists(path: str) -> "DeltaTableBuilder":
        """Create if table doesn't exist."""
    
    @staticmethod
    def replace(path: str) -> "DeltaTableBuilder":
        """Replace existing table."""
```

#### Builder Methods

```python
def add_column(
    self,
    name: str,
    dtype: DataType,
    nullable: bool = True,
    comment: Optional[str] = None
) -> "DeltaTableBuilder":
    """Add a column."""

def partition_by(
    self, 
    *cols: str
) -> "DeltaTableBuilder":
    """Set partition columns."""

def property(
    self, 
    key: str, 
    value: str
) -> "DeltaTableBuilder":
    """Set table property."""

def comment(
    self, 
    comment: str
) -> "DeltaTableBuilder":
    """Set table comment."""

def execute(self) -> DeltaTable:
    """Create the table."""
```

#### Example

```python
from hiveframe.storage import DeltaTableBuilder
from hiveframe import DataType

# Create new table
table = DeltaTableBuilder.create("delta/sales") \
    .add_column("id", DataType.LONG, nullable=False) \
    .add_column("product", DataType.STRING) \
    .add_column("amount", DataType.DOUBLE) \
    .add_column("date", DataType.DATE) \
    .partition_by("date") \
    .property("delta.autoOptimize.optimizeWrite", "true") \
    .comment("Sales transactions table") \
    .execute()
```

### MergeBuilder

Build merge (upsert) operations.

```python
table.merge(updates_df, "target.id = source.id") \
    .when_matched_update(
        condition="source.updated_at > target.updated_at",
        set_values={
            "name": "source.name",
            "amount": "source.amount",
            "updated_at": "source.updated_at"
        }
    ) \
    .when_matched_delete(
        condition="source.deleted = true"
    ) \
    .when_not_matched_insert(
        values={
            "id": "source.id",
            "name": "source.name",
            "amount": "source.amount",
            "created_at": "current_timestamp()"
        }
    ) \
    .execute()
```

### Convenience Methods

```python
import hiveframe as hf
from hiveframe.storage import DeltaTable

# Read
df = hf.read.delta("delta/users")

# Read with time travel
df = hf.read.delta("delta/users", version=5)
df = hf.read.delta("delta/users", timestamp="2024-01-15")

# Write
df.write.format("delta").save("delta/output")
df.write.format("delta").mode("overwrite").save("delta/output")

# Append with merge schema
df.write.format("delta") \
    .option("mergeSchema", "true") \
    .mode("append") \
    .save("delta/output")
```

---

## Complete Example

```python
from hiveframe.storage import DeltaTable, DeltaTableBuilder
import hiveframe as hf

# Create table
table = DeltaTableBuilder.create_if_not_exists("delta/events") \
    .add_column("event_id", hf.DataType.STRING, nullable=False) \
    .add_column("user_id", hf.DataType.LONG) \
    .add_column("event_type", hf.DataType.STRING) \
    .add_column("timestamp", hf.DataType.TIMESTAMP) \
    .add_column("properties", hf.DataType.MAP) \
    .partition_by("event_type") \
    .execute()

# Write initial data
events_df = hf.DataFrame([
    {"event_id": "e1", "user_id": 1, "event_type": "login", ...},
    {"event_id": "e2", "user_id": 2, "event_type": "purchase", ...},
])
events_df.write.format("delta").mode("append").save("delta/events")

# Merge updates
updates_df = hf.read.json("updates.json")
table.merge(updates_df, "target.event_id = source.event_id") \
    .when_matched_update_all() \
    .when_not_matched_insert_all() \
    .execute()

# Time travel query
yesterday = table.as_of_timestamp("2024-01-14T00:00:00Z")
df_yesterday = yesterday.to_dataframe()
df_today = table.to_dataframe()

# Compare
new_events = df_today.join(
    df_yesterday.select("event_id"),
    on="event_id",
    how="left_anti"
)
print(f"New events since yesterday: {new_events.count()}")

# Maintenance
table.optimize(z_order_by=["user_id", "timestamp"])
table.vacuum(retention_hours=168)
```

## See Also

- [Read/Write Parquet](/docs/how-to/read-write-parquet) - Parquet how-to
- [Use Delta Lake](/docs/how-to/use-delta-lake) - Delta Lake how-to
- [Time Travel](/docs/how-to/delta-time-travel) - Version history
