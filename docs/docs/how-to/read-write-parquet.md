---
sidebar_position: 2
---

# Read and Write Parquet Files

Efficiently read and write Parquet files with HiveFrame's optimized storage layer.

## Reading Parquet Files

### Basic Read

```python
import hiveframe as hf

# Read a single file
df = hf.read.parquet("data/sales.parquet")

# Read all files in a directory
df = hf.read.parquet("data/sales/")

# Read with glob pattern
df = hf.read.parquet("data/sales/*.parquet")
```

### Read with Options

```python
from hiveframe.storage import ParquetReadOptions

options = ParquetReadOptions(
    # Read specific columns only (column pruning)
    columns=["id", "amount", "date"],
    
    # Apply predicate pushdown
    filters=[("amount", ">", 100)],
    
    # Row group filtering
    row_groups=[0, 1, 2],
)

df = hf.read.parquet("data/sales.parquet", options=options)
```

### Read from Cloud Storage

```python
# Amazon S3
df = hf.read.parquet("s3://bucket/path/data.parquet")

# Google Cloud Storage
df = hf.read.parquet("gs://bucket/path/data.parquet")

# Azure Blob Storage
df = hf.read.parquet("abfs://container@account.dfs.core.windows.net/path/")
```

Configure credentials:

```python
# S3 credentials
hf.config.set("fs.s3.access_key", "YOUR_ACCESS_KEY")
hf.config.set("fs.s3.secret_key", "YOUR_SECRET_KEY")
hf.config.set("fs.s3.region", "us-east-1")

# Or use environment variables (recommended)
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
```

## Writing Parquet Files

### Basic Write

```python
# Write to a single file
df.write.parquet("output/results.parquet")

# Write partitioned data
df.write.parquet("output/results/", partition_by=["year", "month"])
```

### Write with Options

```python
from hiveframe.storage import ParquetWriteOptions, CompressionCodec

options = ParquetWriteOptions(
    # Compression (SNAPPY, GZIP, LZ4, ZSTD, NONE)
    compression=CompressionCodec.ZSTD,
    compression_level=3,
    
    # Row group size (default: 128MB)
    row_group_size=128 * 1024 * 1024,
    
    # Page size (default: 1MB)
    page_size=1024 * 1024,
    
    # Write statistics
    write_statistics=True,
    
    # Overwrite existing files
    mode="overwrite",  # or "append", "error"
)

df.write.parquet("output/results.parquet", options=options)
```

### Partitioning Strategies

```python
# Single partition column
df.write.parquet("output/", partition_by="date")

# Multiple partition columns (creates nested directories)
df.write.parquet("output/", partition_by=["year", "month", "day"])

# Partition with transformation
df = df.with_column("year", hf.year("timestamp"))
df = df.with_column("month", hf.month("timestamp"))
df.write.parquet("output/", partition_by=["year", "month"])
```

Directory structure:

```
output/
├── year=2026/
│   ├── month=01/
│   │   └── part-00000.parquet
│   └── month=02/
│       └── part-00000.parquet
└── year=2025/
    └── month=12/
        └── part-00000.parquet
```

## Schema Management

### View Schema

```python
df = hf.read.parquet("data/sales.parquet")
df.print_schema()
```

```
root
 |-- id: integer (nullable = true)
 |-- product: string (nullable = true)
 |-- amount: double (nullable = true)
 |-- timestamp: timestamp (nullable = true)
```

### Enforce Schema

```python
from hiveframe import Schema, Field, IntegerType, StringType, DoubleType

schema = Schema([
    Field("id", IntegerType(), nullable=False),
    Field("product", StringType()),
    Field("amount", DoubleType()),
])

df = hf.read.parquet("data/sales.parquet", schema=schema)
```

### Handle Schema Evolution

```python
# Merge schemas from multiple files
options = ParquetReadOptions(
    merge_schema=True  # Combines schemas from all files
)

df = hf.read.parquet("data/sales/", options=options)
```

## Performance Tips

### Column Pruning

Only read the columns you need:

```python
# ❌ Reads all columns
df = hf.read.parquet("data.parquet")
result = df.select("id", "amount")

# ✅ Only reads required columns
options = ParquetReadOptions(columns=["id", "amount"])
df = hf.read.parquet("data.parquet", options=options)
```

### Predicate Pushdown

Filter at read time:

```python
# ❌ Reads all data, then filters
df = hf.read.parquet("data.parquet")
result = df.filter(hf.col("amount") > 1000)

# ✅ Filters during read (skips row groups)
options = ParquetReadOptions(
    filters=[("amount", ">", 1000)]
)
df = hf.read.parquet("data.parquet", options=options)
```

### Choose the Right Compression

| Codec | Speed | Ratio | Use Case |
|-------|-------|-------|----------|
| SNAPPY | Fast | Medium | General purpose |
| LZ4 | Fastest | Lower | Real-time, hot data |
| ZSTD | Medium | Best | Cold storage, archives |
| GZIP | Slow | Good | Compatibility |

### Optimize Row Group Size

```python
# Larger row groups = fewer groups, better compression
# Smaller row groups = finer-grained filtering

# For analytical queries (full scans)
options = ParquetWriteOptions(row_group_size=256 * 1024 * 1024)

# For selective queries (many filters)
options = ParquetWriteOptions(row_group_size=64 * 1024 * 1024)
```

## Common Patterns

### Read and Append

```python
# Read existing data
existing = hf.read.parquet("data/sales/")

# Add new data
new_data = hf.DataFrame([...])

# Append
new_data.write.parquet("data/sales/", mode="append")
```

### Read Latest Partition

```python
import os

# Find latest partition
partitions = sorted(os.listdir("data/sales/"))
latest = partitions[-1]

df = hf.read.parquet(f"data/sales/{latest}/")
```

### Coalesce Small Files

```python
# Read many small files
df = hf.read.parquet("data/sales/")

# Write as fewer large files
df.coalesce(4).write.parquet("data/sales_optimized/", mode="overwrite")
```

## Troubleshooting

### File Not Found

```python
# Check if path exists
import os
print(os.path.exists("data/sales.parquet"))

# For cloud paths, verify credentials
hf.config.get("fs.s3.access_key")  # Should not be None
```

### Schema Mismatch

```python
# View actual schema
df = hf.read.parquet("data.parquet")
df.print_schema()

# Compare with expected
# If mismatch, use merge_schema or explicit schema
```

### Memory Issues

```python
# Process in chunks
for chunk in hf.read.parquet_batches("large_file.parquet", batch_size=10000):
    process(chunk)
```

## See Also

- [Use Delta Lake](./use-delta-lake) - ACID transactions on Parquet
- [Reference: Storage](/docs/reference/storage) - Complete API reference
- [Explanation: Storage Layer](/docs/explanation/architecture-overview#storage) - How storage works
