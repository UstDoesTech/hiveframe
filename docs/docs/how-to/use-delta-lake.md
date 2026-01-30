---
sidebar_position: 3
---

# Use Delta Lake

Work with Delta Lake tables for ACID transactions, schema evolution, and unified batch/streaming.

## Create a Delta Table

### From a DataFrame

```python
import hiveframe as hf
from hiveframe.storage import DeltaTable

# Create sample data
df = hf.DataFrame([
    {"id": 1, "name": "Alice", "balance": 1000.00},
    {"id": 2, "name": "Bob", "balance": 2500.00},
    {"id": 3, "name": "Carol", "balance": 1500.00},
])

# Write as Delta table
df.write.delta("data/accounts")
```

### With Schema Definition

```python
from hiveframe import Schema, Field, IntegerType, StringType, DoubleType

schema = Schema([
    Field("id", IntegerType(), nullable=False),
    Field("name", StringType()),
    Field("balance", DoubleType()),
])

# Create empty table with schema
DeltaTable.create(
    path="data/accounts",
    schema=schema,
    partition_by=["region"]
)
```

## Read Delta Tables

### Basic Read

```python
# Read the latest version
df = hf.read.delta("data/accounts")
df.show()
```

### Read from DeltaTable Object

```python
from hiveframe.storage import DeltaTable

# Open existing table
table = DeltaTable("data/accounts")

# Get as DataFrame
df = table.to_dataframe()

# View table details
print(f"Version: {table.version()}")
print(f"Files: {table.files()}")
```

## Update Data

### Insert New Rows

```python
new_data = hf.DataFrame([
    {"id": 4, "name": "Dave", "balance": 3000.00},
    {"id": 5, "name": "Eve", "balance": 4000.00},
])

# Append to table
new_data.write.delta("data/accounts", mode="append")
```

### Update Existing Rows

```python
table = DeltaTable("data/accounts")

# Update where condition matches
table.update(
    condition="id = 1",
    set={"balance": 1500.00}
)

# Update with expression
table.update(
    condition="balance < 2000",
    set={"balance": "balance * 1.1"}  # 10% increase
)
```

### Delete Rows

```python
table = DeltaTable("data/accounts")

# Delete by condition
table.delete(condition="id = 3")

# Delete with complex condition
table.delete(condition="balance < 1000 AND name != 'Alice'")
```

### Merge (Upsert)

```python
# Source data with updates and inserts
updates = hf.DataFrame([
    {"id": 1, "name": "Alice", "balance": 2000.00},  # Update
    {"id": 6, "name": "Frank", "balance": 500.00},   # Insert
])

table = DeltaTable("data/accounts")

table.merge(
    source=updates,
    condition="target.id = source.id"
).when_matched_update(
    set={"balance": "source.balance"}
).when_not_matched_insert(
    values={
        "id": "source.id",
        "name": "source.name",
        "balance": "source.balance"
    }
).execute()
```

## Schema Evolution

### Add New Columns

```python
# Enable schema evolution
new_data = hf.DataFrame([
    {"id": 7, "name": "Grace", "balance": 3500.00, "tier": "gold"},
])

new_data.write.delta(
    "data/accounts",
    mode="append",
    schema_evolution="merge"  # Adds new columns automatically
)
```

### Change Column Types

```python
from hiveframe.storage import DeltaTable

table = DeltaTable("data/accounts")

# Replace the table schema
table.replace_schema(
    Schema([
        Field("id", IntegerType()),
        Field("name", StringType()),
        Field("balance", DecimalType(10, 2)),  # Changed from DoubleType
        Field("tier", StringType()),
    ]),
    allow_unsafe=True  # Required for type changes
)
```

## Table Maintenance

### Optimize (Compact Files)

```python
table = DeltaTable("data/accounts")

# Compact small files into larger ones
table.optimize()

# Optimize specific partitions
table.optimize(partition_filter="region = 'us-east'")

# Z-order for query optimization
table.optimize(z_order_by=["id", "name"])
```

### Vacuum (Remove Old Files)

```python
table = DeltaTable("data/accounts")

# Remove files older than 7 days (default retention)
table.vacuum()

# Custom retention period
table.vacuum(retention_hours=168)  # 7 days

# Dry run (see what would be deleted)
files_to_delete = table.vacuum(dry_run=True)
print(f"Would delete {len(files_to_delete)} files")
```

### Check Constraints

```python
table = DeltaTable("data/accounts")

# Add constraint
table.add_constraint(
    name="positive_balance",
    expression="balance >= 0"
)

# List constraints
print(table.constraints())

# Drop constraint
table.drop_constraint("positive_balance")
```

## Table Properties

### View Properties

```python
table = DeltaTable("data/accounts")

print(table.properties())
# {'delta.minReaderVersion': '1', 'delta.minWriterVersion': '2'}

print(table.metadata())
# {'name': 'accounts', 'description': None, 'format': 'delta'}
```

### Set Properties

```python
table.set_property("delta.logRetentionDuration", "interval 30 days")
table.set_property("delta.deletedFileRetentionDuration", "interval 7 days")

# Custom properties
table.set_property("owner", "data-team")
table.set_property("description", "Customer account balances")
```

## Partitioning

### Create Partitioned Table

```python
df = hf.DataFrame([
    {"id": 1, "region": "us-east", "amount": 100},
    {"id": 2, "region": "us-west", "amount": 200},
    {"id": 3, "region": "eu-west", "amount": 150},
])

df.write.delta(
    "data/sales",
    partition_by=["region"]
)
```

### Query Partitioned Table

```python
# Partition pruning happens automatically
df = hf.read.delta("data/sales")
result = df.filter(hf.col("region") == "us-east")  # Only reads us-east partition
```

## Best Practices

### Write Performance

```python
# Batch writes for better performance
df.coalesce(4).write.delta("data/table", mode="append")

# Use appropriate partition granularity
# ❌ Too many partitions (one per row)
df.write.delta("data/table", partition_by=["id"])

# ✅ Good partition granularity
df.write.delta("data/table", partition_by=["date"])
```

### Read Performance

```python
# Filter on partition columns first
df = hf.read.delta("data/sales")
result = df.filter(
    (hf.col("date") >= "2026-01-01") &  # Partition pruning
    (hf.col("amount") > 100)             # Then row filtering
)
```

### Maintenance Schedule

```python
import schedule

def maintain_table():
    table = DeltaTable("data/accounts")
    
    # Compact files
    table.optimize()
    
    # Clean up old versions
    table.vacuum(retention_hours=168)
    
    print(f"Maintenance complete. Version: {table.version()}")

# Run daily
schedule.every().day.at("02:00").do(maintain_table)
```

## See Also

- [Delta Time Travel](./delta-time-travel) - Query historical versions
- [Read/Write Parquet](./read-write-parquet) - Parquet file operations
- [Reference: Storage](/docs/reference/storage) - Complete API reference
