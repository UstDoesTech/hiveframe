---
sidebar_position: 4
---

# Delta Lake Time Travel

Query historical versions of your Delta tables for auditing, debugging, and recovering data.

## View Table History

```python
from hiveframe.storage import DeltaTable

table = DeltaTable("data/accounts")

# View all versions
history = table.history()
for entry in history:
    print(f"Version {entry['version']}: {entry['operation']} at {entry['timestamp']}")
```

Output:

```
Version 5: UPDATE at 2026-01-30 14:30:00
Version 4: MERGE at 2026-01-30 12:15:00
Version 3: DELETE at 2026-01-29 18:00:00
Version 2: WRITE at 2026-01-29 10:00:00
Version 1: WRITE at 2026-01-28 09:00:00
Version 0: CREATE at 2026-01-28 08:00:00
```

### Detailed History

```python
# Get full details
history = table.history(limit=10)

for entry in history:
    print(f"""
Version: {entry['version']}
Timestamp: {entry['timestamp']}
Operation: {entry['operation']}
User: {entry.get('userName', 'unknown')}
Parameters: {entry.get('operationParameters', {})}
Metrics: {entry.get('operationMetrics', {})}
""")
```

## Query by Version

### Read Specific Version

```python
import hiveframe as hf

# Read version 3 of the table
df = hf.read.delta("data/accounts", version=3)
df.show()
```

### Using DeltaTable

```python
table = DeltaTable("data/accounts")

# Read as of version
df = table.to_dataframe(version=3)

# Or using the at() method
df = table.at(version=3).to_dataframe()
```

## Query by Timestamp

### Read at Timestamp

```python
from datetime import datetime

# Read the table as it was at a specific time
df = hf.read.delta(
    "data/accounts",
    timestamp="2026-01-29T12:00:00"
)

# Using datetime object
timestamp = datetime(2026, 1, 29, 12, 0, 0)
df = hf.read.delta("data/accounts", timestamp=timestamp)
```

### Using DeltaTable

```python
table = DeltaTable("data/accounts")

# Read as of timestamp
df = table.at(timestamp="2026-01-29T12:00:00").to_dataframe()
```

## Compare Versions

### View Changes Between Versions

```python
table = DeltaTable("data/accounts")

# Get the diff between versions
changes = table.diff(from_version=2, to_version=5)
print(changes)
```

### Manual Comparison

```python
# Read both versions
v2 = hf.read.delta("data/accounts", version=2)
v5 = hf.read.delta("data/accounts", version=5)

# Find added rows
added = v5.join(v2, on="id", how="left_anti")
print("Added rows:")
added.show()

# Find removed rows
removed = v2.join(v5, on="id", how="left_anti")
print("Removed rows:")
removed.show()

# Find changed rows
both = v5.join(v2, on="id", how="inner")
changed = both.filter(hf.col("v5.balance") != hf.col("v2.balance"))
print("Changed rows:")
changed.show()
```

## Restore to Previous Version

### Restore Entire Table

```python
table = DeltaTable("data/accounts")

# Restore to version 3
table.restore(version=3)

# Or restore to a timestamp
table.restore(timestamp="2026-01-29T12:00:00")
```

### Restore Specific Records

```python
# Read the old version
old_data = hf.read.delta("data/accounts", version=2)

# Filter for records to restore
records_to_restore = old_data.filter(hf.col("id").isin([1, 2, 3]))

# Merge back into current table
table = DeltaTable("data/accounts")
table.merge(
    source=records_to_restore,
    condition="target.id = source.id"
).when_matched_update_all(
).execute()
```

## Audit Use Cases

### Find Who Changed a Record

```python
table = DeltaTable("data/accounts")

# Get history and filter for specific record
history = table.history()

for entry in history:
    version = entry['version']
    df = table.at(version=version).to_dataframe()
    
    record = df.filter(hf.col("id") == 42).collect()
    if record:
        print(f"Version {version} ({entry['timestamp']}): {record[0]}")
```

### Track Balance Over Time

```python
table = DeltaTable("data/accounts")
history = table.history()

balance_history = []
for entry in history:
    df = table.at(version=entry['version']).to_dataframe()
    record = df.filter(hf.col("id") == 1).collect()
    
    if record:
        balance_history.append({
            "version": entry['version'],
            "timestamp": entry['timestamp'],
            "balance": record[0]['balance']
        })

# Show balance changes
for h in balance_history:
    print(f"{h['timestamp']}: ${h['balance']:,.2f}")
```

## Configuration

### Retention Settings

```python
table = DeltaTable("data/accounts")

# Set log retention (how long to keep history)
table.set_property(
    "delta.logRetentionDuration",
    "interval 30 days"  # Keep 30 days of history
)

# Set file retention (must be >= log retention)
table.set_property(
    "delta.deletedFileRetentionDuration",
    "interval 7 days"
)
```

### Enable Extended History

```python
# Enable change data feed for detailed tracking
table.set_property("delta.enableChangeDataFeed", "true")

# Now you can query changes
changes = table.changes(
    start_version=1,
    end_version=5
)
changes.show()
```

## Best Practices

### 1. Set Appropriate Retention

```python
# For compliance (long retention)
table.set_property("delta.logRetentionDuration", "interval 365 days")

# For development (shorter retention, save space)
table.set_property("delta.logRetentionDuration", "interval 7 days")
```

### 2. Document Important Versions

```python
# Add comments to important operations
table.update(
    condition="id = 1",
    set={"balance": 5000.00},
    comment="Annual bonus credited"  # Stored in history
)
```

### 3. Regular Vacuuming

```python
# Vacuum removes old files but preserves history metadata
table.vacuum(retention_hours=168)  # 7 days

# Warning: Vacuum makes old versions unreadable if files are removed
# Set retention appropriately before vacuuming
```

### 4. Version Bookmarks

```python
# Save important versions for quick reference
bookmarks = {
    "pre_migration": 45,
    "post_bugfix": 52,
    "quarterly_snapshot_q1": 78,
}

# Query a bookmarked version
df = hf.read.delta("data/accounts", version=bookmarks["pre_migration"])
```

## Troubleshooting

### Version Not Found

```python
# Check available versions
table = DeltaTable("data/accounts")
history = table.history()
versions = [h['version'] for h in history]
print(f"Available versions: {min(versions)} to {max(versions)}")
```

### Files Missing (After Vacuum)

```python
# If you get "file not found" errors on old versions:
# 1. Check vacuum retention settings
# 2. Restore from backup if available
# 3. For future: increase retention before vacuum

# Check current retention
props = table.properties()
print(f"Retention: {props.get('delta.logRetentionDuration', 'default')}")
```

## See Also

- [Use Delta Lake](./use-delta-lake) - Basic Delta Lake operations
- [Reference: Storage](/docs/reference/storage) - Complete API reference
