---
sidebar_position: 7
---

# Handle Errors with Dead Letter Queue

Route failed records to a Dead Letter Queue (DLQ) for later analysis and reprocessing.

## Basic DLQ Setup

```python
from hiveframe.dlq import DeadLetterQueue, DLQConfig

# Create a DLQ
dlq = DeadLetterQueue(DLQConfig(
    path="data/dlq",
    format="parquet",  # or "json", "delta"
))

# Process with DLQ
for record in records:
    try:
        process(record)
    except Exception as e:
        dlq.send(record, error=e)
```

## DLQ Configuration

```python
config = DLQConfig(
    # Storage location
    path="s3://bucket/dlq/",
    format="delta",  # Supports time travel
    
    # Partitioning
    partition_by=["date", "error_type"],
    
    # Retention
    retention_days=30,
    
    # Metadata
    include_stacktrace=True,
    include_timestamp=True,
    max_error_length=1000,
)

dlq = DeadLetterQueue(config)
```

## With DataFrame Operations

```python
import hiveframe as hf
from hiveframe.dlq import DeadLetterQueue

dlq = DeadLetterQueue(DLQConfig(path="data/dlq"))

def process_safely(df: hf.DataFrame) -> hf.DataFrame:
    """Process DataFrame, sending failures to DLQ."""
    
    def process_row(row):
        try:
            # Your transformation logic
            result = transform(row)
            return {"success": True, "data": result}
        except Exception as e:
            dlq.send(row, error=e, context={"operation": "transform"})
            return {"success": False, "data": None}
    
    # Process and filter out failures
    results = df.map(process_row)
    return results.filter(hf.col("success") == True).select("data.*")
```

## With Streaming

```python
from hiveframe.streaming import StreamProcessor
from hiveframe.dlq import DeadLetterQueue

dlq = DeadLetterQueue(DLQConfig(path="data/stream_dlq"))

stream = StreamProcessor(config)

# Configure stream with DLQ
stream.with_error_handler(
    on_error=lambda record, error: dlq.send(record, error=error)
)

# Or per-operation
stream.map(transform).on_error(
    handler=lambda r, e: dlq.send(r, e),
    continue_processing=True  # Don't stop on errors
)
```

## Query the DLQ

```python
# Read DLQ contents
dlq_df = hf.read.delta("data/dlq")

# View recent errors
dlq_df.orderBy("timestamp", ascending=False).show(10)

# Analyze error types
dlq_df.groupBy("error_type").agg(
    hf.count("*").alias("count")
).show()

# Find specific errors
dlq_df.filter(
    hf.col("error_message").contains("ConnectionError")
).show()
```

## Reprocess DLQ

```python
from hiveframe.dlq import DLQReprocessor

reprocessor = DLQReprocessor(
    dlq_path="data/dlq",
    processor=process_function,
    batch_size=100,
)

# Reprocess all records
results = reprocessor.reprocess_all()
print(f"Success: {results.success_count}, Failed: {results.failure_count}")

# Reprocess with filter
results = reprocessor.reprocess(
    filter_condition="error_type = 'TimeoutError' AND timestamp > '2026-01-29'"
)

# Reprocess with manual confirmation
for batch in reprocessor.batches():
    print(f"Batch: {len(batch)} records")
    if confirm("Reprocess this batch?"):
        batch.reprocess()
```

## DLQ Record Structure

```python
# Each DLQ record contains:
{
    "id": "uuid-of-record",
    "timestamp": "2026-01-30T10:15:30.123Z",
    "original_record": {...},  # The failed record
    "error_type": "ValueError",
    "error_message": "Invalid amount: -100",
    "stacktrace": "Traceback (most recent call last)...",
    "context": {
        "operation": "validate",
        "source": "kafka-topic-orders",
        "partition": 3,
        "offset": 12345,
    },
    "retry_count": 0,
    "status": "pending",  # pending, reprocessed, discarded
}
```

## Alerting on DLQ

```python
from hiveframe.dlq import DLQMonitor

monitor = DLQMonitor(dlq)

# Alert when DLQ grows
monitor.on_threshold(
    count=100,
    window_minutes=60,
    callback=lambda stats: alert(f"DLQ has {stats.count} records in the last hour")
)

# Alert on specific error types
monitor.on_error_type(
    error_type="DataCorruptionError",
    callback=lambda record: page_oncall(f"Data corruption detected: {record}")
)
```

## Best Practices

1. **Always include context** - Add source info, timestamps, operation names
2. **Partition by date** - Makes cleanup and analysis easier
3. **Set retention policies** - Don't keep DLQ records forever
4. **Monitor DLQ growth** - Sudden spikes indicate problems
5. **Automate reprocessing** - For transient errors
6. **Review regularly** - DLQ records reveal systemic issues

## See Also

- [Configure Retry](./configure-retry) - Retry before sending to DLQ
- [Use Circuit Breaker](./use-circuit-breaker) - Prevent failures
- [Reference: DLQ](/docs/reference/exceptions) - Complete API
