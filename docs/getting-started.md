# Getting Started with HiveFrame

This guide will help you get started with HiveFrame in just a few minutes.

## Installation

```bash
pip install hiveframe
```

## Basic Usage

### RDD-Style Processing

```python
from hiveframe import create_hive

# Create a hive with 8 workers
hive = create_hive(num_workers=8)

# Map operation
data = list(range(1000))
doubled = hive.map(data, lambda x: x * 2)

# Filter operation
evens = hive.filter(data, lambda x: x % 2 == 0)

# Reduce operation
total = hive.reduce(data, lambda a, b: a + b)
```

### DataFrame API

```python
from hiveframe import HiveDataFrame, col, avg, count

# Create DataFrame from records
records = [
    {'name': 'Alice', 'department': 'Engineering', 'salary': 95000},
    {'name': 'Bob', 'department': 'Engineering', 'salary': 87000},
    {'name': 'Carol', 'department': 'Marketing', 'salary': 78000},
]

df = HiveDataFrame(records)

# Filter and aggregate
result = (df
    .filter(col('salary') > 80000)
    .groupBy('department')
    .agg(avg(col('salary')), count(col('name')))
)

result.show()
```

### SQL Queries with SwarmQL

```python
from hiveframe import HiveDataFrame
from hiveframe.sql import SwarmQLContext

# Create SQL context
ctx = SwarmQLContext(num_workers=4)

# Register tables
users = HiveDataFrame([
    {'id': 1, 'name': 'Alice', 'age': 28},
    {'id': 2, 'name': 'Bob', 'age': 35},
])
ctx.register_table("users", users)

# Run SQL queries
result = ctx.sql("SELECT name, age FROM users WHERE age > 30")
result.show()

# View query plan
print(ctx.explain("SELECT * FROM users WHERE age > 25"))
```

### Streaming

```python
from hiveframe import HiveStream

# Create stream processor
stream = HiveStream(num_workers=4)

def process(record):
    return record['value'] * 2

stream.start(process)

# Submit records
stream.submit('key1', {'value': 10})
stream.submit('key2', {'value': 20})

# Stop when done
stream.stop()
```

### Advanced Streaming with Windows

```python
from hiveframe.streaming import (
    EnhancedStreamProcessor,
    tumbling_window,
    bounded_watermark,
    DeliveryGuarantee,
    StreamRecord
)

# Create windowed processor
processor = EnhancedStreamProcessor(
    num_workers=4,
    window_assigner=tumbling_window(5.0),  # 5-second windows
    watermark_generator=bounded_watermark(2.0),
    delivery_guarantee=DeliveryGuarantee.EXACTLY_ONCE
)

# Process records with aggregation
for event in events:
    record = StreamRecord(
        key=event['sensor_id'],
        value=event['reading'],
        timestamp=event['timestamp']
    )
    processor.process_record(record, aggregator=sum, initial_value=0.0)
```

## Data Storage

### Reading and Writing Parquet

```python
from hiveframe import HiveDataFrame
from hiveframe.storage import read_parquet, write_parquet

# Write DataFrame to Parquet
df = HiveDataFrame(records)
write_parquet(df, "/data/output.parquet")

# Read Parquet file
df = read_parquet("/data/output.parquet")
df.show()
```

### Delta Lake with ACID Transactions

```python
from hiveframe.storage import DeltaTable

# Create Delta table
table = DeltaTable("/data/events")
table.write(df, mode='overwrite')

# Update records
table.update(
    predicate=col('status') == 'pending',
    updates={'status': 'completed'}
)

# Time travel - read historical version
old_df = table.version_as_of(5)

# View change history
for entry in table.history():
    print(f"Version {entry['version']}: {entry['operation']}")
```

## DataFrame Operations

### Joins

```python
employees = HiveDataFrame([...])
departments = HiveDataFrame([...])

# Inner join
result = employees.join(departments, on='dept_id', how='inner')

# Left join
result = employees.join(departments, on='dept_id', how='left')
```

### Unions and Deduplication

```python
# Union two DataFrames
combined = df1.union(df2)

# Remove duplicates
unique = combined.distinct()

# Deduplicate by specific columns
unique = combined.dropDuplicates('id', 'name')
```

### Statistics

```python
# Get summary statistics
df.select('amount', 'quantity').describe().show()
```

## Resilience Patterns

### Retry with Backoff

```python
from hiveframe.resilience import RetryPolicy, with_retry, BackoffStrategy

policy = RetryPolicy(
    max_retries=3,
    base_delay=1.0,
    strategy=BackoffStrategy.EXPONENTIAL
)

@with_retry(policy)
def fetch_data():
    return api.get("/data")
```

### Circuit Breaker

```python
from hiveframe.resilience import CircuitBreaker, CircuitBreakerConfig

breaker = CircuitBreaker("api", CircuitBreakerConfig(
    failure_threshold=5,
    timeout=30.0
))

# Call with circuit breaker protection
result = breaker.call(lambda: api.request())
```

## Monitoring

### Metrics

```python
from hiveframe.monitoring import get_registry

registry = get_registry()

# Create metrics
requests = registry.counter("requests_total")
latency = registry.histogram("request_latency")

# Record values
requests.inc()
latency.observe(0.125)

# Export to Prometheus format
print(registry.to_prometheus())
```

### Logging

```python
from hiveframe.monitoring import get_logger

logger = get_logger("my_app")
logger.info("Processing started", batch_id=123, records=1000)
```

## Kubernetes Deployment

```python
from hiveframe.k8s import HiveCluster, WorkerSpec, HiveOperator

# Define cluster
cluster = HiveCluster(
    name="production",
    workers=WorkerSpec(replicas=10, cpu="2", memory="4Gi")
)

# Deploy to Kubernetes
operator = HiveOperator()
operator.deploy(cluster)
```

## Colony Dashboard

```python
from hiveframe.dashboard import Dashboard

# Start monitoring dashboard
dashboard = Dashboard(port=8080)
dashboard.start()

# Access at http://localhost:8080
```

## Next Steps

- Read [Core Concepts](core-concepts.md) to understand the bee-inspired architecture
- See the [Examples](../examples/) folder for complete demos:
  - `demo.py` - Comprehensive feature demonstration
  - `demo_progressive.py` - Progressive difficulty levels
  - `demo_sql.py` - SwarmQL SQL engine
  - `demo_storage.py` - Parquet and Delta Lake
- See [API Reference](api-reference.md) for complete documentation
