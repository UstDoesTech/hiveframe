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

## Next Steps

- Read [Core Concepts](core-concepts.md) to understand the bee-inspired architecture
- Explore [Examples](examples.md) for more advanced usage
- See [API Reference](api-reference.md) for complete documentation
