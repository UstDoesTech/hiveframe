---
sidebar_position: 1
---

# Getting Started

This tutorial will help you get started with HiveFrame in just a few minutes. You'll learn how to install HiveFrame, create your first hive, and perform basic data processing operations.

## Prerequisites

- Python 3.9 or higher
- pip package manager

## Installation

Install HiveFrame using pip:

```bash
pip install hiveframe
```

For development or to get the latest features, install from source:

```bash
git clone https://github.com/UstDoesTech/hiveframe.git
cd hiveframe
pip install -e .
```

### Optional Dependencies

Install optional features:

```bash
# For Kafka support
pip install hiveframe[kafka]

# For monitoring
pip install hiveframe[monitoring]

# For all production features
pip install hiveframe[production]

# For development
pip install hiveframe[dev]
```

## Your First Hive

Let's create a simple application that processes data using the bee colony pattern.

### Step 1: Create a Hive

```python
from hiveframe import create_hive

# Create a hive with 8 worker bees
hive = create_hive(num_workers=8)
```

The `create_hive` function creates a colony of worker bees that will self-organize to process your data.

### Step 2: Map Operation

Transform data by applying a function to each element:

```python
# Create some sample data
data = list(range(100))

# Double each number
doubled = hive.map(data, lambda x: x * 2)

print(doubled[:5])  # [0, 2, 4, 6, 8]
```

### Step 3: Filter Operation

Filter data based on a condition:

```python
# Keep only even numbers
evens = hive.filter(data, lambda x: x % 2 == 0)

print(evens[:5])  # [0, 2, 4, 6, 8]
```

### Step 4: Reduce Operation

Combine all elements into a single result:

```python
# Calculate the sum
total = hive.reduce(data, lambda a, b: a + b)

print(f"Sum: {total}")  # Sum: 4950
```

## DataFrame API

For more complex operations, use the DataFrame API inspired by Spark:

```python
from hiveframe import HiveDataFrame, col, avg, count

# Create a DataFrame from records
records = [
    {'name': 'Alice', 'department': 'Engineering', 'salary': 95000},
    {'name': 'Bob', 'department': 'Engineering', 'salary': 87000},
    {'name': 'Carol', 'department': 'Marketing', 'salary': 78000},
    {'name': 'Dave', 'department': 'Engineering', 'salary': 92000},
]

df = HiveDataFrame(records)

# Filter and aggregate
result = (df
    .filter(col('salary') > 80000)
    .groupBy('department')
    .agg(
        count(col('name')).alias('count'),
        avg(col('salary')).alias('avg_salary')
    )
)

result.show()
```

Output:
```
+-------------+-------+------------+
| department  | count | avg_salary |
+-------------+-------+------------+
| Engineering |   3   |   91333.33 |
+-------------+-------+------------+
```

## Streaming

Process data in real-time using the streaming API:

```python
from hiveframe import HiveStream

# Create a stream processor with 4 workers
stream = HiveStream(num_workers=4)

# Define processing function
def process_record(record):
    return {'processed_value': record['value'] * 2}

# Start the stream
stream.start(process_record)

# Submit records
for i in range(10):
    stream.submit(f'key_{i}', {'value': i})

# Get results
while True:
    result = stream.get_result(timeout=1.0)
    if result is None:
        break
    print(result)

# Stop the stream
stream.stop()
```

## Understanding the Bee Colony

HiveFrame uses three types of workers inspired by real bee colonies:

- **Employed Bees (50%)**: Process assigned data partitions
- **Onlooker Bees (40%)**: Observe and reinforce high-quality work
- **Scout Bees (10%)**: Explore for new opportunities

This self-organizing behavior provides:
- Automatic load balancing
- Self-healing from failures
- Adaptive backpressure

## Next Steps

Now that you've completed the basics, explore these topics:

- **DataFrame Operations** - Learn more about the DataFrame API
- **Streaming Processing** - Deep dive into streaming
- **Bee Colony Architecture** - Understand how it works
- **API Reference** - Complete API documentation
