---
sidebar_position: 1
---

# API Overview

Complete reference for HiveFrame's API. This page provides an overview of all modules and functions.

## Core Module

The core module provides the fundamental HiveFrame engine and RDD-style operations.

### Classes

- **`HiveFrame`** - Main distributed processing engine
- **`Bee`** - Worker bee implementation
- **`BeeRole`** - Enum for bee roles (EMPLOYED, ONLOOKER, SCOUT)
- **`ColonyState`** - Shared state for bee coordination
- **`DanceFloor`** - Communication hub for waggle dances
- **`FoodSource`** - Data partition with fitness tracking
- **`WaggleDance`** - Quality advertisement data structure
- **`Pheromone`** - Coordination signal

### Factory Functions

```python
def create_hive(
    num_workers: int = 8,
    employed_ratio: float = 0.5,
    onlooker_ratio: float = 0.4,
    scout_ratio: float = 0.1,
    abandonment_limit: int = 20,
    max_cycles: int = 100,
    dance_threshold: float = 0.3
) -> HiveFrame
```

Creates and configures a HiveFrame instance.

**Parameters:**
- `num_workers`: Total number of worker bees
- `employed_ratio`: Fraction of employed bees (0.0-1.0)
- `onlooker_ratio`: Fraction of onlooker bees (0.0-1.0)
- `scout_ratio`: Fraction of scout bees (0.0-1.0)
- `abandonment_limit`: Cycles before abandoning a partition
- `max_cycles`: Maximum processing cycles
- `dance_threshold`: Minimum quality to advertise (0.0-1.0)

**Returns:** Configured HiveFrame instance

### HiveFrame Class

```python
class HiveFrame:
    def map(
        self,
        data: List[T],
        fn: Callable[[T], R]
    ) -> List[R]
```

Apply a function to each element.

**Parameters:**
- `data`: Input data list
- `fn`: Transformation function

**Returns:** List of transformed elements

```python
    def filter(
        self,
        data: List[T],
        predicate: Callable[[T], bool]
    ) -> List[T]
```

Filter elements based on a predicate.

**Parameters:**
- `data`: Input data list
- `predicate`: Boolean function to test each element

**Returns:** List of elements where predicate is True

```python
    def reduce(
        self,
        data: List[T],
        fn: Callable[[T, T], T]
    ) -> T
```

Reduce elements to a single value.

**Parameters:**
- `data`: Input data list
- `fn`: Binary reduction function

**Returns:** Single reduced value

```python
    def flat_map(
        self,
        data: List[T],
        fn: Callable[[T], List[R]]
    ) -> List[R]
```

Map and flatten results.

**Parameters:**
- `data`: Input data list
- `fn`: Function that returns a list for each element

**Returns:** Flattened list of results

```python
    def group_by_key(
        self,
        data: List[Tuple[K, V]]
    ) -> Dict[K, List[V]]
```

Group values by key.

**Parameters:**
- `data`: List of (key, value) tuples

**Returns:** Dictionary mapping keys to lists of values

## DataFrame Module

Spark-like DataFrame API for structured data processing.

### Classes

- **`HiveDataFrame`** - Main DataFrame class
- **`Column`** - Column expression
- **`GroupedData`** - Grouped DataFrame for aggregations

### HiveDataFrame Class

```python
class HiveDataFrame:
    def __init__(self, data: List[Dict[str, Any]])
```

Create a DataFrame from records.

**Parameters:**
- `data`: List of dictionaries (records)

#### Factory Methods

```python
    @staticmethod
    def from_csv(
        path: str,
        delimiter: str = ',',
        header: bool = True
    ) -> 'HiveDataFrame'
```

Load DataFrame from CSV file.

```python
    @staticmethod
    def from_json(
        path: str
    ) -> 'HiveDataFrame'
```

Load DataFrame from JSON file.

```python
    @staticmethod
    def from_records(
        records: List[Dict[str, Any]]
    ) -> 'HiveDataFrame'
```

Create DataFrame from list of records.

#### Transformation Methods

```python
    def select(self, *cols: Union[str, Column]) -> 'HiveDataFrame'
```

Select columns.

```python
    def filter(self, condition: Column) -> 'HiveDataFrame'
```

Filter rows based on condition.

```python
    def withColumn(
        self,
        name: str,
        col: Column
    ) -> 'HiveDataFrame'
```

Add or replace a column.

```python
    def drop(self, *cols: str) -> 'HiveDataFrame'
```

Remove columns.

```python
    def distinct(self) -> 'HiveDataFrame'
```

Remove duplicate rows.

```python
    def orderBy(
        self,
        cols: Union[str, List[str]],
        ascending: bool = True
    ) -> 'HiveDataFrame'
```

Sort by columns.

```python
    def limit(self, n: int) -> 'HiveDataFrame'
```

Limit to first n rows.

```python
    def join(
        self,
        other: 'HiveDataFrame',
        on: str,
        how: str = 'inner'
    ) -> 'HiveDataFrame'
```

Join with another DataFrame.

**Join types:** `'inner'`, `'left'`, `'right'`, `'outer'`

#### Grouping and Aggregation

```python
    def groupBy(self, *cols: str) -> GroupedData
```

Group by columns for aggregation.

#### Action Methods

```python
    def collect(self) -> List[Dict[str, Any]]
```

Collect all rows as list.

```python
    def count(self) -> int
```

Count total rows.

```python
    def show(self, n: int = 20) -> None
```

Display first n rows in table format.

```python
    def to_csv(
        self,
        path: str,
        delimiter: str = ','
    ) -> None
```

Save to CSV file.

```python
    def to_json(self, path: str) -> None
```

Save to JSON file.

### Column Functions

```python
def col(name: str) -> Column
```

Create a column reference.

```python
def lit(value: Any) -> Column
```

Create a literal value column.

### Column Class

```python
class Column:
    # Comparison operators
    def __eq__(self, other) -> Column  # ==
    def __ne__(self, other) -> Column  # !=
    def __lt__(self, other) -> Column  # <
    def __le__(self, other) -> Column  # <=
    def __gt__(self, other) -> Column  # >
    def __ge__(self, other) -> Column  # >=
    
    # Arithmetic operators
    def __add__(self, other) -> Column  # +
    def __sub__(self, other) -> Column  # -
    def __mul__(self, other) -> Column  # *
    def __truediv__(self, other) -> Column  # /
    
    # Logical operators
    def __and__(self, other) -> Column  # &
    def __or__(self, other) -> Column   # |
    
    # String methods
    def contains(self, substr: str) -> Column
    def startswith(self, prefix: str) -> Column
    def endswith(self, suffix: str) -> Column
    
    # Null checks
    def isNull(self) -> Column
    def isNotNull(self) -> Column
    
    # Aliasing
    def alias(self, name: str) -> Column
```

### Aggregation Functions

```python
def count(col: Column) -> AggFunc
```

Count non-null values.

```python
def sum_agg(col: Column) -> AggFunc
```

Sum values.

```python
def avg(col: Column) -> AggFunc
```

Calculate average.

```python
def min_agg(col: Column) -> AggFunc
```

Find minimum value.

```python
def max_agg(col: Column) -> AggFunc
```

Find maximum value.

```python
def collect_list(col: Column) -> AggFunc
```

Collect values into a list.

```python
def collect_set(col: Column) -> AggFunc
```

Collect unique values into a set.

## Streaming Module

Real-time data processing with bee colony coordination.

### Classes

- **`HiveStream`** - Basic stream processor
- **`EnhancedStreamProcessor`** - Production streaming with windows
- **`StreamRecord`** - Stream record wrapper
- **`WindowAssigner`** - Window assignment strategy
- **`WatermarkGenerator`** - Watermark generation strategy

### HiveStream Class

```python
class HiveStream:
    def __init__(
        self,
        num_workers: int = 4,
        buffer_size: int = 10000,
        employed_ratio: float = 0.5,
        onlooker_ratio: float = 0.3,
        scout_ratio: float = 0.2
    )
```

Create a stream processor.

```python
    def start(
        self,
        processor: Callable[[Any], Any]
    ) -> None
```

Start processing with the given function.

```python
    def submit(self, key: str, value: Any) -> None
```

Submit a record for processing.

```python
    def get_result(self, timeout: float = None) -> Optional[Any]
```

Get a processed result (blocking with optional timeout).

```python
    def get_metrics(self) -> Dict[str, Any]
```

Get processing metrics.

```python
    def stop(self) -> None
```

Stop the stream processor.

## Resilience Module

Fault tolerance and error handling.

### Classes

- **`RetryPolicy`** - Configurable retry behavior
- **`CircuitBreaker`** - Circuit breaker pattern
- **`BackoffStrategy`** - Backoff calculation (EXPONENTIAL, LINEAR, CONSTANT)
- **`CircuitState`** - Circuit state (CLOSED, OPEN, HALF_OPEN)

### RetryPolicy Class

```python
class RetryPolicy:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
        jitter: bool = True
    )
```

### CircuitBreaker Class

```python
class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        half_open_timeout: float = 30.0
    )
    
    def call(self, fn: Callable[[], T]) -> T
    
    @property
    def state(self) -> CircuitState
```

## Monitoring Module

Metrics, logging, and profiling.

### Functions

```python
def get_registry() -> MetricsRegistry
```

Get the global metrics registry.

```python
def get_logger(name: str = "hiveframe") -> Logger
```

Get a logger instance.

```python
def get_tracer() -> Tracer
```

Get the distributed tracer.

```python
def get_profiler() -> PerformanceProfiler
```

Get the performance profiler.

### Metric Types

- **`Counter`** - Monotonically increasing value
- **`Gauge`** - Arbitrary value that can go up/down
- **`Histogram`** - Distribution of values

## Exceptions

```python
class HiveFrameError(Exception)
```

Base exception for all HiveFrame errors.

```python
class TransientError(HiveFrameError)
```

Temporary error that can be retried.

```python
class ValidationError(HiveFrameError)
```

Data validation error.

```python
class CircuitOpenError(HiveFrameError)
```

Circuit breaker is open.

```python
class TimeoutError(HiveFrameError)
```

Operation timed out.

## Next Steps

- **Core API** - Detailed core module documentation
- **DataFrame API** - Complete DataFrame reference
- **Streaming API** - Streaming reference
- **Monitoring API** - Metrics and logging
