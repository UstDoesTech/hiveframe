# API Reference

Complete API documentation for HiveFrame.

## Core Module

### HiveFrame

The main entry point for distributed processing.

```python
class HiveFrame:
    def __init__(self, num_workers: int = 8, **config)
    def map(self, data: List[T], fn: Callable[[T], R]) -> List[R]
    def filter(self, data: List[T], predicate: Callable[[T], bool]) -> List[T]
    def reduce(self, data: List[T], fn: Callable[[T, T], T]) -> T
    def process(self, data: List[T], fn: Callable[[T], Tuple[R, float]]) -> List[R]
```

### create_hive

Factory function to create a HiveFrame instance.

```python
def create_hive(
    num_workers: int = 8,
    abandonment_limit: int = 20,
    dance_threshold: float = 0.3,
    **kwargs
) -> HiveFrame
```

## DataFrame Module

### HiveDataFrame

Spark-like DataFrame API.

```python
class HiveDataFrame:
    def __init__(self, data: List[Dict[str, Any]])
    def select(self, *cols) -> HiveDataFrame
    def filter(self, condition: Column) -> HiveDataFrame
    def groupBy(self, *cols) -> GroupedData
    def withColumn(self, name: str, col: Column) -> HiveDataFrame
    def join(self, other: HiveDataFrame, on: str, how: str = 'inner') -> HiveDataFrame
    def count(self) -> int
    def collect(self) -> List[Dict[str, Any]]
    def show(self, n: int = 20)
```

### Column Functions

```python
def col(name: str) -> Column
def lit(value: Any) -> Column
```

### Aggregation Functions

```python
def sum_agg(column: Column) -> AggFunc
def avg(column: Column) -> AggFunc
def count(column: Column) -> AggFunc
def min_agg(column: Column) -> AggFunc
def max_agg(column: Column) -> AggFunc
def collect_list(column: Column) -> AggFunc
def collect_set(column: Column) -> AggFunc
```

## Streaming Module

### HiveStream

Basic streaming processor.

```python
class HiveStream:
    def __init__(self, num_workers: int = 4)
    def start(self, processor: Callable[[StreamRecord], Any])
    def submit(self, key: str, value: Any)
    def stop()
```

### EnhancedStreamProcessor

Production streaming with windows and delivery guarantees.

```python
class EnhancedStreamProcessor:
    def __init__(
        self,
        num_workers: int = 4,
        window_assigner: WindowAssigner = None,
        watermark_generator: WatermarkGenerator = None,
        delivery_guarantee: DeliveryGuarantee = AT_LEAST_ONCE
    )
    def process_record(self, record: StreamRecord, aggregator: Callable, initial_value: Any = None)
    def get_metrics() -> Dict[str, Any]
```

## Resilience Module

### RetryPolicy

```python
class RetryPolicy:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: BackoffStrategy = EXPONENTIAL,
        jitter: bool = True
    )
```

### CircuitBreaker

```python
class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig)
    def call(self, fn: Callable[[], T]) -> T
    @property
    def state(self) -> CircuitState  # CLOSED, OPEN, HALF_OPEN
```

## Connectors Module

### Data Sources

```python
class CSVSource(DataSource)
class JSONSource(DataSource)
class JSONLSource(DataSource)
class HTTPSource(DataSource)
```

### Data Sinks

```python
class CSVSink(DataSink)
class JSONLSink(DataSink)
```

### Message Broker

```python
class MessageBroker:
    def create_topic(self, name: str, num_partitions: int = 4) -> Topic
    
class Topic:
    def produce(self, key: str, value: Any)
    def consume(self, group_id: str, partition: int, max_messages: int = 100) -> List[Message]
```

## Monitoring Module

### Metrics

```python
class Counter:
    def inc(self, value: float = 1.0)
    
class Gauge:
    def set(self, value: float)
    def inc(self, value: float = 1.0)
    def dec(self, value: float = 1.0)
    
class Histogram:
    def observe(self, value: float)
```

### Logger

```python
class Logger:
    def debug(self, message: str, **context)
    def info(self, message: str, **context)
    def warning(self, message: str, **context)
    def error(self, message: str, **context)
```

### Global Accessors

```python
def get_registry() -> MetricsRegistry
def get_logger(name: str = "hiveframe") -> Logger
def get_tracer() -> Tracer
def get_profiler() -> PerformanceProfiler
```

## Exceptions

```python
class HiveFrameError(Exception)  # Base exception
class TransientError(HiveFrameError)  # Retryable errors
class ValidationError(HiveFrameError)  # Data validation errors
class CircuitOpenError(HiveFrameError)  # Circuit breaker open
class TimeoutError(HiveFrameError)  # Operation timeout
```
