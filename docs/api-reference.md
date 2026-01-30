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
    
    # Selection and filtering
    def select(self, *cols) -> HiveDataFrame
    def filter(self, condition: Column) -> HiveDataFrame
    def where(self, condition: Column) -> HiveDataFrame  # Alias for filter
    def limit(self, n: int) -> HiveDataFrame
    
    # Grouping and aggregation
    def groupBy(self, *cols) -> GroupedData
    
    # Joins and set operations
    def join(self, other: HiveDataFrame, on: str, how: str = 'inner') -> HiveDataFrame
    def union(self, other: HiveDataFrame) -> HiveDataFrame
    def crossJoin(self, other: HiveDataFrame) -> HiveDataFrame
    
    # Column operations
    def withColumn(self, name: str, col: Column) -> HiveDataFrame
    def drop(self, *cols) -> HiveDataFrame
    def rename(self, old: str, new: str) -> HiveDataFrame
    
    # Deduplication
    def distinct(self) -> HiveDataFrame
    def dropDuplicates(self, *cols) -> HiveDataFrame
    
    # Sorting
    def orderBy(self, *cols, ascending: bool = True) -> HiveDataFrame
    def sort(self, *cols, ascending: bool = True) -> HiveDataFrame
    
    # Statistics
    def describe(self, *cols) -> HiveDataFrame
    def count(self) -> int
    
    # I/O
    def collect(self) -> List[Dict[str, Any]]
    def show(self, n: int = 20)
    def printSchema()
    
    # File operations
    @classmethod
    def from_records(cls, records: List[Dict], hive: HiveFrame = None) -> HiveDataFrame
    @classmethod
    def read_json(cls, path: str) -> HiveDataFrame
    @classmethod  
    def read_csv(cls, path: str, header: bool = True) -> HiveDataFrame
    def write_json(self, path: str)
    def write_csv(self, path: str, header: bool = True)
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
def first(column: Column) -> AggFunc
def last(column: Column) -> AggFunc
```

## SQL Module (SwarmQL)

### SwarmQLContext

SQL execution context for running queries.

```python
class SwarmQLContext:
    def __init__(self, num_workers: int = 4)
    
    # Table management
    def register_table(self, name: str, data: HiveDataFrame) -> None
    def drop_table(self, name: str) -> None
    def table(self, name: str) -> Optional[HiveDataFrame]
    def tables(self) -> List[str]
    
    # Query execution
    def sql(self, query: str) -> HiveDataFrame
    def explain(self, query: str) -> str
    
    # DataFrame creation
    def create_dataframe(self, data: List[Dict]) -> HiveDataFrame
```

### SQLParser

Parses SQL queries into an AST.

```python
class SQLParser:
    def parse(self, query: str) -> SQLStatement
    
class SQLTokenizer:
    def tokenize(self, query: str) -> List[Token]
```

### SQL Types

```python
class SQLType: pass
class IntegerType(SQLType): pass
class FloatType(SQLType): pass
class StringType(SQLType): pass
class BooleanType(SQLType): pass
class DateType(SQLType): pass
class TimestampType(SQLType): pass
class ArrayType(SQLType): element_type: SQLType
class MapType(SQLType): key_type: SQLType; value_type: SQLType
```

## Query Optimizer

### QueryOptimizer

Bee-inspired query optimization using ABC algorithm.

```python
class QueryOptimizer:
    def __init__(
        self,
        rules: List[OptimizationRule] = None,
        cost_model: CostModel = None,
        max_iterations: int = 100,
        swarm_size: int = 20,
        abandonment_limit: int = 10
    )
    
    def optimize(self, plan: PlanNode, stats: Dict = None) -> OptimizedPlan
```

### Optimization Rules

```python
class OptimizationRule: 
    def apply(self, plan: PlanNode) -> Optional[PlanNode]

class PredicatePushdown(OptimizationRule): pass
class ProjectionPruning(OptimizationRule): pass
class ConstantFolding(OptimizationRule): pass
class FilterCombination(OptimizationRule): pass
class JoinReordering(OptimizationRule): pass
class LimitPushdown(OptimizationRule): pass
```

### Cost Model

```python
class CostModel:
    def estimate(self, plan: PlanNode, stats: Statistics) -> CostEstimate

class SwarmCostModel(CostModel):
    """ABC-algorithm based cost estimation"""
    pass

class CostEstimate:
    cpu_cost: float
    io_cost: float
    memory_cost: float
    total_cost: float
```

## Storage Module

### Parquet Operations

```python
def read_parquet(path: str) -> HiveDataFrame
def write_parquet(
    df: HiveDataFrame, 
    path: str,
    compression: CompressionCodec = CompressionCodec.SNAPPY,
    partition_by: PartitionSpec = None
) -> None

class ParquetReader:
    def __init__(self, path: str)
    def read(self) -> HiveDataFrame
    def read_schema(self) -> ParquetSchema

class ParquetWriter:
    def __init__(self, path: str, schema: ParquetSchema = None)
    def write(self, df: HiveDataFrame)
```

### Delta Lake

```python
def read_delta(path: str) -> HiveDataFrame
def write_delta(df: HiveDataFrame, path: str, mode: str = 'overwrite') -> None

class DeltaTable:
    def __init__(self, path: str)
    
    # Basic operations
    def read(self) -> HiveDataFrame
    def write(self, df: HiveDataFrame, mode: str = 'overwrite') -> None
    
    # CRUD operations
    def update(self, predicate: Column, updates: Dict[str, Column]) -> None
    def delete(self, predicate: Column) -> None
    def merge(
        self, 
        source_df: HiveDataFrame,
        condition: Column,
        when_matched: str = None,
        when_not_matched: str = None
    ) -> None
    
    # Time travel
    def history(self, limit: int = 10) -> List[Dict]
    def version_as_of(self, version: int) -> HiveDataFrame
    def timestamp_as_of(self, timestamp: datetime) -> HiveDataFrame

class DeltaTransaction:
    def __init__(self, table: DeltaTable)
    def update(self, predicate: Column, updates: Dict) -> None
    def delete(self, predicate: Column) -> None
    def commit(self) -> None
    def rollback(self) -> None
```

### Storage Options

```python
class CompressionCodec(Enum):
    NONE = 'none'
    SNAPPY = 'snappy'
    GZIP = 'gzip'
    ZSTD = 'zstd'
    LZ4 = 'lz4'

class PartitionSpec:
    def __init__(self, columns: List[str])

class FileFormat(Enum):
    PARQUET = 'parquet'
    DELTA = 'delta'
    JSON = 'json'
    CSV = 'csv'
```

## Streaming Module

### HiveStream

Basic streaming processor.

```python
class HiveStream:
    def __init__(self, num_workers: int = 4)
    def start(self, processor: Callable[[StreamRecord], Any])
    def submit(self, key: str, value: Any) -> bool
    def get_result(self, timeout: float = None) -> Any
    def get_metrics(self) -> Dict[str, Any]
    def stop()

class AsyncHiveStream(HiveStream):
    """Async-aware streaming processor"""
    async def submit_async(self, key: str, value: Any) -> bool
    async def get_result_async(self, timeout: float = None) -> Any
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
    
    def process_record(
        self, 
        record: StreamRecord, 
        aggregator: Callable, 
        initial_value: Any = None
    ) -> None
    
    def get_metrics(self) -> Dict[str, Any]
```

### Windows

```python
class WindowType(Enum):
    TUMBLING = 'tumbling'
    SLIDING = 'sliding'
    SESSION = 'session'

# Factory functions
def tumbling_window(size_seconds: float) -> TumblingWindowAssigner
def sliding_window(window_size: float, slide_interval: float) -> SlidingWindowAssigner
def session_window(gap_seconds: float) -> SessionWindowAssigner

class TumblingWindowAssigner(WindowAssigner): pass
class SlidingWindowAssigner(WindowAssigner): pass
class SessionWindowAssigner(WindowAssigner): pass
```

### Watermarks

```python
def bounded_watermark(max_out_of_order: float) -> BoundedOutOfOrdernessWatermarkGenerator

class WatermarkGenerator:
    def generate(self, timestamp: float) -> Watermark

class BoundedOutOfOrdernessWatermarkGenerator(WatermarkGenerator):
    def __init__(self, max_out_of_order: float)
```

### Delivery Guarantees

```python
class DeliveryGuarantee(Enum):
    AT_MOST_ONCE = 'at_most_once'
    AT_LEAST_ONCE = 'at_least_once'
    EXACTLY_ONCE = 'exactly_once'
```

### Stream Aggregators

```python
def count_aggregator(acc: int, value: Any) -> int
def sum_aggregator(acc: float, value: float) -> float
def avg_aggregator(acc: Tuple, value: float) -> Tuple
def max_aggregator(acc: float, value: float) -> float
def min_aggregator(acc: float, value: float) -> float
def collect_aggregator(acc: List, value: Any) -> List
```

## Kubernetes Module

### HiveCluster

Kubernetes cluster specification.

```python
class HiveCluster:
    def __init__(
        self,
        name: str,
        namespace: str = 'default',
        workers: WorkerSpec = None,
        service_type: str = 'ClusterIP',
        service_port: int = 8080,
        persistence: bool = False
    )
    
    def to_k8s_manifests(self) -> Dict[str, Any]

class WorkerSpec:
    def __init__(
        self,
        replicas: int = 3,
        resources: ResourceRequirements = None,
        image: str = 'hiveframe:latest'
    )

class ResourceRequirements:
    def __init__(
        self,
        cpu: str = '1',
        memory: str = '2Gi',
        gpu: int = 0
    )
```

### HiveOperator

Kubernetes operator for managing HiveFrame clusters.

```python
class HiveOperator:
    def __init__(self, config: OperatorConfig = None)
    
    def deploy(self, cluster: HiveCluster) -> None
    def scale(self, cluster_name: str, replicas: int) -> None
    def delete(self, cluster_name: str) -> None
    def status(self, cluster_name: str) -> ClusterStatus
```

### Manifest Generation

```python
def generate_deployment(cluster: HiveCluster) -> Dict
def generate_service(cluster: HiveCluster) -> Dict
def generate_configmap(cluster: HiveCluster, config: Dict) -> Dict
def generate_crd() -> Dict
```

## Dashboard Module

### Dashboard

Web UI for monitoring HiveFrame clusters.

```python
class Dashboard:
    def __init__(self, port: int = 8080, config: DashboardConfig = None)
    
    def start(self) -> None
    def stop(self) -> None
    
    @property
    def url(self) -> str

class DashboardConfig:
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 8080,
        refresh_interval: float = 1.0,
        enable_api: bool = True
    )
```

### Dashboard Components

```python
class ColonyMetricsPanel:
    """Displays colony temperature, pheromone levels, throughput"""
    pass

class WorkerStatusPanel:
    """Shows worker roles, status, and load distribution"""
    pass

class DanceFloorPanel:
    """Visualizes waggle dance activity"""
    pass

class QueryHistoryPanel:
    """Tracks query execution history"""
    pass
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

class BackoffStrategy(Enum):
    CONSTANT = 'constant'
    LINEAR = 'linear'
    EXPONENTIAL = 'exponential'

@with_retry(policy: RetryPolicy)
def decorated_function(): ...
```

### CircuitBreaker

```python
class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig)
    def call(self, fn: Callable[[], T]) -> T
    
    @property
    def state(self) -> CircuitState  # CLOSED, OPEN, HALF_OPEN

class CircuitBreakerConfig:
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 30.0
    )

@circuit_breaker(name: str, config: CircuitBreakerConfig = None)
def decorated_function(): ...
```

### Bulkhead

```python
class Bulkhead:
    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_wait: float = 0.0
    )
    
    def execute(self, fn: Callable[[], T]) -> T
```

### Timeout

```python
def with_timeout(seconds: float)
def timeout_decorator(seconds: float)

class TimeoutConfig:
    def __init__(self, timeout: float, cancel_on_timeout: bool = True)
```

### ResilientExecutor

Combines multiple resilience patterns.

```python
class ResilientExecutor:
    def __init__(
        self,
        retry_policy: RetryPolicy = None,
        circuit_breaker: CircuitBreaker = None,
        bulkhead: Bulkhead = None,
        timeout: float = None
    )
    
    def execute(self, fn: Callable[[], T]) -> T
```

## Connectors Module

### Data Sources

```python
class CSVSource(DataSource):
    def __init__(self, path: str, header: bool = True)
    def read(self) -> Iterator[Dict]

class JSONSource(DataSource):
    def __init__(self, path: str)
    def read(self) -> Iterator[Dict]

class JSONLSource(DataSource):
    def __init__(self, path: str)
    def read(self) -> Iterator[Dict]

class HTTPSource(DataSource):
    def __init__(self, url: str, method: str = 'GET', headers: Dict = None)
    def read(self) -> Iterator[Dict]

class DataGenerator(DataSource):
    def __init__(
        self,
        count: int,
        rate: float = None,  # records per second
        error_rate: float = 0.0,
        schema: Dict[str, str] = None
    )
    def read(self) -> Iterator[Dict]
```

### Data Sinks

```python
class CSVSink(DataSink):
    def __init__(self, path: str, header: bool = True)
    def write(self, record: Dict) -> None

class JSONLSink(DataSink):
    def __init__(self, path: str)
    def write(self, record: Dict) -> None
```

### Message Broker

```python
class MessageBroker:
    def create_topic(self, name: str, num_partitions: int = 4) -> Topic
    def get_topic(self, name: str) -> Optional[Topic]
    def delete_topic(self, name: str) -> None
    
class Topic:
    def produce(self, key: str, value: Any) -> None
    def consume(
        self, 
        group_id: str, 
        partition: int = None,
        max_messages: int = 100
    ) -> List[Message]
    def get_lag(self, group_id: str) -> Dict[int, int]

class FileWatcher(DataSource):
    def __init__(self, path: str, pattern: str = '*')
    def watch(self) -> Iterator[str]  # Yields new file paths
```

## Monitoring Module

### Metrics

```python
class Counter:
    def inc(self, value: float = 1.0)
    def get(self) -> float
    
class Gauge:
    def set(self, value: float)
    def inc(self, value: float = 1.0)
    def dec(self, value: float = 1.0)
    def get(self) -> float
    
class Histogram:
    def observe(self, value: float)
    def get_percentile(self, p: float) -> float
    
class Summary:
    def observe(self, value: float)
    def get_quantiles(self) -> Dict[float, float]
```

### MetricsRegistry

```python
class MetricsRegistry:
    def counter(self, name: str, labels: Dict = None) -> Counter
    def gauge(self, name: str, labels: Dict = None) -> Gauge
    def histogram(self, name: str, buckets: List[float] = None) -> Histogram
    def summary(self, name: str, quantiles: List[float] = None) -> Summary
    
    def to_prometheus(self) -> str
    def to_json(self) -> Dict

def get_registry() -> MetricsRegistry
```

### Logger

```python
class LogLevel(Enum):
    DEBUG = 'debug'
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

class Logger:
    def debug(self, message: str, **context)
    def info(self, message: str, **context)
    def warning(self, message: str, **context)
    def error(self, message: str, **context)
    def critical(self, message: str, **context)

def get_logger(name: str = "hiveframe") -> Logger
```

### Tracing

```python
class Tracer:
    def start_span(self, name: str, parent: Span = None) -> Span
    def inject(self, span: Span, carrier: Dict) -> None
    def extract(self, carrier: Dict) -> Optional[Span]

class Span:
    def set_tag(self, key: str, value: Any) -> None
    def log(self, message: str, **fields) -> None
    def finish(self) -> None

def get_tracer() -> Tracer
```

### Profiling

```python
class PerformanceProfiler:
    def start(self, operation: str) -> ProfileContext
    def get_stats(self, operation: str) -> Dict
    def get_all_stats(self) -> Dict[str, Dict]

def get_profiler() -> PerformanceProfiler
```

### Health Monitoring

```python
class ColonyHealthMonitor:
    def __init__(self, colony: ColonyState)
    
    def get_health(self) -> HealthSnapshot
    def get_worker_health(self, worker_id: str) -> WorkerHealth
    def is_healthy(self) -> bool

class HealthSnapshot:
    timestamp: float
    colony_temperature: float
    worker_count: int
    active_workers: int
    throttle_level: float
    alarm_level: float
```

## Exceptions

```python
# Base exception
class HiveFrameError(Exception):
    severity: ErrorSeverity
    category: ErrorCategory
    
# Severity levels
class ErrorSeverity(Enum):
    DEBUG = 'debug'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'
    FATAL = 'fatal'

# Error categories
class ErrorCategory(Enum):
    TRANSIENT = 'transient'
    DATA_QUALITY = 'data_quality'
    CONFIGURATION = 'configuration'
    RESOURCE = 'resource'
    EXTERNAL = 'external'

# Specific exceptions
class TransientError(HiveFrameError): pass  # Retryable errors
class ValidationError(HiveFrameError):      # Data validation
    field: str
    expected: Any
    actual: Any
class NetworkError(TransientError): pass    # Network failures
class RateLimitError(TransientError): pass  # Rate limiting
class ConnectionError(TransientError): pass # Connection failures
class SchemaError(HiveFrameError): pass     # Schema mismatches
class ParseError(HiveFrameError): pass      # Parsing failures
class ResourceExhaustedError(HiveFrameError): pass  # OOM, etc.
class ConfigurationError(HiveFrameError): pass      # Config issues
class ExternalServiceError(HiveFrameError): pass    # External deps
class CircuitOpenError(HiveFrameError): pass        # Circuit breaker
class TimeoutError(HiveFrameError): pass            # Operation timeout

# Dead Letter Queue
class DeadLetterQueue:
    def __init__(self, max_size: int = 1000)
    def push(self, record: DeadLetterRecord) -> None
    def pop(self) -> Optional[DeadLetterRecord]
    def get_stats(self) -> Dict

class DeadLetterRecord:
    original_data: Any
    error: Exception
    partition_id: str
    worker_id: str
    attempt_count: int
    first_failure: float
```
