"""
HiveFrame: Bee-Inspired Data Processing Framework
================================================

A biomimetic alternative to Apache Spark that uses bee colony
intelligence patterns for distributed data processing.

Key Features:
- Decentralized coordination (no driver bottleneck)
- Adaptive load balancing through waggle dance protocol
- Self-healing through ABC abandonment mechanism
- Quality-weighted task reinforcement
- Pheromone-based backpressure

Quick Start:
    from hiveframe import HiveFrame, HiveDataFrame, col
    
    # RDD-style processing
    hive = HiveFrame(num_workers=8)
    results = hive.map(data, lambda x: x * 2)
    
    # DataFrame API
    df = HiveDataFrame.from_csv('data.csv')
    result = df.filter(col('age') > 21).select('name', 'age')
    result.show()
    
    # Streaming
    stream = HiveStream(num_workers=4)
    stream.start(process_fn)
    stream.submit('key', data)

Biomimicry Concepts:
- Waggle Dance: Workers advertise task quality through dance signals
- Three-Tier Workers: Employed (exploit), Onlooker (reinforce), Scout (explore)
- Stigmergic Coordination: Indirect communication through shared state
- Quorum Consensus: Decisions emerge from local interactions
- Adaptive Allocation: Self-organizing based on local stimuli
"""

__version__ = '0.2.0-dev'
__author__ = 'HiveFrame Contributors'

# Core bee-inspired components
from .core import (
    HiveFrame,
    Bee,
    BeeRole,
    WaggleDance,
    DanceFloor,
    ColonyState,
    Pheromone,
    FoodSource,
    create_hive,
)

# DataFrame API (from package)
from .dataframe import (
    HiveDataFrame,
    Column,
    col,
    lit,
    Schema,
    DataType,
    GroupedData,
    createDataFrame,
    # Aggregation functions
    sum_agg,
    avg,
    count,
    count_all,
    min_agg,
    max_agg,
    collect_list,
    collect_set,
)

# Unified Streaming (from package - combines basic and enhanced)
from .streaming import (
    # Core streaming
    HiveStream,
    AsyncHiveStream,
    StreamRecord,
    StreamPartitioner,
    StreamBuffer,
    StreamBee,
    # Enhanced streaming - windowing
    Window,
    WindowType,
    WindowedValue,
    WindowAssigner,
    TumblingWindowAssigner,
    SlidingWindowAssigner,
    SessionWindowAssigner,
    tumbling_window,
    sliding_window,
    session_window,
    # Enhanced streaming - watermarks
    Watermark,
    WatermarkGenerator,
    BoundedOutOfOrdernessWatermarkGenerator,
    PunctuatedWatermarkGenerator,
    bounded_watermark,
    # Enhanced streaming - state
    Checkpoint,
    StateBackend,
    InMemoryStateBackend,
    # Enhanced streaming - delivery
    DeliveryGuarantee,
    ProcessingContext,
    IdempotencyStore,
    # Enhanced streaming - processor
    WindowAggregation,
    EnhancedStreamProcessor,
    # Enhanced streaming - aggregators
    count_aggregator,
    sum_aggregator,
    avg_aggregator,
    max_aggregator,
    min_aggregator,
    collect_aggregator,
)

# Exceptions
from .exceptions import (
    HiveFrameError,
    TransientError,
    NetworkError,
    TimeoutError,
    ValidationError,
    SchemaError,
    ParseError,
    ResourceError,
    ConfigurationError,
    DependencyError,
    CircuitOpenError,
    DeadLetterQueue,
    DeadLetterRecord,
)

# Resilience patterns (from package)
from .resilience import (
    RetryPolicy,
    BackoffStrategy,
    RetryState,
    RetryContext,
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    Bulkhead,
    TimeoutWrapper,
    ResilientExecutor,
    with_retry,
    with_circuit_breaker,
    with_timeout,
)

# Connectors
from .connectors import (
    DataSource,
    DataSink,
    CSVSource,
    JSONSource,
    JSONLSource,
    CSVSink,
    JSONLSink,
    FileWatcher,
    HTTPSource,
    MessageBroker,
    Topic,
    DataGenerator,
)

# Monitoring
from .monitoring import (
    MetricsRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    Logger,
    LogLevel,
    ColonyHealthMonitor,
    Tracer,
    TraceSpan,
    PerformanceProfiler,
    get_registry,
    get_logger,
    get_tracer,
    get_profiler,
)

__all__ = [
    # Core
    'HiveFrame',
    'Bee',
    'BeeRole',
    'WaggleDance',
    'DanceFloor',
    'ColonyState',
    'Pheromone',
    'FoodSource',
    'create_hive',
    # DataFrame
    'HiveDataFrame',
    'Column',
    'col',
    'lit',
    'Schema',
    'DataType',
    'GroupedData',
    'createDataFrame',
    # Aggregations
    'sum_agg',
    'avg',
    'count',
    'count_all',
    'min_agg',
    'max_agg',
    'collect_list',
    'collect_set',
    # Streaming - Core
    'HiveStream',
    'AsyncHiveStream',
    'StreamRecord',
    'StreamPartitioner',
    'StreamBuffer',
    'StreamBee',
    # Streaming - Windows
    'Window',
    'WindowType',
    'WindowedValue',
    'WindowAssigner',
    'TumblingWindowAssigner',
    'SlidingWindowAssigner',
    'SessionWindowAssigner',
    'tumbling_window',
    'sliding_window',
    'session_window',
    # Streaming - Watermarks
    'Watermark',
    'WatermarkGenerator',
    'BoundedOutOfOrdernessWatermarkGenerator',
    'PunctuatedWatermarkGenerator',
    'bounded_watermark',
    # Streaming - State
    'Checkpoint',
    'StateBackend',
    'InMemoryStateBackend',
    # Streaming - Delivery
    'DeliveryGuarantee',
    'ProcessingContext',
    'IdempotencyStore',
    # Streaming - Processor
    'WindowAggregation',
    'EnhancedStreamProcessor',
    # Streaming - Aggregators
    'count_aggregator',
    'sum_aggregator',
    'avg_aggregator',
    'max_aggregator',
    'min_aggregator',
    'collect_aggregator',
    # Exceptions
    'HiveFrameError',
    'TransientError',
    'NetworkError',
    'TimeoutError',
    'ValidationError',
    'SchemaError',
    'ParseError',
    'ResourceError',
    'ConfigurationError',
    'DependencyError',
    'CircuitOpenError',
    'DeadLetterQueue',
    'DeadLetterRecord',
    # Resilience
    'RetryPolicy',
    'BackoffStrategy',
    'RetryState',
    'RetryContext',
    'CircuitBreaker',
    'CircuitState',
    'CircuitBreakerConfig',
    'Bulkhead',
    'TimeoutWrapper',
    'ResilientExecutor',
    'with_retry',
    'with_circuit_breaker',
    'with_timeout',
    # Connectors
    'DataSource',
    'DataSink',
    'CSVSource',
    'JSONSource',
    'JSONLSource',
    'CSVSink',
    'JSONLSink',
    'FileWatcher',
    'HTTPSource',
    'MessageBroker',
    'Topic',
    'DataGenerator',
    # Monitoring
    'MetricsRegistry',
    'Counter',
    'Gauge',
    'Histogram',
    'Summary',
    'Logger',
    'LogLevel',
    'ColonyHealthMonitor',
    'Tracer',
    'TraceSpan',
    'PerformanceProfiler',
    'get_registry',
    'get_logger',
    'get_tracer',
    'get_profiler',
]
