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

from .streaming import (
    HiveStream,
    AsyncHiveStream,
    StreamRecord,
    StreamPartitioner,
    StreamBuffer,
    StreamBee,
)

# Production modules
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

from .resilience import (
    RetryPolicy,
    BackoffStrategy,
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    Bulkhead,
    ResilientExecutor,
    with_retry,
    with_circuit_breaker,
    with_timeout,
)

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

from .streaming_enhanced import (
    EnhancedStreamProcessor,
    Window,
    WindowAssigner,
    TumblingWindowAssigner,
    SlidingWindowAssigner,
    SessionWindowAssigner,
    Watermark,
    WatermarkGenerator,
    BoundedOutOfOrdernessWatermarkGenerator,
    DeliveryGuarantee,
    Checkpoint,
    StateBackend,
    InMemoryStateBackend,
    IdempotencyStore,
    tumbling_window,
    sliding_window,
    session_window,
    bounded_watermark,
    sum_aggregator,
    count_aggregator,
    avg_aggregator,
    min_aggregator,
    max_aggregator,
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
    # Streaming
    'HiveStream',
    'AsyncHiveStream',
    'StreamRecord',
    'StreamPartitioner',
    'StreamBuffer',
    'StreamBee',
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
    'CircuitBreaker',
    'CircuitState',
    'CircuitBreakerConfig',
    'Bulkhead',
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
    # Enhanced Streaming
    'EnhancedStreamProcessor',
    'Window',
    'WindowAssigner',
    'TumblingWindowAssigner',
    'SlidingWindowAssigner',
    'SessionWindowAssigner',
    'Watermark',
    'WatermarkGenerator',
    'BoundedOutOfOrdernessWatermarkGenerator',
    'DeliveryGuarantee',
    'Checkpoint',
    'StateBackend',
    'InMemoryStateBackend',
    'IdempotencyStore',
    'tumbling_window',
    'sliding_window',
    'session_window',
    'bounded_watermark',
    'sum_aggregator',
    'count_aggregator',
    'avg_aggregator',
    'min_aggregator',
    'max_aggregator',
]
