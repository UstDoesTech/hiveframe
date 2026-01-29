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

# Utilities (new unified module)
from .utils import (
    ThreadSafeMixin,
    ManagedResource,
    Serializable,
    Monitorable,
    deprecated_param,
)

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

# Unified Aggregations (new module)
from .aggregations import (
    Aggregator,
    CountAggregator,
    SumAggregator,
    AvgAggregator,
    MinAggregator,
    MaxAggregator,
    CollectListAggregator,
    CollectSetAggregator,
    get_aggregator_registry,
)

# Dead Letter Queue (extracted from exceptions)
from .dlq import (
    DeadLetterQueue,
    DeadLetterRecord,
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

# SQL Engine (SwarmQL) - Phase 1
from .sql import (
    SwarmQLContext,
    SQLCatalog,
    SQLParser,
    SQLStatement,
    SQLTokenizer,
    SQLExecutor,
    QueryPlan,
    PlanNode,
)

# Query Optimizer - Phase 1
from .optimizer import (
    QueryOptimizer,
    OptimizedPlan,
    PlanCandidate,
    CostModel,
    SwarmCostModel,
    CostEstimate,
    Statistics,
    OptimizationRule,
    PredicatePushdown,
    ProjectionPruning,
)

# Storage (Parquet & Delta Lake) - Phase 1
from .storage import (
    ParquetReader,
    ParquetWriter,
    ParquetSchema,
    read_parquet,
    write_parquet,
    DeltaTable,
    DeltaTransaction,
    DeltaLog,
    read_delta,
    write_delta,
    FileFormat,
    StorageOptions,
    CompressionCodec,
)

# Kubernetes Support - Phase 1
from .k8s import (
    HiveCluster,
    WorkerSpec,
    ResourceRequirements,
    ClusterStatus,
    HiveOperator,
    OperatorConfig,
    generate_deployment,
    generate_service,
    generate_configmap,
    generate_crd,
)

# Dashboard (Colony Dashboard) - Phase 1
from .dashboard import (
    Dashboard,
    DashboardConfig,
    DashboardAPI,
)

__all__ = [
    # Utilities (new)
    'ThreadSafeMixin',
    'ManagedResource',
    'Serializable',
    'Monitorable',
    'deprecated_param',
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
    # Aggregations (unified)
    'Aggregator',
    'CountAggregator',
    'SumAggregator',
    'AvgAggregator',
    'MinAggregator',
    'MaxAggregator',
    'CollectListAggregator',
    'CollectSetAggregator',
    'get_aggregator_registry',
    # DataFrame
    'HiveDataFrame',
    'Column',
    'col',
    'lit',
    'Schema',
    'DataType',
    'GroupedData',
    'createDataFrame',
    # DataFrame Aggregations (convenience)
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
    # Streaming - Aggregators (legacy functions)
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
    # Dead Letter Queue
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
    # SQL Engine (SwarmQL)
    'SwarmQLContext',
    'SQLCatalog',
    'SQLParser',
    'SQLStatement',
    'SQLTokenizer',
    'SQLExecutor',
    'QueryPlan',
    'PlanNode',
    # Query Optimizer
    'QueryOptimizer',
    'OptimizedPlan',
    'PlanCandidate',
    'CostModel',
    'SwarmCostModel',
    'CostEstimate',
    'Statistics',
    'OptimizationRule',
    'PredicatePushdown',
    'ProjectionPruning',
    # Storage (Parquet & Delta Lake)
    'ParquetReader',
    'ParquetWriter',
    'ParquetSchema',
    'read_parquet',
    'write_parquet',
    'DeltaTable',
    'DeltaTransaction',
    'DeltaLog',
    'read_delta',
    'write_delta',
    'FileFormat',
    'StorageOptions',
    'CompressionCodec',
    # Kubernetes Support
    'HiveCluster',
    'WorkerSpec',
    'ResourceRequirements',
    'ClusterStatus',
    'HiveOperator',
    'OperatorConfig',
    'generate_deployment',
    'generate_service',
    'generate_configmap',
    'generate_crd',
    # Dashboard
    'Dashboard',
    'DashboardConfig',
    'DashboardAPI',
]
