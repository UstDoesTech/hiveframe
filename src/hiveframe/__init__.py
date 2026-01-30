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

__version__ = "0.2.0-dev"
__author__ = "HiveFrame Contributors"

# Utilities (new unified module)
# Unified Aggregations (new module)
from .aggregations import (
    Aggregator,
    AvgAggregator,
    CollectListAggregator,
    CollectSetAggregator,
    CountAggregator,
    MaxAggregator,
    MinAggregator,
    SumAggregator,
    get_aggregator_registry,
)

# Connectors
from .connectors import (
    CSVSink,
    CSVSource,
    DataGenerator,
    DataSink,
    DataSource,
    FileWatcher,
    HTTPSource,
    JSONLSink,
    JSONLSource,
    JSONSource,
    MessageBroker,
    Topic,
)

# Core bee-inspired components
from .core import (
    Bee,
    BeeRole,
    ColonyState,
    DanceFloor,
    FoodSource,
    HiveFrame,
    Pheromone,
    WaggleDance,
    create_hive,
)

# Dashboard (Colony Dashboard) - Phase 1
from .dashboard import (
    Dashboard,
    DashboardAPI,
    DashboardConfig,
)

# DataFrame API (from package)
from .dataframe import (
    Column,
    DataType,
    GroupedData,
    HiveDataFrame,
    Schema,
    avg,
    col,
    collect_list,
    collect_set,
    count,
    count_all,
    createDataFrame,
    lit,
    max_agg,
    min_agg,
    # Aggregation functions
    sum_agg,
)

# Phase 2: Distributed Execution Engine
from .distributed import (
    # Adaptive Partitioning
    AdaptivePartitioner,
    CrossDatacenterManager,
    DataLocality,
    FederatedHive,
    FederationCoordinator,
    FederationProtocol,
    FitnessPartitioner,
    # Federation
    HiveFederation,
    HiveHealth,
    HiveRegistry,
    # Locality
    LocalityAwareScheduler,
    LocalityHint,
    LocalityLevel,
    PartitionMerger,
    PartitionSplitter,
    PartitionState,
    PartitionStrategy,
    ScoutTaskRunner,
    SlowTaskDetector,
    SpeculativeConfig,
    # Speculative Execution
    SpeculativeExecutor,
    TaskTracker,
)

# Dead Letter Queue (extracted from exceptions)
from .dlq import (
    DeadLetterQueue,
    DeadLetterRecord,
)

# Exceptions
from .exceptions import (
    CircuitOpenError,
    ConfigurationError,
    DeadLetterQueue,
    DeadLetterRecord,
    DependencyError,
    HiveFrameError,
    NetworkError,
    ParseError,
    ResourceError,
    SchemaError,
    TimeoutError,
    TransientError,
    ValidationError,
)

# Kubernetes Support - Phase 1
from .k8s import (
    ClusterStatus,
    HiveCluster,
    HiveOperator,
    OperatorConfig,
    ResourceRequirements,
    WorkerSpec,
    generate_configmap,
    generate_crd,
    generate_deployment,
    generate_service,
)

# Monitoring
from .monitoring import (
    ColonyHealthMonitor,
    Counter,
    Gauge,
    Histogram,
    Logger,
    LogLevel,
    MetricsRegistry,
    PerformanceProfiler,
    Summary,
    Tracer,
    TraceSpan,
    get_logger,
    get_profiler,
    get_registry,
    get_tracer,
)

# Query Optimizer - Phase 1
# Phase 2: Advanced Query Engine (Vectorized + AQE)
from .optimizer import (
    # Adaptive Query Execution
    AdaptiveQueryExecutor,
    AQEContext,
    CostEstimate,
    CostModel,
    JoinStrategy,
    OptimizationRule,
    OptimizedPlan,
    ParallelVectorizedExecutor,
    PlanCandidate,
    PredicatePushdown,
    ProjectionPruning,
    QueryOptimizer,
    RuntimeStatistics,
    Statistics,
    SwarmCostModel,
    # Vectorized Execution
    VectorBatch,
    VectorizedAggregate,
    VectorizedFilter,
    VectorizedJoin,
    VectorizedPipeline,
    VectorizedProject,
    VectorType,
    WaggleDanceFeedback,
)

# Resilience patterns (from package)
from .resilience import (
    BackoffStrategy,
    Bulkhead,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ResilientExecutor,
    RetryContext,
    RetryPolicy,
    RetryState,
    TimeoutWrapper,
    with_circuit_breaker,
    with_retry,
    with_timeout,
)

# SQL Engine (SwarmQL) - Phase 1
from .sql import (
    PlanNode,
    QueryPlan,
    SQLCatalog,
    SQLExecutor,
    SQLParser,
    SQLStatement,
    SQLTokenizer,
    SwarmQLContext,
)

# Storage (Parquet & Delta Lake) - Phase 1
# Phase 2: Storage Layer (HoneyStore, Iceberg, Caching)
from .storage import (
    # Caching Swarm
    CachingSwarm,
    CompressionCodec,
    DeltaLog,
    DeltaTable,
    DeltaTransaction,
    FileFormat,
    HoneyStoreMetadata,
    HoneyStoreReader,
    # HoneyStore
    HoneyStoreWriter,
    IcebergField,
    IcebergSchema,
    # Iceberg
    IcebergTable,
    ParquetReader,
    ParquetSchema,
    ParquetWriter,
    PheromoneCache,
    StorageOptions,
    SwarmPrefetcher,
    read_delta,
    read_honeystore,
    read_iceberg,
    read_parquet,
    write_delta,
    write_honeystore,
    write_iceberg,
    write_parquet,
)

# Unified Streaming (from package - combines basic and enhanced)
from .streaming import (
    AsyncHiveStream,
    BoundedOutOfOrdernessWatermarkGenerator,
    # Enhanced streaming - state
    Checkpoint,
    # Enhanced streaming - delivery
    DeliveryGuarantee,
    EnhancedStreamProcessor,
    # Core streaming
    HiveStream,
    IdempotencyStore,
    InMemoryStateBackend,
    ProcessingContext,
    PunctuatedWatermarkGenerator,
    SessionWindowAssigner,
    SlidingWindowAssigner,
    StateBackend,
    StreamBee,
    StreamBuffer,
    StreamPartitioner,
    StreamRecord,
    TumblingWindowAssigner,
    # Enhanced streaming - watermarks
    Watermark,
    WatermarkGenerator,
    # Enhanced streaming - windowing
    Window,
    # Enhanced streaming - processor
    WindowAggregation,
    WindowAssigner,
    WindowedValue,
    WindowType,
    avg_aggregator,
    bounded_watermark,
    collect_aggregator,
    # Enhanced streaming - aggregators
    count_aggregator,
    max_aggregator,
    min_aggregator,
    session_window,
    sliding_window,
    sum_aggregator,
    tumbling_window,
)
from .utils import (
    ManagedResource,
    Monitorable,
    Serializable,
    ThreadSafeMixin,
    deprecated_param,
)

__all__ = [
    # Utilities (new)
    "ThreadSafeMixin",
    "ManagedResource",
    "Serializable",
    "Monitorable",
    "deprecated_param",
    # Core
    "HiveFrame",
    "Bee",
    "BeeRole",
    "WaggleDance",
    "DanceFloor",
    "ColonyState",
    "Pheromone",
    "FoodSource",
    "create_hive",
    # Aggregations (unified)
    "Aggregator",
    "CountAggregator",
    "SumAggregator",
    "AvgAggregator",
    "MinAggregator",
    "MaxAggregator",
    "CollectListAggregator",
    "CollectSetAggregator",
    "get_aggregator_registry",
    # DataFrame
    "HiveDataFrame",
    "Column",
    "col",
    "lit",
    "Schema",
    "DataType",
    "GroupedData",
    "createDataFrame",
    # DataFrame Aggregations (convenience)
    "sum_agg",
    "avg",
    "count",
    "count_all",
    "min_agg",
    "max_agg",
    "collect_list",
    "collect_set",
    # Streaming - Core
    "HiveStream",
    "AsyncHiveStream",
    "StreamRecord",
    "StreamPartitioner",
    "StreamBuffer",
    "StreamBee",
    # Streaming - Windows
    "Window",
    "WindowType",
    "WindowedValue",
    "WindowAssigner",
    "TumblingWindowAssigner",
    "SlidingWindowAssigner",
    "SessionWindowAssigner",
    "tumbling_window",
    "sliding_window",
    "session_window",
    # Streaming - Watermarks
    "Watermark",
    "WatermarkGenerator",
    "BoundedOutOfOrdernessWatermarkGenerator",
    "PunctuatedWatermarkGenerator",
    "bounded_watermark",
    # Streaming - State
    "Checkpoint",
    "StateBackend",
    "InMemoryStateBackend",
    # Streaming - Delivery
    "DeliveryGuarantee",
    "ProcessingContext",
    "IdempotencyStore",
    # Streaming - Processor
    "WindowAggregation",
    "EnhancedStreamProcessor",
    # Streaming - Aggregators (legacy functions)
    "count_aggregator",
    "sum_aggregator",
    "avg_aggregator",
    "max_aggregator",
    "min_aggregator",
    "collect_aggregator",
    # Exceptions
    "HiveFrameError",
    "TransientError",
    "NetworkError",
    "TimeoutError",
    "ValidationError",
    "SchemaError",
    "ParseError",
    "ResourceError",
    "ConfigurationError",
    "DependencyError",
    "CircuitOpenError",
    # Dead Letter Queue
    "DeadLetterQueue",
    "DeadLetterRecord",
    # Resilience
    "RetryPolicy",
    "BackoffStrategy",
    "RetryState",
    "RetryContext",
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerConfig",
    "Bulkhead",
    "TimeoutWrapper",
    "ResilientExecutor",
    "with_retry",
    "with_circuit_breaker",
    "with_timeout",
    # Connectors
    "DataSource",
    "DataSink",
    "CSVSource",
    "JSONSource",
    "JSONLSource",
    "CSVSink",
    "JSONLSink",
    "FileWatcher",
    "HTTPSource",
    "MessageBroker",
    "Topic",
    "DataGenerator",
    # Monitoring
    "MetricsRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "Logger",
    "LogLevel",
    "ColonyHealthMonitor",
    "Tracer",
    "TraceSpan",
    "PerformanceProfiler",
    "get_registry",
    "get_logger",
    "get_tracer",
    "get_profiler",
    # SQL Engine (SwarmQL)
    "SwarmQLContext",
    "SQLCatalog",
    "SQLParser",
    "SQLStatement",
    "SQLTokenizer",
    "SQLExecutor",
    "QueryPlan",
    "PlanNode",
    # Query Optimizer
    "QueryOptimizer",
    "OptimizedPlan",
    "PlanCandidate",
    "CostModel",
    "SwarmCostModel",
    "CostEstimate",
    "Statistics",
    "OptimizationRule",
    "PredicatePushdown",
    "ProjectionPruning",
    # Storage (Parquet & Delta Lake)
    "ParquetReader",
    "ParquetWriter",
    "ParquetSchema",
    "read_parquet",
    "write_parquet",
    "DeltaTable",
    "DeltaTransaction",
    "DeltaLog",
    "read_delta",
    "write_delta",
    "FileFormat",
    "StorageOptions",
    "CompressionCodec",
    # Kubernetes Support
    "HiveCluster",
    "WorkerSpec",
    "ResourceRequirements",
    "ClusterStatus",
    "HiveOperator",
    "OperatorConfig",
    "generate_deployment",
    "generate_service",
    "generate_configmap",
    "generate_crd",
    # Dashboard
    "Dashboard",
    "DashboardConfig",
    "DashboardAPI",
    # Phase 2: Distributed Execution Engine
    "HiveFederation",
    "FederatedHive",
    "FederationCoordinator",
    "HiveRegistry",
    "FederationProtocol",
    "HiveHealth",
    "AdaptivePartitioner",
    "PartitionStrategy",
    "PartitionState",
    "FitnessPartitioner",
    "PartitionSplitter",
    "PartitionMerger",
    "SpeculativeExecutor",
    "ScoutTaskRunner",
    "TaskTracker",
    "SlowTaskDetector",
    "SpeculativeConfig",
    "LocalityAwareScheduler",
    "DataLocality",
    "LocalityLevel",
    "LocalityHint",
    "CrossDatacenterManager",
    # Phase 2: Vectorized Execution
    "VectorBatch",
    "VectorType",
    "VectorizedPipeline",
    "VectorizedFilter",
    "VectorizedProject",
    "VectorizedAggregate",
    "VectorizedJoin",
    "ParallelVectorizedExecutor",
    # Phase 2: Adaptive Query Execution
    "AdaptiveQueryExecutor",
    "AQEContext",
    "WaggleDanceFeedback",
    "RuntimeStatistics",
    "JoinStrategy",
    # Phase 2: HoneyStore
    "HoneyStoreWriter",
    "HoneyStoreReader",
    "HoneyStoreMetadata",
    "write_honeystore",
    "read_honeystore",
    # Phase 2: Iceberg
    "IcebergTable",
    "IcebergSchema",
    "IcebergField",
    "read_iceberg",
    "write_iceberg",
    # Phase 2: Caching Swarm
    "CachingSwarm",
    "PheromoneCache",
    "SwarmPrefetcher",
]
