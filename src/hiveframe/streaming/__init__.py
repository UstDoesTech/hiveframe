"""
HiveFrame Unified Streaming Package
====================================
Real-time stream processing using bee-inspired patterns.

Combines basic streaming and enhanced production-grade features:
- Continuous foraging with adaptive throttling
- Windowing (tumbling, sliding, session)
- Watermarks and late data handling
- Exactly-once delivery guarantees
- Checkpointing and state management

Phase 3 Real-Time Analytics:
- Structured Streaming 2.0: Sub-millisecond latency streaming
- Complex Event Processing (CEP): Pattern detection in streaming data
- Materialized Views: Automatically maintained aggregate tables
- Change Data Capture (CDC): Database replication and synchronization
"""

# Basic streaming components
# Aggregators
from .aggregators import (
    avg_aggregator,
    collect_aggregator,
    count_aggregator,
    max_aggregator,
    min_aggregator,
    sum_aggregator,
)

# Phase 3: Change Data Capture (CDC)
from .cdc import (
    CaptureMode,
    CDCReplicator,
    CDCStream,
    ChangeCapture,
    ChangeEvent,
    ChangeLog,
    ChangeType,
    ConflictResolution,
    InMemoryCapture,
    QueryBasedCapture,
    ReplicationMode,
    TableCheckpoint,
    create_cdc_stream,
    create_replicator,
)

# Phase 3: Complex Event Processing (CEP)
from .cep import (
    CEPEngine,
    ContiguityType,
    Pattern,
    PatternCondition,
    PatternElement,
    PatternMatch,
    PatternMatcher,
    PatternState,
    QuantifierType,
    begin,
    condition,
    pattern,
)
from .core import (
    AsyncHiveStream,
    HiveStream,
    StreamBee,
    StreamBuffer,
    StreamPartitioner,
    StreamRecord,
)

# Phase 3: Materialized Views
from .materialized_views import (
    IncrementalDelta,
    MaterializedView,
    MaterializedViewManager,
    RefreshStrategy,
    ViewChange,
    ViewMetadata,
    ViewState,
    create_materialized_view,
)

# Phase 3: Structured Streaming 2.0
from .realtime import (
    AdaptiveMicroBatcher,
    AsyncStructuredStreaming2,
    LatencyMetrics,
    LockFreeQueue,
    PriorityLevel,
    PriorityQueue,
    ProcessingMode,
    StreamingRecord,
    StructuredStreaming2,
)

# Delivery guarantees
from .delivery import (
    DeliveryGuarantee,
    IdempotencyStore,
    ProcessingContext,
)

# Enhanced processor
from .processor import (
    EnhancedStreamProcessor,
    WindowAggregation,
)

# State management
from .state import (
    Checkpoint,
    InMemoryStateBackend,
    StateBackend,
)

# Watermarks
from .watermarks import (
    BoundedOutOfOrdernessWatermarkGenerator,
    PunctuatedWatermarkGenerator,
    Watermark,
    WatermarkGenerator,
    bounded_watermark,
)

# Windowing
from .windows import (
    SessionWindowAssigner,
    SlidingWindowAssigner,
    TumblingWindowAssigner,
    Window,
    WindowAssigner,
    WindowedValue,
    WindowType,
    session_window,
    sliding_window,
    tumbling_window,
)

__all__ = [
    # Core
    "StreamRecord",
    "StreamPartitioner",
    "StreamBuffer",
    "StreamBee",
    "HiveStream",
    "AsyncHiveStream",
    # Windows
    "WindowType",
    "Window",
    "WindowedValue",
    "WindowAssigner",
    "TumblingWindowAssigner",
    "SlidingWindowAssigner",
    "SessionWindowAssigner",
    "tumbling_window",
    "sliding_window",
    "session_window",
    # Watermarks
    "Watermark",
    "WatermarkGenerator",
    "BoundedOutOfOrdernessWatermarkGenerator",
    "PunctuatedWatermarkGenerator",
    "bounded_watermark",
    # State
    "Checkpoint",
    "StateBackend",
    "InMemoryStateBackend",
    # Delivery
    "DeliveryGuarantee",
    "ProcessingContext",
    "IdempotencyStore",
    # Processor
    "WindowAggregation",
    "EnhancedStreamProcessor",
    # Aggregators
    "count_aggregator",
    "sum_aggregator",
    "avg_aggregator",
    "max_aggregator",
    "min_aggregator",
    "collect_aggregator",
    # Phase 3: Structured Streaming 2.0
    "StructuredStreaming2",
    "AsyncStructuredStreaming2",
    "StreamingRecord",
    "ProcessingMode",
    "PriorityLevel",
    "LatencyMetrics",
    "LockFreeQueue",
    "PriorityQueue",
    "AdaptiveMicroBatcher",
    # Phase 3: Complex Event Processing (CEP)
    "CEPEngine",
    "Pattern",
    "PatternElement",
    "PatternCondition",
    "PatternMatch",
    "PatternMatcher",
    "PatternState",
    "QuantifierType",
    "ContiguityType",
    "pattern",
    "begin",
    "condition",
    # Phase 3: Materialized Views
    "MaterializedView",
    "MaterializedViewManager",
    "ViewMetadata",
    "ViewState",
    "ViewChange",
    "RefreshStrategy",
    "IncrementalDelta",
    "create_materialized_view",
    # Phase 3: Change Data Capture (CDC)
    "CDCStream",
    "CDCReplicator",
    "ChangeEvent",
    "ChangeLog",
    "ChangeType",
    "ChangeCapture",
    "CaptureMode",
    "ReplicationMode",
    "ConflictResolution",
    "TableCheckpoint",
    "InMemoryCapture",
    "QueryBasedCapture",
    "create_cdc_stream",
    "create_replicator",
]
