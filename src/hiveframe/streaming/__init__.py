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
from .core import (
    AsyncHiveStream,
    HiveStream,
    StreamBee,
    StreamBuffer,
    StreamPartitioner,
    StreamRecord,
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
]
