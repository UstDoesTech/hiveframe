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
from .core import (
    StreamRecord,
    StreamPartitioner,
    StreamBuffer,
    StreamBee,
    HiveStream,
    AsyncHiveStream,
)

# Windowing
from .windows import (
    WindowType,
    Window,
    WindowedValue,
    WindowAssigner,
    TumblingWindowAssigner,
    SlidingWindowAssigner,
    SessionWindowAssigner,
    tumbling_window,
    sliding_window,
    session_window,
)

# Watermarks
from .watermarks import (
    Watermark,
    WatermarkGenerator,
    BoundedOutOfOrdernessWatermarkGenerator,
    PunctuatedWatermarkGenerator,
    bounded_watermark,
)

# State management
from .state import (
    Checkpoint,
    StateBackend,
    InMemoryStateBackend,
)

# Delivery guarantees
from .delivery import (
    DeliveryGuarantee,
    ProcessingContext,
    IdempotencyStore,
)

# Enhanced processor
from .processor import (
    WindowAggregation,
    EnhancedStreamProcessor,
)

# Aggregators
from .aggregators import (
    count_aggregator,
    sum_aggregator,
    avg_aggregator,
    max_aggregator,
    min_aggregator,
    collect_aggregator,
)

__all__ = [
    # Core
    'StreamRecord',
    'StreamPartitioner',
    'StreamBuffer',
    'StreamBee',
    'HiveStream',
    'AsyncHiveStream',
    # Windows
    'WindowType',
    'Window',
    'WindowedValue',
    'WindowAssigner',
    'TumblingWindowAssigner',
    'SlidingWindowAssigner',
    'SessionWindowAssigner',
    'tumbling_window',
    'sliding_window',
    'session_window',
    # Watermarks
    'Watermark',
    'WatermarkGenerator',
    'BoundedOutOfOrdernessWatermarkGenerator',
    'PunctuatedWatermarkGenerator',
    'bounded_watermark',
    # State
    'Checkpoint',
    'StateBackend',
    'InMemoryStateBackend',
    # Delivery
    'DeliveryGuarantee',
    'ProcessingContext',
    'IdempotencyStore',
    # Processor
    'WindowAggregation',
    'EnhancedStreamProcessor',
    # Aggregators
    'count_aggregator',
    'sum_aggregator',
    'avg_aggregator',
    'max_aggregator',
    'min_aggregator',
    'collect_aggregator',
]
