---
sidebar_position: 5
---

# Streaming Module

Real-time stream processing with windows, watermarks, and delivery guarantees.

```python
from hiveframe.streaming import (
    StreamProcessor,
    StreamConfig,
    TumblingWindow,
    SlidingWindow,
    SessionWindow,
    Watermark,
    DeliveryGuarantee
)
```

## StreamProcessor

Core class for building streaming pipelines.

### Class Definition

```python
class StreamProcessor:
    """
    Process continuous data streams with exactly-once semantics.
    
    Supports windowed aggregations, stateful processing,
    and integration with various sources and sinks.
    """
    
    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        colony: Optional[Colony] = None
    ) -> None:
        """
        Initialize stream processor.
        
        Args:
            config: Stream configuration
            colony: Colony for distributed processing
        """
```

### StreamConfig

```python
class StreamConfig:
    """Configuration for stream processing."""
    
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval_ms: int = 10000,
        watermark_delay_ms: int = 5000,
        max_records_per_batch: int = 10000,
        delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.EXACTLY_ONCE
    ) -> None:
        """
        Configure stream processing.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            checkpoint_interval_ms: Checkpoint frequency
            watermark_delay_ms: Default watermark delay
            max_records_per_batch: Batch size limit
            delivery_guarantee: Processing guarantee level
        """
```

### Methods

#### `from_source()`

```python
def from_source(
    self,
    source: Source
) -> "StreamBuilder":
    """
    Create stream from a source.
    
    Args:
        source: Data source (Kafka, file, etc.)
        
    Returns:
        StreamBuilder for pipeline construction
        
    Example:
        stream = StreamProcessor()
        stream.from_source(KafkaSource(
            bootstrap_servers="localhost:9092",
            topic="events"
        ))
    """
```

#### `start()`

```python
def start(self) -> "StreamProcessor":
    """
    Start stream processing.
    
    Returns:
        Self for method chaining
        
    Example:
        processor.start()
        processor.await_termination()
    """
```

#### `stop()`

```python
def stop(self, graceful: bool = True) -> None:
    """
    Stop stream processing.
    
    Args:
        graceful: Wait for in-flight records
        
    Example:
        processor.stop(graceful=True)
    """
```

#### `await_termination()`

```python
def await_termination(
    self,
    timeout: Optional[float] = None
) -> bool:
    """
    Wait for stream to terminate.
    
    Args:
        timeout: Maximum wait time in seconds
        
    Returns:
        True if terminated, False if timeout
        
    Example:
        processor.start()
        if not processor.await_termination(timeout=3600):
            processor.stop()
    """
```

### Properties

```python
@property
def status(self) -> StreamStatus:
    """Current status (CREATED, RUNNING, STOPPED, FAILED)."""

@property
def metrics(self) -> StreamMetrics:
    """Current streaming metrics."""

@property
def exception(self) -> Optional[Exception]:
    """Exception if stream failed."""
```

---

## StreamBuilder

Fluent API for building streaming pipelines.

### Transformation Methods

#### `map()`

```python
def map(
    self,
    func: Callable[[T], U]
) -> "StreamBuilder[U]":
    """
    Transform each record.
    
    Example:
        stream.map(lambda x: x.upper())
    """
```

#### `filter()`

```python
def filter(
    self,
    predicate: Callable[[T], bool]
) -> "StreamBuilder[T]":
    """
    Filter records.
    
    Example:
        stream.filter(lambda x: x["status"] == "active")
    """
```

#### `flatMap()`

```python
def flatMap(
    self,
    func: Callable[[T], Iterable[U]]
) -> "StreamBuilder[U]":
    """
    Transform and flatten.
    
    Example:
        stream.flatMap(lambda x: x["tags"])
    """
```

### Windowing Methods

#### `window()`

```python
def window(
    self,
    window_spec: Window
) -> "WindowedStream":
    """
    Apply windowing.
    
    Args:
        window_spec: Window specification
        
    Returns:
        WindowedStream for aggregation
        
    Example:
        stream.window(TumblingWindow(
            size_ms=60000,
            time_field="event_time"
        ))
    """
```

### Output Methods

#### `to_sink()`

```python
def to_sink(
    self,
    sink: Sink
) -> "StreamBuilder":
    """
    Write to a sink.
    
    Args:
        sink: Output destination
        
    Example:
        stream.to_sink(KafkaSink(
            bootstrap_servers="localhost:9092",
            topic="output"
        ))
    """
```

#### `foreach()`

```python
def foreach(
    self,
    func: Callable[[T], None]
) -> "StreamBuilder":
    """
    Apply side effect to each record.
    
    Example:
        stream.foreach(lambda x: print(x))
    """
```

---

## Window Types

### TumblingWindow

Fixed-size, non-overlapping windows.

```python
class TumblingWindow:
    """
    Fixed-size tumbling window.
    
    Events are assigned to exactly one window.
    
    |---Window 1---|---Window 2---|---Window 3---|
    """
    
    def __init__(
        self,
        size_ms: int,
        time_field: str = "timestamp",
        timezone: str = "UTC"
    ) -> None:
        """
        Create tumbling window.
        
        Args:
            size_ms: Window size in milliseconds
            time_field: Field containing event time
            timezone: Timezone for window boundaries
            
        Example:
            # 1-hour windows
            TumblingWindow(size_ms=3600000)
            
            # 5-minute windows
            TumblingWindow(size_ms=300000)
        """
```

### SlidingWindow

Fixed-size, overlapping windows.

```python
class SlidingWindow:
    """
    Sliding window with overlap.
    
    Events may belong to multiple windows.
    
    |---Window 1---|
         |---Window 2---|
              |---Window 3---|
    """
    
    def __init__(
        self,
        size_ms: int,
        slide_ms: int,
        time_field: str = "timestamp"
    ) -> None:
        """
        Create sliding window.
        
        Args:
            size_ms: Window size
            slide_ms: Slide interval
            time_field: Event time field
            
        Example:
            # 1-hour window, sliding every 15 minutes
            SlidingWindow(
                size_ms=3600000,
                slide_ms=900000
            )
        """
```

### SessionWindow

Dynamic windows based on activity gaps.

```python
class SessionWindow:
    """
    Session window based on activity gaps.
    
    Window closes after inactivity timeout.
    
    |--Session 1--|  gap  |--Session 2--|
    """
    
    def __init__(
        self,
        gap_ms: int,
        time_field: str = "timestamp",
        max_duration_ms: Optional[int] = None
    ) -> None:
        """
        Create session window.
        
        Args:
            gap_ms: Inactivity gap to close session
            time_field: Event time field
            max_duration_ms: Maximum session length
            
        Example:
            # 30-minute inactivity timeout
            SessionWindow(gap_ms=1800000)
            
            # With max duration
            SessionWindow(
                gap_ms=1800000,
                max_duration_ms=86400000  # 24 hours max
            )
        """
```

---

## Watermark

Handle late-arriving data.

```python
class Watermark:
    """
    Track event time progress and handle late data.
    """
    
    def __init__(
        self,
        delay_ms: int,
        time_field: str = "timestamp"
    ) -> None:
        """
        Create watermark specification.
        
        Args:
            delay_ms: Maximum expected lateness
            time_field: Event time field
            
        Example:
            # Allow 5 seconds of lateness
            watermark = Watermark(delay_ms=5000)
        """
```

### Using Watermarks

```python
stream = StreamProcessor(config)
stream.from_source(source) \
    .with_watermark(Watermark(
        delay_ms=10000,
        time_field="event_time"
    )) \
    .window(TumblingWindow(60000)) \
    .groupBy("category") \
    .agg(hf.count("*")) \
    .to_sink(sink)
```

---

## DeliveryGuarantee

Processing guarantee levels.

```python
class DeliveryGuarantee(Enum):
    """Delivery guarantee levels."""
    
    AT_MOST_ONCE = "at_most_once"
    """Records may be lost but never duplicated."""
    
    AT_LEAST_ONCE = "at_least_once"
    """Records may be duplicated but never lost."""
    
    EXACTLY_ONCE = "exactly_once"
    """Records processed exactly once."""
```

### Choosing a Guarantee

| Guarantee | Performance | Use Case |
|-----------|-------------|----------|
| AT_MOST_ONCE | Highest | Metrics, logs where loss is OK |
| AT_LEAST_ONCE | High | Idempotent operations |
| EXACTLY_ONCE | Lower | Financial, critical data |

### Configuration

```python
config = StreamConfig(
    delivery_guarantee=DeliveryGuarantee.EXACTLY_ONCE,
    checkpoint_dir="/checkpoints",
    checkpoint_interval_ms=5000
)
```

---

## WindowedStream

Stream with windowing applied.

### Aggregation Methods

```python
def groupBy(self, *cols: str) -> "GroupedWindowedStream":
    """
    Group within windows.
    
    Example:
        stream.window(TumblingWindow(60000)) \
            .groupBy("user_id", "action_type")
    """

def agg(self, *exprs: Column) -> "StreamBuilder":
    """
    Aggregate without grouping.
    
    Example:
        stream.window(TumblingWindow(60000)) \
            .agg(hf.count("*"), hf.sum("amount"))
    """
```

### GroupedWindowedStream

```python
def agg(self, *exprs: Column) -> "StreamBuilder":
    """
    Aggregate grouped windows.
    
    Example:
        stream.window(TumblingWindow(60000)) \
            .groupBy("category") \
            .agg(
                hf.count("*").alias("count"),
                hf.sum("amount").alias("total")
            )
    """
```

---

## Complete Example

```python
from hiveframe.streaming import (
    StreamProcessor,
    StreamConfig,
    TumblingWindow,
    Watermark,
    DeliveryGuarantee
)
from hiveframe.connectors import KafkaSource, KafkaSink
import hiveframe as hf

# Configuration
config = StreamConfig(
    checkpoint_dir="/data/checkpoints",
    checkpoint_interval_ms=10000,
    delivery_guarantee=DeliveryGuarantee.EXACTLY_ONCE
)

# Source
source = KafkaSource(
    bootstrap_servers="kafka:9092",
    topic="user-events",
    group_id="analytics-processor",
    value_deserializer="json"
)

# Sink
sink = KafkaSink(
    bootstrap_servers="kafka:9092",
    topic="user-metrics",
    value_serializer="json"
)

# Build pipeline
processor = StreamProcessor(config)

processor.from_source(source) \
    .filter(lambda e: e["event_type"] == "purchase") \
    .with_watermark(Watermark(
        delay_ms=30000,
        time_field="event_time"
    )) \
    .window(TumblingWindow(
        size_ms=300000,  # 5 minutes
        time_field="event_time"
    )) \
    .groupBy("user_id") \
    .agg(
        hf.count("*").alias("purchase_count"),
        hf.sum("amount").alias("total_spent"),
        hf.avg("amount").alias("avg_purchase")
    ) \
    .map(lambda r: {
        "user_id": r["user_id"],
        "window_start": r["window_start"],
        "window_end": r["window_end"],
        "metrics": {
            "count": r["purchase_count"],
            "total": r["total_spent"],
            "average": r["avg_purchase"]
        }
    }) \
    .to_sink(sink)

# Run
processor.start()
processor.await_termination()
```

## See Also

- [Connectors](./connectors) - Source and sink connectors
- [Configure Windows](/docs/how-to/configure-windows) - Window configuration
- [Streaming Concepts](/docs/explanation/streaming-windows-watermarks) - How streaming works
