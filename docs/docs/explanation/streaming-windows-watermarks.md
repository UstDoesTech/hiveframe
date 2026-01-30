---
sidebar_position: 6
---

# Streaming Windows and Watermarks

Understanding time-based event processing in HiveFrame streaming.

## The Time Problem

In streaming, events can arrive out of order:

```mermaid
sequenceDiagram
    participant S as Source
    participant P as Processor
    
    S->>P: Event A (time=100)
    S->>P: Event C (time=120)
    S->>P: Event B (time=110) ← Late!
    S->>P: Event D (time=130)
```

How do we handle this? When can we close a window and emit results?

## Event Time vs Processing Time

```mermaid
flowchart LR
    subgraph Event["Event Time"]
        E1["When event occurred"]
        E2["Embedded in event data"]
    end
    
    subgraph Processing["Processing Time"]
        P1["When event processed"]
        P2["System clock time"]
    end
```

| Aspect | Event Time | Processing Time |
|--------|------------|-----------------|
| Source | Event timestamp field | System clock |
| Ordering | May be out of order | Always ordered |
| Accuracy | Reflects reality | Reflects processing |
| Use case | Analytics, billing | Monitoring, alerts |

HiveFrame primarily uses **event time** for accurate analytics.

## Window Types

### Tumbling Windows

Non-overlapping, fixed-size windows:

```mermaid
gantt
    title Tumbling Window (5 min)
    dateFormat HH:mm
    axisFormat %H:%M
    
    section Windows
    Window 1 :00:00, 5m
    Window 2 :00:05, 5m
    Window 3 :00:10, 5m
```

```python
from hiveframe.streaming import TumblingWindow

stream.window(TumblingWindow(duration_seconds=300))  # 5 minutes
```

**Use when:** You need non-overlapping time periods (hourly aggregations, daily reports).

### Sliding Windows

Overlapping windows that "slide" forward:

```mermaid
gantt
    title Sliding Window (5 min window, 1 min slide)
    dateFormat HH:mm
    axisFormat %H:%M
    
    section Windows
    Window 1 :00:00, 5m
    Window 2 :00:01, 5m
    Window 3 :00:02, 5m
    Window 4 :00:03, 5m
```

```python
from hiveframe.streaming import SlidingWindow

stream.window(SlidingWindow(
    duration_seconds=300,  # 5 minute window
    slide_seconds=60       # Slide every 1 minute
))
```

**Use when:** You need moving averages or trends (last 5 minutes, updated every minute).

### Session Windows

Dynamic windows based on activity gaps:

```mermaid
gantt
    title Session Window (2 min gap)
    dateFormat HH:mm
    axisFormat %H:%M
    
    section User A
    Session 1 :00:00, 3m
    Gap :00:03, 2m
    Session 2 :00:05, 4m
    
    section User B
    Session 1 :00:01, 7m
```

```python
from hiveframe.streaming import SessionWindow

stream.window(SessionWindow(gap_seconds=120))  # 2 minute gap
```

**Use when:** Analyzing user sessions, activity bursts, or variable-length sequences.

## Watermarks

Watermarks track the progress of event time and handle late data:

```mermaid
sequenceDiagram
    participant E as Events
    participant W as Watermark
    participant P as Processor
    
    E->>P: Event (time=100)
    W->>W: Watermark = 100 - 30 = 70
    Note over W: Allow 30s lateness
    
    E->>P: Event (time=110)
    W->>W: Watermark = 110 - 30 = 80
    
    E->>P: Event (time=75)
    Note over P: 75 > 70, accept late event ✓
    
    E->>P: Event (time=60)
    Note over P: 60 < 70, too late! ✗
```

### Watermark Definition

```
Watermark(t) = max(event_times) - threshold
```

```python
from hiveframe.streaming import Watermark

stream.with_watermark(
    Watermark(
        event_time_column="timestamp",
        delay_threshold_seconds=30
    )
)
```

### How Watermarks Trigger Windows

Windows are closed when the watermark passes their end time:

```mermaid
flowchart LR
    subgraph Window["Window [0:00 - 0:05]"]
        E1[Event 0:01]
        E2[Event 0:03]
        E3[Event 0:04]
    end
    
    W1["Watermark: 0:04"] --> Wait
    W2["Watermark: 0:05"] --> Close["Close & Emit"]
    W3["Watermark: 0:06"] --> Done["Window Done"]
    
    style Close fill:#22c55e
```

## Late Data Handling

### Drop Late Events

```python
stream.with_watermark(
    Watermark(
        event_time_column="timestamp",
        delay_threshold_seconds=30,
        late_data_policy="drop"
    )
)
```

### Allowed Lateness

Allow late updates to already-emitted windows:

```python
stream.with_watermark(
    Watermark(
        event_time_column="timestamp",
        delay_threshold_seconds=30,
        allowed_lateness_seconds=3600  # Accept 1 hour late
    )
)
```

This emits:
1. Initial result when window closes (watermark passes)
2. Updated results as late data arrives
3. Final result when allowed lateness expires

### Side Output for Late Data

```python
stream.with_watermark(
    Watermark(
        event_time_column="timestamp",
        delay_threshold_seconds=30,
        late_data_output="late_events"
    )
)

# Process late data separately
late_stream = stream.get_side_output("late_events")
late_stream.to_sink(late_data_sink)
```

## Putting It Together

```python
from hiveframe.streaming import (
    StreamProcessor,
    StreamConfig,
    TumblingWindow,
    Watermark,
    DeliveryGuarantee,
)

stream = StreamProcessor(StreamConfig(
    name="clickstream-analytics",
    delivery=DeliveryGuarantee.EXACTLY_ONCE,
))

(
    stream
    # Handle late data (up to 5 minutes)
    .with_watermark(
        Watermark(
            event_time_column="click_time",
            delay_threshold_seconds=300
        )
    )
    # 1-hour tumbling windows
    .window(TumblingWindow(duration_seconds=3600))
    # Aggregate clicks by page
    .groupBy("page_url")
    .agg(
        hf.count("*").alias("clicks"),
        hf.count_distinct("user_id").alias("unique_users"),
        hf.window_start().alias("hour_start"),
        hf.window_end().alias("hour_end"),
    )
    .to_sink(results_sink)
)
```

## Choosing Parameters

### Window Size

| Window Size | Use Case | Trade-off |
|-------------|----------|-----------|
| Seconds | Real-time monitoring | High output volume |
| Minutes | Operational dashboards | Balanced |
| Hours | Business reporting | Delayed insights |

### Watermark Delay

| Delay | Use Case | Trade-off |
|-------|----------|-----------|
| Low (seconds) | Strict real-time | May drop late data |
| Medium (minutes) | Most applications | Balanced |
| High (hours) | Complete data | High latency |

Consider your data's lateness characteristics:
- Network delays
- Mobile devices going offline
- Batch uploads

## Monitoring

```python
# Get window statistics
stats = stream.get_window_stats()

print(f"Open windows: {stats.open_count}")
print(f"Late events: {stats.late_count}")
print(f"Current watermark: {stats.watermark}")
```

## See Also

- [How-To: Configure Windows](/docs/how-to/configure-windows) - Practical setup
- [How-To: Manage Watermarks](/docs/how-to/manage-watermarks) - Late data handling
- [Tutorial: Streaming](/docs/tutorials/streaming-application) - Full tutorial
