---
sidebar_position: 13
---

# Configure Delivery Guarantees

Choose the right processing semantics for your streaming application.

## Delivery Guarantee Levels

| Guarantee | Data Loss | Duplicates | Performance | Use Case |
|-----------|-----------|------------|-------------|----------|
| At-most-once | Possible | No | Fastest | Metrics, non-critical |
| At-least-once | No | Possible | Medium | Most applications |
| Exactly-once | No | No | Slowest | Financial, critical |

## At-Most-Once

Events are processed at most one time. May lose data on failures.

```python
from hiveframe.streaming import StreamConfig, DeliveryGuarantee

config = StreamConfig(
    name="metrics-stream",
    delivery=DeliveryGuarantee.AT_MOST_ONCE
)
```

**When to use:**
- Real-time metrics and monitoring
- Non-critical event logging
- When performance matters more than accuracy

## At-Least-Once

Events are processed at least one time. May produce duplicates.

```python
config = StreamConfig(
    name="order-stream",
    delivery=DeliveryGuarantee.AT_LEAST_ONCE,
    # Configure checkpointing for recovery
    checkpoint_dir="/checkpoints",
    checkpoint_interval_ms=10000,
)
```

**When to use:**
- Most production applications
- When you can handle duplicates (idempotent operations)
- When data loss is unacceptable

### Handle Duplicates

```python
# Use idempotent operations
stream.map(lambda e: {**e, "processed": True})

# Or deduplicate
stream.deduplicate(
    key="event_id",
    window_seconds=3600  # Remember IDs for 1 hour
)
```

## Exactly-Once

Events are processed exactly one time. No loss, no duplicates.

```python
config = StreamConfig(
    name="payment-stream",
    delivery=DeliveryGuarantee.EXACTLY_ONCE,
    checkpoint_dir="/checkpoints",
    checkpoint_interval_ms=5000,
    # Transaction support
    transaction_timeout_ms=60000,
)
```

**Requirements:**
- Checkpointing enabled
- Transactional sinks (Kafka, Delta Lake)
- Higher latency tolerance

**When to use:**
- Financial transactions
- Billing and invoicing
- Any data where accuracy is critical

## Checkpointing

Enable checkpointing for fault tolerance:

```python
config = StreamConfig(
    name="my-stream",
    delivery=DeliveryGuarantee.AT_LEAST_ONCE,
    
    # Checkpoint location
    checkpoint_dir="s3://bucket/checkpoints/",
    
    # Checkpoint frequency
    checkpoint_interval_ms=30000,  # Every 30 seconds
    
    # Checkpoint timeout
    checkpoint_timeout_ms=60000,
    
    # Minimum pause between checkpoints
    min_pause_between_checkpoints_ms=5000,
)
```

## End-to-End Exactly-Once

For true exactly-once from source to sink:

```python
from hiveframe.streaming import StreamProcessor, StreamConfig
from hiveframe.connectors import KafkaSource, KafkaSink

# Both source and sink must support transactions
config = StreamConfig(
    name="e2e-exactly-once",
    delivery=DeliveryGuarantee.EXACTLY_ONCE,
)

stream = StreamProcessor(config)

# Transactional source
source = KafkaSource(
    bootstrap_servers="kafka:9092",
    topics=["input-topic"],
    group_id="my-consumer",
    isolation_level="read_committed"  # Only read committed messages
)

# Transactional sink
sink = KafkaSink(
    bootstrap_servers="kafka:9092",
    topic="output-topic",
    transactional_id="my-producer"  # Enable transactions
)

stream.from_source(source)
stream.to_sink(sink)
```

## See Also

- [Configure Windows](./configure-windows) - Window aggregations
- [Manage Watermarks](./manage-watermarks) - Late data handling
- [Reference: Streaming](/docs/reference/streaming) - Complete API
