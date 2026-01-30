---
sidebar_position: 14
---

# Connect to Kafka

Read from and write to Apache Kafka topics.

## Installation

```bash
pip install hiveframe[kafka]
```

## Read from Kafka

```python
from hiveframe.connectors import KafkaSource

source = KafkaSource(
    bootstrap_servers="localhost:9092",
    topics=["events"],
    group_id="my-consumer-group",
)

# With streaming
stream.from_source(source)

# Or batch read
df = source.read(timeout_seconds=30)
```

### Configuration Options

```python
source = KafkaSource(
    bootstrap_servers="kafka1:9092,kafka2:9092",
    topics=["topic1", "topic2"],
    group_id="my-group",
    
    # Starting position
    auto_offset_reset="earliest",  # or "latest"
    
    # Deserialization
    key_deserializer="string",
    value_deserializer="json",  # or "avro", "protobuf"
    
    # Consumer settings
    max_poll_records=500,
    session_timeout_ms=30000,
    
    # Security
    security_protocol="SASL_SSL",
    sasl_mechanism="PLAIN",
    sasl_username="user",
    sasl_password="password",
)
```

## Write to Kafka

```python
from hiveframe.connectors import KafkaSink

sink = KafkaSink(
    bootstrap_servers="localhost:9092",
    topic="output-events",
)

# With streaming
stream.to_sink(sink)

# Or batch write
df.write.kafka(sink)
```

### Configuration Options

```python
sink = KafkaSink(
    bootstrap_servers="localhost:9092",
    topic="output",
    
    # Key selection
    key_column="id",  # Use this column as message key
    
    # Serialization
    key_serializer="string",
    value_serializer="json",
    
    # Producer settings
    acks="all",  # or "0", "1"
    retries=3,
    batch_size=16384,
    linger_ms=5,
    
    # Exactly-once
    transactional_id="my-producer",  # Enable transactions
)
```

## Schema Registry

```python
from hiveframe.connectors import KafkaSource, SchemaRegistry

registry = SchemaRegistry("http://localhost:8081")

source = KafkaSource(
    bootstrap_servers="localhost:9092",
    topics=["avro-events"],
    value_deserializer="avro",
    schema_registry=registry,
)
```

## Error Handling

```python
from hiveframe.connectors import KafkaSource
from hiveframe.dlq import DeadLetterQueue

dlq = DeadLetterQueue(DLQConfig(path="data/kafka_dlq"))

source = KafkaSource(
    bootstrap_servers="localhost:9092",
    topics=["events"],
    on_deserialization_error=lambda e, msg: dlq.send(msg, error=e),
)
```

## See Also

- [Connect to PostgreSQL](./connect-postgres) - Database connector
- [Reference: Connectors](/docs/reference/connectors) - Complete API
