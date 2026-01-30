---
sidebar_position: 8
---

# Connectors Module

Sources and sinks for external systems: Kafka, PostgreSQL, HTTP, and more.

```python
from hiveframe.connectors import (
    # Kafka
    KafkaSource, KafkaSink,
    # PostgreSQL
    PostgresSource, PostgresSink,
    # HTTP
    HTTPSource, HTTPSink,
    # File
    FileSource, FileSink
)
```

## Kafka Connectors

### KafkaSource

Read from Kafka topics.

```python
class KafkaSource:
    """
    Kafka consumer source.
    """
    
    def __init__(
        self,
        bootstrap_servers: str,
        topic: Union[str, List[str]],
        group_id: str,
        auto_offset_reset: str = "latest",
        value_deserializer: str = "json",
        key_deserializer: str = "string",
        **kafka_config: Any
    ) -> None:
        """
        Create Kafka source.
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic(s) to consume
            group_id: Consumer group ID
            auto_offset_reset: Start position
                - "latest": Only new messages
                - "earliest": From beginning
            value_deserializer: How to decode values
                - "json": JSON to dict
                - "string": UTF-8 string
                - "bytes": Raw bytes
                - "avro": Avro with schema registry
            key_deserializer: How to decode keys
            **kafka_config: Additional Kafka config
        """
```

#### Configuration Options

| Key | Type | Description |
|-----|------|-------------|
| `max_poll_records` | int | Max records per poll |
| `session_timeout_ms` | int | Session timeout |
| `heartbeat_interval_ms` | int | Heartbeat interval |
| `enable_auto_commit` | bool | Auto-commit offsets |
| `schema_registry_url` | str | For Avro deserialization |

#### Example

```python
from hiveframe.connectors import KafkaSource
from hiveframe.streaming import StreamProcessor

source = KafkaSource(
    bootstrap_servers="kafka-1:9092,kafka-2:9092",
    topic=["events", "clicks"],
    group_id="my-consumer-group",
    auto_offset_reset="earliest",
    value_deserializer="json",
    max_poll_records=500,
    enable_auto_commit=False
)

processor = StreamProcessor()
processor.from_source(source) \
    .map(process_event) \
    .to_sink(output_sink)
```

### KafkaSink

Write to Kafka topics.

```python
class KafkaSink:
    """
    Kafka producer sink.
    """
    
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        value_serializer: str = "json",
        key_serializer: str = "string",
        key_field: Optional[str] = None,
        partition_field: Optional[str] = None,
        **kafka_config: Any
    ) -> None:
        """
        Create Kafka sink.
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Target topic
            value_serializer: How to encode values
            key_serializer: How to encode keys
            key_field: Record field to use as message key
            partition_field: Field for custom partitioning
            **kafka_config: Additional Kafka config
        """
```

#### Configuration Options

| Key | Type | Description |
|-----|------|-------------|
| `acks` | str | Acknowledgment level ("all", "1", "0") |
| `retries` | int | Send retries |
| `linger_ms` | int | Batching delay |
| `batch_size` | int | Batch size bytes |
| `compression_type` | str | Compression ("gzip", "snappy", "lz4") |

#### Example

```python
from hiveframe.connectors import KafkaSink

sink = KafkaSink(
    bootstrap_servers="kafka:9092",
    topic="processed-events",
    value_serializer="json",
    key_field="event_id",
    acks="all",
    compression_type="snappy"
)

processor.from_source(source) \
    .map(transform) \
    .to_sink(sink)
```

---

## PostgreSQL Connectors

### PostgresSource

Read from PostgreSQL.

```python
class PostgresSource:
    """
    PostgreSQL database source.
    """
    
    def __init__(
        self,
        host: str,
        port: int = 5432,
        database: str,
        user: str,
        password: str,
        query: Optional[str] = None,
        table: Optional[str] = None,
        schema: str = "public",
        batch_size: int = 10000,
        ssl: bool = False
    ) -> None:
        """
        Create PostgreSQL source.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
            query: Custom SQL query
            table: Table name (alternative to query)
            schema: Database schema
            batch_size: Rows per batch
            ssl: Use SSL connection
        """
```

#### Methods

```python
def read(self) -> DataFrame:
    """
    Read data as DataFrame.
    
    Example:
        source = PostgresSource(
            host="localhost",
            database="mydb",
            user="user",
            password="pass",
            query="SELECT * FROM users WHERE active = true"
        )
        df = source.read()
    """

def read_stream(
    self,
    change_tracking_column: str
) -> StreamBuilder:
    """
    Read as stream using change tracking.
    
    Args:
        change_tracking_column: Column to track changes
        
    Example:
        source.read_stream(
            change_tracking_column="updated_at"
        )
    """
```

#### Example

```python
from hiveframe.connectors import PostgresSource

# Query-based
source = PostgresSource(
    host="db.example.com",
    database="analytics",
    user="reader",
    password="secret",
    query="""
        SELECT 
            u.id, u.name, u.email,
            COUNT(o.id) as order_count
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        GROUP BY u.id, u.name, u.email
    """
)

df = source.read()

# Table-based with partitioning
source = PostgresSource(
    host="db.example.com",
    database="analytics",
    user="reader",
    password="secret",
    table="large_table",
    partition_column="id",
    num_partitions=10
)
```

### PostgresSink

Write to PostgreSQL.

```python
class PostgresSink:
    """
    PostgreSQL database sink.
    """
    
    def __init__(
        self,
        host: str,
        port: int = 5432,
        database: str,
        user: str,
        password: str,
        table: str,
        schema: str = "public",
        mode: str = "append",
        batch_size: int = 1000,
        on_conflict: Optional[str] = None
    ) -> None:
        """
        Create PostgreSQL sink.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
            table: Target table
            schema: Database schema
            mode: Write mode
                - "append": Insert rows
                - "overwrite": Truncate and insert
                - "upsert": Insert or update
            batch_size: Rows per batch insert
            on_conflict: Conflict resolution for upsert
        """
```

#### Example

```python
from hiveframe.connectors import PostgresSink

# Append mode
sink = PostgresSink(
    host="db.example.com",
    database="warehouse",
    user="writer",
    password="secret",
    table="processed_data",
    mode="append",
    batch_size=5000
)

# Upsert mode
sink = PostgresSink(
    host="db.example.com",
    database="warehouse",
    user="writer",
    password="secret",
    table="users",
    mode="upsert",
    on_conflict="(id) DO UPDATE SET name = EXCLUDED.name"
)

# Write DataFrame
df.write.to_sink(sink)
```

---

## HTTP Connectors

### HTTPSource

Read from HTTP endpoints.

```python
class HTTPSource:
    """
    HTTP REST API source.
    """
    
    def __init__(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        auth: Optional[Tuple[str, str]] = None,
        timeout_ms: int = 30000,
        pagination: Optional[PaginationConfig] = None
    ) -> None:
        """
        Create HTTP source.
        
        Args:
            url: API endpoint URL
            method: HTTP method
            headers: Request headers
            params: Query parameters
            auth: Basic auth (username, password)
            timeout_ms: Request timeout
            pagination: Pagination configuration
        """
```

#### PaginationConfig

```python
class PaginationConfig:
    """Configure API pagination."""
    
    def __init__(
        self,
        type: str = "offset",
        page_param: str = "page",
        size_param: str = "limit",
        page_size: int = 100,
        total_field: Optional[str] = None,
        next_url_field: Optional[str] = None
    ):
        """
        Args:
            type: Pagination type
                - "offset": page/offset based
                - "cursor": cursor/token based
                - "link": next URL in response
            page_param: Parameter for page/offset
            size_param: Parameter for page size
            page_size: Records per page
            total_field: JSON path to total count
            next_url_field: JSON path to next URL
        """
```

#### Example

```python
from hiveframe.connectors import HTTPSource, PaginationConfig

# Simple GET
source = HTTPSource(
    url="https://api.example.com/users",
    headers={"Authorization": "Bearer token123"}
)

# With pagination
source = HTTPSource(
    url="https://api.example.com/events",
    pagination=PaginationConfig(
        type="cursor",
        page_param="cursor",
        size_param="limit",
        page_size=100,
        next_url_field="meta.next_cursor"
    )
)

df = source.read()
```

### HTTPSink

Write to HTTP endpoints.

```python
class HTTPSink:
    """
    HTTP REST API sink.
    """
    
    def __init__(
        self,
        url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Tuple[str, str]] = None,
        batch_size: int = 100,
        timeout_ms: int = 30000,
        retry_policy: Optional[RetryPolicy] = None
    ) -> None:
        """
        Create HTTP sink.
        
        Args:
            url: API endpoint URL
            method: HTTP method (POST, PUT, PATCH)
            headers: Request headers
            auth: Basic auth
            batch_size: Records per request
            timeout_ms: Request timeout
            retry_policy: Retry configuration
        """
```

#### Example

```python
from hiveframe.connectors import HTTPSink
from hiveframe.resilience import RetryPolicy

sink = HTTPSink(
    url="https://api.example.com/ingest",
    method="POST",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer token123"
    },
    batch_size=50,
    retry_policy=RetryPolicy(max_attempts=3)
)

df.write.to_sink(sink)
```

---

## File Connectors

### FileSource

Read from files (local or cloud).

```python
class FileSource:
    """
    File system source.
    """
    
    def __init__(
        self,
        path: str,
        format: str = "auto",
        watch: bool = False,
        poll_interval_ms: int = 1000
    ) -> None:
        """
        Create file source.
        
        Args:
            path: File or directory path
            format: File format (auto, parquet, csv, json)
            watch: Watch for new files (streaming)
            poll_interval_ms: Poll interval for watching
        """
```

### FileSink

Write to files.

```python
class FileSink:
    """
    File system sink.
    """
    
    def __init__(
        self,
        path: str,
        format: str = "parquet",
        mode: str = "append",
        partition_by: Optional[List[str]] = None
    ) -> None:
        """
        Create file sink.
        
        Args:
            path: Output path
            format: Output format
            mode: Write mode
            partition_by: Partition columns
        """
```

---

## Connection Strings

For convenience, connectors can be created from connection strings:

```python
from hiveframe.connectors import create_source, create_sink

# Kafka
source = create_source("kafka://broker:9092/topic?group_id=mygroup")

# PostgreSQL
source = create_source(
    "postgresql://user:pass@host:5432/db?table=users"
)

# HTTP
source = create_source(
    "https://api.example.com/data",
    headers={"Auth": "Bearer token"}
)
```

## See Also

- [Connect Kafka](/docs/how-to/connect-kafka) - Kafka how-to
- [Connect PostgreSQL](/docs/how-to/connect-postgres) - PostgreSQL how-to
- [Connect HTTP](/docs/how-to/connect-http) - HTTP API how-to
