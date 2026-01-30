---
sidebar_position: 15
---

# Connect to PostgreSQL

Read from and write to PostgreSQL databases.

## Installation

```bash
pip install hiveframe[postgres]
```

## Read from PostgreSQL

```python
from hiveframe.connectors import PostgresSource

source = PostgresSource(
    host="localhost",
    port=5432,
    database="mydb",
    user="user",
    password="password",
)

# Read entire table
df = source.read("SELECT * FROM users")

# With parameters
df = source.read(
    "SELECT * FROM users WHERE created_at > %s",
    params=["2026-01-01"]
)
```

### Streaming Changes (CDC)

```python
source = PostgresSource(
    host="localhost",
    database="mydb",
    user="user",
    password="password",
    # Enable CDC
    replication_slot="hiveframe_slot",
    publication="my_publication",
)

# Stream changes
stream.from_source(source)
```

## Write to PostgreSQL

```python
from hiveframe.connectors import PostgresSink

sink = PostgresSink(
    host="localhost",
    database="mydb",
    user="user",
    password="password",
    table="results",
)

# Write DataFrame
df.write.jdbc(sink)

# With options
df.write.jdbc(
    sink,
    mode="append",  # or "overwrite", "upsert"
    batch_size=1000,
)
```

### Upsert (Insert or Update)

```python
sink = PostgresSink(
    host="localhost",
    database="mydb",
    table="users",
    user="user",
    password="password",
    primary_key=["id"],  # For upsert
)

df.write.jdbc(sink, mode="upsert")
```

## Connection Pooling

```python
source = PostgresSource(
    host="localhost",
    database="mydb",
    user="user",
    password="password",
    # Pool settings
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
)
```

## SSL/TLS Connection

```python
source = PostgresSource(
    host="db.example.com",
    database="mydb",
    user="user",
    password="password",
    ssl_mode="require",  # or "verify-ca", "verify-full"
    ssl_ca="/path/to/ca.pem",
)
```

## See Also

- [Connect to Kafka](./connect-kafka) - Kafka connector
- [Connect via HTTP](./connect-http) - REST API connector
- [Reference: Connectors](/docs/reference/connectors) - Complete API
