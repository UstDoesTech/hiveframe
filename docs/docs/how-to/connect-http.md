---
sidebar_position: 16
---

# Connect via HTTP

Read from and write to REST APIs and HTTP endpoints.

## Installation

```bash
pip install hiveframe[http]
```

## Read from HTTP API

```python
from hiveframe.connectors import HTTPSource

source = HTTPSource(
    url="https://api.example.com/data",
    method="GET",
)

# Single request
df = source.read()

# Paginated API
source = HTTPSource(
    url="https://api.example.com/data",
    pagination="offset",
    page_size=100,
    max_pages=10,
)
df = source.read()
```

### Streaming (Polling)

```python
source = HTTPSource(
    url="https://api.example.com/events",
    poll_interval_seconds=5,
)

stream.from_source(source)
```

### Server-Sent Events (SSE)

```python
source = HTTPSource(
    url="https://api.example.com/stream",
    stream_type="sse",
)

stream.from_source(source)
```

## Write to HTTP API

```python
from hiveframe.connectors import HTTPSink

sink = HTTPSink(
    url="https://api.example.com/ingest",
    method="POST",
    batch_size=100,
)

df.write.http(sink)
```

## Authentication

```python
# Bearer token
source = HTTPSource(
    url="https://api.example.com/data",
    auth={"type": "bearer", "token": "your-token"},
)

# Basic auth
source = HTTPSource(
    url="https://api.example.com/data",
    auth={"type": "basic", "username": "user", "password": "pass"},
)

# API key
source = HTTPSource(
    url="https://api.example.com/data",
    headers={"X-API-Key": "your-api-key"},
)

# OAuth2
source = HTTPSource(
    url="https://api.example.com/data",
    auth={
        "type": "oauth2",
        "token_url": "https://auth.example.com/token",
        "client_id": "client-id",
        "client_secret": "client-secret",
    },
)
```

## Response Parsing

```python
source = HTTPSource(
    url="https://api.example.com/data",
    response_format="json",
    data_path="results.items",  # JSONPath to data array
)

# Custom parser
def parse_response(response):
    data = response.json()
    return [transform(item) for item in data["items"]]

source = HTTPSource(
    url="https://api.example.com/data",
    parser=parse_response,
)
```

## Error Handling

```python
from hiveframe.resilience import RetryConfig, CircuitBreaker

source = HTTPSource(
    url="https://api.example.com/data",
    retry=RetryConfig(max_attempts=3),
    circuit_breaker=CircuitBreaker(failure_threshold=5),
    timeout_seconds=30,
)
```

## See Also

- [Connect to Kafka](./connect-kafka) - Kafka connector
- [Connect to PostgreSQL](./connect-postgres) - Database connector
- [Reference: Connectors](/docs/reference/connectors) - Complete API
