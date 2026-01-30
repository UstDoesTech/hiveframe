---
sidebar_position: 5
---

# Configure Retry Policies

Set up automatic retry with exponential backoff for resilient data processing.

## Basic Retry

```python
from hiveframe.resilience import Retry, RetryConfig

# Simple retry with defaults
@Retry()
def fetch_data():
    return external_api.get_data()

# Configure retry behavior
config = RetryConfig(
    max_attempts=3,
    initial_delay_ms=100,
    max_delay_ms=5000,
    backoff_multiplier=2.0,
)

@Retry(config)
def fetch_data():
    return external_api.get_data()
```

## Retry Strategies

### Exponential Backoff

```python
config = RetryConfig(
    max_attempts=5,
    initial_delay_ms=100,    # Start with 100ms
    max_delay_ms=10000,      # Cap at 10 seconds
    backoff_multiplier=2.0,  # Double each time
    # Delays: 100ms, 200ms, 400ms, 800ms, 1600ms (capped at 10s)
)
```

### Fixed Delay

```python
config = RetryConfig(
    max_attempts=3,
    initial_delay_ms=1000,
    backoff_multiplier=1.0,  # No increase
    # Delays: 1s, 1s, 1s
)
```

### With Jitter

```python
config = RetryConfig(
    max_attempts=5,
    initial_delay_ms=100,
    backoff_multiplier=2.0,
    jitter=True,  # Add randomness to prevent thundering herd
    jitter_factor=0.5,  # ±50% variation
)
```

## Selective Retry

### Retry Specific Exceptions

```python
from hiveframe.resilience import Retry, RetryConfig
import requests

config = RetryConfig(
    max_attempts=3,
    retry_on=[
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        ConnectionRefusedError,
    ]
)

@Retry(config)
def call_api():
    return requests.get("https://api.example.com/data", timeout=5)
```

### Don't Retry Certain Errors

```python
config = RetryConfig(
    max_attempts=3,
    retry_on_exception=lambda e: not isinstance(e, (
        ValueError,       # Don't retry validation errors
        PermissionError,  # Don't retry auth errors
    ))
)
```

### Retry Based on Result

```python
config = RetryConfig(
    max_attempts=3,
    retry_on_result=lambda r: r.status_code >= 500,  # Retry server errors
)

@Retry(config)
def call_api():
    return requests.get("https://api.example.com/data")
```

## Retry with Context

### Async Support

```python
from hiveframe.resilience import AsyncRetry

@AsyncRetry(config)
async def fetch_data_async():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com/data") as response:
            return await response.json()
```

### With DataFrame Operations

```python
from hiveframe.resilience import RetryExecutor

executor = RetryExecutor(config)

# Wrap a function
def write_to_database(df):
    df.write.jdbc("jdbc:postgresql://localhost/db", "table")

executor.execute(write_to_database, df)

# Or use as context manager
with executor:
    df.write.jdbc("jdbc:postgresql://localhost/db", "table")
```

## Callbacks and Logging

```python
def on_retry(attempt, exception, delay):
    print(f"Attempt {attempt} failed with {exception}. Retrying in {delay}ms...")

def on_success(result, attempts):
    print(f"Succeeded after {attempts} attempts")

def on_failure(exception, attempts):
    print(f"Failed after {attempts} attempts: {exception}")

config = RetryConfig(
    max_attempts=3,
    on_retry=on_retry,
    on_success=on_success,
    on_failure=on_failure,
)
```

## Integration with HiveFrame

```python
import hiveframe as hf
from hiveframe.resilience import Retry, RetryConfig

# Configure global retry policy
hf.config.set_retry_policy(RetryConfig(
    max_attempts=3,
    initial_delay_ms=500,
))

# Now all operations use retry
df = hf.read.parquet("s3://bucket/data/")  # Retries on network errors
```

## See Also

- [Use Circuit Breaker](./use-circuit-breaker) - Prevent cascade failures
- [Handle Errors with DLQ](./handle-errors-dlq) - Failed record handling
- [Reference: Resilience](/docs/reference/resilience) - Complete API
