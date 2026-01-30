---
sidebar_position: 9
---

# Configure Logging

Set up structured logging for HiveFrame applications.

## Basic Configuration

```python
from hiveframe.monitoring import configure_logging, LogLevel

# Simple setup
configure_logging(level=LogLevel.INFO)

# With file output
configure_logging(
    level=LogLevel.INFO,
    file="logs/hiveframe.log",
    rotation="10 MB",
    retention="7 days",
)
```

## Log Levels

```python
from hiveframe.monitoring import LogLevel

configure_logging(level=LogLevel.DEBUG)  # Most verbose
configure_logging(level=LogLevel.INFO)   # Normal operation
configure_logging(level=LogLevel.WARNING)  # Warnings and errors
configure_logging(level=LogLevel.ERROR)  # Errors only
```

## Structured Logging

```python
from hiveframe.monitoring import get_logger

logger = get_logger(__name__)

# Add context to logs
logger.info(
    "Processing batch",
    batch_id="abc123",
    record_count=1000,
    source="kafka"
)
```

Output (JSON format):
```json
{
  "timestamp": "2026-01-30T10:15:30.123Z",
  "level": "INFO",
  "logger": "myapp.processor",
  "message": "Processing batch",
  "batch_id": "abc123",
  "record_count": 1000,
  "source": "kafka"
}
```

## Log Formats

```python
from hiveframe.monitoring import configure_logging, LogFormat

# JSON (recommended for production)
configure_logging(format=LogFormat.JSON)

# Human-readable (for development)
configure_logging(format=LogFormat.CONSOLE)

# Custom format
configure_logging(
    format="{timestamp} [{level}] {logger}: {message} {extra}"
)
```

## Context Propagation

```python
from hiveframe.monitoring import LogContext, get_logger

logger = get_logger(__name__)

# Add context for a block
with LogContext(request_id="req-123", user_id="user-456"):
    logger.info("Starting request")  # Includes request_id and user_id
    process_request()
    logger.info("Request complete")  # Still has context

# Or set context globally
LogContext.set(environment="production", service="processor")
```

## Per-Module Logging

```python
from hiveframe.monitoring import configure_logging

configure_logging(
    level="INFO",
    module_levels={
        "hiveframe.streaming": "DEBUG",  # More verbose for streaming
        "hiveframe.storage": "WARNING",  # Less verbose for storage
        "urllib3": "ERROR",  # Silence noisy libraries
    }
)
```

## Log to Multiple Destinations

```python
from hiveframe.monitoring import configure_logging, LogHandler

configure_logging(
    handlers=[
        LogHandler.console(level="INFO", format="CONSOLE"),
        LogHandler.file(
            path="logs/app.log",
            level="DEBUG",
            format="JSON"
        ),
        LogHandler.syslog(
            host="logs.example.com",
            port=514,
            level="WARNING"
        ),
    ]
)
```

## Integration with Colony

```python
import hiveframe as hf
from hiveframe.monitoring import configure_logging

# Configure logging before creating colony
configure_logging(level="INFO")

colony = hf.Colony(name="my-colony")

# Colony automatically logs:
# - Worker lifecycle events
# - Waggle dance communications
# - Pheromone level changes
# - Food source quality updates
```

## See Also

- [Setup Monitoring](./setup-monitoring) - Prometheus metrics
- [Enable Tracing](./enable-tracing) - Distributed tracing
- [Reference: Monitoring](/docs/reference/monitoring) - Complete API
