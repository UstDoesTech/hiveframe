---
sidebar_position: 9
---

# Monitoring Module

Metrics, logging, health checks, and distributed tracing.

```python
from hiveframe.monitoring import (
    MetricsCollector,
    HealthChecker,
    Logger,
    Tracer
)
```

## MetricsCollector

Collect and export metrics.

### Class Definition

```python
class MetricsCollector:
    """
    Collect application and system metrics.
    """
    
    def __init__(
        self,
        name: str = "hiveframe",
        export_interval_ms: int = 10000,
        exporters: Optional[List[MetricsExporter]] = None
    ) -> None:
        """
        Create metrics collector.
        
        Args:
            name: Metrics namespace
            export_interval_ms: Export frequency
            exporters: Output destinations
        """
```

### Methods

#### Counter

```python
def counter(
    self,
    name: str,
    description: str = "",
    labels: Optional[List[str]] = None
) -> Counter:
    """
    Create a counter metric.
    
    Args:
        name: Metric name
        description: Help text
        labels: Label names
        
    Example:
        requests = collector.counter(
            "requests_total",
            "Total requests processed",
            labels=["endpoint", "status"]
        )
        requests.inc(labels={"endpoint": "/api", "status": "200"})
    """
```

#### Gauge

```python
def gauge(
    self,
    name: str,
    description: str = "",
    labels: Optional[List[str]] = None
) -> Gauge:
    """
    Create a gauge metric.
    
    Example:
        queue_size = collector.gauge(
            "queue_size",
            "Current queue depth",
            labels=["queue_name"]
        )
        queue_size.set(42, labels={"queue_name": "events"})
    """
```

#### Histogram

```python
def histogram(
    self,
    name: str,
    description: str = "",
    labels: Optional[List[str]] = None,
    buckets: Optional[List[float]] = None
) -> Histogram:
    """
    Create a histogram metric.
    
    Args:
        buckets: Bucket boundaries
        
    Example:
        latency = collector.histogram(
            "request_duration_seconds",
            "Request latency",
            labels=["endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        latency.observe(0.042, labels={"endpoint": "/api"})
    """
```

### Exporters

#### Prometheus

```python
from hiveframe.monitoring import PrometheusExporter

exporter = PrometheusExporter(
    port=9090,
    path="/metrics"
)

collector = MetricsCollector(exporters=[exporter])
```

#### StatsD

```python
from hiveframe.monitoring import StatsDExporter

exporter = StatsDExporter(
    host="localhost",
    port=8125,
    prefix="hiveframe"
)
```

#### Console

```python
from hiveframe.monitoring import ConsoleExporter

exporter = ConsoleExporter(
    format="json"  # or "text"
)
```

### Built-in Metrics

HiveFrame automatically collects:

| Metric | Type | Description |
|--------|------|-------------|
| `hiveframe_tasks_total` | Counter | Tasks processed |
| `hiveframe_task_duration_seconds` | Histogram | Task execution time |
| `hiveframe_workers_active` | Gauge | Active workers |
| `hiveframe_memory_bytes` | Gauge | Memory usage |
| `hiveframe_stream_records_total` | Counter | Records processed |
| `hiveframe_stream_lag_seconds` | Gauge | Processing lag |

---

## HealthChecker

Monitor application health.

### Class Definition

```python
class HealthChecker:
    """
    Health check management.
    """
    
    def __init__(
        self,
        name: str = "hiveframe"
    ) -> None:
        """
        Create health checker.
        
        Args:
            name: Application name
        """
```

### Methods

```python
def add_check(
    self,
    name: str,
    check_func: Callable[[], HealthStatus],
    timeout_ms: int = 5000,
    critical: bool = True
) -> None:
    """
    Register a health check.
    
    Args:
        name: Check name
        check_func: Function returning health status
        timeout_ms: Check timeout
        critical: If false, failure doesn't affect overall health
        
    Example:
        def check_database():
            try:
                db.execute("SELECT 1")
                return HealthStatus.HEALTHY
            except:
                return HealthStatus.UNHEALTHY
        
        checker.add_check("database", check_database)
    """

def check(self) -> HealthReport:
    """
    Run all health checks.
    
    Returns:
        HealthReport with status of all checks
    """

def start_server(
    self,
    port: int = 8080,
    path: str = "/health"
) -> None:
    """
    Start HTTP health endpoint.
    
    Endpoints:
    - /health - Overall health
    - /health/live - Liveness probe
    - /health/ready - Readiness probe
    """
```

### HealthStatus

```python
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
```

### Example

```python
from hiveframe.monitoring import HealthChecker, HealthStatus

checker = HealthChecker("my-app")

# Add checks
checker.add_check("database", lambda: check_db())
checker.add_check("kafka", lambda: check_kafka())
checker.add_check("cache", lambda: check_redis(), critical=False)

# Start health server
checker.start_server(port=8080)

# Manual check
report = checker.check()
print(f"Status: {report.status}")
for check_name, result in report.checks.items():
    print(f"  {check_name}: {result.status}")
```

---

## Logger

Structured logging with context.

### Class Definition

```python
class Logger:
    """
    Structured logging with context propagation.
    """
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        format: str = "json",
        handlers: Optional[List[LogHandler]] = None
    ) -> None:
        """
        Create logger.
        
        Args:
            name: Logger name
            level: Minimum log level
            format: Output format (json, text)
            handlers: Output handlers
        """
```

### Methods

```python
def debug(self, message: str, **context: Any) -> None:
    """Log debug message."""

def info(self, message: str, **context: Any) -> None:
    """Log info message."""

def warning(self, message: str, **context: Any) -> None:
    """Log warning message."""

def error(
    self, 
    message: str, 
    exc: Optional[Exception] = None,
    **context: Any
) -> None:
    """Log error message with optional exception."""

def with_context(self, **context: Any) -> "Logger":
    """Create child logger with additional context."""
```

### Example

```python
from hiveframe.monitoring import Logger

logger = Logger("my-app", level="INFO", format="json")

# Basic logging
logger.info("Processing started", batch_id="batch-123")

# With context
request_logger = logger.with_context(
    request_id="req-456",
    user_id="user-789"
)
request_logger.info("Request received")
request_logger.info("Processing complete", duration_ms=42)

# Error logging
try:
    process_data()
except Exception as e:
    logger.error("Processing failed", exc=e, batch_id="batch-123")
```

### Output

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "my-app",
  "message": "Processing complete",
  "request_id": "req-456",
  "user_id": "user-789",
  "duration_ms": 42
}
```

---

## Tracer

Distributed tracing with OpenTelemetry.

### Class Definition

```python
class Tracer:
    """
    Distributed tracing with span context.
    """
    
    def __init__(
        self,
        service_name: str,
        exporter: Optional[TraceExporter] = None,
        sampler: Optional[Sampler] = None
    ) -> None:
        """
        Create tracer.
        
        Args:
            service_name: Service identifier
            exporter: Trace exporter (Jaeger, Zipkin, OTLP)
            sampler: Sampling strategy
        """
```

### Methods

```python
def start_span(
    self,
    name: str,
    parent: Optional[SpanContext] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> Span:
    """
    Start a new span.
    
    Args:
        name: Span name
        parent: Parent span context
        attributes: Span attributes
        
    Returns:
        Active span
    """

def current_span(self) -> Optional[Span]:
    """Get current active span."""

@contextmanager
def span(
    self,
    name: str,
    **attributes: Any
) -> Generator[Span, None, None]:
    """
    Context manager for spans.
    
    Example:
        with tracer.span("process_batch", batch_size=100) as span:
            result = process()
            span.set_attribute("result_count", len(result))
    """
```

### Exporters

```python
from hiveframe.monitoring import (
    JaegerExporter,
    ZipkinExporter,
    OTLPExporter
)

# Jaeger
exporter = JaegerExporter(
    agent_host="localhost",
    agent_port=6831
)

# Zipkin
exporter = ZipkinExporter(
    endpoint="http://localhost:9411/api/v2/spans"
)

# OTLP
exporter = OTLPExporter(
    endpoint="http://localhost:4317"
)

tracer = Tracer("my-service", exporter=exporter)
```

### Example

```python
from hiveframe.monitoring import Tracer, JaegerExporter

tracer = Tracer(
    "data-pipeline",
    exporter=JaegerExporter(agent_host="localhost")
)

def process_batch(batch_id: str, data: list):
    with tracer.span("process_batch", batch_id=batch_id) as span:
        # Child span for validation
        with tracer.span("validate"):
            validated = validate(data)
            span.set_attribute("valid_count", len(validated))
        
        # Child span for transformation
        with tracer.span("transform"):
            transformed = transform(validated)
        
        # Child span for storage
        with tracer.span("store"):
            store(transformed)
        
        span.set_attribute("total_processed", len(data))
```

---

## Complete Example

```python
from hiveframe.monitoring import (
    MetricsCollector,
    HealthChecker,
    Logger,
    Tracer,
    PrometheusExporter,
    JaegerExporter,
    HealthStatus
)

# Setup metrics
metrics = MetricsCollector(
    name="my-app",
    exporters=[PrometheusExporter(port=9090)]
)
requests_counter = metrics.counter("requests_total", labels=["status"])
latency_histogram = metrics.histogram("request_duration_seconds")

# Setup logging
logger = Logger("my-app", level="INFO", format="json")

# Setup tracing
tracer = Tracer(
    "my-app",
    exporter=JaegerExporter(agent_host="jaeger")
)

# Setup health checks
health = HealthChecker("my-app")
health.add_check("database", check_database)
health.start_server(port=8080)

# Use in application
def handle_request(request):
    with tracer.span("handle_request", path=request.path) as span:
        request_logger = logger.with_context(
            request_id=request.id,
            trace_id=span.trace_id
        )
        
        start_time = time.time()
        try:
            request_logger.info("Request received")
            result = process(request)
            
            duration = time.time() - start_time
            latency_histogram.observe(duration)
            requests_counter.inc(labels={"status": "success"})
            
            request_logger.info("Request complete", duration_ms=duration*1000)
            return result
            
        except Exception as e:
            requests_counter.inc(labels={"status": "error"})
            request_logger.error("Request failed", exc=e)
            raise
```

## See Also

- [Setup Monitoring](/docs/how-to/setup-monitoring) - Monitoring how-to
- [Configure Logging](/docs/how-to/configure-logging) - Logging how-to
- [Enable Tracing](/docs/how-to/enable-tracing) - Tracing how-to
