---
sidebar_position: 10
---

# Enable Distributed Tracing

Set up OpenTelemetry tracing to track requests across your HiveFrame application.

## Basic Setup

```python
from hiveframe.monitoring import configure_tracing, TracingConfig

configure_tracing(TracingConfig(
    service_name="my-hiveframe-app",
    exporter="jaeger",  # or "zipkin", "otlp"
    endpoint="http://localhost:14268/api/traces",
))
```

## Auto-Instrumentation

```python
from hiveframe.monitoring import enable_auto_tracing

# Automatically traces:
# - DataFrame operations
# - SQL queries
# - Storage reads/writes
# - Stream processing
enable_auto_tracing()
```

## Manual Spans

```python
from hiveframe.monitoring import Tracer

tracer = Tracer("my-service")

with tracer.span("process_batch") as span:
    span.set_attribute("batch_size", 1000)
    
    with tracer.span("transform"):
        transform_data()
    
    with tracer.span("write"):
        write_results()
```

## Context Propagation

```python
from hiveframe.monitoring import TraceContext

# Extract context from incoming request
context = TraceContext.extract(request.headers)

# Process with context
with TraceContext.use(context):
    process_request()

# Inject context for outgoing calls
headers = TraceContext.inject({})
requests.post(url, headers=headers)
```

## Exporters

### Jaeger

```python
configure_tracing(TracingConfig(
    service_name="my-app",
    exporter="jaeger",
    endpoint="http://jaeger:14268/api/traces",
))
```

### OTLP (OpenTelemetry Protocol)

```python
configure_tracing(TracingConfig(
    service_name="my-app",
    exporter="otlp",
    endpoint="http://otel-collector:4317",
))
```

### Console (Development)

```python
configure_tracing(TracingConfig(
    service_name="my-app",
    exporter="console",  # Prints to stdout
))
```

## Sampling

```python
configure_tracing(TracingConfig(
    service_name="my-app",
    # Sample 10% of traces
    sampler="ratio",
    sample_rate=0.1,
))

# Or always sample errors
configure_tracing(TracingConfig(
    service_name="my-app",
    sampler="parent_based",
    always_sample_errors=True,
))
```

## See Also

- [Setup Monitoring](./setup-monitoring) - Prometheus metrics
- [Configure Logging](./configure-logging) - Structured logging
- [Reference: Monitoring](/docs/reference/monitoring) - Complete API
