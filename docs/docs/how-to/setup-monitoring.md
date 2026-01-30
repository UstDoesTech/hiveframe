---
sidebar_position: 8
---

# Setup Monitoring

Configure Prometheus metrics, alerting, and observability for HiveFrame.

## Enable Metrics

```python
import hiveframe as hf
from hiveframe.monitoring import MetricsConfig, enable_metrics

# Enable with defaults
enable_metrics()

# Or with custom configuration
config = MetricsConfig(
    port=9090,
    path="/metrics",
    prefix="hiveframe_",
    include_process_metrics=True,
)
enable_metrics(config)
```

## Available Metrics

### Colony Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `hiveframe_colony_temperature` | Gauge | System load (target: 35-38) |
| `hiveframe_colony_workers` | Gauge | Active worker count |
| `hiveframe_food_source_quality` | Histogram | Task quality scores |
| `hiveframe_waggle_dance_count` | Counter | Communication events |
| `hiveframe_pheromone_level` | Gauge | Backpressure indicator |

### Processing Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `hiveframe_records_processed` | Counter | Total records processed |
| `hiveframe_records_failed` | Counter | Failed records |
| `hiveframe_processing_latency_seconds` | Histogram | Processing time |
| `hiveframe_batch_size` | Histogram | Records per batch |

### Storage Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `hiveframe_bytes_read` | Counter | Bytes read from storage |
| `hiveframe_bytes_written` | Counter | Bytes written to storage |
| `hiveframe_parquet_row_groups` | Gauge | Row groups in memory |

## Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'hiveframe'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9090']
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        regex: '(.+):\d+'
        replacement: '${1}'
```

## Custom Metrics

```python
from hiveframe.monitoring import Metrics

metrics = Metrics()

# Counter
orders_counter = metrics.counter(
    "orders_processed",
    "Total orders processed",
    labels=["region", "status"]
)
orders_counter.labels(region="us-east", status="success").inc()

# Gauge
queue_depth = metrics.gauge(
    "queue_depth",
    "Current queue depth"
)
queue_depth.set(42)

# Histogram
latency = metrics.histogram(
    "api_latency_seconds",
    "API call latency",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)
with latency.time():
    call_api()
```

## Grafana Dashboard

Import the HiveFrame dashboard:

```bash
# Download dashboard JSON
curl -o hiveframe-dashboard.json \
  https://raw.githubusercontent.com/hiveframe/hiveframe/main/grafana/dashboard.json

# Import via Grafana API
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @hiveframe-dashboard.json
```

Key panels:
- Colony health (temperature, workers, food sources)
- Throughput (records/second)
- Latency percentiles (p50, p95, p99)
- Error rate
- Resource utilization

## Alerting Rules

```yaml
# prometheus-alerts.yml
groups:
  - name: hiveframe
    rules:
      - alert: HighColonyTemperature
        expr: hiveframe_colony_temperature > 40
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Colony temperature high"
          
      - alert: HighErrorRate
        expr: rate(hiveframe_records_failed[5m]) / rate(hiveframe_records_processed[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate above 5%"
          
      - alert: ProcessingLatencyHigh
        expr: histogram_quantile(0.99, rate(hiveframe_processing_latency_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p99 latency above 5 seconds"
```

## Health Checks

```python
from hiveframe.monitoring import HealthCheck, HealthStatus

health = HealthCheck()

# Add custom checks
@health.check("database")
def check_database():
    try:
        db.ping()
        return HealthStatus.HEALTHY
    except:
        return HealthStatus.UNHEALTHY

@health.check("storage")
def check_storage():
    if disk_usage() > 0.9:
        return HealthStatus.DEGRADED
    return HealthStatus.HEALTHY

# Expose health endpoint
health.serve(port=8080, path="/health")
```

```bash
# Check health
curl http://localhost:8080/health
```

```json
{
  "status": "healthy",
  "checks": {
    "database": "healthy",
    "storage": "healthy",
    "colony": "healthy"
  },
  "timestamp": "2026-01-30T10:15:30Z"
}
```

## See Also

- [Configure Logging](./configure-logging) - Structured logging
- [Enable Tracing](./enable-tracing) - Distributed tracing
- [Reference: Monitoring](/docs/reference/monitoring) - Complete API
