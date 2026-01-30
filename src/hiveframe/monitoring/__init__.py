"""
HiveFrame Monitoring & Observability
====================================
Prometheus-style metrics, structured logging, and colony health monitoring.

Enables debugging, performance analysis, and production observability
for bee-inspired distributed processing.
"""

# Metrics
# Health Monitoring
from .health import (
    ColonyHealthMonitor,
    ColonyHealthReport,
    WorkerHealthSnapshot,
)

# Logging
from .logging import (
    BufferedHandler,
    ConsoleHandler,
    Logger,
    LogHandler,
    LogLevel,
    LogRecord,
    get_logger,
)
from .metrics import (
    Counter,
    Gauge,
    Histogram,
    Metric,
    MetricLabels,
    MetricsRegistry,
    MetricType,
    Summary,
    get_registry,
)

# Performance Profiling
from .profiling import (
    PerformanceProfiler,
    get_profiler,
)

# Distributed Tracing
from .tracing import (
    Tracer,
    TraceSpan,
    get_tracer,
)

__all__ = [
    # Metrics
    "MetricType",
    "MetricLabels",
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "MetricsRegistry",
    "get_registry",
    # Logging
    "LogLevel",
    "LogRecord",
    "LogHandler",
    "ConsoleHandler",
    "BufferedHandler",
    "Logger",
    "get_logger",
    # Health
    "WorkerHealthSnapshot",
    "ColonyHealthReport",
    "ColonyHealthMonitor",
    # Tracing
    "TraceSpan",
    "Tracer",
    "get_tracer",
    # Profiling
    "PerformanceProfiler",
    "get_profiler",
]
