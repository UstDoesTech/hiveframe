"""
HiveFrame Monitoring & Observability
====================================
Prometheus-style metrics, structured logging, and colony health monitoring.

Enables debugging, performance analysis, and production observability
for bee-inspired distributed processing.
"""

# Metrics
from .metrics import (
    MetricType,
    MetricLabels,
    Metric,
    Counter,
    Gauge,
    Histogram,
    Summary,
    MetricsRegistry,
    get_registry,
)

# Logging
from .logging import (
    LogLevel,
    LogRecord,
    LogHandler,
    ConsoleHandler,
    BufferedHandler,
    Logger,
    get_logger,
)

# Health Monitoring
from .health import (
    WorkerHealthSnapshot,
    ColonyHealthReport,
    ColonyHealthMonitor,
)

# Distributed Tracing
from .tracing import (
    TraceSpan,
    Tracer,
    get_tracer,
)

# Performance Profiling
from .profiling import (
    PerformanceProfiler,
    get_profiler,
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
