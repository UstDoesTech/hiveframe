"""
HiveFrame Monitoring & Observability
====================================
Prometheus-style metrics, structured logging, and colony health monitoring.

Enables debugging, performance analysis, and production observability
for bee-inspired distributed processing.
"""

import time
import threading
import json
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from enum import Enum, auto
from contextlib import contextmanager
import sys
import traceback


T = TypeVar('T')


# ============================================================================
# Metric Types (Prometheus-Compatible)
# ============================================================================

class MetricType(Enum):
    """Types of metrics following Prometheus conventions."""
    COUNTER = auto()     # Monotonically increasing value
    GAUGE = auto()       # Value that can go up and down
    HISTOGRAM = auto()   # Distribution of values in buckets
    SUMMARY = auto()     # Similar to histogram with quantiles


@dataclass
class MetricLabels:
    """Labels for metric identification."""
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(tuple(sorted(self.labels.items())))
        
    def __eq__(self, other):
        if not isinstance(other, MetricLabels):
            return False
        return self.labels == other.labels
        
    def to_prometheus(self) -> str:
        """Format labels for Prometheus output."""
        if not self.labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(self.labels.items())]
        return "{" + ",".join(parts) + "}"


class Metric(ABC):
    """Base class for all metrics."""
    
    def __init__(self, name: str, help_text: str, metric_type: MetricType):
        self.name = name
        self.help_text = help_text
        self.metric_type = metric_type
        self._lock = threading.Lock()
        
    @abstractmethod
    def collect(self) -> List[Dict[str, Any]]:
        """Collect all metric values."""
        pass
        
    @abstractmethod
    def to_prometheus(self) -> str:
        """Format as Prometheus exposition format."""
        pass


class Counter(Metric):
    """
    Counter metric - monotonically increasing.
    
    Use for: request counts, error counts, bytes processed
    """
    
    def __init__(self, name: str, help_text: str = ""):
        super().__init__(name, help_text, MetricType.COUNTER)
        self._values: Dict[MetricLabels, float] = defaultdict(float)
        
    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")
            
        key = MetricLabels(labels or {})
        with self._lock:
            self._values[key] += value
            
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        key = MetricLabels(labels or {})
        with self._lock:
            return self._values[key]
            
    def collect(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {'labels': k.labels, 'value': v}
                for k, v in self._values.items()
            ]
            
    def to_prometheus(self) -> str:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} counter"
        ]
        with self._lock:
            for labels, value in self._values.items():
                lines.append(f"{self.name}{labels.to_prometheus()} {value}")
        return "\n".join(lines)


class Gauge(Metric):
    """
    Gauge metric - can go up and down.
    
    Use for: queue sizes, active workers, temperature
    """
    
    def __init__(self, name: str, help_text: str = ""):
        super().__init__(name, help_text, MetricType.GAUGE)
        self._values: Dict[MetricLabels, float] = defaultdict(float)
        
    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value."""
        key = MetricLabels(labels or {})
        with self._lock:
            self._values[key] = value
            
    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment gauge."""
        key = MetricLabels(labels or {})
        with self._lock:
            self._values[key] += value
            
    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement gauge."""
        key = MetricLabels(labels or {})
        with self._lock:
            self._values[key] -= value
            
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        key = MetricLabels(labels or {})
        with self._lock:
            return self._values[key]
            
    def collect(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {'labels': k.labels, 'value': v}
                for k, v in self._values.items()
            ]
            
    def to_prometheus(self) -> str:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} gauge"
        ]
        with self._lock:
            for labels, value in self._values.items():
                lines.append(f"{self.name}{labels.to_prometheus()} {value}")
        return "\n".join(lines)


class Histogram(Metric):
    """
    Histogram metric - distribution of values.
    
    Use for: latency, request sizes
    """
    
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float('inf'))
    
    def __init__(
        self,
        name: str,
        help_text: str = "",
        buckets: tuple = DEFAULT_BUCKETS
    ):
        super().__init__(name, help_text, MetricType.HISTOGRAM)
        self.buckets = buckets
        self._bucket_counts: Dict[MetricLabels, List[int]] = defaultdict(
            lambda: [0] * len(buckets)
        )
        self._sums: Dict[MetricLabels, float] = defaultdict(float)
        self._counts: Dict[MetricLabels, int] = defaultdict(int)
        
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observation."""
        key = MetricLabels(labels or {})
        with self._lock:
            # Update buckets
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    self._bucket_counts[key][i] += 1
                    
            self._sums[key] += value
            self._counts[key] += 1
            
    @contextmanager
    def time(self, labels: Optional[Dict[str, str]] = None):
        """Context manager to time a block of code."""
        start = time.time()
        try:
            yield
        finally:
            self.observe(time.time() - start, labels)
            
    def collect(self) -> List[Dict[str, Any]]:
        with self._lock:
            results = []
            for key in self._counts.keys():
                results.append({
                    'labels': key.labels,
                    'buckets': dict(zip(self.buckets, self._bucket_counts[key])),
                    'sum': self._sums[key],
                    'count': self._counts[key]
                })
            return results
            
    def to_prometheus(self) -> str:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} histogram"
        ]
        with self._lock:
            for key in self._counts.keys():
                label_str = key.to_prometheus()
                cumulative = 0
                for i, bucket in enumerate(self.buckets):
                    cumulative += self._bucket_counts[key][i]
                    le = "+Inf" if bucket == float('inf') else bucket
                    if label_str:
                        bucket_label = label_str[:-1] + f',le="{le}"' + "}"
                    else:
                        bucket_label = f'{{le="{le}"}}'
                    lines.append(f"{self.name}_bucket{bucket_label} {cumulative}")
                    
                lines.append(f"{self.name}_sum{label_str} {self._sums[key]}")
                lines.append(f"{self.name}_count{label_str} {self._counts[key]}")
                
        return "\n".join(lines)


class Summary(Metric):
    """
    Summary metric - quantiles over sliding time window.
    
    Use for: latency percentiles
    """
    
    def __init__(
        self,
        name: str,
        help_text: str = "",
        quantiles: tuple = (0.5, 0.9, 0.95, 0.99),
        max_age_seconds: float = 60.0
    ):
        super().__init__(name, help_text, MetricType.SUMMARY)
        self.quantiles = quantiles
        self.max_age = max_age_seconds
        self._observations: Dict[MetricLabels, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self._sums: Dict[MetricLabels, float] = defaultdict(float)
        self._counts: Dict[MetricLabels, int] = defaultdict(int)
        
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observation."""
        key = MetricLabels(labels or {})
        now = time.time()
        
        with self._lock:
            self._observations[key].append((now, value))
            self._sums[key] += value
            self._counts[key] += 1
            
            # Clean old observations
            cutoff = now - self.max_age
            obs = self._observations[key]
            while obs and obs[0][0] < cutoff:
                obs.popleft()
                
    def _calculate_quantiles(self, key: MetricLabels) -> Dict[float, float]:
        """Calculate current quantiles."""
        with self._lock:
            values = [v for _, v in self._observations[key]]
            
        if not values:
            return {q: 0 for q in self.quantiles}
            
        values.sort()
        result = {}
        
        for q in self.quantiles:
            idx = int(q * len(values))
            idx = min(idx, len(values) - 1)
            result[q] = values[idx]
            
        return result
        
    def collect(self) -> List[Dict[str, Any]]:
        with self._lock:
            results = []
            for key in self._counts.keys():
                results.append({
                    'labels': key.labels,
                    'quantiles': self._calculate_quantiles(key),
                    'sum': self._sums[key],
                    'count': self._counts[key]
                })
            return results
            
    def to_prometheus(self) -> str:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} summary"
        ]
        
        with self._lock:
            for key in self._counts.keys():
                label_str = key.to_prometheus()
                quantiles = self._calculate_quantiles(key)
                
                for q, v in quantiles.items():
                    if label_str:
                        q_label = label_str[:-1] + f',quantile="{q}"' + "}"
                    else:
                        q_label = f'{{quantile="{q}"}}'
                    lines.append(f"{self.name}{q_label} {v}")
                    
                lines.append(f"{self.name}_sum{label_str} {self._sums[key]}")
                lines.append(f"{self.name}_count{label_str} {self._counts[key]}")
                
        return "\n".join(lines)


# ============================================================================
# Metrics Registry
# ============================================================================

class MetricsRegistry:
    """
    Central registry for all metrics.
    
    Provides:
    - Metric creation and registration
    - Prometheus exposition endpoint
    - JSON export for dashboards
    """
    
    def __init__(self, prefix: str = "hiveframe"):
        self.prefix = prefix
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()
        
    def _full_name(self, name: str) -> str:
        return f"{self.prefix}_{name}"
        
    def counter(self, name: str, help_text: str = "") -> Counter:
        """Get or create a counter metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Counter(full_name, help_text)
            return self._metrics[full_name]
            
    def gauge(self, name: str, help_text: str = "") -> Gauge:
        """Get or create a gauge metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Gauge(full_name, help_text)
            return self._metrics[full_name]
            
    def histogram(
        self,
        name: str,
        help_text: str = "",
        buckets: tuple = Histogram.DEFAULT_BUCKETS
    ) -> Histogram:
        """Get or create a histogram metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Histogram(full_name, help_text, buckets)
            return self._metrics[full_name]
            
    def summary(
        self,
        name: str,
        help_text: str = "",
        quantiles: tuple = (0.5, 0.9, 0.95, 0.99)
    ) -> Summary:
        """Get or create a summary metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Summary(full_name, help_text, quantiles)
            return self._metrics[full_name]
            
    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus format."""
        with self._lock:
            return "\n\n".join(m.to_prometheus() for m in self._metrics.values())
            
    def to_json(self) -> Dict[str, Any]:
        """Export all metrics as JSON."""
        with self._lock:
            return {
                name: {
                    'type': m.metric_type.name,
                    'help': m.help_text,
                    'values': m.collect()
                }
                for name, m in self._metrics.items()
            }


# Global registry instance
_default_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Get the default metrics registry."""
    return _default_registry


# ============================================================================
# Structured Logging
# ============================================================================

class LogLevel(Enum):
    """Log levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogRecord:
    """Structured log record."""
    timestamp: float
    level: LogLevel
    message: str
    logger_name: str
    extra: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    
    def to_json(self) -> str:
        """Format as JSON."""
        data = {
            'timestamp': self.timestamp,
            'level': self.level.name,
            'message': self.message,
            'logger': self.logger_name,
            **self.extra
        }
        if self.exception:
            data['exception'] = self.exception
        return json.dumps(data)
        
    def to_text(self) -> str:
        """Format as human-readable text."""
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))
        extra_str = ' '.join(f'{k}={v}' for k, v in self.extra.items())
        line = f"[{ts}] {self.level.name:8} {self.logger_name}: {self.message}"
        if extra_str:
            line += f" | {extra_str}"
        if self.exception:
            line += f"\n{self.exception}"
        return line


class LogHandler(ABC):
    """Base class for log handlers."""
    
    @abstractmethod
    def handle(self, record: LogRecord) -> None:
        """Process a log record."""
        pass


class ConsoleHandler(LogHandler):
    """Write logs to console."""
    
    def __init__(self, format: str = 'text', stream=None):
        self.format = format
        self.stream = stream or sys.stderr
        self._lock = threading.Lock()
        
    def handle(self, record: LogRecord) -> None:
        with self._lock:
            if self.format == 'json':
                self.stream.write(record.to_json() + '\n')
            else:
                self.stream.write(record.to_text() + '\n')
            self.stream.flush()


class BufferedHandler(LogHandler):
    """Buffer logs in memory for testing/inspection."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        
    def handle(self, record: LogRecord) -> None:
        with self._lock:
            self._buffer.append(record)
            
    def get_logs(self, level: Optional[LogLevel] = None, limit: int = 100) -> List[LogRecord]:
        """Get buffered logs."""
        with self._lock:
            logs = list(self._buffer)
            
        if level:
            logs = [l for l in logs if l.level.value >= level.value]
            
        return logs[-limit:]
        
    def clear(self) -> None:
        """Clear buffer."""
        with self._lock:
            self._buffer.clear()


class Logger:
    """
    Structured logger with context support.
    
    Features:
    - Structured key-value logging
    - Context propagation
    - Multiple handlers
    - Level filtering
    """
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        handlers: Optional[List[LogHandler]] = None
    ):
        self.name = name
        self.level = level
        self.handlers = handlers or [ConsoleHandler()]
        self._context: Dict[str, Any] = {}
        
    def with_context(self, **kwargs) -> 'Logger':
        """Create child logger with additional context."""
        child = Logger(self.name, self.level, self.handlers)
        child._context = {**self._context, **kwargs}
        return child
        
    def _log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Internal logging method."""
        if level.value < self.level.value:
            return
            
        exc_info = kwargs.pop('exc_info', False)
        exception = None
        if exc_info:
            exception = traceback.format_exc()
            
        record = LogRecord(
            timestamp=time.time(),
            level=level,
            message=message,
            logger_name=self.name,
            extra={**self._context, **kwargs},
            exception=exception
        )
        
        for handler in self.handlers:
            try:
                handler.handle(record)
            except Exception:
                pass  # Don't let logging failures break the app
                
    def debug(self, message: str, **kwargs) -> None:
        self._log(LogLevel.DEBUG, message, **kwargs)
        
    def info(self, message: str, **kwargs) -> None:
        self._log(LogLevel.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs) -> None:
        self._log(LogLevel.WARNING, message, **kwargs)
        
    def error(self, message: str, **kwargs) -> None:
        self._log(LogLevel.ERROR, message, **kwargs)
        
    def critical(self, message: str, **kwargs) -> None:
        self._log(LogLevel.CRITICAL, message, **kwargs)
        
    def exception(self, message: str, **kwargs) -> None:
        """Log error with exception traceback."""
        self._log(LogLevel.ERROR, message, exc_info=True, **kwargs)


def get_logger(name: str) -> Logger:
    """Get a logger instance."""
    return Logger(name)


# ============================================================================
# Colony Health Monitor
# ============================================================================

@dataclass
class WorkerHealthSnapshot:
    """Point-in-time health snapshot of a worker."""
    worker_id: str
    role: str
    processed_count: int
    error_count: int
    last_activity: float
    current_load: float
    avg_latency: float
    status: str  # 'healthy', 'degraded', 'unhealthy', 'dead'


@dataclass
class ColonyHealthReport:
    """Overall colony health report."""
    timestamp: float
    total_workers: int
    healthy_workers: int
    degraded_workers: int
    unhealthy_workers: int
    dead_workers: int
    overall_status: str
    temperature: float
    throughput: float
    error_rate: float
    worker_snapshots: List[WorkerHealthSnapshot]
    alerts: List[str]


class ColonyHealthMonitor:
    """
    Monitors colony health using bee-inspired metrics.
    
    Health indicators:
    - Temperature: Overall colony load/stress
    - Worker distribution: Balance across roles
    - Waggle dance frequency: Communication activity
    - Pheromone levels: Coordination signals
    - Abandonment rate: Food source quality
    """
    
    def __init__(
        self,
        colony,  # ColonyState from core
        check_interval: float = 5.0,
        unhealthy_threshold: float = 0.7,
        dead_threshold: float = 30.0  # Seconds without activity
    ):
        self.colony = colony
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.dead_threshold = dead_threshold
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._history: deque = deque(maxlen=100)
        self._alerts: List[str] = []
        self._lock = threading.Lock()
        
        # Metrics
        self._registry = get_registry()
        self._temperature_gauge = self._registry.gauge(
            "colony_temperature", "Current colony temperature (load)"
        )
        self._worker_gauge = self._registry.gauge(
            "workers_by_status", "Workers by health status"
        )
        self._throughput_gauge = self._registry.gauge(
            "colony_throughput", "Records processed per second"
        )
        self._error_rate_gauge = self._registry.gauge(
            "colony_error_rate", "Error rate (errors per record)"
        )
        
    def start(self) -> None:
        """Start background health monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                report = self._collect_health()
                
                with self._lock:
                    self._history.append(report)
                    
                # Update metrics
                self._temperature_gauge.set(report.temperature)
                self._worker_gauge.set(report.healthy_workers, {'status': 'healthy'})
                self._worker_gauge.set(report.degraded_workers, {'status': 'degraded'})
                self._worker_gauge.set(report.unhealthy_workers, {'status': 'unhealthy'})
                self._worker_gauge.set(report.dead_workers, {'status': 'dead'})
                self._throughput_gauge.set(report.throughput)
                self._error_rate_gauge.set(report.error_rate)
                
            except Exception:
                pass
                
            time.sleep(self.check_interval)
            
    def _collect_health(self) -> ColonyHealthReport:
        """Collect current health metrics."""
        now = time.time()
        workers = []
        alerts = []
        
        total_processed = 0
        total_errors = 0
        total_load = 0
        
        # Collect worker stats (simulated - in real implementation
        # this would query actual worker state)
        temperature = self.colony.get_colony_temperature()
        
        # Analyze temperature
        if temperature > 0.9:
            alerts.append("CRITICAL: Colony temperature > 90% - severe overload")
        elif temperature > 0.8:
            alerts.append("WARNING: Colony temperature > 80% - high load")
            
        # Analyze pheromone levels
        throttle_level = self.colony.sense_pheromone('throttle')
        if throttle_level > 0.5:
            alerts.append(f"WARNING: High throttle pheromone ({throttle_level:.2f}) - backpressure active")
            
        alarm_level = self.colony.sense_pheromone('alarm')
        if alarm_level > 0.3:
            alerts.append(f"WARNING: Alarm pheromones detected ({alarm_level:.2f}) - errors occurring")
            
        # Analyze food source health
        abandoned = self.colony.get_abandoned_sources()
        if abandoned:
            alerts.append(f"INFO: {len(abandoned)} food sources abandoned - scout bees reassigning")
            
        # Calculate overall status
        if temperature > 0.9 or alarm_level > 0.5:
            overall_status = 'unhealthy'
        elif temperature > 0.7 or throttle_level > 0.5:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
            
        return ColonyHealthReport(
            timestamp=now,
            total_workers=len(self.colony.temperature),
            healthy_workers=sum(1 for t in self.colony.temperature.values() if t < 0.5),
            degraded_workers=sum(1 for t in self.colony.temperature.values() if 0.5 <= t < 0.8),
            unhealthy_workers=sum(1 for t in self.colony.temperature.values() if t >= 0.8),
            dead_workers=0,  # Would need to track last activity times
            overall_status=overall_status,
            temperature=temperature,
            throughput=0,  # Would need to calculate from time series
            error_rate=0,  # Would need to track errors
            worker_snapshots=workers,
            alerts=alerts
        )
        
    def get_current_health(self) -> ColonyHealthReport:
        """Get latest health report."""
        with self._lock:
            if self._history:
                return self._history[-1]
        return self._collect_health()
        
    def get_health_history(self, limit: int = 20) -> List[ColonyHealthReport]:
        """Get historical health reports."""
        with self._lock:
            return list(self._history)[-limit:]


# ============================================================================
# Execution Tracer
# ============================================================================

@dataclass
class TraceSpan:
    """A span in a distributed trace."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = 'ok'  # 'ok', 'error'
    
    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """
    Distributed tracing for colony operations.
    
    Tracks execution flow across workers and partitions.
    """
    
    def __init__(self):
        self._spans: Dict[str, List[TraceSpan]] = defaultdict(list)
        self._lock = threading.Lock()
        self._span_counter = 0
        
    def _generate_id(self) -> str:
        """Generate unique span ID."""
        with self._lock:
            self._span_counter += 1
            return f"{time.time_ns()}_{self._span_counter}"
            
    def start_trace(self, operation: str, **tags) -> TraceSpan:
        """Start a new trace."""
        trace_id = self._generate_id()
        span = TraceSpan(
            trace_id=trace_id,
            span_id=self._generate_id(),
            parent_span_id=None,
            operation=operation,
            start_time=time.time(),
            tags=tags
        )
        
        with self._lock:
            self._spans[trace_id].append(span)
            
        return span
        
    def start_span(
        self,
        trace_id: str,
        operation: str,
        parent_span_id: Optional[str] = None,
        **tags
    ) -> TraceSpan:
        """Start a new span within a trace."""
        span = TraceSpan(
            trace_id=trace_id,
            span_id=self._generate_id(),
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time(),
            tags=tags
        )
        
        with self._lock:
            self._spans[trace_id].append(span)
            
        return span
        
    def end_span(self, span: TraceSpan, status: str = 'ok') -> None:
        """End a span."""
        span.end_time = time.time()
        span.status = status
        
    def log_to_span(self, span: TraceSpan, message: str, **fields) -> None:
        """Add log entry to span."""
        span.logs.append({
            'timestamp': time.time(),
            'message': message,
            **fields
        })
        
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        with self._lock:
            return list(self._spans.get(trace_id, []))
            
    @contextmanager
    def trace_operation(self, operation: str, **tags):
        """Context manager for tracing an operation."""
        span = self.start_trace(operation, **tags)
        try:
            yield span
            self.end_span(span, 'ok')
        except Exception as e:
            self.log_to_span(span, f"Error: {e}")
            self.end_span(span, 'error')
            raise


# Global tracer instance
_default_tracer = Tracer()


def get_tracer() -> Tracer:
    """Get the default tracer."""
    return _default_tracer


# ============================================================================
# Performance Profiler
# ============================================================================

class PerformanceProfiler:
    """
    Profile performance of colony operations.
    
    Tracks timing, resource usage, and bottlenecks.
    """
    
    def __init__(self):
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
    @contextmanager
    def profile(self, operation: str):
        """Profile a block of code."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                self._timings[operation].append(elapsed)
                # Keep only recent timings
                if len(self._timings[operation]) > 1000:
                    self._timings[operation] = self._timings[operation][-1000:]
                    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get timing statistics for an operation."""
        with self._lock:
            timings = self._timings.get(operation, [])
            
        if not timings:
            return {
                'count': 0,
                'mean': 0,
                'min': 0,
                'max': 0,
                'p50': 0,
                'p95': 0,
                'p99': 0
            }
            
        timings = sorted(timings)
        return {
            'count': len(timings),
            'mean': statistics.mean(timings),
            'min': min(timings),
            'max': max(timings),
            'p50': timings[int(len(timings) * 0.50)],
            'p95': timings[int(len(timings) * 0.95)] if len(timings) > 20 else max(timings),
            'p99': timings[int(len(timings) * 0.99)] if len(timings) > 100 else max(timings)
        }
        
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get stats for all profiled operations."""
        with self._lock:
            operations = list(self._timings.keys())
        return {op: self.get_stats(op) for op in operations}
        
    def report(self) -> str:
        """Generate human-readable performance report."""
        stats = self.get_all_stats()
        
        lines = ["Performance Report", "=" * 50]
        
        for op, s in sorted(stats.items(), key=lambda x: -x[1]['mean']):
            lines.append(f"\n{op}:")
            lines.append(f"  Count: {s['count']}")
            lines.append(f"  Mean:  {s['mean']*1000:.2f}ms")
            lines.append(f"  Min:   {s['min']*1000:.2f}ms")
            lines.append(f"  Max:   {s['max']*1000:.2f}ms")
            lines.append(f"  P95:   {s['p95']*1000:.2f}ms")
            
        return "\n".join(lines)


# Global profiler
_default_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the default profiler."""
    return _default_profiler
