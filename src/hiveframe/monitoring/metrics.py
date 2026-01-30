"""
HiveFrame Metrics
=================
Prometheus-style metrics for monitoring.

Provides:
- Counter: Monotonically increasing value
- Gauge: Value that can go up and down
- Histogram: Distribution of values in buckets
- Summary: Similar to histogram with quantiles
- MetricsRegistry: Central registry for all metrics
"""

import time
import threading
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum, auto
from contextlib import contextmanager


class MetricType(Enum):
    """Types of metrics following Prometheus conventions."""

    COUNTER = auto()  # Monotonically increasing value
    GAUGE = auto()  # Value that can go up and down
    HISTOGRAM = auto()  # Distribution of values in buckets
    SUMMARY = auto()  # Similar to histogram with quantiles


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
            return [{"labels": k.labels, "value": v} for k, v in self._values.items()]

    def to_prometheus(self) -> str:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} counter"]
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
            return [{"labels": k.labels, "value": v} for k, v in self._values.items()]

    def to_prometheus(self) -> str:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} gauge"]
        with self._lock:
            for labels, value in self._values.items():
                lines.append(f"{self.name}{labels.to_prometheus()} {value}")
        return "\n".join(lines)


class Histogram(Metric):
    """
    Histogram metric - distribution of values.

    Use for: latency, request sizes
    """

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float("inf"))

    def __init__(self, name: str, help_text: str = "", buckets: tuple = DEFAULT_BUCKETS):
        super().__init__(name, help_text, MetricType.HISTOGRAM)
        self.buckets = buckets
        self._bucket_counts: Dict[MetricLabels, List[int]] = defaultdict(lambda: [0] * len(buckets))
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
                results.append(
                    {
                        "labels": key.labels,
                        "buckets": dict(zip(self.buckets, self._bucket_counts[key])),
                        "sum": self._sums[key],
                        "count": self._counts[key],
                    }
                )
            return results

    def to_prometheus(self) -> str:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} histogram"]
        with self._lock:
            for key in self._counts.keys():
                label_str = key.to_prometheus()
                cumulative = 0
                for i, bucket in enumerate(self.buckets):
                    cumulative += self._bucket_counts[key][i]
                    le = "+Inf" if bucket == float("inf") else bucket
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
        max_age_seconds: float = 60.0,
    ):
        super().__init__(name, help_text, MetricType.SUMMARY)
        self.quantiles = quantiles
        self.max_age = max_age_seconds
        self._observations: Dict[MetricLabels, deque] = defaultdict(lambda: deque(maxlen=10000))
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
                results.append(
                    {
                        "labels": key.labels,
                        "quantiles": self._calculate_quantiles(key),
                        "sum": self._sums[key],
                        "count": self._counts[key],
                    }
                )
            return results

    def to_prometheus(self) -> str:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} summary"]

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
        self, name: str, help_text: str = "", buckets: tuple = Histogram.DEFAULT_BUCKETS
    ) -> Histogram:
        """Get or create a histogram metric."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Histogram(full_name, help_text, buckets)
            return self._metrics[full_name]

    def summary(
        self, name: str, help_text: str = "", quantiles: tuple = (0.5, 0.9, 0.95, 0.99)
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
                name: {"type": m.metric_type.name, "help": m.help_text, "values": m.collect()}
                for name, m in self._metrics.items()
            }


# Global registry instance
_default_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Get the default metrics registry."""
    return _default_registry
