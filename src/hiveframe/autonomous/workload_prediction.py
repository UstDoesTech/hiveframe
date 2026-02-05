"""
Workload Prediction Module

Predict future workloads and pre-warm resources based on usage patterns,
inspired by how bee colonies anticipate seasonal changes and prepare accordingly.
"""

import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class WorkloadSample:
    """Single workload measurement"""

    timestamp: float
    query_count: int
    cpu_percent: float
    memory_mb: float
    io_operations: int
    active_users: int


@dataclass
class WorkloadForecast:
    """Workload forecast"""

    start_time: float
    end_time: float
    predicted_query_count: int
    predicted_cpu_percent: float
    predicted_memory_mb: float
    confidence: float
    pattern_type: str  # 'periodic', 'trending', 'stable', 'unknown'


@dataclass
class UsagePattern:
    """Detected usage pattern"""

    pattern_type: str
    periodicity_hours: Optional[float]
    trend: float  # Positive = increasing, Negative = decreasing
    peak_hours: List[int]
    low_hours: List[int]


class UsageAnalyzer:
    """
    Analyze usage patterns using swarm intelligence.

    Like bees that learn flower blooming patterns, this analyzer
    learns workload patterns to enable proactive resource management.
    """

    def __init__(self, history_hours: int = 168):  # Default 7 days
        self.history_hours = history_hours
        self.samples: deque = deque(maxlen=history_hours * 60)  # Minute-level samples

    def record_sample(self, sample: WorkloadSample) -> None:
        """Record a workload sample"""
        self.samples.append(sample)

    def detect_patterns(self) -> UsagePattern:
        """
        Detect usage patterns from historical data.

        Returns:
            UsagePattern with detected characteristics
        """
        if len(self.samples) < 24 * 60:  # Need at least 24 hours
            return UsagePattern(
                pattern_type="unknown",
                periodicity_hours=None,
                trend=0.0,
                peak_hours=[],
                low_hours=[],
            )

        samples = list(self.samples)

        # Detect daily periodicity (bee-inspired: daily foraging cycles)
        hourly_avg_queries = self._compute_hourly_averages(samples)
        peak_hours = self._find_peak_hours(hourly_avg_queries)
        low_hours = self._find_low_hours(hourly_avg_queries)

        # Detect trend
        trend = self._compute_trend(samples)

        # Determine pattern type
        if self._is_periodic(hourly_avg_queries):
            pattern_type = "periodic"
            periodicity_hours = 24.0
        elif abs(trend) > 0.1:
            pattern_type = "trending"
            periodicity_hours = None
        else:
            pattern_type = "stable"
            periodicity_hours = None

        return UsagePattern(
            pattern_type=pattern_type,
            periodicity_hours=periodicity_hours,
            trend=trend,
            peak_hours=peak_hours,
            low_hours=low_hours,
        )

    def _compute_hourly_averages(self, samples: List[WorkloadSample]) -> Dict[int, float]:
        """Compute average query count by hour of day"""
        hourly_data: Dict[int, List[int]] = {h: [] for h in range(24)}

        for sample in samples:
            hour = int((sample.timestamp % 86400) // 3600)  # Hour of day (0-23)
            hourly_data[hour].append(sample.query_count)

        return {
            hour: statistics.mean(queries) if queries else 0
            for hour, queries in hourly_data.items()
        }

    def _find_peak_hours(
        self, hourly_avg: Dict[int, float], threshold_factor: float = 1.3
    ) -> List[int]:
        """Find hours with above-average activity"""
        if not hourly_avg:
            return []

        avg = statistics.mean(hourly_avg.values())
        return [hour for hour, value in hourly_avg.items() if value > avg * threshold_factor]

    def _find_low_hours(
        self, hourly_avg: Dict[int, float], threshold_factor: float = 0.7
    ) -> List[int]:
        """Find hours with below-average activity"""
        if not hourly_avg:
            return []

        avg = statistics.mean(hourly_avg.values())
        return [hour for hour, value in hourly_avg.items() if value < avg * threshold_factor]

    def _is_periodic(self, hourly_avg: Dict[int, float]) -> bool:
        """Check if pattern is periodic"""
        values = list(hourly_avg.values())
        if len(values) < 24:
            return False

        # Simple check: significant variation between hours
        stddev = statistics.stdev(values)
        mean_val = statistics.mean(values)
        cv = stddev / mean_val if mean_val > 0 else 0

        return cv > 0.3  # Coefficient of variation > 30%

    def _compute_trend(self, samples: List[WorkloadSample]) -> float:
        """Compute overall trend in workload"""
        if len(samples) < 100:
            return 0.0

        # Compare first and last quarters
        quarter_size = len(samples) // 4
        first_quarter = samples[:quarter_size]
        last_quarter = samples[-quarter_size:]

        first_avg = statistics.mean([s.query_count for s in first_quarter])
        last_avg = statistics.mean([s.query_count for s in last_quarter])

        return (last_avg - first_avg) / first_avg if first_avg > 0 else 0.0


class WorkloadPredictor:
    """
    Predict future workload using bee-inspired forecasting.

    Combines pattern recognition with adaptive learning to forecast
    resource needs, similar to how colonies predict resource requirements.
    """

    def __init__(self):
        self.analyzer = UsageAnalyzer()
        self.forecasts: List[WorkloadForecast] = []

    def record_workload(self, sample: WorkloadSample) -> None:
        """Record workload sample"""
        self.analyzer.record_sample(sample)

    def predict(self, hours_ahead: int = 1) -> WorkloadForecast:
        """
        Predict workload for future time period.

        Args:
            hours_ahead: Number of hours to predict ahead

        Returns:
            WorkloadForecast for the specified period
        """
        pattern = self.analyzer.detect_patterns()
        samples = list(self.analyzer.samples)

        if not samples:
            # No data - return conservative estimate
            return WorkloadForecast(
                start_time=time.time(),
                end_time=time.time() + hours_ahead * 3600,
                predicted_query_count=0,
                predicted_cpu_percent=10.0,
                predicted_memory_mb=100.0,
                confidence=0.1,
                pattern_type="unknown",
            )

        current_time = time.time()
        future_time = current_time + hours_ahead * 3600
        future_hour = int((future_time % 86400) // 3600)

        # Base prediction on recent average
        recent = samples[-60:] if len(samples) >= 60 else samples
        base_queries = statistics.mean([s.query_count for s in recent])
        base_cpu = statistics.mean([s.cpu_percent for s in recent])
        base_memory = statistics.mean([s.memory_mb for s in recent])

        # Adjust based on pattern
        multiplier = 1.0
        confidence = 0.5

        if pattern.pattern_type == "periodic" and future_hour in pattern.peak_hours:
            multiplier = 1.5
            confidence = 0.8
        elif pattern.pattern_type == "periodic" and future_hour in pattern.low_hours:
            multiplier = 0.6
            confidence = 0.8
        elif pattern.pattern_type == "trending":
            # Apply trend
            multiplier = 1.0 + (pattern.trend * hours_ahead / 24)
            confidence = 0.7

        forecast = WorkloadForecast(
            start_time=future_time,
            end_time=future_time + 3600,
            predicted_query_count=int(base_queries * multiplier),
            predicted_cpu_percent=base_cpu * multiplier,
            predicted_memory_mb=base_memory * multiplier,
            confidence=confidence,
            pattern_type=pattern.pattern_type,
        )

        self.forecasts.append(forecast)
        return forecast


class ResourcePrewarmer:
    """
    Pre-warm resources based on workload predictions.

    Proactively allocates resources before demand spikes, like bees
    preparing the hive for incoming foragers.
    """

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self.prewarm_actions: List[Dict[str, Any]] = []

    def plan_prewarming(self, forecast: WorkloadForecast) -> Optional[Dict[str, Any]]:
        """
        Plan resource pre-warming based on forecast.

        Args:
            forecast: Workload forecast

        Returns:
            Pre-warming plan if warranted, None otherwise
        """
        if forecast.confidence < self.min_confidence:
            return None

        # Determine if pre-warming is needed
        predicted_load = forecast.predicted_query_count

        if predicted_load < 100:
            # Low load - no pre-warming needed
            return None

        # Calculate required resources
        estimated_workers = max(1, predicted_load // 100)
        estimated_memory_mb = forecast.predicted_memory_mb

        plan = {
            "action": "prewarm",
            "scheduled_time": forecast.start_time - 300,  # 5 minutes before
            "duration_sec": 300,
            "workers": estimated_workers,
            "memory_mb": estimated_memory_mb,
            "confidence": forecast.confidence,
            "reason": f"Predicted {predicted_load} queries ({forecast.pattern_type})",
        }

        self.prewarm_actions.append(plan)
        return plan

    def execute_prewarm(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute pre-warming plan.

        In a real implementation, this would:
        - Spin up additional workers
        - Pre-allocate memory buffers
        - Warm up caches
        - Open database connections

        Returns:
            Execution result
        """
        return {
            "status": "completed",
            "workers_started": plan["workers"],
            "memory_allocated_mb": plan["memory_mb"],
            "timestamp": time.time(),
        }
