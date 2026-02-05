"""
Predictive Maintenance Module

Anticipate and prevent failures before they occur using bee-inspired health monitoring
and swarm intelligence for anomaly detection.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
from enum import Enum
import statistics


class HealthStatus(Enum):
    """Health status levels"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILING = "failing"


@dataclass
class HealthMetric:
    """Individual health metric"""

    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    timestamp: float = field(default_factory=time.time)

    @property
    def status(self) -> HealthStatus:
        """Determine status based on thresholds"""
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.WARNING
        return HealthStatus.HEALTHY


@dataclass
class FailurePrediction:
    """Failure prediction result"""

    component: str
    probability: float
    estimated_time_to_failure_hours: Optional[float]
    confidence: float
    contributing_factors: List[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemHealth:
    """Overall system health assessment"""

    status: HealthStatus
    metrics: Dict[str, HealthMetric]
    score: float  # 0-100
    issues: List[str]
    timestamp: float = field(default_factory=time.time)


class HealthMonitor:
    """
    Continuous health monitoring using bee-inspired inspection patterns.

    Like scout bees that patrol the hive checking for issues, this monitor
    continuously tracks system health indicators and raises alerts.
    """

    def __init__(self):
        self.metrics: Dict[str, deque] = {}
        self.alert_history: List[Dict[str, Any]] = []

    def record_metric(self, metric: HealthMetric) -> None:
        """Record a health metric"""
        if metric.name not in self.metrics:
            self.metrics[metric.name] = deque(maxlen=100)
        self.metrics[metric.name].append(metric)

    def check_health(self) -> SystemHealth:
        """
        Perform comprehensive health check.

        Returns:
            SystemHealth object with current status
        """
        if not self.metrics:
            return SystemHealth(
                status=HealthStatus.HEALTHY,
                metrics={},
                score=100.0,
                issues=[],
            )

        # Get latest metric for each type
        latest_metrics = {}
        statuses = []
        issues = []

        for name, history in self.metrics.items():
            if not history:
                continue
            latest = history[-1]
            latest_metrics[name] = latest
            statuses.append(latest.status)

            if latest.status == HealthStatus.CRITICAL:
                issues.append(f"{name} is critical: {latest.value:.2f}")
            elif latest.status == HealthStatus.WARNING:
                issues.append(f"{name} is warning: {latest.value:.2f}")

        # Determine overall status (worst status wins)
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.FAILING in statuses:
            overall_status = HealthStatus.FAILING
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY

        # Calculate health score (bee-inspired: colony vitality)
        healthy_count = statuses.count(HealthStatus.HEALTHY)
        score = (healthy_count / len(statuses) * 100) if statuses else 100.0

        return SystemHealth(
            status=overall_status,
            metrics=latest_metrics,
            score=score,
            issues=issues,
        )

    def get_trends(self, metric_name: str, window_size: int = 20) -> Dict[str, float]:
        """
        Analyze trends for a specific metric.

        Returns:
            Dictionary with trend analysis
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) < 2:
            return {"trend": 0.0, "confidence": 0.0}

        recent = list(self.metrics[metric_name])[-window_size:]
        values = [m.value for m in recent]

        if len(values) < 2:
            return {"trend": 0.0, "confidence": 0.0}

        # Simple linear trend (could be enhanced with proper regression)
        first_half_avg = statistics.mean(values[: len(values) // 2])
        second_half_avg = statistics.mean(values[len(values) // 2 :])

        trend = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0.0
        confidence = min(len(values) / window_size, 1.0)

        return {
            "trend": trend,
            "confidence": confidence,
            "increasing": trend > 0.1,
            "decreasing": trend < -0.1,
        }


class FailurePredictor:
    """
    Predict failures before they occur using swarm intelligence.

    Analyzes patterns and anomalies to forecast potential failures,
    similar to how bee colonies sense environmental threats.
    """

    def __init__(self):
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.predictions: List[FailurePrediction] = []

    def analyze_component(
        self,
        component: str,
        metrics: Dict[str, deque],
    ) -> Optional[FailurePrediction]:
        """
        Analyze a component for potential failure.

        Args:
            component: Component name
            metrics: Historical metrics for the component

        Returns:
            FailurePrediction if failure is predicted, None otherwise
        """
        if not metrics:
            return None

        contributing_factors = []
        risk_score = 0.0

        # Check for concerning patterns
        for metric_name, history in metrics.items():
            if len(history) < 10:
                continue

            recent = list(history)[-20:]
            values = [m.value for m in recent if isinstance(m, HealthMetric)]

            if not values:
                continue

            # Check for degradation trend
            if len(values) >= 4:
                first_quarter = statistics.mean(values[: len(values) // 4])
                last_quarter = statistics.mean(values[-len(values) // 4 :])

                if last_quarter > first_quarter * 1.5:
                    contributing_factors.append(f"{metric_name}_increasing")
                    risk_score += 0.2

            # Check for high variability (instability)
            if len(values) >= 2:
                stddev = statistics.stdev(values)
                mean_val = statistics.mean(values)
                cv = stddev / mean_val if mean_val > 0 else 0

                if cv > 0.5:
                    contributing_factors.append(f"{metric_name}_unstable")
                    risk_score += 0.15

            # Check for threshold breaches
            critical_breaches = sum(
                1 for m in recent if isinstance(m, HealthMetric) and m.value >= m.threshold_critical
            )
            if critical_breaches > len(recent) * 0.3:
                contributing_factors.append(f"{metric_name}_critical_breaches")
                risk_score += 0.3

        # Predict failure if risk is significant
        if risk_score > 0.4:
            # Estimate time to failure based on trend severity
            estimated_hours = max(1.0, (1.0 - risk_score) * 48)  # 1-48 hours

            prediction = FailurePrediction(
                component=component,
                probability=min(risk_score, 0.95),
                estimated_time_to_failure_hours=estimated_hours,
                confidence=min(risk_score * 1.2, 0.9),
                contributing_factors=contributing_factors,
            )

            self.predictions.append(prediction)
            return prediction

        return None

    def predict_failures(
        self,
        system_metrics: Dict[str, Dict[str, deque]],
    ) -> List[FailurePrediction]:
        """
        Predict failures across all system components.

        Args:
            system_metrics: Metrics organized by component

        Returns:
            List of failure predictions
        """
        predictions = []

        for component, metrics in system_metrics.items():
            prediction = self.analyze_component(component, metrics)
            if prediction:
                predictions.append(prediction)

        return predictions


class PredictiveMaintenance:
    """
    Predictive maintenance orchestrator.

    Coordinates health monitoring and failure prediction to enable
    proactive maintenance, inspired by how bee colonies maintain
    hive integrity through continuous inspection and repair.
    """

    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.failure_predictor = FailurePredictor()
        self.maintenance_schedule: List[Dict[str, Any]] = []

    def record_metric(self, metric: HealthMetric) -> None:
        """Record a health metric"""
        self.health_monitor.record_metric(metric)

    def assess_system(self) -> Dict[str, Any]:
        """
        Perform comprehensive system assessment.

        Returns:
            Dictionary with health status and predictions
        """
        # Check current health
        health = self.health_monitor.check_health()

        # Predict potential failures
        component_metrics = {"system": self.health_monitor.metrics}
        predictions = self.failure_predictor.predict_failures(component_metrics)

        # Generate maintenance recommendations
        recommendations = []

        if health.status in [HealthStatus.CRITICAL, HealthStatus.FAILING]:
            recommendations.append(
                {
                    "priority": "immediate",
                    "action": "investigate_critical_issues",
                    "issues": health.issues,
                }
            )

        for prediction in predictions:
            if prediction.probability > 0.7:
                recommendations.append(
                    {
                        "priority": "high",
                        "action": f"preventive_maintenance_{prediction.component}",
                        "estimated_time": prediction.estimated_time_to_failure_hours,
                    }
                )

        return {
            "health": {
                "status": health.status.value,
                "score": health.score,
                "issues": health.issues,
            },
            "predictions": [
                {
                    "component": p.component,
                    "probability": p.probability,
                    "time_to_failure_hours": p.estimated_time_to_failure_hours,
                    "factors": p.contributing_factors,
                }
                for p in predictions
            ],
            "recommendations": recommendations,
            "timestamp": time.time(),
        }

    def schedule_maintenance(
        self,
        component: str,
        action: str,
        scheduled_time: float,
    ) -> None:
        """Schedule a maintenance action"""
        self.maintenance_schedule.append(
            {
                "component": component,
                "action": action,
                "scheduled_time": scheduled_time,
                "status": "scheduled",
            }
        )
