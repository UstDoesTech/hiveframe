"""
Cost Optimization Engine

Minimize cloud spend while meeting SLAs using bee-inspired resource efficiency.
"""

import statistics
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class OptimizationStrategy(Enum):
    """Cost optimization strategies"""

    AGGRESSIVE = "aggressive"  # Minimize cost, may risk SLA
    BALANCED = "balanced"  # Balance cost and SLA
    CONSERVATIVE = "conservative"  # Prioritize SLA over cost


@dataclass
class CostMetrics:
    """Cost and usage metrics"""

    timestamp: float
    compute_cost_per_hour: float
    storage_cost_per_hour: float
    network_cost_per_hour: float
    total_cost_per_hour: float
    resource_utilization: float  # 0.0 to 1.0
    active_workers: int
    storage_gb: float


@dataclass
class SLAMetrics:
    """SLA performance metrics"""

    timestamp: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate: float
    availability: float


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""

    action: str
    estimated_savings_per_hour: float
    risk_level: str  # 'low', 'medium', 'high'
    description: str
    parameters: Dict[str, Any]


class SpendAnalyzer:
    """
    Analyze spending patterns and identify optimization opportunities.

    Like bees that optimize nectar collection efficiency, this analyzer
    finds ways to reduce resource consumption while maintaining productivity.
    """

    def __init__(self, budget_per_hour: Optional[float] = None):
        self.budget_per_hour = budget_per_hour
        self.cost_history: deque = deque(maxlen=168)  # 7 days hourly
        self.sla_history: deque = deque(maxlen=168)

    def record_metrics(self, cost: CostMetrics, sla: SLAMetrics) -> None:
        """Record cost and SLA metrics"""
        self.cost_history.append(cost)
        self.sla_history.append(sla)

    def analyze_spend(self) -> Dict[str, Any]:
        """
        Analyze spending patterns.

        Returns:
            Analysis with trends and insights
        """
        if not self.cost_history:
            return {
                "status": "insufficient_data",
                "total_cost": 0.0,
            }

        costs = list(self.cost_history)

        # Calculate totals and averages
        total_cost = sum(c.total_cost_per_hour for c in costs)
        avg_hourly = statistics.mean([c.total_cost_per_hour for c in costs])

        # Analyze utilization
        avg_utilization = statistics.mean([c.resource_utilization for c in costs])

        # Check for waste (low utilization with high cost)
        waste_hours = sum(
            1
            for c in costs
            if c.resource_utilization < 0.3 and c.total_cost_per_hour > avg_hourly * 0.5
        )

        # Budget status
        budget_status = "within_budget"
        if self.budget_per_hour:
            if avg_hourly > self.budget_per_hour:
                budget_status = "over_budget"
            elif avg_hourly > self.budget_per_hour * 0.9:
                budget_status = "approaching_limit"

        return {
            "status": "analyzed",
            "total_cost": total_cost,
            "avg_hourly_cost": avg_hourly,
            "avg_utilization": avg_utilization,
            "waste_hours": waste_hours,
            "budget_status": budget_status,
            "hours_analyzed": len(costs),
        }

    def identify_waste(self) -> List[Dict[str, Any]]:
        """
        Identify wasteful spending patterns.

        Returns:
            List of waste sources
        """
        waste_sources = []

        if len(self.cost_history) < 10:
            return waste_sources

        costs = list(self.cost_history)

        # Identify idle resources
        low_util_costs = [c for c in costs if c.resource_utilization < 0.2]
        if len(low_util_costs) > len(costs) * 0.2:  # 20% of time
            avg_idle_cost = statistics.mean([c.total_cost_per_hour for c in low_util_costs])
            waste_sources.append(
                {
                    "type": "idle_resources",
                    "frequency": len(low_util_costs) / len(costs),
                    "avg_cost": avg_idle_cost,
                    "estimated_savings": avg_idle_cost * 0.7,  # Could save 70%
                }
            )

        # Identify overprovisioning
        max_workers = max(c.active_workers for c in costs)
        avg_workers = statistics.mean([c.active_workers for c in costs])

        if max_workers > avg_workers * 2:
            waste_sources.append(
                {
                    "type": "overprovisioning",
                    "max_workers": max_workers,
                    "avg_workers": avg_workers,
                    "estimated_savings": (max_workers - avg_workers * 1.2) * 0.1,
                }
            )

        return waste_sources


class SLAOptimizer:
    """
    Optimize resources to meet SLAs at minimum cost.

    Balances cost and performance like bees balance energy expenditure
    with nectar collection efficiency.
    """

    def __init__(
        self,
        target_p95_ms: float = 1000.0,
        target_availability: float = 0.99,
        max_error_rate: float = 0.01,
    ):
        self.target_p95_ms = target_p95_ms
        self.target_availability = target_availability
        self.max_error_rate = max_error_rate

    def check_sla_compliance(self, sla: SLAMetrics) -> Dict[str, bool]:
        """Check if SLAs are being met"""
        return {
            "response_time": sla.p95_response_time_ms <= self.target_p95_ms,
            "availability": sla.availability >= self.target_availability,
            "error_rate": sla.error_rate <= self.max_error_rate,
            "overall": (
                sla.p95_response_time_ms <= self.target_p95_ms
                and sla.availability >= self.target_availability
                and sla.error_rate <= self.max_error_rate
            ),
        }

    def optimize_for_sla(
        self,
        current_cost: CostMetrics,
        current_sla: SLAMetrics,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    ) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations.

        Args:
            current_cost: Current cost metrics
            current_sla: Current SLA metrics
            strategy: Optimization strategy

        Returns:
            List of recommendations
        """
        recommendations = []
        compliance = self.check_sla_compliance(current_sla)

        if compliance["overall"]:
            # Meeting SLAs - can optimize for cost
            if current_cost.resource_utilization < 0.4:
                # Significant underutilization
                reduction = 0.3 if strategy == OptimizationStrategy.AGGRESSIVE else 0.2
                recommendations.append(
                    OptimizationRecommendation(
                        action="scale_down",
                        estimated_savings_per_hour=current_cost.total_cost_per_hour * reduction,
                        risk_level=(
                            "low" if strategy == OptimizationStrategy.AGGRESSIVE else "medium"
                        ),
                        description=f"Reduce workers by {int(reduction * 100)}% - low util",
                        parameters={
                            "worker_reduction": int(current_cost.active_workers * reduction)
                        },
                    )
                )
        else:
            # Not meeting SLAs - need to scale up
            if not compliance["response_time"]:
                recommendations.append(
                    OptimizationRecommendation(
                        action="scale_up",
                        estimated_savings_per_hour=-current_cost.total_cost_per_hour * 0.2,
                        risk_level="low",
                        description="Increase workers to improve response time",
                        parameters={
                            "worker_increase": max(1, int(current_cost.active_workers * 0.3))
                        },
                    )
                )

        # Check for expensive resources
        if current_cost.compute_cost_per_hour > current_cost.total_cost_per_hour * 0.7:
            # Compute-heavy workload
            recommendations.append(
                OptimizationRecommendation(
                    action="optimize_compute",
                    estimated_savings_per_hour=current_cost.compute_cost_per_hour * 0.15,
                    risk_level="low",
                    description="Consider spot instances or reserved capacity",
                    parameters={"recommendation": "use_spot_instances"},
                )
            )

        return recommendations


class CostOptimizer:
    """
    Cost optimization orchestrator.

    Coordinates spend analysis and SLA optimization to minimize costs
    while maintaining service quality, inspired by bee colony efficiency.
    """

    def __init__(
        self,
        budget_per_hour: Optional[float] = None,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    ):
        self.spend_analyzer = SpendAnalyzer(budget_per_hour)
        self.sla_optimizer = SLAOptimizer()
        self.strategy = strategy
        self.recommendations_history: List[OptimizationRecommendation] = []

    def record_metrics(self, cost: CostMetrics, sla: SLAMetrics) -> None:
        """Record cost and SLA metrics"""
        self.spend_analyzer.record_metrics(cost, sla)

    def optimize(self) -> Dict[str, Any]:
        """
        Perform comprehensive cost optimization.

        Returns:
            Optimization results with recommendations
        """
        # Analyze spending
        spend_analysis = self.spend_analyzer.analyze_spend()

        if spend_analysis["status"] == "insufficient_data":
            return {
                "status": "insufficient_data",
                "recommendations": [],
            }

        # Get recent metrics
        recent_cost = list(self.spend_analyzer.cost_history)[-1]
        recent_sla = list(self.spend_analyzer.sla_history)[-1]

        # Generate recommendations
        recommendations = self.sla_optimizer.optimize_for_sla(
            recent_cost,
            recent_sla,
            self.strategy,
        )

        # Identify waste
        waste_sources = self.spend_analyzer.identify_waste()

        # Calculate potential savings
        total_savings = sum(
            r.estimated_savings_per_hour
            for r in recommendations
            if r.estimated_savings_per_hour > 0
        )

        self.recommendations_history.extend(recommendations)

        return {
            "status": "optimized",
            "timestamp": time.time(),
            "spend_analysis": spend_analysis,
            "waste_sources": waste_sources,
            "recommendations": [
                {
                    "action": r.action,
                    "savings_per_hour": r.estimated_savings_per_hour,
                    "risk": r.risk_level,
                    "description": r.description,
                    "parameters": r.parameters,
                }
                for r in recommendations
            ],
            "total_potential_savings": total_savings,
            "strategy": self.strategy.value,
        }

    def apply_recommendation(self, recommendation: OptimizationRecommendation) -> Dict[str, Any]:
        """
        Apply an optimization recommendation.

        In a real implementation, this would:
        - Scale resources up or down
        - Switch to spot instances
        - Adjust cache sizes
        - etc.

        Returns:
            Application result
        """
        return {
            "status": "applied",
            "action": recommendation.action,
            "timestamp": time.time(),
            "parameters": recommendation.parameters,
        }
