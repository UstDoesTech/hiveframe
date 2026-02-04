"""
Autonomous Operations Module

This module provides self-managing capabilities for the HiveFrame platform,
including self-tuning, predictive maintenance, workload prediction, and cost optimization.
"""

from .self_tuning import (
    SelfTuningColony,
    MemoryManager,
    ResourceAllocator,
    QueryPredictor,
    MemoryStats,
    ResourceMetrics,
    QueryPerformance,
)
from .predictive_maintenance import (
    PredictiveMaintenance,
    HealthMonitor,
    FailurePredictor,
    HealthMetric,
    HealthStatus,
    FailurePrediction,
    SystemHealth,
)
from .workload_prediction import (
    WorkloadPredictor,
    UsageAnalyzer,
    ResourcePrewarmer,
    WorkloadSample,
    WorkloadForecast,
    UsagePattern,
)
from .cost_optimization import (
    CostOptimizer,
    SpendAnalyzer,
    SLAOptimizer,
    CostMetrics,
    SLAMetrics,
    OptimizationStrategy,
    OptimizationRecommendation,
)

__all__ = [
    "SelfTuningColony",
    "MemoryManager",
    "ResourceAllocator",
    "QueryPredictor",
    "MemoryStats",
    "ResourceMetrics",
    "QueryPerformance",
    "PredictiveMaintenance",
    "HealthMonitor",
    "FailurePredictor",
    "HealthMetric",
    "HealthStatus",
    "FailurePrediction",
    "SystemHealth",
    "WorkloadPredictor",
    "UsageAnalyzer",
    "ResourcePrewarmer",
    "WorkloadSample",
    "WorkloadForecast",
    "UsagePattern",
    "CostOptimizer",
    "SpendAnalyzer",
    "SLAOptimizer",
    "CostMetrics",
    "SLAMetrics",
    "OptimizationStrategy",
    "OptimizationRecommendation",
]
