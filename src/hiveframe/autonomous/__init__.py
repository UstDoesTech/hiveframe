"""
Autonomous Operations Module

This module provides self-managing capabilities for the HiveFrame platform,
including self-tuning, predictive maintenance, workload prediction, and cost optimization.
"""

from .cost_optimization import (
    CostMetrics,
    CostOptimizer,
    OptimizationRecommendation,
    OptimizationStrategy,
    SLAMetrics,
    SLAOptimizer,
    SpendAnalyzer,
)
from .predictive_maintenance import (
    FailurePrediction,
    FailurePredictor,
    HealthMetric,
    HealthMonitor,
    HealthStatus,
    PredictiveMaintenance,
    SystemHealth,
)
from .self_tuning import (
    MemoryManager,
    MemoryStats,
    QueryPerformance,
    QueryPredictor,
    ResourceAllocator,
    ResourceMetrics,
    SelfTuningColony,
)
from .workload_prediction import (
    ResourcePrewarmer,
    UsageAnalyzer,
    UsagePattern,
    WorkloadForecast,
    WorkloadPredictor,
    WorkloadSample,
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
