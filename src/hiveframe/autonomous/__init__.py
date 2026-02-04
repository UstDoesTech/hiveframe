"""
Autonomous Operations Module

This module provides self-managing capabilities for the HiveFrame platform,
including self-tuning, predictive maintenance, workload prediction, and cost optimization.
"""

from .self_tuning import SelfTuningColony, MemoryManager, ResourceAllocator, QueryPredictor
from .predictive_maintenance import PredictiveMaintenance, HealthMonitor, FailurePredictor
from .workload_prediction import WorkloadPredictor, UsageAnalyzer, ResourcePrewarmer
from .cost_optimization import CostOptimizer, SpendAnalyzer, SLAOptimizer

__all__ = [
    "SelfTuningColony",
    "MemoryManager",
    "ResourceAllocator",
    "QueryPredictor",
    "PredictiveMaintenance",
    "HealthMonitor",
    "FailurePredictor",
    "WorkloadPredictor",
    "UsageAnalyzer",
    "ResourcePrewarmer",
    "CostOptimizer",
    "SpendAnalyzer",
    "SLAOptimizer",
]
