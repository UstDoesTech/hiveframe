"""
HiveFrame Query Optimizer
=========================

A bee-inspired query optimizer that uses swarm intelligence
principles to optimize query execution plans.

Key Features:
- Rule-based optimization (predicate pushdown, projection pruning)
- Cost-based optimization using fitness functions
- Adaptive optimization based on runtime feedback
- Waggle dance protocol for plan quality signaling

This is HiveFrame's equivalent to Apache Spark's Catalyst optimizer,
but uses bio-inspired algorithms instead of traditional approaches.
"""

from .rules import (
    OptimizationRule,
    PredicatePushdown,
    ProjectionPruning,
    ConstantFolding,
    FilterCombination,
    JoinReordering,
    LimitPushdown,
)
from .cost import (
    CostModel,
    SwarmCostModel,
    CostEstimate,
    Statistics,
)
from .planner import (
    QueryOptimizer,
    OptimizedPlan,
    PlanCandidate,
)

__all__ = [
    # Rules
    'OptimizationRule',
    'PredicatePushdown',
    'ProjectionPruning',
    'ConstantFolding',
    'FilterCombination',
    'JoinReordering',
    'LimitPushdown',
    # Cost
    'CostModel',
    'SwarmCostModel',
    'CostEstimate',
    'Statistics',
    # Planner
    'QueryOptimizer',
    'OptimizedPlan',
    'PlanCandidate',
]
