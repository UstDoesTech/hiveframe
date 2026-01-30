"""
HiveFrame Query Optimizer
=========================

A bee-inspired query optimizer that uses swarm intelligence
principles to optimize query execution plans.

Key Features (Phase 1):
- Rule-based optimization (predicate pushdown, projection pruning)
- Cost-based optimization using fitness functions
- Adaptive optimization based on runtime feedback
- Waggle dance protocol for plan quality signaling

Phase 2 Additions:
- Vectorized Execution: SIMD-accelerated processing for numerical workloads
- Adaptive Query Execution: Real-time plan modification based on runtime stats
- Enhanced Cost-Based Optimizer: Fitness functions with swarm intelligence

This is HiveFrame's equivalent to Apache Spark's Catalyst optimizer,
but uses bio-inspired algorithms instead of traditional approaches.
"""

# Phase 2: Adaptive Query Execution
from .aqe import (
    AdaptationRule,
    AdaptationTrigger,
    AdaptiveQueryExecutor,
    AQEContext,
    JoinStrategy,
    JoinStrategySelector,
    PartitionCoalescer,
    RuntimeStatistics,
    SkewHandler,
    WaggleDanceFeedback,
)
from .cost import (
    CostEstimate,
    CostModel,
    Statistics,
    SwarmCostModel,
)
from .planner import (
    OptimizedPlan,
    PlanCandidate,
    QueryOptimizer,
)
from .rules import (
    ConstantFolding,
    FilterCombination,
    JoinReordering,
    LimitPushdown,
    OptimizationRule,
    PredicatePushdown,
    ProjectionPruning,
)

# Phase 2: Vectorized Execution
from .vectorized import (
    ParallelVectorizedExecutor,
    VectorBatch,
    VectorizedAggregate,
    VectorizedFilter,
    VectorizedJoin,
    VectorizedLimit,
    VectorizedOp,
    VectorizedPipeline,
    VectorizedProject,
    VectorizedSort,
    VectorType,
    create_vectorized_aggregate,
    create_vectorized_filter,
    create_vectorized_project,
)

__all__ = [
    # Rules
    "OptimizationRule",
    "PredicatePushdown",
    "ProjectionPruning",
    "ConstantFolding",
    "FilterCombination",
    "JoinReordering",
    "LimitPushdown",
    # Cost
    "CostModel",
    "SwarmCostModel",
    "CostEstimate",
    "Statistics",
    # Planner
    "QueryOptimizer",
    "OptimizedPlan",
    "PlanCandidate",
    # Phase 2: Vectorized Execution
    "VectorBatch",
    "VectorType",
    "VectorizedOp",
    "VectorizedFilter",
    "VectorizedProject",
    "VectorizedAggregate",
    "VectorizedJoin",
    "VectorizedSort",
    "VectorizedLimit",
    "VectorizedPipeline",
    "ParallelVectorizedExecutor",
    "create_vectorized_filter",
    "create_vectorized_project",
    "create_vectorized_aggregate",
    # Phase 2: Adaptive Query Execution
    "AdaptiveQueryExecutor",
    "AQEContext",
    "WaggleDanceFeedback",
    "RuntimeStatistics",
    "AdaptationTrigger",
    "AdaptationRule",
    "JoinStrategy",
    "JoinStrategySelector",
    "PartitionCoalescer",
    "SkewHandler",
]
