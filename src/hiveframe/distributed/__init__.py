"""
HiveFrame Distributed Execution Engine (Phase 2)
================================================

A swarm intelligence-based distributed execution engine that enables
multi-hive federation, cross-datacenter coordination, adaptive partitioning,
and speculative execution.

Key Features:
- Multi-hive Federation: Connect multiple HiveFrame clusters
- Cross-datacenter Swarm: Global task distribution with locality awareness
- Adaptive Partitioning: Dynamic partition splitting/merging based on fitness
- Speculative Execution: Scout bees proactively retry slow tasks

This module is part of Phase 2: Swarm Intelligence.
"""

from .federation import (
    HiveFederation,
    FederatedHive,
    FederationCoordinator,
    HiveRegistry,
    FederationProtocol,
    HiveHealth,
)
from .partitioning import (
    AdaptivePartitioner,
    PartitionStrategy,
    PartitionState,
    FitnessPartitioner,
    PartitionSplitter,
    PartitionMerger,
)
from .speculative import (
    SpeculativeExecutor,
    ScoutTaskRunner,
    TaskTracker,
    SlowTaskDetector,
    SpeculativeConfig,
)
from .locality import (
    LocalityAwareScheduler,
    DataLocality,
    LocalityLevel,
    LocalityHint,
    CrossDatacenterManager,
)

__all__ = [
    # Federation
    "HiveFederation",
    "FederatedHive",
    "FederationCoordinator",
    "HiveRegistry",
    "FederationProtocol",
    "HiveHealth",
    # Adaptive Partitioning
    "AdaptivePartitioner",
    "PartitionStrategy",
    "PartitionState",
    "FitnessPartitioner",
    "PartitionSplitter",
    "PartitionMerger",
    # Speculative Execution
    "SpeculativeExecutor",
    "ScoutTaskRunner",
    "TaskTracker",
    "SlowTaskDetector",
    "SpeculativeConfig",
    # Locality
    "LocalityAwareScheduler",
    "DataLocality",
    "LocalityLevel",
    "LocalityHint",
    "CrossDatacenterManager",
]
