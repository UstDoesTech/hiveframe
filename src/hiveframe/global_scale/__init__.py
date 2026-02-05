"""
HiveFrame Global Scale Platform - Phase 5

Planet-scale infrastructure, edge computing, satellite integration,
and 5G/6G optimization for worldwide distributed data processing.
"""

from .edge_computing import (
    EdgeCloudSync,
    EdgeNodeManager,
    OfflineOperationSupport,
)
from .mesh_architecture import (
    CrossRegionReplicator,
    GlobalMeshCoordinator,
    LatencyAwareRouter,
)
from .mobile_optimization import (
    HandoffHandler,
    MobileAwareScheduler,
    NetworkSliceIntegration,
)
from .satellite_integration import (
    BandwidthOptimizer,
    DataBufferingStrategy,
    HighLatencyProtocol,
)

__all__ = [
    # Global Mesh Architecture
    "GlobalMeshCoordinator",
    "CrossRegionReplicator",
    "LatencyAwareRouter",
    # Edge Computing
    "EdgeNodeManager",
    "EdgeCloudSync",
    "OfflineOperationSupport",
    # Satellite Integration
    "HighLatencyProtocol",
    "BandwidthOptimizer",
    "DataBufferingStrategy",
    # Mobile Optimization
    "MobileAwareScheduler",
    "NetworkSliceIntegration",
    "HandoffHandler",
]
