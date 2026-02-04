"""
HiveFrame Global Scale Platform - Phase 5

Planet-scale infrastructure, edge computing, satellite integration,
and 5G/6G optimization for worldwide distributed data processing.
"""

from .mesh_architecture import (
    GlobalMeshCoordinator,
    CrossRegionReplicator,
    LatencyAwareRouter,
)
from .edge_computing import (
    EdgeNodeManager,
    EdgeCloudSync,
    OfflineOperationSupport,
)
from .satellite_integration import (
    HighLatencyProtocol,
    BandwidthOptimizer,
    DataBufferingStrategy,
)
from .mobile_optimization import (
    MobileAwareScheduler,
    NetworkSliceIntegration,
    HandoffHandler,
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
