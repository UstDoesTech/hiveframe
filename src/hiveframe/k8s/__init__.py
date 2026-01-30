"""
HiveFrame Kubernetes Support
============================

Kubernetes deployment and operator support for HiveFrame.
Enables running HiveFrame clusters on Kubernetes with:

- Custom Resource Definitions (CRDs) for HiveFrame clusters
- Auto-scaling based on workload
- Resource management and quotas
- Integration with Kubernetes services

Usage:
    from hiveframe.k8s import HiveCluster, HiveOperator

    # Define a cluster
    cluster = HiveCluster(
        name="my-cluster",
        workers=10,
        worker_memory="4Gi",
        worker_cpu="2"
    )

    # Deploy to Kubernetes
    operator = HiveOperator()
    operator.deploy(cluster)
"""

from .cluster import (
    ClusterPhase,
    ClusterStatus,
    HiveCluster,
    ResourceRequirements,
    WorkerSpec,
)
from .manifests import (
    generate_configmap,
    generate_crd,
    generate_deployment,
    generate_service,
)
from .operator import (
    HiveOperator,
    OperatorConfig,
)

__all__ = [
    # Cluster
    "HiveCluster",
    "WorkerSpec",
    "ResourceRequirements",
    "ClusterStatus",
    "ClusterPhase",
    # Operator
    "HiveOperator",
    "OperatorConfig",
    # Manifests
    "generate_deployment",
    "generate_service",
    "generate_configmap",
    "generate_crd",
]
