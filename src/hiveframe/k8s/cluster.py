"""
HiveFrame Cluster Definition
============================

Defines the HiveFrame cluster specification for Kubernetes deployment.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ClusterPhase(Enum):
    """Cluster lifecycle phases."""

    PENDING = "Pending"
    CREATING = "Creating"
    RUNNING = "Running"
    SCALING = "Scaling"
    UPDATING = "Updating"
    FAILED = "Failed"
    TERMINATING = "Terminating"


@dataclass
class ResourceRequirements:
    """
    Resource Requirements
    ---------------------
    Kubernetes resource specifications for workers.
    """

    cpu: str = "1"  # CPU cores (e.g., "1", "500m")
    memory: str = "2Gi"  # Memory (e.g., "2Gi", "512Mi")
    ephemeral_storage: Optional[str] = None
    gpu: Optional[int] = None  # Number of GPUs

    def to_k8s_resources(self) -> Dict[str, Any]:
        """Convert to Kubernetes resource specification."""
        resources = {
            "requests": {
                "cpu": self.cpu,
                "memory": self.memory,
            },
            "limits": {
                "cpu": self.cpu,
                "memory": self.memory,
            },
        }

        if self.ephemeral_storage:
            resources["requests"]["ephemeral-storage"] = self.ephemeral_storage
            resources["limits"]["ephemeral-storage"] = self.ephemeral_storage

        if self.gpu:
            resources["limits"]["nvidia.com/gpu"] = str(self.gpu)

        return resources


@dataclass
class WorkerSpec:
    """
    Worker Specification
    --------------------
    Specification for HiveFrame worker pods.
    """

    replicas: int = 3
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    image: str = "hiveframe/worker:latest"
    image_pull_policy: str = "IfNotPresent"
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    affinity: Optional[Dict[str, Any]] = None

    # Worker role distribution (sum should be 1.0)
    employed_ratio: float = 0.5
    onlooker_ratio: float = 0.4
    scout_ratio: float = 0.1

    # ABC algorithm parameters
    abandonment_limit: int = 10
    max_cycles: int = 100

    def to_k8s_spec(self) -> Dict[str, Any]:
        """Convert to Kubernetes pod spec."""
        spec = {
            "replicas": self.replicas,
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "hiveframe-worker",
                            "image": self.image,
                            "imagePullPolicy": self.image_pull_policy,
                            "resources": self.resources.to_k8s_resources(),
                            "env": [
                                {"name": "HIVE_EMPLOYED_RATIO", "value": str(self.employed_ratio)},
                                {"name": "HIVE_ONLOOKER_RATIO", "value": str(self.onlooker_ratio)},
                                {"name": "HIVE_SCOUT_RATIO", "value": str(self.scout_ratio)},
                                {
                                    "name": "HIVE_ABANDONMENT_LIMIT",
                                    "value": str(self.abandonment_limit),
                                },
                                {"name": "HIVE_MAX_CYCLES", "value": str(self.max_cycles)},
                            ],
                            "ports": [
                                {"containerPort": 8080, "name": "http"},
                                {"containerPort": 9090, "name": "metrics"},
                            ],
                        }
                    ]
                }
            },
        }

        if self.node_selector:
            spec["template"]["spec"]["nodeSelector"] = self.node_selector  # type: ignore[index]

        if self.tolerations:
            spec["template"]["spec"]["tolerations"] = self.tolerations  # type: ignore[index]

        if self.affinity:
            spec["template"]["spec"]["affinity"] = self.affinity  # type: ignore[index]

        return spec


@dataclass
class ClusterStatus:
    """
    Cluster Status
    --------------
    Current status of a HiveFrame cluster.
    """

    phase: ClusterPhase = ClusterPhase.PENDING
    ready_workers: int = 0
    total_workers: int = 0
    message: str = ""
    conditions: List[Dict[str, Any]] = field(default_factory=list)

    # Colony metrics
    colony_temperature: float = 0.0
    active_food_sources: int = 0
    average_fitness: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "readyWorkers": self.ready_workers,
            "totalWorkers": self.total_workers,
            "message": self.message,
            "conditions": self.conditions,
            "colonyMetrics": {
                "temperature": self.colony_temperature,
                "activeFoodSources": self.active_food_sources,
                "averageFitness": self.average_fitness,
            },
        }


@dataclass
class HiveCluster:
    """
    HiveFrame Cluster
    -----------------
    Complete specification for a HiveFrame cluster on Kubernetes.

    This is the main configuration object used to define and
    deploy a HiveFrame cluster.

    Usage:
        cluster = HiveCluster(
            name="production-hive",
            namespace="data-platform",
            workers=WorkerSpec(
                replicas=20,
                resources=ResourceRequirements(
                    cpu="4",
                    memory="16Gi"
                )
            )
        )

        # Generate Kubernetes manifests
        manifests = cluster.to_k8s_manifests()
    """

    name: str
    namespace: str = "default"
    workers: WorkerSpec = field(default_factory=WorkerSpec)

    # Cluster configuration
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # Service configuration
    service_type: str = "ClusterIP"  # ClusterIP, NodePort, LoadBalancer
    service_port: int = 8080
    metrics_port: int = 9090

    # Storage configuration
    persistence_enabled: bool = False
    storage_class: Optional[str] = None
    storage_size: str = "10Gi"

    # Monitoring
    prometheus_enabled: bool = True
    grafana_dashboard: bool = True

    # Auto-scaling
    autoscaling_enabled: bool = False
    min_workers: int = 1
    max_workers: int = 100
    target_cpu_utilization: int = 80

    # Status (set by operator)
    status: ClusterStatus = field(default_factory=ClusterStatus)

    def __post_init__(self):
        """Initialize default labels."""
        default_labels = {
            "app.kubernetes.io/name": "hiveframe",
            "app.kubernetes.io/instance": self.name,
            "app.kubernetes.io/component": "cluster",
        }
        self.labels = {**default_labels, **self.labels}

    def to_crd(self) -> Dict[str, Any]:
        """
        Convert to Kubernetes Custom Resource Definition.

        Returns:
            Dictionary representing the CRD YAML
        """
        return {
            "apiVersion": "hiveframe.io/v1alpha1",
            "kind": "HiveCluster",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": self.labels,
                "annotations": self.annotations,
            },
            "spec": {
                "workers": self.workers.to_k8s_spec(),
                "service": {
                    "type": self.service_type,
                    "port": self.service_port,
                    "metricsPort": self.metrics_port,
                },
                "persistence": {
                    "enabled": self.persistence_enabled,
                    "storageClass": self.storage_class,
                    "size": self.storage_size,
                },
                "monitoring": {
                    "prometheus": self.prometheus_enabled,
                    "grafanaDashboard": self.grafana_dashboard,
                },
                "autoscaling": {
                    "enabled": self.autoscaling_enabled,
                    "minWorkers": self.min_workers,
                    "maxWorkers": self.max_workers,
                    "targetCPUUtilization": self.target_cpu_utilization,
                },
            },
            "status": self.status.to_dict(),
        }

    def to_k8s_manifests(self) -> List[Dict[str, Any]]:
        """
        Generate all Kubernetes manifests for the cluster.

        Returns:
            List of Kubernetes resource dictionaries
        """
        from .manifests import (
            generate_configmap,
            generate_deployment,
            generate_service,
        )

        manifests = []

        # ConfigMap
        manifests.append(generate_configmap(self))

        # Deployment
        manifests.append(generate_deployment(self))

        # Service
        manifests.append(generate_service(self))

        # HPA if autoscaling enabled
        if self.autoscaling_enabled:
            manifests.append(self._generate_hpa())

        # PVC if persistence enabled
        if self.persistence_enabled:
            manifests.append(self._generate_pvc())

        # ServiceMonitor if Prometheus enabled
        if self.prometheus_enabled:
            manifests.append(self._generate_service_monitor())

        return manifests

    def _generate_hpa(self) -> Dict[str, Any]:
        """Generate HorizontalPodAutoscaler."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.name}-hpa",
                "namespace": self.namespace,
                "labels": self.labels,
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{self.name}-workers",
                },
                "minReplicas": self.min_workers,
                "maxReplicas": self.max_workers,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.target_cpu_utilization,
                            },
                        },
                    }
                ],
            },
        }

    def _generate_pvc(self) -> Dict[str, Any]:
        """Generate PersistentVolumeClaim."""
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{self.name}-data",
                "namespace": self.namespace,
                "labels": self.labels,
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": self.storage_class,
                "resources": {
                    "requests": {
                        "storage": self.storage_size,
                    }
                },
            },
        }

    def _generate_service_monitor(self) -> Dict[str, Any]:
        """Generate ServiceMonitor for Prometheus."""
        return {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {
                "name": f"{self.name}-monitor",
                "namespace": self.namespace,
                "labels": self.labels,
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/instance": self.name,
                    }
                },
                "endpoints": [
                    {
                        "port": "metrics",
                        "interval": "30s",
                        "path": "/metrics",
                    }
                ],
            },
        }
