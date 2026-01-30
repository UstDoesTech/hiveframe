"""
HiveFrame Kubernetes Operator
=============================

Operator for managing HiveFrame clusters on Kubernetes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import subprocess
import time

from .cluster import HiveCluster, ClusterStatus, ClusterPhase
from .manifests import generate_crd, generate_rbac, to_yaml


@dataclass
class OperatorConfig:
    """
    Operator Configuration
    ----------------------
    Configuration for the HiveFrame Kubernetes operator.
    """

    namespace: str = "hiveframe-system"
    image: str = "hiveframe/operator:latest"
    reconcile_interval: int = 30  # seconds
    leader_election: bool = True
    metrics_port: int = 8080
    health_port: int = 8081

    # Resource limits for operator
    cpu_limit: str = "500m"
    memory_limit: str = "256Mi"


class HiveOperator:
    """
    HiveFrame Operator
    ------------------

    Manages the lifecycle of HiveFrame clusters on Kubernetes.

    The operator watches for HiveCluster custom resources and
    ensures the actual state matches the desired state.

    Key responsibilities:
    - Create and update worker deployments
    - Manage services and networking
    - Handle scaling operations
    - Monitor cluster health
    - Coordinate with Prometheus for metrics

    Usage:
        operator = HiveOperator()

        # Install CRDs and operator
        operator.install()

        # Deploy a cluster
        cluster = HiveCluster(name="my-cluster", workers=10)
        operator.deploy(cluster)

        # Get cluster status
        status = operator.get_status("my-cluster")

        # Scale cluster
        operator.scale("my-cluster", 20)

        # Delete cluster
        operator.delete("my-cluster")
    """

    def __init__(self, config: Optional[OperatorConfig] = None):
        """
        Initialize operator.

        Args:
            config: Operator configuration
        """
        self.config = config or OperatorConfig()
        self._kubectl_available = self._check_kubectl()

    def _check_kubectl(self) -> bool:
        """Check if kubectl is available."""
        try:
            subprocess.run(["kubectl", "version", "--client"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _kubectl(self, *args: str, input_data: Optional[str] = None) -> subprocess.CompletedProcess:
        """Run kubectl command."""
        if not self._kubectl_available:
            raise RuntimeError("kubectl not available")

        cmd = ["kubectl"] + list(args)
        return subprocess.run(
            cmd, input=input_data.encode() if input_data else None, capture_output=True, text=True
        )

    def install(self, dry_run: bool = False) -> List[Dict[str, Any]]:
        """
        Install the operator and CRDs.

        Args:
            dry_run: If True, only generate manifests without applying

        Returns:
            List of generated manifests
        """
        manifests = []

        # Generate CRD
        crd = generate_crd()
        manifests.append(crd)

        # Generate RBAC
        rbac = generate_rbac(self.config.namespace)
        manifests.extend(rbac)

        # Generate operator deployment
        operator_deployment = self._generate_operator_deployment()
        manifests.append(operator_deployment)

        if not dry_run and self._kubectl_available:
            yaml_content = to_yaml(manifests)
            result = self._kubectl("apply", "-f", "-", input_data=yaml_content)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to install operator: {result.stderr}")

        return manifests

    def _generate_operator_deployment(self) -> Dict[str, Any]:
        """Generate operator deployment manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "hiveframe-operator",
                "namespace": self.config.namespace,
                "labels": {
                    "app.kubernetes.io/name": "hiveframe-operator",
                    "app.kubernetes.io/component": "operator",
                },
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": "hiveframe-operator",
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app.kubernetes.io/name": "hiveframe-operator",
                        }
                    },
                    "spec": {
                        "serviceAccountName": "hiveframe-operator",
                        "containers": [
                            {
                                "name": "operator",
                                "image": self.config.image,
                                "args": [
                                    f"--reconcile-interval={self.config.reconcile_interval}",
                                    f"--leader-election={str(self.config.leader_election).lower()}",
                                ],
                                "ports": [
                                    {"containerPort": self.config.metrics_port, "name": "metrics"},
                                    {"containerPort": self.config.health_port, "name": "health"},
                                ],
                                "resources": {
                                    "limits": {
                                        "cpu": self.config.cpu_limit,
                                        "memory": self.config.memory_limit,
                                    },
                                    "requests": {
                                        "cpu": "100m",
                                        "memory": "64Mi",
                                    },
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/healthz",
                                        "port": self.config.health_port,
                                    },
                                    "initialDelaySeconds": 15,
                                    "periodSeconds": 20,
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/readyz",
                                        "port": self.config.health_port,
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 10,
                                },
                            }
                        ],
                    },
                },
            },
        }

    def uninstall(self) -> None:
        """Uninstall the operator and CRDs."""
        if not self._kubectl_available:
            raise RuntimeError("kubectl not available")

        # Delete operator
        self._kubectl(
            "delete",
            "deployment",
            "hiveframe-operator",
            "-n",
            self.config.namespace,
            "--ignore-not-found",
        )

        # Delete RBAC
        self._kubectl("delete", "clusterrolebinding", "hiveframe-operator", "--ignore-not-found")
        self._kubectl("delete", "clusterrole", "hiveframe-operator", "--ignore-not-found")
        self._kubectl(
            "delete",
            "serviceaccount",
            "hiveframe-operator",
            "-n",
            self.config.namespace,
            "--ignore-not-found",
        )

        # Delete CRD (this will delete all HiveClusters)
        self._kubectl("delete", "crd", "hiveclusters.hiveframe.io", "--ignore-not-found")

    def deploy(self, cluster: HiveCluster, wait: bool = True, timeout: int = 300) -> ClusterStatus:
        """
        Deploy a HiveFrame cluster.

        Args:
            cluster: Cluster specification
            wait: Wait for cluster to be ready
            timeout: Timeout in seconds for wait

        Returns:
            Cluster status
        """
        # Generate and apply manifests
        manifests = cluster.to_k8s_manifests()

        if self._kubectl_available:
            yaml_content = to_yaml(manifests)
            result = self._kubectl("apply", "-f", "-", input_data=yaml_content)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to deploy cluster: {result.stderr}")

            # Also apply the CRD resource
            crd_yaml = to_yaml([cluster.to_crd()])
            self._kubectl("apply", "-f", "-", input_data=crd_yaml)

            if wait:
                return self._wait_for_ready(cluster.name, cluster.namespace, timeout)
        else:
            # Return generated status for dry-run
            cluster.status.phase = ClusterPhase.CREATING
            cluster.status.total_workers = cluster.workers.replicas

        return cluster.status

    def _wait_for_ready(self, name: str, namespace: str, timeout: int) -> ClusterStatus:
        """Wait for cluster to be ready."""
        start = time.time()

        while time.time() - start < timeout:
            status = self.get_status(name, namespace)

            if status.phase == ClusterPhase.RUNNING:
                return status
            if status.phase == ClusterPhase.FAILED:
                raise RuntimeError(f"Cluster failed: {status.message}")

            time.sleep(5)

        raise TimeoutError(f"Cluster {name} did not become ready within {timeout}s")

    def get_status(self, name: str, namespace: str = "default") -> ClusterStatus:
        """
        Get cluster status.

        Args:
            name: Cluster name
            namespace: Kubernetes namespace

        Returns:
            Cluster status
        """
        if not self._kubectl_available:
            return ClusterStatus(phase=ClusterPhase.PENDING, message="kubectl not available")

        # Get deployment status
        result = self._kubectl(
            "get", "deployment", f"{name}-workers", "-n", namespace, "-o", "json"
        )

        if result.returncode != 0:
            return ClusterStatus(
                phase=ClusterPhase.PENDING, message=f"Deployment not found: {result.stderr}"
            )

        deployment = json.loads(result.stdout)
        spec_replicas = deployment.get("spec", {}).get("replicas", 0)
        status_replicas = deployment.get("status", {}).get("readyReplicas", 0)

        status = ClusterStatus(
            ready_workers=status_replicas or 0,
            total_workers=spec_replicas,
        )

        if status.ready_workers >= status.total_workers and status.total_workers > 0:
            status.phase = ClusterPhase.RUNNING
            status.message = "All workers ready"
        elif status.ready_workers > 0:
            status.phase = ClusterPhase.CREATING
            status.message = f"{status.ready_workers}/{status.total_workers} workers ready"
        else:
            status.phase = ClusterPhase.CREATING
            status.message = "Starting workers"

        return status

    def scale(self, name: str, replicas: int, namespace: str = "default") -> ClusterStatus:
        """
        Scale a cluster.

        Args:
            name: Cluster name
            replicas: New worker count
            namespace: Kubernetes namespace

        Returns:
            Updated cluster status
        """
        if not self._kubectl_available:
            raise RuntimeError("kubectl not available")

        result = self._kubectl(
            "scale", "deployment", f"{name}-workers", f"--replicas={replicas}", "-n", namespace
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to scale cluster: {result.stderr}")

        return self.get_status(name, namespace)

    def delete(self, name: str, namespace: str = "default", wait: bool = True) -> None:
        """
        Delete a cluster.

        Args:
            name: Cluster name
            namespace: Kubernetes namespace
            wait: Wait for deletion to complete
        """
        if not self._kubectl_available:
            raise RuntimeError("kubectl not available")

        # Delete all cluster resources
        resources = [
            ("deployment", f"{name}-workers"),
            ("service", f"{name}-service"),
            ("configmap", f"{name}-config"),
            ("hpa", f"{name}-hpa"),
            ("pvc", f"{name}-data"),
            ("servicemonitor", f"{name}-monitor"),
            ("hivecluster", name),
        ]

        for kind, resource_name in resources:
            self._kubectl("delete", kind, resource_name, "-n", namespace, "--ignore-not-found")

        if wait:
            # Wait for pods to terminate
            start = time.time()
            while time.time() - start < 60:
                result = self._kubectl(
                    "get",
                    "pods",
                    "-l",
                    f"app.kubernetes.io/instance={name}",
                    "-n",
                    namespace,
                    "-o",
                    "json",
                )
                if result.returncode == 0:
                    pods = json.loads(result.stdout)
                    if not pods.get("items"):
                        break
                time.sleep(2)

    def list_clusters(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all HiveFrame clusters.

        Args:
            namespace: Optional namespace filter

        Returns:
            List of cluster info dictionaries
        """
        if not self._kubectl_available:
            return []

        args = ["get", "hiveclusters", "-o", "json"]
        if namespace:
            args.extend(["-n", namespace])
        else:
            args.append("--all-namespaces")

        result = self._kubectl(*args)

        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
        return data.get("items", [])

    def get_logs(self, name: str, namespace: str = "default", tail: int = 100) -> str:
        """
        Get logs from cluster workers.

        Args:
            name: Cluster name
            namespace: Kubernetes namespace
            tail: Number of lines to return

        Returns:
            Combined log output
        """
        if not self._kubectl_available:
            return "kubectl not available"

        result = self._kubectl(
            "logs",
            f"-l",
            f"app.kubernetes.io/instance={name}",
            "-n",
            namespace,
            f"--tail={tail}",
            "--all-containers",
        )

        return result.stdout if result.returncode == 0 else result.stderr
