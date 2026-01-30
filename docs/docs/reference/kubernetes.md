---
sidebar_position: 10
---

# Kubernetes Module

Deploy and manage HiveFrame clusters on Kubernetes.

```python
from hiveframe.k8s import (
    HiveFrameCluster,
    ClusterConfig,
    HiveFrameOperator
)
```

## HiveFrameCluster

Kubernetes-native cluster management.

### Class Definition

```python
class HiveFrameCluster:
    """
    Manage HiveFrame cluster on Kubernetes.
    """
    
    def __init__(
        self,
        name: str,
        namespace: str = "default",
        config: Optional[ClusterConfig] = None,
        kubeconfig: Optional[str] = None
    ) -> None:
        """
        Create cluster manager.
        
        Args:
            name: Cluster name
            namespace: Kubernetes namespace
            config: Cluster configuration
            kubeconfig: Path to kubeconfig file
        """
```

### Methods

#### Lifecycle

```python
def create(self) -> "HiveFrameCluster":
    """
    Create cluster resources.
    
    Creates:
    - ConfigMap for configuration
    - Deployment for queen (coordinator)
    - StatefulSet for workers
    - Services for networking
    - HPA for autoscaling
    
    Example:
        cluster = HiveFrameCluster(
            name="production",
            config=ClusterConfig(workers=5)
        )
        cluster.create()
    """

def delete(self, wait: bool = True) -> None:
    """
    Delete cluster resources.
    
    Args:
        wait: Wait for cleanup
        
    Example:
        cluster.delete(wait=True)
    """

def scale(self, workers: int) -> None:
    """
    Scale worker count.
    
    Args:
        workers: Target worker count
        
    Example:
        cluster.scale(workers=10)
    """

def upgrade(
    self,
    version: Optional[str] = None,
    config: Optional[ClusterConfig] = None
) -> None:
    """
    Rolling upgrade cluster.
    
    Args:
        version: New HiveFrame version
        config: New configuration
        
    Example:
        cluster.upgrade(version="0.3.0")
    """
```

#### Monitoring

```python
def status(self) -> ClusterStatus:
    """
    Get cluster status.
    
    Returns:
        ClusterStatus with health and metrics
        
    Example:
        status = cluster.status()
        print(f"State: {status.state}")
        print(f"Workers: {status.ready_workers}/{status.desired_workers}")
    """

def logs(
    self,
    component: str = "all",
    since: Optional[str] = None,
    follow: bool = False
) -> Iterator[str]:
    """
    Stream cluster logs.
    
    Args:
        component: "queen", "worker", or "all"
        since: Time filter (e.g., "1h", "30m")
        follow: Stream new logs
        
    Example:
        for line in cluster.logs(component="queen", since="10m"):
            print(line)
    """

def events(self) -> List[K8sEvent]:
    """Get Kubernetes events for cluster."""
```

### Properties

```python
@property
def endpoint(self) -> str:
    """Cluster API endpoint."""

@property
def workers(self) -> List[WorkerInfo]:
    """Information about workers."""

@property
def metrics(self) -> ClusterMetrics:
    """Current cluster metrics."""
```

---

## ClusterConfig

Configure cluster deployment.

```python
class ClusterConfig:
    """
    Kubernetes cluster configuration.
    """
    
    def __init__(
        self,
        # Worker configuration
        workers: int = 3,
        min_workers: int = 1,
        max_workers: int = 10,
        
        # Resource limits
        worker_cpu: str = "1",
        worker_memory: str = "2Gi",
        queen_cpu: str = "500m",
        queen_memory: str = "1Gi",
        
        # Storage
        storage_class: Optional[str] = None,
        storage_size: str = "10Gi",
        
        # Networking
        service_type: str = "ClusterIP",
        ingress_enabled: bool = False,
        ingress_host: Optional[str] = None,
        
        # Autoscaling
        autoscale_enabled: bool = True,
        autoscale_cpu_target: int = 70,
        autoscale_memory_target: int = 80,
        
        # High availability
        queen_replicas: int = 1,
        anti_affinity: bool = True,
        
        # Image
        image: str = "hiveframe/hiveframe",
        image_tag: str = "latest",
        image_pull_policy: str = "IfNotPresent",
        image_pull_secrets: Optional[List[str]] = None,
        
        # Custom
        env: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None,
        node_selector: Optional[Dict[str, str]] = None,
        tolerations: Optional[List[Dict]] = None
    ) -> None:
        """
        Configure cluster.
        
        See parameters above for all options.
        """
```

### Example Configurations

#### Development

```python
dev_config = ClusterConfig(
    workers=2,
    worker_cpu="500m",
    worker_memory="1Gi",
    autoscale_enabled=False
)
```

#### Production

```python
prod_config = ClusterConfig(
    workers=5,
    min_workers=3,
    max_workers=20,
    worker_cpu="2",
    worker_memory="4Gi",
    queen_cpu="1",
    queen_memory="2Gi",
    autoscale_enabled=True,
    autoscale_cpu_target=70,
    anti_affinity=True,
    storage_class="fast-ssd",
    storage_size="100Gi",
    image_tag="0.2.0"
)
```

#### With Ingress

```python
config = ClusterConfig(
    workers=5,
    service_type="ClusterIP",
    ingress_enabled=True,
    ingress_host="hiveframe.example.com",
    annotations={
        "kubernetes.io/ingress.class": "nginx",
        "cert-manager.io/cluster-issuer": "letsencrypt"
    }
)
```

---

## HiveFrameOperator

Kubernetes operator for custom resources.

### Installation

```bash
# Install CRDs and operator
kubectl apply -f https://hiveframe.io/k8s/operator.yaml

# Or via Helm
helm install hiveframe-operator hiveframe/operator
```

### Custom Resource Definition

```yaml
apiVersion: hiveframe.io/v1
kind: HiveFrameCluster
metadata:
  name: my-cluster
  namespace: data-processing
spec:
  version: "0.2.0"
  workers:
    replicas: 5
    minReplicas: 2
    maxReplicas: 20
    resources:
      cpu: "2"
      memory: "4Gi"
  queen:
    resources:
      cpu: "1"
      memory: "2Gi"
  storage:
    class: fast-ssd
    size: 50Gi
  autoscaling:
    enabled: true
    cpuTarget: 70
    memoryTarget: 80
  monitoring:
    enabled: true
    prometheus: true
```

### Python Integration

```python
from hiveframe.k8s import HiveFrameOperator

operator = HiveFrameOperator(namespace="data-processing")

# Create from spec
operator.create_cluster({
    "apiVersion": "hiveframe.io/v1",
    "kind": "HiveFrameCluster",
    "metadata": {"name": "my-cluster"},
    "spec": {
        "version": "0.2.0",
        "workers": {"replicas": 5}
    }
})

# List clusters
clusters = operator.list_clusters()
for cluster in clusters:
    print(f"{cluster.name}: {cluster.status}")

# Get cluster
cluster = operator.get_cluster("my-cluster")

# Delete cluster
operator.delete_cluster("my-cluster")
```

---

## Manifests Module

Generate Kubernetes manifests without deploying.

```python
from hiveframe.k8s import generate_manifests

manifests = generate_manifests(
    name="my-cluster",
    namespace="production",
    config=ClusterConfig(workers=5)
)

# Write to file
with open("hiveframe-cluster.yaml", "w") as f:
    f.write(manifests)

# Or apply programmatically
from kubernetes import client, config
config.load_kube_config()

for manifest in yaml.safe_load_all(manifests):
    apply_manifest(manifest)
```

### Generated Resources

| Resource | Name | Description |
|----------|------|-------------|
| ConfigMap | `{name}-config` | Cluster configuration |
| Deployment | `{name}-queen` | Coordinator deployment |
| StatefulSet | `{name}-workers` | Worker pods |
| Service | `{name}` | Main service |
| Service | `{name}-headless` | Worker discovery |
| HPA | `{name}-workers` | Autoscaler |
| PDB | `{name}-pdb` | Pod disruption budget |
| ServiceAccount | `{name}` | RBAC account |

---

## Complete Example

```python
from hiveframe.k8s import HiveFrameCluster, ClusterConfig

# Configure production cluster
config = ClusterConfig(
    # Workers
    workers=5,
    min_workers=3,
    max_workers=15,
    worker_cpu="2",
    worker_memory="4Gi",
    
    # Queen
    queen_cpu="1",
    queen_memory="2Gi",
    
    # Storage
    storage_class="gp3",
    storage_size="50Gi",
    
    # Autoscaling
    autoscale_enabled=True,
    autoscale_cpu_target=70,
    
    # HA
    anti_affinity=True,
    
    # Image
    image="myregistry/hiveframe",
    image_tag="0.2.0",
    image_pull_secrets=["registry-secret"],
    
    # Environment
    env={
        "HIVEFRAME_LOG_LEVEL": "INFO",
        "KAFKA_BOOTSTRAP_SERVERS": "kafka:9092"
    },
    
    # Node placement
    node_selector={"node-type": "compute"},
    tolerations=[{
        "key": "dedicated",
        "value": "hiveframe",
        "effect": "NoSchedule"
    }]
)

# Create cluster
cluster = HiveFrameCluster(
    name="production-cluster",
    namespace="data-processing",
    config=config
)

# Deploy
cluster.create()

# Wait for ready
import time
while True:
    status = cluster.status()
    if status.ready:
        print("Cluster ready!")
        break
    print(f"Waiting... {status.ready_workers}/{status.desired_workers}")
    time.sleep(5)

# Connect application
from hiveframe import Colony

colony = Colony(
    name="production",
    config={"k8s.cluster": cluster.endpoint}
)

# Use cluster...

# Cleanup
cluster.delete()
```

## See Also

- [Kubernetes Tutorial](/docs/tutorials/kubernetes-deployment) - Deployment guide
- [Architecture](/docs/explanation/architecture-overview) - System architecture
- [Monitoring](/docs/how-to/setup-monitoring) - Monitor cluster
