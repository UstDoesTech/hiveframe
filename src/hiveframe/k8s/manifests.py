"""
Kubernetes Manifests Generator
==============================

Generates Kubernetes manifests for HiveFrame clusters.
"""

from typing import Any, Dict, List, Optional
import yaml


def generate_deployment(cluster: "HiveCluster") -> Dict[str, Any]:
    """
    Generate Deployment manifest for workers.

    Args:
        cluster: HiveCluster specification

    Returns:
        Kubernetes Deployment dictionary
    """
    worker_spec = cluster.workers.to_k8s_spec()

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"{cluster.name}-workers",
            "namespace": cluster.namespace,
            "labels": {
                **cluster.labels,
                "app.kubernetes.io/component": "worker",
            },
            "annotations": cluster.annotations,
        },
        "spec": {
            "replicas": worker_spec["replicas"],
            "selector": {
                "matchLabels": {
                    "app.kubernetes.io/instance": cluster.name,
                    "app.kubernetes.io/component": "worker",
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        **cluster.labels,
                        "app.kubernetes.io/component": "worker",
                    },
                    "annotations": {
                        "prometheus.io/scrape": "true" if cluster.prometheus_enabled else "false",
                        "prometheus.io/port": str(cluster.metrics_port),
                        "prometheus.io/path": "/metrics",
                    },
                },
                "spec": worker_spec["template"]["spec"],
            },
        },
    }

    # Add volume mounts if persistence enabled
    if cluster.persistence_enabled:
        deployment["spec"]["template"]["spec"]["volumes"] = [
            {
                "name": "data",
                "persistentVolumeClaim": {
                    "claimName": f"{cluster.name}-data",
                },
            }
        ]

        deployment["spec"]["template"]["spec"]["containers"][0]["volumeMounts"] = [
            {
                "name": "data",
                "mountPath": "/data",
            }
        ]

    # Add configmap volume
    deployment["spec"]["template"]["spec"].setdefault("volumes", []).append(
        {
            "name": "config",
            "configMap": {
                "name": f"{cluster.name}-config",
            },
        }
    )

    deployment["spec"]["template"]["spec"]["containers"][0].setdefault("volumeMounts", []).append(
        {
            "name": "config",
            "mountPath": "/etc/hiveframe",
        }
    )

    return deployment


def generate_service(cluster: "HiveCluster") -> Dict[str, Any]:
    """
    Generate Service manifest.

    Args:
        cluster: HiveCluster specification

    Returns:
        Kubernetes Service dictionary
    """
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{cluster.name}-service",
            "namespace": cluster.namespace,
            "labels": cluster.labels,
        },
        "spec": {
            "type": cluster.service_type,
            "selector": {
                "app.kubernetes.io/instance": cluster.name,
                "app.kubernetes.io/component": "worker",
            },
            "ports": [
                {
                    "name": "http",
                    "port": cluster.service_port,
                    "targetPort": 8080,
                    "protocol": "TCP",
                },
                {
                    "name": "metrics",
                    "port": cluster.metrics_port,
                    "targetPort": 9090,
                    "protocol": "TCP",
                },
            ],
        },
    }


def generate_configmap(cluster: "HiveCluster") -> Dict[str, Any]:
    """
    Generate ConfigMap manifest.

    Args:
        cluster: HiveCluster specification

    Returns:
        Kubernetes ConfigMap dictionary
    """
    config = {
        "cluster_name": cluster.name,
        "namespace": cluster.namespace,
        "workers": {
            "replicas": cluster.workers.replicas,
            "employed_ratio": cluster.workers.employed_ratio,
            "onlooker_ratio": cluster.workers.onlooker_ratio,
            "scout_ratio": cluster.workers.scout_ratio,
            "abandonment_limit": cluster.workers.abandonment_limit,
            "max_cycles": cluster.workers.max_cycles,
        },
        "monitoring": {
            "prometheus_enabled": cluster.prometheus_enabled,
            "metrics_port": cluster.metrics_port,
        },
    }

    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": f"{cluster.name}-config",
            "namespace": cluster.namespace,
            "labels": cluster.labels,
        },
        "data": {
            "hiveframe.yaml": yaml.dump(config),
        },
    }


def generate_crd() -> Dict[str, Any]:
    """
    Generate CustomResourceDefinition for HiveCluster.

    Returns:
        Kubernetes CRD dictionary
    """
    return {
        "apiVersion": "apiextensions.k8s.io/v1",
        "kind": "CustomResourceDefinition",
        "metadata": {
            "name": "hiveclusters.hiveframe.io",
        },
        "spec": {
            "group": "hiveframe.io",
            "names": {
                "kind": "HiveCluster",
                "listKind": "HiveClusterList",
                "plural": "hiveclusters",
                "singular": "hivecluster",
                "shortNames": ["hive", "hc"],
            },
            "scope": "Namespaced",
            "versions": [
                {
                    "name": "v1alpha1",
                    "served": True,
                    "storage": True,
                    "schema": {
                        "openAPIV3Schema": {
                            "type": "object",
                            "properties": {
                                "spec": {
                                    "type": "object",
                                    "properties": {
                                        "workers": {
                                            "type": "object",
                                            "properties": {
                                                "replicas": {
                                                    "type": "integer",
                                                    "minimum": 1,
                                                    "default": 3,
                                                },
                                                "template": {
                                                    "type": "object",
                                                    "x-kubernetes-preserve-unknown-fields": True,
                                                },
                                            },
                                        },
                                        "service": {
                                            "type": "object",
                                            "properties": {
                                                "type": {
                                                    "type": "string",
                                                    "enum": [
                                                        "ClusterIP",
                                                        "NodePort",
                                                        "LoadBalancer",
                                                    ],
                                                    "default": "ClusterIP",
                                                },
                                                "port": {
                                                    "type": "integer",
                                                    "default": 8080,
                                                },
                                                "metricsPort": {
                                                    "type": "integer",
                                                    "default": 9090,
                                                },
                                            },
                                        },
                                        "persistence": {
                                            "type": "object",
                                            "properties": {
                                                "enabled": {
                                                    "type": "boolean",
                                                    "default": False,
                                                },
                                                "storageClass": {
                                                    "type": "string",
                                                },
                                                "size": {
                                                    "type": "string",
                                                    "default": "10Gi",
                                                },
                                            },
                                        },
                                        "monitoring": {
                                            "type": "object",
                                            "properties": {
                                                "prometheus": {
                                                    "type": "boolean",
                                                    "default": True,
                                                },
                                                "grafanaDashboard": {
                                                    "type": "boolean",
                                                    "default": True,
                                                },
                                            },
                                        },
                                        "autoscaling": {
                                            "type": "object",
                                            "properties": {
                                                "enabled": {
                                                    "type": "boolean",
                                                    "default": False,
                                                },
                                                "minWorkers": {
                                                    "type": "integer",
                                                    "minimum": 1,
                                                    "default": 1,
                                                },
                                                "maxWorkers": {
                                                    "type": "integer",
                                                    "minimum": 1,
                                                    "default": 100,
                                                },
                                                "targetCPUUtilization": {
                                                    "type": "integer",
                                                    "minimum": 1,
                                                    "maximum": 100,
                                                    "default": 80,
                                                },
                                            },
                                        },
                                    },
                                },
                                "status": {
                                    "type": "object",
                                    "properties": {
                                        "phase": {
                                            "type": "string",
                                            "enum": [
                                                "Pending",
                                                "Creating",
                                                "Running",
                                                "Scaling",
                                                "Updating",
                                                "Failed",
                                                "Terminating",
                                            ],
                                        },
                                        "readyWorkers": {
                                            "type": "integer",
                                        },
                                        "totalWorkers": {
                                            "type": "integer",
                                        },
                                        "message": {
                                            "type": "string",
                                        },
                                        "colonyMetrics": {
                                            "type": "object",
                                            "properties": {
                                                "temperature": {
                                                    "type": "number",
                                                },
                                                "activeFoodSources": {
                                                    "type": "integer",
                                                },
                                                "averageFitness": {
                                                    "type": "number",
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        }
                    },
                    "subresources": {
                        "status": {},
                        "scale": {
                            "specReplicasPath": ".spec.workers.replicas",
                            "statusReplicasPath": ".status.totalWorkers",
                            "labelSelectorPath": ".status.selector",
                        },
                    },
                    "additionalPrinterColumns": [
                        {
                            "name": "Phase",
                            "type": "string",
                            "jsonPath": ".status.phase",
                        },
                        {
                            "name": "Ready",
                            "type": "string",
                            "jsonPath": ".status.readyWorkers",
                        },
                        {
                            "name": "Workers",
                            "type": "integer",
                            "jsonPath": ".status.totalWorkers",
                        },
                        {
                            "name": "Age",
                            "type": "date",
                            "jsonPath": ".metadata.creationTimestamp",
                        },
                    ],
                }
            ],
        },
    }


def generate_rbac(namespace: str = "hiveframe-system") -> List[Dict[str, Any]]:
    """
    Generate RBAC resources for the operator.

    Args:
        namespace: Operator namespace

    Returns:
        List of RBAC resources
    """
    resources = []

    # ServiceAccount
    resources.append(
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": "hiveframe-operator",
                "namespace": namespace,
            },
        }
    )

    # ClusterRole
    resources.append(
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {
                "name": "hiveframe-operator",
            },
            "rules": [
                {
                    "apiGroups": ["hiveframe.io"],
                    "resources": ["hiveclusters"],
                    "verbs": ["*"],
                },
                {
                    "apiGroups": ["hiveframe.io"],
                    "resources": ["hiveclusters/status"],
                    "verbs": ["get", "update", "patch"],
                },
                {
                    "apiGroups": ["apps"],
                    "resources": ["deployments"],
                    "verbs": ["*"],
                },
                {
                    "apiGroups": [""],
                    "resources": ["services", "configmaps", "persistentvolumeclaims"],
                    "verbs": ["*"],
                },
                {
                    "apiGroups": ["autoscaling"],
                    "resources": ["horizontalpodautoscalers"],
                    "verbs": ["*"],
                },
                {
                    "apiGroups": ["monitoring.coreos.com"],
                    "resources": ["servicemonitors"],
                    "verbs": ["*"],
                },
            ],
        }
    )

    # ClusterRoleBinding
    resources.append(
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {
                "name": "hiveframe-operator",
            },
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": "hiveframe-operator",
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": "hiveframe-operator",
                    "namespace": namespace,
                }
            ],
        }
    )

    return resources


def to_yaml(manifests: List[Dict[str, Any]]) -> str:
    """
    Convert manifests to YAML string.

    Args:
        manifests: List of Kubernetes resources

    Returns:
        YAML string with document separators
    """
    return "\n---\n".join(yaml.dump(m, default_flow_style=False) for m in manifests)
