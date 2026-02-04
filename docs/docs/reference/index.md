---
sidebar_position: 1
---

# API Reference

Complete reference documentation for HiveFrame v0.3.0.

## Overview

This reference section provides comprehensive documentation for all HiveFrame APIs. Each page includes:

- **Module overview** - Purpose and key concepts
- **Class/function signatures** - Complete API with type hints
- **Parameters** - Detailed parameter descriptions
- **Examples** - Practical usage patterns
- **Related APIs** - Links to related functionality

## Module Index

### Core Modules

| Module | Description |
|--------|-------------|
| [Core](./core) | Main `Colony` and `Cell` classes |
| [DataFrame](./dataframe) | Spark-like DataFrame API |
| [SQL](./sql) | SQL context and execution |
| [Streaming](./streaming) | Real-time stream processing |

### Infrastructure

| Module | Description |
|--------|-------------|
| [Storage](./storage) | Parquet and Delta Lake support |
| [Connectors](./connectors) | Kafka, PostgreSQL, HTTP sources/sinks |
| [Resilience](./resilience) | Retry, circuit breaker, timeout patterns |
| [Monitoring](./monitoring) | Metrics, logging, health checks |

### Phase 3: Enterprise Platform

| Module | Description | Status |
|--------|-------------|--------|
| Lakehouse | Unity Hive Catalog, access control, lineage | ✅ Implemented |
| ML (HiveMind) | AutoML Swarm, Feature Store, Model Serving | ✅ Implemented |
| Notebooks | Multi-language interactive execution environment | ✅ Implemented |

> **Phase 3 Status**: Lakehouse, ML Platform, and Notebooks modules are implemented!
> Documentation pages are being written. For now, see `examples/demo_phase3.py`.

### Phase 4: Autonomous Data Intelligence (New!)

| Module | Description | Status |
|--------|-------------|--------|
| [Autonomous](./autonomous) | Self-tuning, predictive maintenance, cost optimization | ✅ Documented |
| [AI Integration](./ai) | Natural language queries, data prep, code generation | ✅ Documented |
| [Advanced Swarm](./advanced-swarm) | Hybrid swarm, quantum-ready, federated learning | ✅ Documented |

> **Phase 4 Status**: All autonomous intelligence features are implemented and documented!
> See comprehensive examples in `examples/demo_phase4.py`.

### Deployment

| Module | Description |
|--------|-------------|
| [Kubernetes](./kubernetes) | K8s operator and cluster management |
| [Dashboard](./dashboard) | Monitoring dashboard API |
| [Exceptions](./exceptions) | Error types and handling |

## Quick Import Reference

```python
# Core
from hiveframe import Colony, Cell, create_colony

# DataFrame
import hiveframe as hf
df = hf.read.parquet("data.parquet")

# SQL
from hiveframe.sql import SQLContext

# Streaming
from hiveframe.streaming import StreamProcessor, TumblingWindow

# Storage
from hiveframe.storage import DeltaTable, ParquetReader

# Connectors
from hiveframe.connectors import KafkaSource, KafkaSink, PostgresSource

# Resilience
from hiveframe.resilience import (
    RetryPolicy, CircuitBreaker, Timeout, Bulkhead
)

# Monitoring
from hiveframe.monitoring import MetricsCollector, HealthChecker

# Kubernetes
from hiveframe.k8s import HiveFrameCluster, ClusterConfig

# Dashboard
from hiveframe.dashboard import DashboardServer

# Phase 3: Lakehouse
from hiveframe.lakehouse import (
    UnityHiveCatalog, AccessControl, LineageTracker,
    PIIDetector, DeltaSharing
)

# Phase 3: Machine Learning
from hiveframe.ml import (
    AutoMLSwarm, FeatureHive, ModelServer,
    DistributedTrainer, MLflowIntegration
)

# Phase 3: Notebooks
from hiveframe.notebooks import (
    NotebookKernel, NotebookSession, CollaborationManager,
    GPUCell
)

# Phase 4: Autonomous Operations
from hiveframe.autonomous import (
    SelfTuningColony, PredictiveMaintenance, WorkloadPredictor,
    CostOptimizer, MemoryStats, ResourceMetrics, HealthMetric
)

# Phase 4: AI Integration
from hiveframe.ai import (
    NaturalLanguageQuery, AIDataPrep, DataDiscovery,
    HiveFrameCodeGen, LLMFineTuner
)

# Phase 4: Advanced Swarm
from hiveframe.advanced_swarm import (
    HybridSwarmOptimizer, ParticleSwarmOptimizer, AntColonyOptimizer,
    FireflyAlgorithm, HybridQuantumClassical, CrossOrgTrainer
)

# Exceptions
from hiveframe.exceptions import (
    HiveFrameError, ValidationError, TimeoutError
)
```

## Type Conventions

Throughout this documentation:

| Type | Description |
|------|-------------|
| `T` | Generic type parameter |
| `DataFrame` | HiveFrame DataFrame object |
| `Column` | Column expression |
| `Schema` | DataFrame schema definition |
| `Dict[str, Any]` | Configuration dictionary |
| `Callable[..., T]` | Function returning type T |
| `Optional[T]` | Type T or None |
| `Union[A, B]` | Either type A or B |

## Version Compatibility

```python
import hiveframe
print(hiveframe.__version__)  # "0.3.0-dev"
```

| Version | Python | Status |
|---------|--------|--------|
| 0.3.x | 3.9+ | Current |
| 0.2.x | 3.9+ | Stable |
| 0.1.x | 3.8+ | Deprecated |

## See Also

- [Getting Started](/docs/tutorials/getting-started) - First steps
- [Architecture](/docs/explanation/architecture-overview) - How it works
- [How-To Guides](/docs/how-to) - Task-oriented guides
