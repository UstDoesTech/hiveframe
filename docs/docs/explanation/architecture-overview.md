---
sidebar_position: 2
---

# Architecture Overview

A high-level view of HiveFrame's architecture and how the components work together.

## System Architecture

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        API[Python API]
        SQL[SwarmQL]
        DF[DataFrame API]
    end
    
    subgraph Colony["Colony (Coordination)"]
        Queen[Queen Coordinator]
        Dance[Waggle Dance Protocol]
        Pheromone[Pheromone Controller]
    end
    
    subgraph Workers["Worker Bees"]
        direction LR
        E1[Employed Bee 1]
        E2[Employed Bee 2]
        E3[Employed Bee 3]
        O1[Onlooker Bee 1]
        O2[Onlooker Bee 2]
        S1[Scout Bee]
    end
    
    subgraph Processing["Processing Engine"]
        Optimizer[ABC Query Optimizer]
        Executor[Task Executor]
        Stream[Stream Processor]
    end
    
    subgraph Storage["Storage Layer"]
        Parquet[Parquet]
        Delta[Delta Lake]
        Memory[In-Memory]
    end
    
    subgraph External["External Systems"]
        Kafka[Kafka]
        Postgres[PostgreSQL]
        HTTP[HTTP APIs]
    end
    
    API --> Queen
    SQL --> Optimizer
    DF --> Executor
    
    Queen --> Dance
    Dance --> Workers
    Pheromone --> Workers
    
    Workers --> Executor
    Workers --> Stream
    
    Executor --> Storage
    Stream --> Storage
    
    Workers --> External
    
    style Queen fill:#fbbf24
    style E1 fill:#f59e0b
    style E2 fill:#f59e0b
    style E3 fill:#f59e0b
    style O1 fill:#fcd34d
    style O2 fill:#fcd34d
    style S1 fill:#fef3c7
```

## Core Components

### Colony

The Colony is the central coordination unit, analogous to a beehive. It manages:

- **Worker lifecycle** - Starting, stopping, and monitoring workers
- **Task distribution** - Assigning work to appropriate workers
- **Colony state** - Shared state for stigmergic coordination
- **Health monitoring** - Detecting and recovering from failures

```python
import hiveframe as hf

# Create a colony
colony = hf.Colony(
    name="my-colony",
    workers=8,           # Number of worker threads/processes
    coordinator="local"  # or "distributed" for cluster mode
)
```

### Worker Bees

Workers are the computational units that process data. HiveFrame uses a three-tier worker system:

| Worker Type | Role | Percentage | Behavior |
|-------------|------|------------|----------|
| **Employed** | Exploitation | ~50% | Process assigned tasks, report quality |
| **Onlooker** | Reinforcement | ~40% | Choose high-quality tasks based on dances |
| **Scout** | Exploration | ~10% | Search for new opportunities, recover abandoned tasks |

See [Three-Tier Workers](./three-tier-workers) for details.

### Waggle Dance Protocol

Workers communicate task quality through a virtual "waggle dance":

```mermaid
sequenceDiagram
    participant E as Employed Bee
    participant D as Dance Floor
    participant O as Onlooker Bee
    
    E->>E: Process food source
    E->>E: Calculate quality score
    E->>D: Perform waggle dance
    Note over D: Dance intensity ∝ quality
    O->>D: Observe dances
    O->>O: Choose based on probability
    O->>E: Follow to food source
```

See [Waggle Dance Protocol](./waggle-dance-protocol) for details.

### Query Optimizer

The ABC (Artificial Bee Colony) optimizer generates and evaluates query plans:

```mermaid
flowchart LR
    subgraph Input
        Q[SQL Query]
    end
    
    subgraph ABC["ABC Optimizer"]
        Init[Initialize Plans]
        Employed[Employed Phase]
        Onlooker[Onlooker Phase]
        Scout[Scout Phase]
        Best[Best Plan]
    end
    
    subgraph Output
        Plan[Physical Plan]
    end
    
    Q --> Init
    Init --> Employed
    Employed --> Onlooker
    Onlooker --> Scout
    Scout --> Best
    Best --> Plan
    
    Scout -.-> Init
    
    style Employed fill:#f59e0b
    style Onlooker fill:#fcd34d
    style Scout fill:#fef3c7
```

See [ABC Optimization](./abc-optimization) for details.

### Storage Layer

HiveFrame supports multiple storage formats:

| Format | Use Case | Features |
|--------|----------|----------|
| **Parquet** | Analytics | Columnar, compressed, predicate pushdown |
| **Delta Lake** | Lakehouse | ACID, time travel, schema evolution |
| **In-Memory** | Processing | Fast iteration, caching |

### Stream Processor

The streaming engine handles unbounded data with:

- **Windows** - Tumbling, sliding, session
- **Watermarks** - Late data handling
- **State management** - Checkpointing, recovery
- **Delivery guarantees** - At-most-once to exactly-once

## Data Flow

### Batch Processing

```mermaid
flowchart LR
    subgraph Read
        S[Source]
        P[Partitioner]
    end
    
    subgraph Process
        W1[Worker 1]
        W2[Worker 2]
        W3[Worker 3]
    end
    
    subgraph Write
        C[Collector]
        D[Destination]
    end
    
    S --> P
    P --> W1
    P --> W2
    P --> W3
    W1 --> C
    W2 --> C
    W3 --> C
    C --> D
    
    style W1 fill:#f59e0b
    style W2 fill:#f59e0b
    style W3 fill:#f59e0b
```

1. **Source** reads data (Parquet, database, etc.)
2. **Partitioner** distributes data to workers
3. **Workers** process partitions in parallel
4. **Collector** gathers results
5. **Destination** writes output

### Stream Processing

```mermaid
flowchart LR
    subgraph Input
        E[Events]
    end
    
    subgraph Process
        W[Watermark]
        T[Transform]
        WI[Window]
        A[Aggregate]
    end
    
    subgraph Output
        S[Sink]
        C[Checkpoint]
    end
    
    E --> W
    W --> T
    T --> WI
    WI --> A
    A --> S
    A -.-> C
    
    style W fill:#fef3c7
    style WI fill:#fbbf24
```

1. **Events** arrive continuously
2. **Watermark** tracks event-time progress
3. **Transform** applies map/filter operations
4. **Window** groups events by time
5. **Aggregate** computes results per window
6. **Sink** outputs results
7. **Checkpoint** saves state for recovery

## Deployment Models

### Local Mode

Single machine, multiple threads:

```python
colony = hf.Colony(name="local", coordinator="local")
```

### Cluster Mode

Distributed across machines:

```python
colony = hf.Colony(
    name="cluster",
    coordinator="distributed",
    coordinator_url="hiveframe://coordinator:5000"
)
```

### Kubernetes Mode

Managed by the HiveFrame Operator:

```yaml
apiVersion: hiveframe.io/v1
kind: HiveFrameCluster
metadata:
  name: production
spec:
  colony:
    name: prod-colony
  bees:
    employed:
      replicas: 10
```

## Key Design Principles

### 1. Decentralization

No single point of failure. Workers self-organize without a central driver.

### 2. Emergent Behavior

Complex system behavior emerges from simple worker rules:
- Report quality honestly
- Follow high-quality dances
- Abandon poor sources
- Explore when needed

### 3. Self-Healing

The colony automatically recovers from failures:
- Scout bees discover abandoned tasks
- Workers redistribute load
- Checkpoints enable recovery

### 4. Adaptive Resource Allocation

Resources flow to high-quality tasks:
- Better tasks attract more workers
- Poor tasks are abandoned
- System naturally load-balances

## Next Steps

- [Waggle Dance Protocol](./waggle-dance-protocol) - How workers communicate
- [Three-Tier Workers](./three-tier-workers) - Worker types and roles
- [ABC Optimization](./abc-optimization) - Query optimization algorithm
