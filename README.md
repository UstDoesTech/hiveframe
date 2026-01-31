# 🐝 HiveFrame

**A bee-inspired distributed data processing framework — a biomimetic alternative to Apache Spark**

[![Version](https://img.shields.io/badge/version-0.2.0--dev-green.svg)](https://github.com/hiveframe/hiveframe)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

HiveFrame replaces Spark's centralized driver model with **decentralized bee colony coordination**. Instead of a single driver scheduling tasks across executors, HiveFrame uses autonomous "bee" workers that self-organize through:

| Bee Behavior | Software Pattern |
|-------------|-----------------|
| **Waggle Dance** | Workers advertise task quality through dance signals |
| **Three-Tier Workers** | Employed (exploit), Onlooker (reinforce), Scout (explore) |
| **Stigmergic Coordination** | Indirect communication through shared colony state |
| **Pheromone Signaling** | Backpressure and rate limiting |
| **Abandonment Mechanism** | Self-healing through ABC algorithm |

## Why Bee-Inspired Processing?

Traditional distributed frameworks (Spark, Flink) use centralized schedulers that become bottlenecks. Bee colonies solve the same coordination problems without any central controller:

```
Spark Model:                    HiveFrame Model:
                                
    ┌─────────┐                     🐝 ←→ 🐝 ←→ 🐝
    │ Driver  │                      ↑     ↑     ↑
    └────┬────┘                      │  Dance    │
         │                           │  Floor    │
    ┌────┴────┐                      ↓     ↓     ↓
    ▼    ▼    ▼                     🐝 ←→ 🐝 ←→ 🐝
   [E]  [E]  [E]                   (Self-organizing)
  (Executors)
```

**Key advantages:**
- ✓ No single point of failure (no driver)
- ✓ Quality-weighted work distribution
- ✓ Self-healing through abandonment
- ✓ Adaptive backpressure via pheromones
- ✓ Emergent load balancing

## Key Features

### Phase 1 (Complete)
| Feature | Description |
|---------|-------------|
| **SwarmQL** | Full SQL engine with table registration and query optimization |
| **Query Optimizer** | Catalyst-equivalent optimizer with swarm-based cost estimation |
| **Parquet & Delta Lake** | Production-grade columnar storage with ACID transactions |
| **Kubernetes Native** | Deploy and scale colonies on K8s with custom operator |
| **Colony Dashboard** | Real-time web UI for monitoring colony health and metrics |
| **Monitoring Suite** | Prometheus metrics, distributed tracing, and profiling |
| **Resilience Patterns** | Circuit breakers, retry policies, bulkheads, and timeouts |
| **Advanced Streaming** | Windowing, watermarks, and exactly-once delivery guarantees |
| **Dead Letter Queue** | Failed record management with full error context |

### Phase 2 (Complete) ✅
| Feature | Description |
|---------|-------------|
| **Multi-Hive Federation** | Connect multiple clusters that coordinate like allied bee colonies |
| **Adaptive Partitioning** | Dynamic partition splitting/merging based on swarm fitness |
| **Speculative Execution** | Scout bees proactively retry slow tasks |
| **Vectorized Execution** | SIMD-accelerated processing for numerical workloads |
| **Adaptive Query Execution** | Real-time plan modification based on waggle dance feedback |
| **HoneyStore** | Native columnar format optimized for swarm access patterns |
| **Iceberg Support** | Open table format compatibility |
| **Caching Swarm** | Distributed intelligent caching with pheromone trails |

### Phase 3 (In Progress) 🚧
| Feature | Description |
|---------|-------------|
| **Unity Hive Catalog** | Unified data governance with fine-grained access control and lineage tracking |
| **AutoML Swarm** | Hyperparameter optimization using bee-inspired ABC algorithm |
| **Feature Hive** | Centralized feature store with versioning and automatic feature engineering |
| **Model Server** | Production inference with swarm-based load balancing |
| **HiveFrame Notebooks** | Multi-language interactive environment (Python, SQL, R, Scala) |
| **Delta Sharing** | Secure data sharing across organizations |
| **MLflow Integration** | Experiment tracking and model registry |

## Installation

```bash
pip install hiveframe
```

With optional dependencies:

```bash
# Full installation with all features
pip install hiveframe[all]

# Specific extras
pip install hiveframe[monitoring]   # Prometheus metrics, tracing
pip install hiveframe[kafka]        # Kafka connector support
pip install hiveframe[postgres]     # PostgreSQL sink support
pip install hiveframe[http]         # HTTP source connector
pip install hiveframe[dashboard]    # Colony web dashboard
```

Or from source:

```bash
git clone https://github.com/hiveframe/hiveframe.git
cd hiveframe
pip install -e ".[all]"
```

## Quick Start

### RDD-Style API

```python
from hiveframe import HiveFrame, create_hive

# Create a hive with 8 workers
hive = create_hive(num_workers=8)

# Map operation
data = list(range(1000))
doubled = hive.map(data, lambda x: x * 2)

# Filter
filtered = hive.filter(doubled, lambda x: x > 500)

# Reduce
total = hive.reduce(filtered, lambda a, b: a + b)
print(f"Sum: {total}")
```

### DataFrame API (Spark-like)

```python
from hiveframe import HiveDataFrame, col, sum_agg, avg, count

# Load data
df = HiveDataFrame.from_csv('transactions.csv')

# Query with familiar Spark syntax
result = (df
    .filter(col('amount') > 100)
    .filter(col('category') == 'Electronics')
    .groupBy('region')
    .agg(
        count(col('id')),
        sum_agg(col('amount')),
        avg(col('amount'))
    )
    .orderBy('region'))

result.show()
```

### Streaming

```python
from hiveframe import HiveStream

# Create stream processor
stream = HiveStream(num_workers=6)

def process(record):
    return {'processed': record['value'] * 2}

# Start processing
stream.start(process)

# Submit records
for i in range(1000):
    stream.submit(f"key_{i}", {'value': i})

# Get results
while True:
    result = stream.get_result(timeout=1.0)
    if result is None:
        break
    print(result)

stream.stop()
```

### SQL Queries (SwarmQL)

```python
from hiveframe.sql import SwarmQLContext

# Create SQL context
ctx = SwarmQLContext()

# Register DataFrames as tables
ctx.register_table("users", users_df)
ctx.register_table("orders", orders_df)

# Execute SQL queries
result = ctx.sql("""
    SELECT u.name, SUM(o.amount) as total_spent
    FROM users u
    JOIN orders o ON u.id = o.user_id
    WHERE o.status = 'completed'
    GROUP BY u.name
    ORDER BY total_spent DESC
""")

result.show()
```

### Parquet & Delta Lake Storage

```python
from hiveframe.storage import ParquetTable, DeltaTable

# Write to Parquet
df.write_parquet('data/output.parquet', compression='snappy')

# Read from Parquet
df = ParquetTable.read('data/output.parquet')

# Delta Lake with ACID transactions
delta = DeltaTable('data/delta_table')
delta.write(df, mode='append')
delta.optimize()  # Compact small files
delta.vacuum(hours=168)  # Clean old versions

# Time travel
df_yesterday = delta.as_of(version=5)
```

### Kubernetes Deployment

```python
from hiveframe.k8s import HiveCluster, HiveOperator

# Define cluster
cluster = HiveCluster(
    name="production-hive",
    workers=20,
    worker_memory="8Gi",
    worker_cpu="4",
    queen_memory="16Gi"
)

# Deploy with operator
operator = HiveOperator(namespace="hiveframe")
operator.deploy(cluster)
operator.scale(cluster, workers=50)  # Scale up
operator.status(cluster)  # Check health
```

### Monitoring & Observability

```python
from hiveframe.monitoring import MetricsCollector, HealthMonitor, SpanTracer

# Prometheus metrics
metrics = MetricsCollector(port=9090)
metrics.register_counter('records_processed')
metrics.register_histogram('processing_latency')

# Health monitoring
health = HealthMonitor()
health.register_check('kafka', kafka_health_check)
health.register_check('storage', storage_health_check)
print(health.status())  # {'status': 'healthy', 'checks': {...}}

# Distributed tracing
tracer = SpanTracer(service_name="hiveframe-etl")
with tracer.span("process_batch") as span:
    span.set_attribute("batch_size", 1000)
    process_records(batch)
```

### Resilience Patterns

```python
from hiveframe.resilience import (
    RetryPolicy, CircuitBreaker, BulkheadPattern,
    retry, circuit_breaker, with_timeout
)

# Retry with exponential backoff
@retry(max_attempts=3, backoff='exponential', base_delay=1.0)
def fetch_data(url):
    return requests.get(url).json()

# Circuit breaker for external services
@circuit_breaker(failure_threshold=5, recovery_timeout=30)
def call_external_api(payload):
    return api_client.post(payload)

# Bulkhead pattern for resource isolation
bulkhead = BulkheadPattern(max_concurrent=10, max_queued=100)
with bulkhead.acquire():
    process_expensive_operation()

# Timeout enforcement
@with_timeout(seconds=30)
def long_running_task():
    ...
```

### Colony Dashboard

```python
from hiveframe.dashboard import Dashboard

# Start web UI
dashboard = Dashboard(port=8080)
dashboard.start()

# Access at http://localhost:8080
# Features:
#   - Real-time colony health metrics
#   - Worker distribution visualization
#   - Task queue monitoring
#   - Pheromone trail activity
```

## Phase 2: Swarm Intelligence Features

### Multi-Hive Federation

```python
from hiveframe.distributed import HiveFederation, FederatedHive, HiveRegistry

# Create a federation of multiple hives
federation = HiveFederation(name="global-analytics")

# Register hives in different data centers
hive_east = FederatedHive(name="east-dc", endpoint="east.example.com:9000", workers=20)
hive_west = FederatedHive(name="west-dc", endpoint="west.example.com:9000", workers=15)

federation.register_hive(hive_east)
federation.register_hive(hive_west)

# Distribute work across federated hives
result = federation.execute_distributed(
    data=large_dataset,
    fn=process_function,
    locality_hints={"region": "east"}  # Prefer east datacenter
)
```

### Adaptive Partitioning

```python
from hiveframe.distributed import AdaptivePartitioner, PartitionStrategy

# Create adaptive partitioner
partitioner = AdaptivePartitioner(
    strategy=PartitionStrategy.FITNESS_BASED,
    min_partition_size=100_000,
    max_partition_size=10_000_000
)

# Partitioner automatically splits/merges based on:
# - Processing time fitness
# - Data distribution patterns
# - Worker load balancing
df = HiveDataFrame.from_csv('large_file.csv')
partitioned_df = partitioner.partition(df)

# Partitions adapt during processing based on swarm feedback
result = partitioned_df.groupBy('category').agg(sum_agg(col('amount')))
```

### Speculative Execution

```python
from hiveframe.distributed import SpeculativeExecutor, SpeculativeConfig

# Configure speculative execution
config = SpeculativeConfig(
    enabled=True,
    slow_task_threshold=1.5,  # 1.5x median time
    speculation_fraction=0.1   # Speculate on slowest 10%
)

executor = SpeculativeExecutor(config=config, num_scouts=4)

# Scout bees automatically detect and retry slow tasks
result = executor.execute(
    data=data,
    fn=expensive_computation,
    timeout=300
)

# Metrics show speculative task improvements
print(executor.get_metrics())
# {'tasks_total': 1000, 'tasks_speculated': 42, 'time_saved_seconds': 127}
```

### Vectorized Execution

```python
from hiveframe.optimizer import VectorizedPipeline, VectorBatch

# Create vectorized pipeline
pipeline = VectorizedPipeline()

# SIMD-accelerated operations on numeric data
df = HiveDataFrame.from_csv('numerical_data.csv')

# Vectorized filter, project, and aggregate
result = (df
    .filter(col('value') > 100)         # Vectorized comparison
    .select(col('value') * 2.5)         # Vectorized arithmetic
    .agg(sum_agg(col('value')))         # Vectorized aggregation
)

# 5-10x speedup on numerical workloads
```

### Adaptive Query Execution (AQE)

```python
from hiveframe.optimizer import AdaptiveQueryExecutor, AQEContext

# Enable adaptive query execution
ctx = SwarmQLContext(aqe_enabled=True)

# Query plans adapt during execution based on runtime statistics
result = ctx.sql("""
    SELECT customer_id, SUM(amount) as total
    FROM orders
    JOIN customers ON orders.customer_id = customers.id
    WHERE status = 'completed'
    GROUP BY customer_id
""")

# AQE automatically:
# - Switches join strategies based on actual data sizes
# - Adjusts partition counts dynamically
# - Reoptimizes based on waggle dance feedback
# - Handles data skew automatically
```

### HoneyStore (Native Columnar Format)

```python
from hiveframe.storage import write_honeystore, read_honeystore, HoneyStoreWriter

# Write data in HoneyStore format
df = HiveDataFrame.from_csv('sales.csv')
write_honeystore(df, 'sales.honey', compression='adaptive')

# HoneyStore features:
# - Columnar layout optimized for swarm access
# - Adaptive compression based on data patterns
# - Honeycomb blocks for balanced parallel I/O
# - Nectar encoding for efficient null handling

# Read with column pruning and predicate pushdown
df = read_honeystore(
    'sales.honey',
    columns=['customer_id', 'amount'],
    filters={'region': 'West'}
)

# Native integration with swarm optimizer
result = df.groupBy('customer_id').agg(sum_agg(col('amount')))
```

### Iceberg Support

```python
from hiveframe.storage import IcebergTable, read_iceberg, write_iceberg

# Create Iceberg table
table = IcebergTable('warehouse/sales_iceberg')

# Write with schema evolution
df = HiveDataFrame.from_csv('sales.csv')
write_iceberg(df, table, mode='append')

# Time travel
df_yesterday = read_iceberg(table, snapshot_id=12345)
df_last_week = read_iceberg(table, timestamp='2024-01-15T00:00:00Z')

# Schema evolution
new_df = df.withColumn('new_field', col('amount') * 1.1)
write_iceberg(new_df, table, mode='append')  # Schema automatically evolves

# Hidden partitioning (Iceberg handles partitioning transparently)
partitioned_df = read_iceberg(
    table,
    filters={'date': '2024-01-20', 'region': 'East'}
)
```

### Caching Swarm

```python
from hiveframe.storage import CachingSwarm, PheromoneCache

# Initialize caching swarm with pheromone-based eviction
cache = CachingSwarm(
    max_size_gb=10.0,
    eviction_policy='pheromone',  # Swarm intelligence-based
    l1_size_mb=512,
    l2_size_gb=2.0,
    l3_distributed=True
)

# Cache frequently accessed data
df = HiveDataFrame.from_csv('large_dataset.csv')
cache.put('dataset_1', df)

# Pheromone trails track access patterns
# - Frequently accessed data gets stronger pheromone
# - Pheromones decay over time
# - Eviction prioritizes weak pheromone trails

# Automatic prefetching by scout bees
result = cache.get('dataset_1')  # Fast cache hit
related = cache.get('dataset_2')  # Prefetched by swarm intelligence

# Distributed cache coordination across cluster
cache.enable_distributed_coordination()

# Cache metrics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Avg pheromone: {stats['avg_pheromone']:.2f}")
```

## Phase 3: Enterprise Platform Features

### Unity Hive Catalog

```python
from hiveframe.lakehouse import (
    UnityHiveCatalog, AccessControl, LineageTracker, 
    PIIDetector, PermissionType
)

# Create unified data catalog
catalog = UnityHiveCatalog()

# Register tables with metadata
catalog.register_table(
    name="users",
    schema={"id": "int", "name": "string", "email": "string"},
    location="/data/users",
    format="parquet",
    owner="data_team",
    tags={"production", "pii"}
)

# Fine-grained access control
acl = AccessControl(catalog)
acl.grant("analyst@company.com", "users", [PermissionType.SELECT])
acl.grant("data_engineer@company.com", "users", [PermissionType.ALL])

# Check permissions
has_access = acl.check_permission("analyst@company.com", "users", PermissionType.SELECT)

# Data lineage tracking
lineage = LineageTracker(catalog)
lineage.record_lineage(
    output_table="user_summary",
    input_tables=["users", "orders"],
    operation="join"
)

# Get lineage graph
upstream = lineage.get_upstream("user_summary", recursive=True)
downstream = lineage.get_downstream("users")

# Automatic PII detection
pii_detector = PIIDetector()
pii_fields = pii_detector.scan_table(catalog, "users")
print(f"PII fields detected: {pii_fields}")
# {'email': 'EMAIL', 'phone': 'PHONE_NUMBER'}
```

### AutoML Swarm

```python
from hiveframe.ml import AutoMLSwarm, HyperparameterSpace, TaskType

# Define hyperparameter search space
search_space = HyperparameterSpace({
    'learning_rate': (0.001, 0.1, 'log'),
    'max_depth': (3, 10, 'int'),
    'n_estimators': (50, 500, 'int'),
    'min_samples_split': (2, 20, 'int')
})

# Create AutoML swarm with bee-inspired optimization
automl = AutoMLSwarm(
    n_workers=8,           # Number of bee workers
    max_iterations=50,     # ABC algorithm iterations
    task=TaskType.CLASSIFICATION
)

# Fit using ABC algorithm for hyperparameter optimization
best_model = automl.fit(
    X_train, y_train,
    search_space=search_space,
    cv=5,  # Cross-validation folds
    metric='accuracy'
)

# Get optimization results
print(f"Best accuracy: {automl.best_score_:.3f}")
print(f"Best params: {automl.best_params_}")
print(f"Total models evaluated: {automl.n_models_evaluated_}")

# Make predictions with best model
predictions = best_model.predict(X_test)

# Get swarm optimization history
history = automl.get_optimization_history()
```

### Feature Hive (Feature Store)

```python
from hiveframe.ml import FeatureHive, FeatureType
from datetime import datetime, timedelta

# Create feature store
feature_store = FeatureHive(storage_path="features/")

# Register feature with compute function
def compute_user_activity_7d(user_ids, timestamp):
    """Compute 7-day user activity features."""
    # Query activity data
    activity = query_user_activity(user_ids, timestamp - timedelta(days=7), timestamp)
    return {
        'page_views_7d': activity.groupby('user_id')['views'].sum(),
        'sessions_7d': activity.groupby('user_id')['session_id'].nunique(),
        'avg_session_duration_7d': activity.groupby('user_id')['duration'].mean()
    }

feature_store.register_feature(
    name="user_activity_7d",
    feature_type=FeatureType.BATCH,
    compute_fn=compute_user_activity_7d,
    version="v1",
    dependencies=["raw_events"]
)

# Get features for training
user_ids = [101, 102, 103]
features = feature_store.get_features(
    feature_names=["user_activity_7d", "user_demographics"],
    entity_ids=user_ids,
    timestamp=datetime.now()
)

# Point-in-time correct features for historical training
training_features = feature_store.get_historical_features(
    feature_names=["user_activity_7d"],
    entity_ids=user_ids,
    timestamps=training_timestamps
)

# Feature versioning and lineage
version_info = feature_store.get_feature_version("user_activity_7d", version="v1")
```

### Model Server

```python
from hiveframe.ml import ModelServer, DistributedTrainer

# Serve model with swarm-based load balancing
server = ModelServer(
    model=trained_model,
    num_workers=4,
    batch_size=32,
    timeout=1.0
)

# Start inference server
server.start(port=8080)

# Make predictions (load balanced across swarm)
predictions = server.predict(input_data)

# Get serving metrics
metrics = server.get_metrics()
print(f"Requests/sec: {metrics['requests_per_sec']}")
print(f"Avg latency: {metrics['avg_latency_ms']:.1f}ms")
print(f"Worker utilization: {metrics['worker_utilization']:.1%}")

# Distributed training
trainer = DistributedTrainer(
    model_class=MyModel,
    num_workers=8,
    strategy='data_parallel'
)

# Train across multiple nodes
history = trainer.fit(
    train_data=train_loader,
    val_data=val_loader,
    epochs=10
)
```

### HiveFrame Notebooks

```python
from hiveframe.notebooks import NotebookKernel, NotebookSession, CollaborationManager

# Create multi-language kernel
kernel = NotebookKernel(language='python')
session = NotebookSession(kernel)

# Execute Python cell
result = session.execute_cell("""
from hiveframe import HiveDataFrame
df = HiveDataFrame.from_csv('data.csv')
df.filter(col('amount') > 100).show()
""")

# Switch to SQL kernel
sql_kernel = NotebookKernel(language='sql')
sql_session = NotebookSession(sql_kernel)

# Execute SQL cell
result = sql_session.execute_cell("""
SELECT customer_id, SUM(amount) as total
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
""")

# Real-time collaboration
collab = CollaborationManager()
collab.create_session("notebook-123", owner="user@example.com")
collab.join_session("notebook-123", user="analyst@example.com")

# Share execution state across users
shared_state = collab.get_shared_state("notebook-123")

# GPU-accelerated cells
from hiveframe.notebooks import GPUCell

gpu_cell = GPUCell(device='cuda:0')
result = gpu_cell.execute("""
import torch
model = torch.nn.Linear(10, 1).cuda()
output = model(torch.randn(100, 10).cuda())
""")
```

### Delta Sharing

```python
from hiveframe.lakehouse import DeltaSharing, ShareAccessLevel

# Create Delta Sharing server
sharing = DeltaSharing(catalog=catalog)

# Create a share
share = sharing.create_share(
    name="analytics_share",
    description="Shared analytics tables"
)

# Add table to share
sharing.add_table_to_share(
    share_name="analytics_share",
    table_name="users",
    access_level=ShareAccessLevel.READ
)

# Create recipient
recipient = sharing.create_recipient(
    name="partner_org",
    email="data@partner.com"
)

# Grant access
sharing.grant_access(
    share_name="analytics_share",
    recipient_name="partner_org",
    expiration_days=90
)

# Recipients can access shared data
shared_table = sharing.read_shared_table(
    share_url="https://sharing.hiveframe.io/analytics_share",
    table_name="users",
    credential_token="<token>"
)
```

### MLflow Integration

```python
from hiveframe.ml import MLflowIntegration, ModelStage

# Initialize MLflow tracking
mlflow_client = MLflowIntegration(
    tracking_uri="http://mlflow.example.com",
    experiment_name="hiveframe_experiments"
)

# Log training run
with mlflow_client.start_run() as run:
    # Train model
    model = automl.fit(X_train, y_train, search_space=search_space)
    
    # Log parameters
    mlflow_client.log_params(automl.best_params_)
    
    # Log metrics
    mlflow_client.log_metrics({
        'accuracy': automl.best_score_,
        'training_time': automl.training_time_
    })
    
    # Log model
    mlflow_client.log_model(model, "best_model")
    
    # Log artifacts
    mlflow_client.log_artifact("feature_importance.png")

# Register model in model registry
model_uri = f"runs:/{run.run_id}/best_model"
mlflow_client.register_model(
    model_uri=model_uri,
    name="customer_churn_predictor",
    stage=ModelStage.STAGING
)

# Promote to production
mlflow_client.transition_model_stage(
    model_name="customer_churn_predictor",
    version=1,
    stage=ModelStage.PRODUCTION
)

# Load production model
production_model = mlflow_client.load_model(
    model_name="customer_churn_predictor",
    stage=ModelStage.PRODUCTION
)
```

## Architecture

### Colony Structure

```
┌─────────────────────────────────────────────────────────────┐
│                      COLONY STATE                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Food Sources│  │ Dance Floor │  │ Pheromone Trails    │ │
│  │ (Partitions)│  │ (Comms Hub) │  │ (Coordination)      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐        ┌─────────┐        ┌─────────┐
   │EMPLOYED │        │ONLOOKER │        │ SCOUT   │
   │  50%    │        │  40%    │        │  10%    │
   │ Exploit │        │Reinforce│        │ Explore │
   └─────────┘        └─────────┘        └─────────┘
```

### Worker Roles

| Role | Percentage | Behavior |
|------|-----------|----------|
| **Employed** | 50% | Process assigned partitions, perform waggle dances |
| **Onlooker** | 40% | Observe dances, select high-quality partitions |
| **Scout** | 10% | Replace abandoned partitions, explore new territory |

### Waggle Dance Protocol

When a worker completes processing, it "dances" to advertise:

```python
@dataclass
class WaggleDance:
    partition_id: str      # Which partition (direction)
    quality_score: float   # How good the result (nectar quality)
    processing_time: float # How long it took (distance)
    result_size: int       # Output volume (throughput)
    worker_id: str         # Who's dancing
```

Other workers observe dances and probabilistically select partitions based on quality — high-quality partitions receive more attention.

### ABC Algorithm Implementation

HiveFrame implements the **Artificial Bee Colony (ABC)** algorithm:

1. **Employed Phase**: Workers process assigned partitions
2. **Onlooker Phase**: Workers select partitions based on observed quality (roulette wheel)
3. **Scout Phase**: Abandoned partitions (no improvement after N cycles) are reset

```python
# Position update (neighborhood search)
v_ij = x_ij + φ_ij * (x_ij - x_kj)

# Selection probability (fitness-proportional)
p_i = fitness_i / Σ(fitness_n)

# Abandonment: trials >= limit → reset with random exploration
```

## API Reference

### HiveFrame (Core Engine)

```python
hive = HiveFrame(
    num_workers=8,           # Total bee count
    employed_ratio=0.5,      # Fraction exploiting
    onlooker_ratio=0.4,      # Fraction reinforcing
    scout_ratio=0.1,         # Fraction exploring
    abandonment_limit=10,    # Cycles before abandonment
    max_cycles=100           # Maximum processing cycles
)

# Operations
results = hive.map(data, transform_fn)
filtered = hive.filter(data, predicate_fn)
total = hive.reduce(data, combine_fn)
groups = hive.group_by_key(pairs)
flat = hive.flat_map(data, expand_fn)
```

### HiveDataFrame

```python
# Construction
df = HiveDataFrame.from_csv('data.csv')
df = HiveDataFrame.from_json('data.json')
df = HiveDataFrame.from_records([{'a': 1}, {'a': 2}])

# Transformations
df.select('col1', 'col2')
df.filter(col('age') > 21)
df.withColumn('new_col', col('a') + col('b'))
df.drop('unwanted_col')
df.distinct()
df.orderBy('col', ascending=False)
df.limit(100)

# Grouping & Aggregation
df.groupBy('category').agg(sum_agg(col('amount')), avg(col('price')))

# Joins
df1.join(df2, on='key', how='inner')  # inner, left, right, outer

# Output
df.show(n=20)
df.collect()
df.to_csv('output.csv')
df.to_json('output.json')
```

### Column Expressions

```python
from hiveframe import col, lit

# Comparisons
col('age') > 21
col('name') == 'Alice'
col('status').isNull()

# Arithmetic
col('price') * col('quantity')
col('total') / 100

# String operations
col('name').contains('Smith')
col('email').endswith('@gmail.com')

# Aliasing
(col('a') + col('b')).alias('sum')
```

### Aggregation Functions

```python
from hiveframe import sum_agg, avg, count, min_agg, max_agg, collect_list

df.groupBy('category').agg(
    count(col('id')),           # Count non-null
    sum_agg(col('amount')),     # Sum values
    avg(col('price')),          # Average
    min_agg(col('date')),       # Minimum
    max_agg(col('score')),      # Maximum
    collect_list(col('tags'))   # Collect into list
)
```

### HiveStream

```python
stream = HiveStream(
    num_workers=8,
    buffer_size=10000,
    employed_ratio=0.5,
    onlooker_ratio=0.3,
    scout_ratio=0.2
)

stream.start(process_fn)
stream.submit(key, value)
result = stream.get_result(timeout=1.0)
metrics = stream.get_metrics()
stream.stop()
```

### Advanced Streaming

```python
from hiveframe.streaming import (
    EnhancedStreamProcessor,
    tumbling_window, sliding_window, session_window,
    bounded_watermark, DeliveryGuarantee,
    WindowAggregation, StateStore
)

# Windowed processing with exactly-once delivery
processor = EnhancedStreamProcessor(
    window=tumbling_window(seconds=60),
    watermark=bounded_watermark(max_lateness=10),
    delivery=DeliveryGuarantee.EXACTLY_ONCE
)

# Window types
tumbling = tumbling_window(seconds=30)      # Fixed, non-overlapping
sliding = sliding_window(size=60, slide=10)  # Overlapping windows
session = session_window(gap=300)            # Activity-based

# Stateful processing
state = StateStore(backend='rocksdb')
processor.with_state(state)
```

### SwarmQL (SQL Engine)

```python
from hiveframe.sql import SwarmQLContext

ctx = SwarmQLContext()

# Table management
ctx.register_table('users', users_df)
ctx.register_temp_view('active_users', active_df)
ctx.list_tables()  # ['users', 'active_users']
ctx.drop_table('active_users')

# Query execution
result = ctx.sql("SELECT * FROM users WHERE age > 21")
result = ctx.sql("SELECT dept, AVG(salary) FROM employees GROUP BY dept")

# Supported SQL: SELECT, FROM, WHERE, JOIN, GROUP BY, ORDER BY, LIMIT
# Aggregates: COUNT, SUM, AVG, MIN, MAX
# Joins: INNER, LEFT, RIGHT, OUTER
```

### Query Optimizer

```python
from hiveframe.optimizer import SwarmOptimizer

optimizer = SwarmOptimizer(
    colony_size=50,
    max_iterations=100
)

# Automatic optimization (applied by default)
# Optimizations include:
#   - Predicate pushdown
#   - Projection pruning
#   - Join reordering
#   - Filter merging
#   - Constant folding

# Manual optimization
optimized_plan = optimizer.optimize(logical_plan)
stats = optimizer.explain(optimized_plan)
```

### Storage Layer

```python
from hiveframe.storage import ParquetTable, DeltaTable, StorageFormat

# Parquet operations
ParquetTable.write(df, 'output.parquet', compression='snappy')
df = ParquetTable.read('output.parquet', columns=['col1', 'col2'])
schema = ParquetTable.schema('output.parquet')

# Delta Lake operations
delta = DeltaTable('path/to/table')
delta.write(df, mode='overwrite')  # overwrite, append, merge
delta.merge(source_df, condition="target.id = source.id")
delta.delete("status = 'expired'")
delta.update({"status": "'active'"}, "created_at > '2024-01-01'")

# Time travel
df = delta.as_of(version=10)
df = delta.as_of(timestamp='2024-01-15T00:00:00')
history = delta.history()

# Maintenance
delta.optimize()       # Compact small files
delta.vacuum(hours=168) # Remove old versions
```

### Kubernetes Deployment

```python
from hiveframe.k8s import HiveCluster, HiveOperator, generate_manifests

# Cluster configuration
cluster = HiveCluster(
    name="my-hive",
    namespace="production",
    workers=10,
    worker_memory="4Gi",
    worker_cpu="2",
    queen_memory="8Gi",
    queen_cpu="4",
    storage_class="fast-ssd",
    image="hiveframe/hiveframe:0.2.0"
)

# Operator deployment
operator = HiveOperator()
operator.deploy(cluster)
operator.scale(cluster, workers=20)
operator.rolling_update(cluster, image="hiveframe/hiveframe:0.2.1")
status = operator.status(cluster)

# Generate raw manifests
manifests = generate_manifests(cluster)
print(manifests['deployment'])
print(manifests['service'])
print(manifests['configmap'])
```

### Monitoring

```python
from hiveframe.monitoring import (
    MetricsCollector, HealthMonitor, SpanTracer,
    StructuredLogger, Profiler
)

# Metrics (Prometheus-compatible)
metrics = MetricsCollector()
metrics.register_counter('events_processed', labels=['source'])
metrics.register_gauge('queue_depth')
metrics.register_histogram('latency_ms', buckets=[10, 50, 100, 500])
metrics.increment('events_processed', labels={'source': 'kafka'})
metrics.expose(port=9090)  # /metrics endpoint

# Health checks
health = HealthMonitor()
health.register_check('database', db_check, interval=30)
health.register_check('kafka', kafka_check, critical=True)
status = health.status()  # {"healthy": true, "checks": {...}}

# Distributed tracing (OpenTelemetry)
tracer = SpanTracer(service_name="hive-etl", exporter="jaeger")
with tracer.span("process") as span:
    span.set_attribute("batch_id", batch_id)
    with tracer.span("transform"):
        transform(data)

# Structured logging
logger = StructuredLogger("hiveframe", format="json")
logger.info("Processing started", batch_id=123, records=1000)

# Profiling
profiler = Profiler()
with profiler.profile("expensive_operation"):
    run_computation()
profiler.report()
```

### Resilience

```python
from hiveframe.resilience import (
    RetryPolicy, BackoffStrategy,
    CircuitBreaker, CircuitState,
    BulkheadPattern, TimeoutEnforcer,
    ResilientExecutor
)

# Retry policies
policy = RetryPolicy(
    max_attempts=5,
    backoff=BackoffStrategy.EXPONENTIAL,
    base_delay=1.0,
    max_delay=60.0,
    retryable_exceptions=[ConnectionError, TimeoutError]
)
result = policy.execute(risky_operation)

# Circuit breaker
breaker = CircuitBreaker(
    failure_threshold=5,
    success_threshold=3,
    recovery_timeout=30,
    half_open_requests=1
)
if breaker.state == CircuitState.CLOSED:
    result = breaker.call(external_service)

# Bulkhead (concurrency limiting)
bulkhead = BulkheadPattern(
    max_concurrent=10,
    max_queued=50,
    timeout=5.0
)

# Combined resilience
executor = ResilientExecutor(
    retry_policy=policy,
    circuit_breaker=breaker,
    timeout=30.0
)
result = executor.execute(operation)
```

### Exception Handling

```python
from hiveframe.exceptions import (
    HiveFrameError, TransientError, DataQualityError,
    ResourceExhaustedError, ConfigurationError,
    ErrorSeverity, ErrorCategory
)
from hiveframe.dlq import DeadLetterQueue, FailedRecord

# Rich exception hierarchy
try:
    process_record(record)
except TransientError as e:
    # Retry-able errors (network, timeout)
    logger.warning(f"Transient error: {e}, retrying...")
except DataQualityError as e:
    # Bad data (schema mismatch, validation failure)
    dlq.push(FailedRecord(record, error=e))
except ResourceExhaustedError as e:
    # Backpressure needed
    apply_backpressure()

# Dead Letter Queue
dlq = DeadLetterQueue(storage_path='dlq/')
dlq.push(FailedRecord(
    record=failed_record,
    error=exception,
    timestamp=datetime.now(),
    source='kafka-topic-1'
))

# DLQ inspection
failed = dlq.pop(count=10)
stats = dlq.stats()  # {"total": 150, "by_error": {...}}
```

### Lakehouse (Unity Hive Catalog)

```python
from hiveframe.lakehouse import (
    UnityHiveCatalog, AccessControl, LineageTracker,
    PIIDetector, DeltaSharing, PermissionType
)

# Catalog management
catalog = UnityHiveCatalog()
catalog.register_table(name, schema, location, format='parquet')
tables = catalog.list_tables()
metadata = catalog.get_table_metadata(name)
catalog.search_by_tags({'production', 'analytics'})

# Access control
acl = AccessControl(catalog)
acl.grant(user, table, [PermissionType.SELECT, PermissionType.INSERT])
acl.revoke(user, table, [PermissionType.DELETE])
has_access = acl.check_permission(user, table, PermissionType.SELECT)

# Lineage tracking
tracker = LineageTracker(catalog)
tracker.record_lineage(output_table, input_tables, operation)
upstream = tracker.get_upstream(table, recursive=True)
downstream = tracker.get_downstream(table)

# PII detection
detector = PIIDetector()
pii_fields = detector.scan_table(catalog, table)
# Returns: {'email': 'EMAIL', 'ssn': 'SSN', ...}

# Delta sharing
sharing = DeltaSharing(catalog)
share = sharing.create_share(name, description)
sharing.add_table_to_share(share_name, table_name, access_level)
recipient = sharing.create_recipient(name, email)
sharing.grant_access(share_name, recipient_name, expiration_days)
```

### Machine Learning (HiveMind ML)

```python
from hiveframe.ml import (
    AutoMLSwarm, FeatureHive, ModelServer, 
    DistributedTrainer, MLflowIntegration,
    HyperparameterSpace, TaskType, FeatureType
)

# AutoML with bee-inspired optimization
search_space = HyperparameterSpace({
    'learning_rate': (0.001, 0.1, 'log'),
    'max_depth': (3, 10, 'int'),
})
automl = AutoMLSwarm(n_workers=8, max_iterations=50, task=TaskType.CLASSIFICATION)
best_model = automl.fit(X_train, y_train, search_space=search_space)
print(automl.best_score_, automl.best_params_)

# Feature store
feature_store = FeatureHive(storage_path='features/')
feature_store.register_feature(name, feature_type, compute_fn, version)
features = feature_store.get_features(feature_names, entity_ids, timestamp)
historical = feature_store.get_historical_features(feature_names, entity_ids, timestamps)

# Model serving
server = ModelServer(model, num_workers=4, batch_size=32)
server.start(port=8080)
predictions = server.predict(input_data)
metrics = server.get_metrics()  # requests/sec, latency, utilization

# Distributed training
trainer = DistributedTrainer(model_class, num_workers=8, strategy='data_parallel')
history = trainer.fit(train_data, val_data, epochs=10)

# MLflow integration
mlflow_client = MLflowIntegration(tracking_uri, experiment_name)
with mlflow_client.start_run() as run:
    mlflow_client.log_params(params)
    mlflow_client.log_metrics(metrics)
    mlflow_client.log_model(model, 'model')
mlflow_client.register_model(model_uri, name, stage)
```

### Notebooks

```python
from hiveframe.notebooks import (
    NotebookKernel, NotebookSession, CollaborationManager,
    NotebookFormat, GPUCell, KernelLanguage
)

# Kernel and session
kernel = NotebookKernel(language=KernelLanguage.PYTHON)
session = NotebookSession(kernel)
result = session.execute_cell(code_string)

# Multi-language support
sql_kernel = NotebookKernel(language=KernelLanguage.SQL)
r_kernel = NotebookKernel(language=KernelLanguage.R)
scala_kernel = NotebookKernel(language=KernelLanguage.SCALA)

# Real-time collaboration
collab = CollaborationManager()
collab.create_session(notebook_id, owner)
collab.join_session(notebook_id, user)
shared_state = collab.get_shared_state(notebook_id)

# Notebook format
nb_format = NotebookFormat()
notebook = nb_format.read('notebook.ipynb')
nb_format.write(notebook, 'output.ipynb')

# GPU-accelerated cells
gpu_cell = GPUCell(device='cuda:0')
result = gpu_cell.execute(gpu_code)
```

## Biomimicry Mapping

| Bee Behavior | HiveFrame Implementation |
|-------------|-------------------------|
| Waggle dance | `WaggleDance` dataclass, `DanceFloor` communication hub |
| Food source | `FoodSource` (data partition with fitness tracking) |
| Nectar quality | Quality score from processing function |
| Dance vigor | Composite metric: quality × throughput |
| Foraging | `Bee.forage()` method |
| Colony temperature | Aggregate worker load (homeostatic regulation) |
| Pheromone trail | `Pheromone` signals for throttling, alarms |
| Abandonment | Reset partition after N failed improvement cycles |
| Quorum sensing | Threshold-based decisions emerge from local interactions |

## Comparison with Spark

| Feature | Apache Spark | HiveFrame |
|---------|-------------|-----------|
| Architecture | Centralized driver | Decentralized colony |
| Scheduling | Central scheduler | Self-organizing workers |
| Fault tolerance | Task retry from driver | Abandonment + scout exploration |
| Load balancing | Round-robin / hash | Quality-weighted probabilistic |
| Backpressure | Rate limiters | Pheromone signals |
| Coordination | Direct messaging | Stigmergic (via environment) |

## Running the Demos

```bash
# Basic demo - core features
python examples/demo.py

# Progressive challenge demo - production scenarios
python examples/demo_progressive.py

# SQL queries (SwarmQL 1.0)
python examples/demo_sql.py

# Storage features (Parquet & Delta Lake)
python examples/demo_storage.py

# SwarmQL 2.0 features (CTEs, subqueries, window functions)
python examples/swarmql_2_demo.py

# Phase 2 demos - swarm intelligence features:
python examples/demo_phase2_federation.py   # Multi-hive federation
python examples/demo_phase2_adaptive.py     # Adaptive partitioning & speculative execution
python examples/demo_phase2_storage.py      # HoneyStore & caching swarm

# Phase 3 demo - enterprise platform features:
python examples/demo_phase3.py              # Unity Catalog, AutoML, Feature Store, Notebooks
```

**demo.py** runs five demonstrations:
1. **Core API** - RDD-style map/filter/reduce
2. **DataFrame API** - Spark-like queries
3. **Streaming** - Real-time processing with windowing
4. **Benchmarks** - Performance comparison
5. **Colony Behavior** - Visualization of bee dynamics

**demo_progressive.py** demonstrates production features:
1. **Error Handling** - Exception hierarchy and recovery
2. **Resilience Patterns** - Circuit breakers, retries, bulkheads
3. **Monitoring Integration** - Metrics, tracing, health checks
4. **Scale Testing** - Concurrent load and backpressure

**demo_sql.py** demonstrates SwarmQL 1.0:
1. **Table Registration** - Catalog management
2. **SELECT Queries** - Basic queries with WHERE, GROUP BY
3. **Aggregations** - COUNT, SUM, AVG, MIN, MAX
4. **Joins** - INNER, LEFT, RIGHT joins

**demo_storage.py** demonstrates storage layer:
1. **Parquet** - Read/write operations with compression
2. **Delta Lake** - ACID transactions and time travel
3. **Schema Evolution** - Handling schema changes

**swarmql_2_demo.py** demonstrates SwarmQL 2.0:
1. **CTEs** - Common Table Expressions (WITH clause)
2. **Set Operations** - UNION, INTERSECT, EXCEPT
3. **Subqueries** - IN, EXISTS, scalar subqueries
4. **Window Functions** - ROW_NUMBER, RANK, LAG, LEAD
5. **String Functions** - UPPER, LOWER, CONCAT, SUBSTRING
6. **Date Functions** - CURRENT_DATE, DATE_ADD, EXTRACT
7. **Bee-inspired Extensions** - WAGGLE JOIN demonstrations

**demo_phase2_federation.py** demonstrates multi-hive coordination:
1. **Federation Setup** - Registering multiple hives
2. **Locality-Aware Scheduling** - Data-local task placement
3. **Federated Execution** - Distributed query processing
4. **Health Monitoring** - Cross-hive health tracking
5. **Automatic Failover** - Handling hive failures

**demo_phase2_adaptive.py** demonstrates intelligent partitioning:
1. **Adaptive Partitioning** - Fitness-based partition sizing
2. **Partition Splitting** - Handling data skew
3. **Partition Merging** - Reducing overhead
4. **Speculative Execution** - Scout bee straggler mitigation
5. **Real-world Scenarios** - E-commerce analytics example

**demo_phase2_storage.py** demonstrates storage innovations:
1. **HoneyStore Basics** - Native columnar format
2. **Adaptive Compression** - Intelligent codec selection
3. **Query Pushdown** - Predicate and projection optimization
4. **Pheromone Caching** - Trail-based cache management
5. **Cache Eviction** - Fitness-based eviction policy
6. **Intelligent Prefetching** - Scout bee data prediction

**demo_phase3.py** demonstrates enterprise platform features:
1. **Unity Hive Catalog** - Data governance and access control
2. **Data Lineage** - Transformation tracking and dependency graphs
3. **PII Detection** - Automatic sensitive data classification
4. **AutoML Swarm** - Bee-inspired hyperparameter optimization
5. **Feature Hive** - Centralized feature store with versioning
6. **Model Serving** - Swarm-based inference load balancing
7. **HiveFrame Notebooks** - Multi-language interactive execution
8. **Delta Sharing** - Secure cross-organization data sharing

## Contributing

Contributions welcome! Areas of interest:
- GPU acceleration for fitness evaluation
- Additional swarm algorithms (PSO, ACO hybrid)
- Machine learning pipeline integration
- Cloud provider integrations (AWS EMR, Azure HDInsight, GCP Dataproc)
- Additional storage connectors (Iceberg, Hudi)
- Performance benchmarking suite

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for our ambitious vision to build the world's first bio-inspired unified data intelligence platform — with the goal of matching and surpassing platforms like Databricks through the power of swarm intelligence.

## License

MIT License - see LICENSE file.

## References

- Karaboga, D. (2005). An Idea Based On Honey Bee Swarm for Numerical Optimization
- Wedde, H.F. et al. (2004). BeeHive: An Efficient Fault-Tolerant Routing Algorithm
- Seeley, T.D. (2010). Honeybee Democracy

---

*"What can 50,000 bees teach us about distributed computing? Everything."*
