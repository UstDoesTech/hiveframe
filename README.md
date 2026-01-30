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
```

**demo.py** runs five demonstrations:
1. **Core API** - RDD-style map/filter/reduce
2. **DataFrame API** - Spark-like queries
3. **Streaming** - Real-time processing
4. **Benchmarks** - Performance comparison
5. **Colony Behavior** - Visualization of bee dynamics

**demo_progressive.py** demonstrates production features:
1. **Error Handling** - Exception hierarchy and recovery
2. **Resilience Patterns** - Circuit breakers, retries, bulkheads
3. **Monitoring Integration** - Metrics, tracing, health checks
4. **Scale Testing** - Concurrent load and backpressure

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
