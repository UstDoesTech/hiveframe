# Core Concepts

HiveFrame is built on bee colony intelligence patterns. This guide explains the key concepts.

## The Bee Colony Metaphor

### Waggle Dance Protocol

In real bee colonies, forager bees communicate the location and quality of food sources through a "waggle dance". In HiveFrame:

- Workers report task quality through dance signals
- Higher quality tasks attract more workers
- The dance floor is a shared data structure for coordination

```python
from hiveframe import WaggleDance, DanceFloor

dance = WaggleDance(
    quality=0.9,      # Task quality score
    source_id='task_1',
    worker_id='bee_1'
)

floor = DanceFloor()
floor.register_dance(dance)
```

### Three-Tier Worker System

HiveFrame uses three types of workers, inspired by the Artificial Bee Colony (ABC) algorithm:

| Role | Behavior | Purpose |
|------|----------|---------|
| **Employed** | Exploit known good sources | Process high-quality tasks |
| **Onlooker** | Reinforce based on dances | Help with popular tasks |
| **Scout** | Explore for new sources | Find new work opportunities |

```python
from hiveframe import BeeRole

# Roles are automatically assigned based on colony needs
# Employed bees: ~50%
# Onlooker bees: ~40%
# Scout bees: ~10%
```

### Pheromone Signaling

Bees communicate through chemical pheromones. In HiveFrame:

- **Throttle pheromone**: Signals backpressure when overloaded
- **Alarm pheromone**: Signals errors or failures
- Pheromones decay over time (stigmergic coordination)

```python
from hiveframe import ColonyState, Pheromone

colony = ColonyState()

# Emit backpressure signal
colony.emit_pheromone(Pheromone(
    signal_type='throttle',
    intensity=0.8,
    source_worker='worker_1'
))

# Sense current throttle level
level = colony.sense_pheromone('throttle')
```

## Self-Healing Mechanism

The ABC algorithm includes an "abandonment" mechanism:

1. If a food source (task) isn't improving, increment a counter
2. When counter exceeds threshold, abandon the source
3. Scout bees will find new sources to replace it

This provides automatic recovery from stuck or failed tasks.

## Colony Temperature

The colony tracks overall "temperature" - a measure of system load:

- Low temperature (< 0.5): Normal operation
- Medium temperature (0.5-0.8): Elevated load
- High temperature (> 0.8): Critical, throttling engaged

```python
colony = ColonyState()
temp = colony.get_colony_temperature()
```

## Food Sources

Tasks in HiveFrame are modeled as "food sources":

```python
from hiveframe import FoodSource

source = FoodSource(
    source_id='batch_1',
    data=[1, 2, 3, 4, 5],
    quality=0.7
)
```

Quality is updated based on processing success, following the waggle dance protocol.

## SwarmQL: SQL Query Execution

HiveFrame includes SwarmQL, a SQL engine that translates queries into distributed operations.

### How SwarmQL Works

1. **Parse**: SQL query is tokenized and parsed into an AST
2. **Plan**: Query plan is generated from the AST
3. **Optimize**: Bee-inspired optimizer improves the plan
4. **Execute**: Plan is executed across the colony

```python
from hiveframe.sql import SwarmQLContext

ctx = SwarmQLContext(num_workers=4)
ctx.register_table("users", df)

# Execute SQL query
result = ctx.sql("SELECT name, age FROM users WHERE age > 21")

# View the execution plan
plan = ctx.explain("SELECT * FROM users")
print(plan)
```

### Supported SQL Features

- **SELECT**: Column selection and expressions
- **FROM**: Single table and joins
- **WHERE**: Filter conditions with AND/OR
- **GROUP BY**: Aggregations with COUNT, SUM, AVG, MIN, MAX
- **ORDER BY**: Result sorting
- **LIMIT**: Result limiting

## Query Optimization with ABC Algorithm

HiveFrame's query optimizer uses the Artificial Bee Colony (ABC) algorithm instead of traditional cost-based optimization:

### How It Works

1. **Employed Bees**: Each explores a candidate query plan
2. **Onlooker Bees**: Evaluate plans based on fitness (cost estimate)
3. **Scout Bees**: Generate new plans when others are abandoned
4. **Selection**: Best plan is selected after iterations

```python
from hiveframe.optimizer import QueryOptimizer, SwarmCostModel

optimizer = QueryOptimizer(
    cost_model=SwarmCostModel(),
    max_iterations=100,
    swarm_size=20
)

optimized = optimizer.optimize(query_plan)
```

### Optimization Rules

| Rule | Description |
|------|-------------|
| **Predicate Pushdown** | Move filters closer to data source |
| **Projection Pruning** | Remove unnecessary columns early |
| **Constant Folding** | Evaluate constant expressions at compile time |
| **Filter Combination** | Merge adjacent filter operations |
| **Join Reordering** | Optimize join order for smaller intermediates |
| **Limit Pushdown** | Apply limits early when possible |

## Storage Layer

HiveFrame provides a unified storage layer supporting multiple formats.

### Parquet Support

Columnar storage format optimized for analytics:

```python
from hiveframe.storage import read_parquet, write_parquet

# Write DataFrame to Parquet
write_parquet(df, "/data/output.parquet")

# Read Parquet file
df = read_parquet("/data/output.parquet")
```

### Delta Lake Integration

ACID transactions and time travel for data lakes:

```python
from hiveframe.storage import DeltaTable

# Create or open Delta table
table = DeltaTable("/data/events")

# CRUD operations
table.write(df, mode='append')
table.update(condition, updates)
table.delete(condition)

# Time travel
old_data = table.version_as_of(5)
history = table.history()
```

### ACID Guarantees

Delta Lake provides:

- **Atomicity**: Transactions succeed completely or not at all
- **Consistency**: Data always in valid state
- **Isolation**: Concurrent transactions don't interfere
- **Durability**: Committed data is permanent

## Stream Processing

HiveFrame supports continuous stream processing with bee-inspired patterns.

### Windowing

Group streaming data into time-based windows:

| Window Type | Description | Use Case |
|-------------|-------------|----------|
| **Tumbling** | Fixed, non-overlapping intervals | Batch metrics every 5 minutes |
| **Sliding** | Fixed size, overlapping | Moving averages |
| **Session** | Activity-based, variable length | User sessions |

```python
from hiveframe.streaming import (
    tumbling_window, sliding_window, session_window,
    EnhancedStreamProcessor
)

# 5-second tumbling windows
processor = EnhancedStreamProcessor(
    window_assigner=tumbling_window(5.0)
)

# 10-second window sliding every 2 seconds
processor = EnhancedStreamProcessor(
    window_assigner=sliding_window(10.0, 2.0)
)
```

### Watermarks and Late Data

Handle out-of-order events with watermarks:

```python
from hiveframe.streaming import bounded_watermark

# Allow up to 5 seconds of out-of-order data
processor = EnhancedStreamProcessor(
    watermark_generator=bounded_watermark(5.0)
)
```

### Delivery Guarantees

| Guarantee | Description | Trade-off |
|-----------|-------------|-----------|
| **At-Most-Once** | Fire and forget | Fastest, may lose data |
| **At-Least-Once** | Retry until success | May duplicate |
| **Exactly-Once** | Idempotent processing | Highest overhead |

```python
from hiveframe.streaming import DeliveryGuarantee

processor = EnhancedStreamProcessor(
    delivery_guarantee=DeliveryGuarantee.EXACTLY_ONCE
)
```

## Kubernetes Deployment

HiveFrame clusters can be deployed to Kubernetes using the built-in operator.

### Cluster Definition

```python
from hiveframe.k8s import HiveCluster, WorkerSpec, ResourceRequirements

cluster = HiveCluster(
    name="production-hive",
    namespace="data-platform",
    workers=WorkerSpec(
        replicas=10,
        resources=ResourceRequirements(
            cpu="2",
            memory="4Gi"
        )
    )
)
```

### Deployment

```python
from hiveframe.k8s import HiveOperator

operator = HiveOperator()
operator.deploy(cluster)

# Scale the cluster
operator.scale("production-hive", replicas=20)

# Check status
status = operator.status("production-hive")
```

### Manifest Generation

Generate Kubernetes YAML manifests:

```python
from hiveframe.k8s import generate_deployment, generate_service

deployment = generate_deployment(cluster)
service = generate_service(cluster)
```

## Colony Dashboard

Monitor your HiveFrame cluster with the built-in web UI.

### Starting the Dashboard

```python
from hiveframe.dashboard import Dashboard

dashboard = Dashboard(port=8080)
dashboard.start()

print(f"Dashboard available at: {dashboard.url}")
```

### Dashboard Features

- **Colony Metrics**: Temperature, throughput, latency
- **Worker Status**: Role distribution, health, load
- **Dance Floor**: Waggle dance activity visualization
- **Query History**: Executed queries and performance
- **Pheromone Levels**: Throttle and alarm signals

## Resilience Patterns

HiveFrame includes production-grade resilience patterns.

### Retry with Backoff

```python
from hiveframe.resilience import RetryPolicy, with_retry, BackoffStrategy

policy = RetryPolicy(
    max_retries=3,
    base_delay=1.0,
    strategy=BackoffStrategy.EXPONENTIAL,
    jitter=True
)

@with_retry(policy)
def flaky_operation():
    # May fail transiently
    pass
```

### Circuit Breaker

Prevent cascade failures:

```python
from hiveframe.resilience import CircuitBreaker, CircuitBreakerConfig

breaker = CircuitBreaker(
    "external-api",
    CircuitBreakerConfig(
        failure_threshold=5,
        timeout=30.0
    )
)

result = breaker.call(lambda: api.request())
```

### Bulkhead

Isolate failures to prevent resource exhaustion:

```python
from hiveframe.resilience import Bulkhead

bulkhead = Bulkhead("db-pool", max_concurrent=10)

result = bulkhead.execute(lambda: db.query())
```

### Combined Resilience

Use `ResilientExecutor` to combine patterns:

```python
from hiveframe.resilience import ResilientExecutor

executor = ResilientExecutor(
    retry_policy=policy,
    circuit_breaker=breaker,
    bulkhead=bulkhead,
    timeout=10.0
)

result = executor.execute(risky_operation)
```
