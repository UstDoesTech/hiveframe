# 🐝 HiveFrame

**A bee-inspired distributed data processing framework — a biomimetic alternative to Apache Spark**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-Docusaurus-blue.svg)](https://ustdoestech.github.io/hiveframe/)

---

## 📚 Documentation

Full documentation is available at **[ustdoestech.github.io/hiveframe](https://ustdoestech.github.io/hiveframe/)**

Our documentation follows the [Diataxis](https://diataxis.fr/) framework:
- **[Tutorials](https://ustdoestech.github.io/hiveframe/docs/tutorials/getting-started)** - Learning-oriented guides
- **[How-to Guides](https://ustdoestech.github.io/hiveframe/docs/how-to-guides/overview)** - Problem-oriented recipes
- **[Reference](https://ustdoestech.github.io/hiveframe/docs/reference/api-overview)** - Technical API documentation
- **[Explanation](https://ustdoestech.github.io/hiveframe/docs/explanation/bee-colony-metaphor)** - Understanding-oriented discussions

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

## Installation

```bash
pip install hiveframe
```

Or from source:

```bash
git clone https://github.com/hiveframe/hiveframe.git
cd hiveframe
pip install -e .
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

## Running the Demo

```bash
python demo.py
```

This runs five demonstrations:
1. **Core API** - RDD-style map/filter/reduce
2. **DataFrame API** - Spark-like queries
3. **Streaming** - Real-time processing
4. **Benchmarks** - Performance comparison
5. **Colony Behavior** - Visualization of bee dynamics

## Contributing

Contributions welcome! Areas of interest:
- GPU acceleration for fitness evaluation
- Kubernetes-native deployment
- Additional swarm algorithms (PSO, ACO hybrid)
- Visualization dashboard for colony state

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
