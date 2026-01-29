---
sidebar_position: 1
---

# Welcome to HiveFrame

**A bee-inspired distributed data processing framework — a biomimetic alternative to Apache Spark**

HiveFrame replaces Spark's centralized driver model with **decentralized bee colony coordination**. Instead of a single driver scheduling tasks across executors, HiveFrame uses autonomous "bee" workers that self-organize through swarm intelligence patterns.

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
git clone https://github.com/UstDoesTech/hiveframe.git
cd hiveframe
pip install -e .
```

## Quick Start

```python
from hiveframe import create_hive

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

## Documentation Structure

This documentation follows the [Diataxis](https://diataxis.fr/) framework:

- **[Tutorials](./tutorials/getting-started.md)** - Learning-oriented guides to get you started
- **[How-to Guides](./how-to-guides/overview.md)** - Problem-oriented recipes for common tasks
- **[Reference](./reference/api-overview.md)** - Information-oriented technical descriptions
- **[Explanation](./explanation/bee-colony-metaphor.md)** - Understanding-oriented discussions of key topics

## Next Steps

- Follow the [Getting Started Tutorial](./tutorials/getting-started.md) to build your first application
- Learn about [Core Concepts](./explanation/bee-colony-metaphor.md) to understand the architecture
- Check the [API Reference](./reference/api-overview.md) for detailed documentation
- See [How-to Guides](./how-to-guides/overview.md) for practical examples
