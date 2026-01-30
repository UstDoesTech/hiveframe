# HiveFrame Documentation

Welcome to the HiveFrame documentation. HiveFrame is a bee-inspired distributed data processing framework - a biomimetic alternative to Apache Spark.

## Contents

- [Getting Started](getting-started.md) - Quick start guide
- [Core Concepts](core-concepts.md) - Understanding the bee-inspired architecture
- [API Reference](api-reference.md) - Complete API documentation

## Examples

Find complete working examples in the `examples/` directory:

| Example | Description |
|---------|-------------|
| [demo.py](../examples/demo.py) | Comprehensive feature demonstration |
| [demo_progressive.py](../examples/demo_progressive.py) | Progressive difficulty challenges |
| [demo_sql.py](../examples/demo_sql.py) | SwarmQL SQL engine examples |
| [demo_storage.py](../examples/demo_storage.py) | Parquet and Delta Lake storage |

## Key Features

| Feature | Description |
|---------|-------------|
| **DataFrame API** | Spark-like DataFrame operations with bee-inspired execution |
| **SwarmQL** | SQL query engine with ABC-algorithm optimization |
| **Stream Processing** | Windowed streaming with delivery guarantees |
| **Delta Lake** | ACID transactions and time travel |
| **Kubernetes** | Native K8s deployment support |
| **Dashboard** | Real-time colony monitoring UI |
| **Resilience** | Circuit breakers, retries, bulkheads |

## Quick Links

- [GitHub Repository](https://github.com/hiveframe/hiveframe)
- [PyPI Package](https://pypi.org/project/hiveframe/)
- [Issue Tracker](https://github.com/hiveframe/hiveframe/issues)
- [Roadmap](../ROADMAP.md) - Our vision and future plans

## Overview

HiveFrame uses bee colony intelligence patterns for distributed data processing:

- **Waggle Dance Protocol**: Workers advertise task quality through dance signals
- **Three-Tier Workers**: Employed (exploit), Onlooker (reinforce), Scout (explore)
- **Stigmergic Coordination**: Indirect communication through shared state
- **Self-Healing**: Automatic recovery through ABC abandonment mechanism
- **Swarm Optimization**: Query plans optimized using Artificial Bee Colony algorithm

## Installation

```bash
pip install hiveframe
```

With optional dependencies:

```bash
pip install hiveframe[kafka,monitoring,dashboard]
```
