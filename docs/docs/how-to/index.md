---
sidebar_position: 1
---

# 📋 How-To Guides

Task-oriented recipes for accomplishing specific goals with HiveFrame.

## What Are How-To Guides?

How-To Guides are **problem-oriented** - they take you through the steps needed to solve a real-world problem. Unlike tutorials, they assume you already have some knowledge and need to accomplish a specific task.

## Categories

### 📦 Data Storage

| Guide | Description |
|-------|-------------|
| [Read/Write Parquet](./read-write-parquet) | Work with Parquet files efficiently |
| [Use Delta Lake](./use-delta-lake) | ACID transactions and schema evolution |
| [Delta Time Travel](./delta-time-travel) | Query historical versions of data |

### 🛡️ Resilience

| Guide | Description |
|-------|-------------|
| [Configure Retry](./configure-retry) | Automatic retry with backoff strategies |
| [Use Circuit Breaker](./use-circuit-breaker) | Prevent cascade failures |
| [Handle Errors with DLQ](./handle-errors-dlq) | Dead Letter Queue for failed records |

### 📊 Monitoring

| Guide | Description |
|-------|-------------|
| [Setup Monitoring](./setup-monitoring) | Prometheus metrics and alerting |
| [Configure Logging](./configure-logging) | Structured logging setup |
| [Enable Tracing](./enable-tracing) | Distributed tracing with OpenTelemetry |

### 🌊 Streaming

| Guide | Description |
|-------|-------------|
| [Configure Windows](./configure-windows) | Tumbling, sliding, and session windows |
| [Manage Watermarks](./manage-watermarks) | Handle late-arriving data |
| [Delivery Guarantees](./delivery-guarantees) | At-most-once to exactly-once |

### 🔌 Connectors

| Guide | Description |
|-------|-------------|
| [Connect to Kafka](./connect-kafka) | Read and write Kafka topics |
| [Connect to PostgreSQL](./connect-postgres) | Database source and sink |
| [Connect via HTTP](./connect-http) | REST API integration |

### 🐝 Phase 2: Swarm Intelligence (New!)

| Guide | Description | Status |
|-------|-------------|--------|
| Configure Multi-Hive Federation | Coordinate multiple clusters across datacenters | 📝 Coming Soon |
| Use Adaptive Partitioning | Dynamic partition management based on fitness | 📝 Coming Soon |
| Enable Speculative Execution | Scout bees for straggler mitigation | 📝 Coming Soon |
| Work with HoneyStore | Native columnar format optimized for swarms | 📝 Coming Soon |
| Use Caching Swarm | Pheromone-based intelligent caching | 📝 Coming Soon |
| Enable Vectorized Execution | SIMD-accelerated numerical processing | 📝 Coming Soon |
| Use Adaptive Query Execution | Real-time query plan optimization | 📝 Coming Soon |
| Integrate Iceberg Tables | Open table format with schema evolution | 📝 Coming Soon |

> **Note**: Phase 2 features are fully implemented and available! The how-to guides are being written.
> For now, see the comprehensive examples in `examples/demo_phase2_*.py` files.

## Finding the Right Guide

- **New to HiveFrame?** Start with [Tutorials](/docs/tutorials) instead
- **Need to understand concepts?** Check [Explanation](/docs/explanation)
- **Looking for API details?** See [Reference](/docs/reference)

## Contributing

Missing a how-to guide? [Open an issue](https://github.com/hiveframe/hiveframe/issues) or submit a PR!
