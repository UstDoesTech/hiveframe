---
sidebar_position: 1
---

# 💡 Explanation

Deep-dive articles to understand how HiveFrame works and why.

## What Are Explanations?

Explanations are **understanding-oriented** - they clarify and illuminate a topic. They broaden the reader's knowledge rather than teach a skill or provide steps.

## Available Explanations

### Architecture

| Article | Description |
|---------|-------------|
| [Architecture Overview](./architecture-overview) | High-level system design and components |
| [Waggle Dance Protocol](./waggle-dance-protocol) | How workers communicate task quality |
| [Three-Tier Workers](./three-tier-workers) | Employed, Onlooker, and Scout bees |
| [ABC Optimization](./abc-optimization) | Query optimization using bee colony algorithm |

### Core Concepts

| Article | Description |
|---------|-------------|
| [Pheromone Signaling](./pheromone-signaling) | Backpressure and rate limiting mechanism |
| [Colony Temperature](./colony-temperature) | System load regulation (homeostasis) |
| [Streaming Windows & Watermarks](./streaming-windows-watermarks) | Time-based event processing |

### Context

| Article | Description |
|---------|-------------|
| [Comparison with Spark](./comparison-spark) | How HiveFrame differs from Apache Spark |

## The Biomimicry Philosophy

HiveFrame is built on **biomimicry** - the practice of learning from and emulating nature's solutions. Bee colonies have evolved over millions of years to solve distributed coordination problems that are remarkably similar to modern computing challenges:

- **No central controller** - The queen doesn't direct workers
- **Emergent intelligence** - Complex behavior from simple rules
- **Self-healing** - Automatic recovery from failures
- **Adaptive resource allocation** - Based on task quality

By modeling these behaviors in software, HiveFrame achieves properties that are difficult to engineer directly.

## Reading Path

**New to HiveFrame?** Start with:
1. [Architecture Overview](./architecture-overview)
2. [Waggle Dance Protocol](./waggle-dance-protocol)
3. [Three-Tier Workers](./three-tier-workers)

**Optimizing performance?** Read:
1. [ABC Optimization](./abc-optimization)
2. [Colony Temperature](./colony-temperature)
3. [Pheromone Signaling](./pheromone-signaling)

**Building streaming apps?** See:
1. [Streaming Windows & Watermarks](./streaming-windows-watermarks)

**Coming from Spark?** Check:
1. [Comparison with Spark](./comparison-spark)
