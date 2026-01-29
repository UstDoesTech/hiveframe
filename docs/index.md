# HiveFrame Documentation

Welcome to the HiveFrame documentation. HiveFrame is a bee-inspired distributed data processing framework - a biomimetic alternative to Apache Spark.

## Contents

- [Getting Started](getting-started.md) - Quick start guide
- [Core Concepts](core-concepts.md) - Understanding the bee-inspired architecture
- [API Reference](api-reference.md) - Complete API documentation
- [Examples](examples.md) - Code examples and tutorials
- [Production Guide](production.md) - Deployment and monitoring
- [Roadmap](../ROADMAP.md) - Our vision and future plans

## Quick Links

- [GitHub Repository](https://github.com/hiveframe/hiveframe)
- [PyPI Package](https://pypi.org/project/hiveframe/)
- [Issue Tracker](https://github.com/hiveframe/hiveframe/issues)

## Overview

HiveFrame uses bee colony intelligence patterns for distributed data processing:

- **Waggle Dance Protocol**: Workers advertise task quality through dance signals
- **Three-Tier Workers**: Employed (exploit), Onlooker (reinforce), Scout (explore)
- **Stigmergic Coordination**: Indirect communication through shared state
- **Self-Healing**: Automatic recovery through ABC abandonment mechanism

## Installation

```bash
pip install hiveframe
```

With optional dependencies:

```bash
pip install hiveframe[kafka,monitoring]
```
