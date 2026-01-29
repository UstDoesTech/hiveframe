# Changelog

All notable changes to HiveFrame will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Production-ready repository structure with `src/` layout
- Comprehensive test suite with pytest
- Documentation in `docs/` directory
- Example scripts in `examples/` directory

## [0.1.0] - 2026-01-29

### Added
- Initial release of HiveFrame
- Core bee-inspired processing engine
  - Waggle Dance Protocol for task quality signaling
  - Three-tier worker system (Employed, Onlooker, Scout)
  - Stigmergic coordination via pheromone signaling
  - ABC algorithm for self-healing
- DataFrame API (Spark-like interface)
  - `select`, `filter`, `groupBy`, `agg`, `join`
  - Column expressions and aggregation functions
  - Schema inference
- Streaming module
  - Basic `HiveStream` for real-time processing
  - Enhanced streaming with windowing (Tumbling, Sliding, Session)
  - Watermark support for late data
  - Delivery guarantees (at-most-once, at-least-once, exactly-once)
- Resilience patterns
  - Retry policies with configurable backoff
  - Circuit breaker pattern
  - Bulkhead isolation
  - Timeout handling
- Data connectors
  - CSV, JSON, JSONL sources and sinks
  - HTTP source with rate limiting
  - Message broker (Kafka-like interface)
  - File watcher for incremental processing
- Monitoring and observability
  - Prometheus-style metrics (Counter, Gauge, Histogram, Summary)
  - Structured logging
  - Distributed tracing
  - Performance profiling
- Exception hierarchy for error categorization
- Dead Letter Queue for failed record management
- Challenge test scenarios for error handling, scale, and data quality

### Notes
- Zero runtime dependencies for core functionality
- Optional dependencies for production integrations (kafka, postgres, http, monitoring)
- Python 3.9+ required
