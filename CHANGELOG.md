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

### Phase 2 Features (Complete)

#### Distributed Execution Engine
- Multi-hive Federation for cluster coordination
- Cross-datacenter swarm coordination
- Adaptive partitioning with fitness-based strategies
- Speculative execution with scout bees
- Locality-aware scheduling

#### Advanced Query Engine
- SwarmQL 2.0 with full ANSI SQL compliance
- Common Table Expressions (CTEs)
- Set operations (UNION, INTERSECT, EXCEPT)
- Subqueries (IN, EXISTS, scalar)
- Window functions (ROW_NUMBER, RANK, LAG, LEAD)
- 20+ new SQL functions (string, date/time, etc.)
- Bee-inspired extensions (WAGGLE JOIN, etc.)
- Adaptive Query Execution (AQE)
- Real-time plan modification based on waggle dance feedback
- Vectorized execution for numerical workloads

#### Storage Layer
- HoneyStore native columnar format
- Adaptive compression
- Honeycomb block structure
- Nectar encoding for nulls
- Iceberg table format support
- Schema evolution
- Hidden partitioning
- Caching Swarm with pheromone-based eviction
- Intelligent prefetching by scout bees
- Distributed cache coordination

#### Documentation & Examples
- Comprehensive Phase 2 feature documentation in README
- Updated ROADMAP to reflect Phase 2 completion
- New demo files:
  - `demo_phase2_federation.py` - Multi-hive coordination
  - `demo_phase2_adaptive.py` - Adaptive partitioning & speculative execution
  - `demo_phase2_storage.py` - HoneyStore & caching swarm
  - `swarmql_2_demo.py` - SwarmQL 2.0 features
- API examples for all Phase 2 features

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
