"""
Self-Tuning Colony

Zero-configuration performance optimization through autonomous colony management.
Inspired by how bee colonies automatically adjust their behavior based on
environmental conditions and colony needs.
"""

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MemoryStats:
    """Memory usage statistics"""

    total_mb: float
    used_mb: float
    available_mb: float
    cache_mb: float
    buffer_mb: float
    timestamp: float = field(default_factory=time.time)

    @property
    def usage_ratio(self) -> float:
        """Memory usage as a ratio (0.0 to 1.0)"""
        return self.used_mb / self.total_mb if self.total_mb > 0 else 0.0


@dataclass
class ResourceMetrics:
    """Resource allocation metrics"""

    cpu_percent: float
    memory_mb: float
    disk_io_mb_per_sec: float
    network_mb_per_sec: float
    active_workers: int
    queued_tasks: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class QueryPerformance:
    """Query performance metrics"""

    query_id: str
    execution_time_ms: float
    rows_processed: int
    bytes_scanned: int
    workers_used: int
    memory_peak_mb: float
    timestamp: float = field(default_factory=time.time)

    @property
    def throughput_rows_per_sec(self) -> float:
        """Rows processed per second"""
        return (
            self.rows_processed / (self.execution_time_ms / 1000)
            if self.execution_time_ms > 0
            else 0.0
        )


class MemoryManager:
    """
    Automatic memory management using swarm intelligence.

    Monitors memory usage patterns and automatically adjusts cache sizes,
    buffer allocations, and memory limits to optimize performance while
    preventing OOM conditions.
    """

    def __init__(self, total_memory_mb: float = 8192, max_cache_ratio: float = 0.3):
        self.total_memory_mb = total_memory_mb
        self.max_cache_ratio = max_cache_ratio
        self.history: deque = deque(maxlen=100)
        self.cache_size_mb = total_memory_mb * max_cache_ratio
        self.buffer_size_mb = total_memory_mb * 0.1

    def record_usage(self, stats: MemoryStats) -> None:
        """Record memory usage statistics"""
        self.history.append(stats)

    def optimize(self) -> Dict[str, float]:
        """
        Optimize memory allocation based on usage patterns.

        Returns:
            Dictionary with optimized memory settings
        """
        if len(self.history) < 10:
            return {
                "cache_mb": self.cache_size_mb,
                "buffer_mb": self.buffer_size_mb,
                "max_allocation_mb": self.total_memory_mb * 0.8,
            }

        # Analyze recent usage patterns
        recent_usage = [s.usage_ratio for s in list(self.history)[-20:]]
        avg_usage = statistics.mean(recent_usage)
        peak_usage = max(recent_usage)

        # Adjust cache size based on usage patterns (bee-inspired adaptive behavior)
        if avg_usage > 0.8:
            # High memory pressure - reduce cache (bees prioritize essential activities)
            self.cache_size_mb *= 0.9
        elif avg_usage < 0.5 and peak_usage < 0.7:
            # Low memory pressure - increase cache (bees store more honey when abundant)
            self.cache_size_mb = min(
                self.cache_size_mb * 1.1, self.total_memory_mb * self.max_cache_ratio
            )

        # Dynamic buffer sizing based on variability
        usage_stddev = statistics.stdev(recent_usage) if len(recent_usage) > 1 else 0
        if usage_stddev > 0.2:
            # High variability - increase buffer (prepare for spikes)
            self.buffer_size_mb = self.total_memory_mb * 0.15
        else:
            # Low variability - reduce buffer (free up memory for cache)
            self.buffer_size_mb = self.total_memory_mb * 0.05

        return {
            "cache_mb": self.cache_size_mb,
            "buffer_mb": self.buffer_size_mb,
            "max_allocation_mb": self.total_memory_mb * 0.8,
            "avg_usage": avg_usage,
            "peak_usage": peak_usage,
        }


class ResourceAllocator:
    """
    Dynamic resource allocation using bee colony decision-making.

    Automatically adjusts worker counts, parallelism levels, and resource
    limits based on workload characteristics and system health.
    """

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 100,
        target_cpu_percent: float = 75.0,
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu_percent = target_cpu_percent
        self.current_workers = min_workers
        self.metrics_history: deque = deque(maxlen=50)

    def record_metrics(self, metrics: ResourceMetrics) -> None:
        """Record resource usage metrics"""
        self.metrics_history.append(metrics)

    def allocate(self) -> Dict[str, Any]:
        """
        Determine optimal resource allocation.

        Uses bee-inspired decision making: scout bees explore resource levels,
        employed bees process tasks, and the colony adapts based on waggle dance signals.

        Returns:
            Dictionary with allocation decisions
        """
        if len(self.metrics_history) < 5:
            return {
                "workers": self.current_workers,
                "parallelism": self.current_workers,
                "reason": "insufficient_data",
            }

        recent = list(self.metrics_history)[-10:]
        avg_cpu = statistics.mean([m.cpu_percent for m in recent])
        avg_queued = statistics.mean([m.queued_tasks for m in recent])

        # Bee-inspired scaling logic
        decision = "maintain"

        if avg_queued > self.current_workers * 2 and avg_cpu < self.target_cpu_percent:
            # Tasks accumulating but CPU available - add workers (recruit more foragers)
            new_workers = min(int(self.current_workers * 1.5), self.max_workers)
            if new_workers > self.current_workers:
                self.current_workers = new_workers
                decision = "scale_up"

        elif avg_cpu > self.target_cpu_percent * 1.2:
            # CPU overloaded - reduce workers to prevent thrashing (reduce colony activity)
            new_workers = max(int(self.current_workers * 0.8), self.min_workers)
            if new_workers < self.current_workers:
                self.current_workers = new_workers
                decision = "scale_down"

        elif avg_queued < 5 and avg_cpu < self.target_cpu_percent * 0.5:
            # Underutilized - reduce workers to save resources (bees rest when no work)
            new_workers = max(int(self.current_workers * 0.9), self.min_workers)
            if new_workers < self.current_workers:
                self.current_workers = new_workers
                decision = "optimize"

        return {
            "workers": self.current_workers,
            "parallelism": max(self.current_workers, 1),
            "decision": decision,
            "avg_cpu": avg_cpu,
            "avg_queued": avg_queued,
        }


class QueryPredictor:
    """
    Query performance prediction using swarm intelligence.

    Learns from historical query patterns to predict execution time,
    resource requirements, and optimal execution strategies.
    """

    def __init__(self):
        self.query_history: List[QueryPerformance] = []
        self.patterns: Dict[str, List[QueryPerformance]] = {}

    def record_query(self, perf: QueryPerformance) -> None:
        """Record query performance"""
        self.query_history.append(perf)

        # Extract pattern signature (simplified - could use query plan fingerprinting)
        pattern_key = f"rows_{perf.rows_processed//1000}k_scan_{perf.bytes_scanned//1000000}mb"

        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = []
        self.patterns[pattern_key].append(perf)

        # Keep only recent patterns (bees forget old dance information)
        if len(self.patterns[pattern_key]) > 20:
            self.patterns[pattern_key] = self.patterns[pattern_key][-20:]

    def predict(
        self,
        query_signature: str,
        rows_estimate: int,
        bytes_estimate: int,
    ) -> Dict[str, float]:
        """
        Predict query performance based on historical patterns.

        Returns:
            Dictionary with predictions
        """
        pattern_key = f"rows_{rows_estimate//1000}k_scan_{bytes_estimate//1000000}mb"

        if pattern_key not in self.patterns or len(self.patterns[pattern_key]) < 3:
            # No pattern data - use conservative estimates
            return {
                "estimated_time_ms": rows_estimate * 0.01,  # 0.01ms per row
                "estimated_memory_mb": bytes_estimate / (1024 * 1024) * 1.5,  # 1.5x data size
                "recommended_workers": max(1, rows_estimate // 100000),
                "confidence": 0.3,
            }

        # Use swarm wisdom (aggregate from similar historical queries)
        similar = self.patterns[pattern_key]
        avg_time = statistics.mean([q.execution_time_ms for q in similar])
        avg_memory = statistics.mean([q.memory_peak_mb for q in similar])
        avg_workers = statistics.mean([q.workers_used for q in similar])

        # Calculate confidence based on sample size and consistency
        time_stddev = (
            statistics.stdev([q.execution_time_ms for q in similar]) if len(similar) > 1 else 0
        )
        confidence = min(0.95, len(similar) / 20) * (1 - min(time_stddev / avg_time, 0.5))

        return {
            "estimated_time_ms": avg_time,
            "estimated_memory_mb": avg_memory,
            "recommended_workers": int(avg_workers),
            "confidence": confidence,
        }


class SelfTuningColony:
    """
    Self-tuning colony orchestrator.

    Coordinates all autonomous tuning components to maintain optimal
    performance without manual intervention, inspired by how bee colonies
    self-organize and adapt to changing conditions.
    """

    def __init__(
        self,
        total_memory_mb: float = 8192,
        max_workers: int = 100,
    ):
        self.memory_manager = MemoryManager(total_memory_mb)
        self.resource_allocator = ResourceAllocator(max_workers=max_workers)
        self.query_predictor = QueryPredictor()
        self.last_tune_time = time.time()
        self.tune_interval_sec = 30

    def tune(self) -> Dict[str, Any]:
        """
        Perform self-tuning optimization.

        Returns:
            Dictionary with tuning decisions and metrics
        """
        current_time = time.time()

        # Check if tuning interval has elapsed (like bee inspection rounds)
        if current_time - self.last_tune_time < self.tune_interval_sec:
            return {"status": "skipped", "reason": "too_soon"}

        self.last_tune_time = current_time

        # Get optimization decisions from each component
        memory_config = self.memory_manager.optimize()
        resource_config = self.resource_allocator.allocate()

        return {
            "status": "tuned",
            "timestamp": current_time,
            "memory": memory_config,
            "resources": resource_config,
            "interval_sec": self.tune_interval_sec,
        }

    def get_query_recommendation(
        self,
        query_signature: str,
        rows_estimate: int,
        bytes_estimate: int,
    ) -> Dict[str, Any]:
        """Get performance prediction and execution recommendations for a query"""
        return self.query_predictor.predict(query_signature, rows_estimate, bytes_estimate)
