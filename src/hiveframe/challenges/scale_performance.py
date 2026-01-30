"""
Challenge Scenario: Scale & Performance
=======================================
Tests HiveFrame under increasing load and scale.

Scenarios:
1. Throughput scaling (linear increase)
2. Partition skew handling
3. Memory pressure under large datasets
4. Worker saturation and backpressure
5. Sustained high-load operation
"""

import gc
import random
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..core import ColonyState, HiveFrame
from ..monitoring import get_logger, get_profiler, get_registry

logger = get_logger("challenge.scale")
metrics = get_registry()
profiler = get_profiler()


@dataclass
class ScaleScenarioConfig:
    """Configuration for scale testing."""

    initial_records: int = 1000
    max_records: int = 100000
    step_multiplier: float = 2.0
    num_workers: int = 8
    timeout_seconds: float = 300.0
    memory_limit_mb: int = 500


@dataclass
class ScaleResult:
    """Results from a scale scenario."""

    scenario_name: str
    record_count: int
    elapsed_seconds: float
    throughput: float
    peak_memory_mb: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    worker_utilization: float
    backpressure_events: int
    success_rate: float

    def summary(self) -> str:
        return f"""
=== {self.scenario_name} ===
Records:           {self.record_count:,}
Elapsed Time:      {self.elapsed_seconds:.2f}s
Throughput:        {self.throughput:,.0f} rec/s
Peak Memory:       {self.peak_memory_mb:.1f} MB
Avg Latency:       {self.avg_latency_ms:.2f}ms
P95 Latency:       {self.p95_latency_ms:.2f}ms
P99 Latency:       {self.p99_latency_ms:.2f}ms
Worker Util:       {100*self.worker_utilization:.1f}%
Backpressure:      {self.backpressure_events}
Success Rate:      {100*self.success_rate:.2f}%
"""


class MemoryTracker:
    """Track memory usage during tests."""

    def __init__(self):
        self._peak_mb = 0
        self._samples = []
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._peak_mb = 0
        self._samples = []
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self._peak_mb

    def _monitor(self):
        import sys

        while self._running:
            try:
                # Estimate memory usage
                gc.collect()
                # This is a rough estimate - in production you'd use psutil
                current = sum(sys.getsizeof(obj) for obj in gc.get_objects()) / (1024 * 1024)
                self._peak_mb = max(self._peak_mb, current)
                self._samples.append(current)
            except Exception:
                pass
            time.sleep(0.1)


class LatencyTracker:
    """Track operation latencies."""

    def __init__(self):
        self._latencies = []
        self._lock = threading.Lock()

    def record(self, latency_seconds: float):
        with self._lock:
            self._latencies.append(latency_seconds * 1000)  # Convert to ms

    def get_stats(self) -> Dict[str, float]:
        with self._lock:
            if not self._latencies:
                return {"avg": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}

            sorted_lat = sorted(self._latencies)
            n = len(sorted_lat)

            return {
                "avg": statistics.mean(sorted_lat),
                "p50": sorted_lat[int(n * 0.50)],
                "p95": sorted_lat[int(n * 0.95)] if n > 20 else sorted_lat[-1],
                "p99": sorted_lat[int(n * 0.99)] if n > 100 else sorted_lat[-1],
                "max": sorted_lat[-1],
            }


def run_throughput_scaling_scenario(
    initial_records: int = 1000,
    max_records: int = 50000,
    step_multiplier: float = 2.0,
    num_workers: int = 8,
) -> List[ScaleResult]:
    """
    Scenario 1: Throughput Scaling

    Tests how throughput scales with increasing data volume.

    Expected behavior:
    - Near-linear throughput scaling up to worker saturation
    - Graceful degradation beyond saturation point
    - Consistent latency until saturation
    """
    logger.info("Starting throughput scaling scenario", initial=initial_records, max=max_records)

    results = []
    current_size = initial_records

    while current_size <= max_records:
        logger.info(f"Testing with {current_size:,} records...")

        # Generate test data
        data = [{"id": i, "value": random.random() * 1000} for i in range(current_size)]

        latency_tracker = LatencyTracker()
        memory_tracker = MemoryTracker()
        memory_tracker.start()

        successful = 0
        backpressure_events = [0]

        def process_item(item: Dict) -> Dict:
            start = time.perf_counter()

            # Simulate varying work complexity
            work_time = 0.0001 + random.random() * 0.0005
            time.sleep(work_time)

            result = {"id": item["id"], "result": item["value"] ** 2, "processed_at": time.time()}

            latency_tracker.record(time.perf_counter() - start)
            return result

        hive = HiveFrame(num_workers=num_workers)

        start_time = time.time()

        try:
            results_data = hive.process(data, lambda item: (process_item(item), 1.0))
            successful = len([r for r in results_data.values() if r is not None])

        except Exception as e:
            logger.error(f"Processing failed: {e}")

        elapsed = time.time() - start_time
        peak_memory = memory_tracker.stop()
        lat_stats = latency_tracker.get_stats()

        result = ScaleResult(
            scenario_name=f"Throughput Scaling ({current_size:,} records)",
            record_count=current_size,
            elapsed_seconds=elapsed,
            throughput=current_size / elapsed if elapsed > 0 else 0,
            peak_memory_mb=peak_memory,
            avg_latency_ms=lat_stats["avg"],
            p95_latency_ms=lat_stats["p95"],
            p99_latency_ms=lat_stats["p99"],
            worker_utilization=min(1.0, (current_size / elapsed) / (num_workers * 1000)),
            backpressure_events=backpressure_events[0],
            success_rate=successful / current_size if current_size > 0 else 0,
        )

        results.append(result)
        print(result.summary())

        current_size = int(current_size * step_multiplier)

        # Force garbage collection between tests
        gc.collect()
        time.sleep(0.5)

    return results


def run_partition_skew_scenario(
    num_records: int = 10000,
    num_partitions: int = 8,
    skew_factor: float = 0.8,  # 80% of data in 20% of partitions
) -> ScaleResult:
    """
    Scenario 2: Partition Skew Handling

    Tests system behavior when data is unevenly distributed
    across partitions (hot spots).

    Expected behavior:
    - Bee colony should naturally rebalance (onlookers help busy partitions)
    - Overall throughput should remain acceptable
    - No single partition should become a bottleneck
    """
    logger.info(
        "Starting partition skew scenario", num_records=num_records, skew_factor=skew_factor
    )

    # Create skewed data distribution
    data = []
    hot_partitions = max(1, int(num_partitions * 0.2))  # 20% are hot

    for i in range(num_records):
        # 80% chance to go to hot partitions
        if random.random() < skew_factor:
            partition = random.randint(0, hot_partitions - 1)
        else:
            partition = random.randint(hot_partitions, num_partitions - 1)

        data.append({"id": i, "partition": partition, "value": random.random() * 100})

    # Track per-partition processing
    partition_counts = {i: 0 for i in range(num_partitions)}
    partition_times = {i: [] for i in range(num_partitions)}
    latency_tracker = LatencyTracker()

    def process_with_partition_tracking(item: Dict) -> Tuple[Dict, float]:
        start = time.perf_counter()
        partition = item["partition"]

        # Hot partitions have more complex processing
        base_time = 0.001
        if partition < int(num_partitions * 0.2):
            base_time *= 2  # Hot partitions take longer

        time.sleep(base_time + random.random() * 0.0005)

        result = {"id": item["id"], "result": item["value"] ** 2}

        elapsed = time.perf_counter() - start
        latency_tracker.record(elapsed)
        partition_times[partition].append(elapsed)
        partition_counts[partition] += 1

        return result, 1.0

    hive = HiveFrame(
        num_workers=num_partitions,
        employed_ratio=0.4,
        onlooker_ratio=0.5,  # More onlookers to handle skew
        scout_ratio=0.1,
    )

    memory_tracker = MemoryTracker()
    memory_tracker.start()
    start_time = time.time()

    results_data = hive.process(data, lambda item: process_with_partition_tracking(item))

    elapsed = time.time() - start_time
    peak_memory = memory_tracker.stop()
    lat_stats = latency_tracker.get_stats()

    # Calculate partition imbalance
    counts = list(partition_counts.values())
    imbalance = max(counts) / (sum(counts) / len(counts)) if counts else 1.0

    logger.info("Partition distribution", counts=partition_counts, imbalance_ratio=imbalance)

    successful = len([r for r in results_data.values() if r is not None])

    return ScaleResult(
        scenario_name=f"Partition Skew ({skew_factor*100:.0f}% skew)",
        record_count=num_records,
        elapsed_seconds=elapsed,
        throughput=num_records / elapsed,
        peak_memory_mb=peak_memory,
        avg_latency_ms=lat_stats["avg"],
        p95_latency_ms=lat_stats["p95"],
        p99_latency_ms=lat_stats["p99"],
        worker_utilization=0.0,  # Would need to track actual worker time
        backpressure_events=0,
        success_rate=successful / num_records,
    )


def run_sustained_load_scenario(
    records_per_second: int = 500, duration_seconds: int = 30, num_workers: int = 8
) -> ScaleResult:
    """
    Scenario 3: Sustained High Load

    Tests system stability under continuous high-throughput operation.

    Expected behavior:
    - Stable throughput over time
    - No memory leaks
    - Consistent latency profile
    """
    logger.info(
        "Starting sustained load scenario", rps=records_per_second, duration=duration_seconds
    )

    latency_tracker = LatencyTracker()
    memory_tracker = MemoryTracker()

    throughput_samples = []
    successful = [0]
    failed = [0]
    backpressure = [0]

    def process_record(record: Dict) -> Dict:
        start = time.perf_counter()

        # Simulate realistic processing
        time.sleep(0.0005 + random.random() * 0.001)

        result = {"id": record["id"], "timestamp": time.time(), "result": record["value"] * 2}

        latency_tracker.record(time.perf_counter() - start)
        return result

    colony = ColonyState()

    memory_tracker.start()
    start_time = time.time()
    last_sample_time = start_time
    records_since_sample = 0

    record_id = 0

    while time.time() - start_time < duration_seconds:
        # Generate batch of records
        batch_size = min(100, records_per_second // 10)
        batch = [{"id": record_id + i, "value": random.random() * 1000} for i in range(batch_size)]
        record_id += batch_size

        # Process batch
        for record in batch:
            try:
                process_record(record)
                successful[0] += 1
                records_since_sample += 1
            except Exception:
                failed[0] += 1

        # Check for backpressure (simulated)
        if colony.get_colony_temperature() > 0.8:
            backpressure[0] += 1
            time.sleep(0.01)

        # Sample throughput every second
        if time.time() - last_sample_time >= 1.0:
            throughput_samples.append(records_since_sample)
            records_since_sample = 0
            last_sample_time = time.time()

        # Rate limiting to achieve target RPS
        expected_records = (time.time() - start_time) * records_per_second
        actual_records = successful[0] + failed[0]

        if actual_records > expected_records:
            sleep_time = (actual_records - expected_records) / records_per_second
            time.sleep(min(0.1, sleep_time))

    elapsed = time.time() - start_time
    peak_memory = memory_tracker.stop()
    lat_stats = latency_tracker.get_stats()

    # Analyze throughput stability
    if throughput_samples:
        avg_throughput = statistics.mean(throughput_samples)
        throughput_stddev = (
            statistics.stdev(throughput_samples) if len(throughput_samples) > 1 else 0
        )
        logger.info(
            "Throughput stability",
            avg=avg_throughput,
            stddev=throughput_stddev,
            samples=len(throughput_samples),
        )

    return ScaleResult(
        scenario_name=f"Sustained Load ({records_per_second} RPS × {duration_seconds}s)",
        record_count=successful[0] + failed[0],
        elapsed_seconds=elapsed,
        throughput=(successful[0] + failed[0]) / elapsed,
        peak_memory_mb=peak_memory,
        avg_latency_ms=lat_stats["avg"],
        p95_latency_ms=lat_stats["p95"],
        p99_latency_ms=lat_stats["p99"],
        worker_utilization=0.0,
        backpressure_events=backpressure[0],
        success_rate=successful[0] / max(1, successful[0] + failed[0]),
    )


def run_worker_saturation_scenario(
    num_records: int = 5000, processing_time_ms: float = 10.0, worker_counts: List[int] = None
) -> List[ScaleResult]:
    """
    Scenario 4: Worker Saturation

    Tests how the system behaves as workers become saturated.

    Expected behavior:
    - Linear scaling until saturation
    - Graceful throughput plateau at saturation
    - Backpressure mechanisms should activate
    """
    if worker_counts is None:
        worker_counts = [1, 2, 4, 8, 16, 32]

    logger.info(
        "Starting worker saturation scenario", num_records=num_records, worker_counts=worker_counts
    )

    results = []
    data = [{"id": i, "value": random.random()} for i in range(num_records)]

    for num_workers in worker_counts:
        logger.info(f"Testing with {num_workers} workers...")

        latency_tracker = LatencyTracker()

        def process_with_fixed_time(item: Dict) -> Tuple[Dict, float]:
            start = time.perf_counter()

            # Fixed processing time to clearly show saturation
            time.sleep(processing_time_ms / 1000)

            result = {"id": item["id"], "result": item["value"] * 2}
            latency_tracker.record(time.perf_counter() - start)

            return result, 1.0

        hive = HiveFrame(num_workers=num_workers)

        start_time = time.time()
        results_data = hive.process(data, process_with_fixed_time)
        elapsed = time.time() - start_time

        lat_stats = latency_tracker.get_stats()
        successful = len([r for r in results_data.values() if r is not None])

        # Calculate theoretical max throughput
        theoretical_max = num_workers * (1000 / processing_time_ms)
        actual_throughput = num_records / elapsed
        utilization = actual_throughput / theoretical_max if theoretical_max > 0 else 0

        result = ScaleResult(
            scenario_name=f"Worker Saturation ({num_workers} workers)",
            record_count=num_records,
            elapsed_seconds=elapsed,
            throughput=actual_throughput,
            peak_memory_mb=0,
            avg_latency_ms=lat_stats["avg"],
            p95_latency_ms=lat_stats["p95"],
            p99_latency_ms=lat_stats["p99"],
            worker_utilization=min(1.0, utilization),
            backpressure_events=0,
            success_rate=successful / num_records,
        )

        results.append(result)
        print(result.summary())

        gc.collect()
        time.sleep(0.5)

    return results


def run_all_scale_scenarios() -> Dict[str, List[ScaleResult]]:
    """Run all scale testing scenarios."""
    results = {}

    print("\n" + "=" * 60)
    print("HiveFrame Scale & Performance Challenge Suite")
    print("=" * 60)

    # Scenario 1: Throughput scaling
    print("\n--- Throughput Scaling ---")
    results["throughput"] = run_throughput_scaling_scenario(
        initial_records=500, max_records=8000, step_multiplier=2.0
    )

    # Scenario 2: Partition skew
    print("\n--- Partition Skew ---")
    results["skew"] = [run_partition_skew_scenario(num_records=5000, skew_factor=0.8)]

    # Scenario 3: Sustained load
    print("\n--- Sustained Load ---")
    results["sustained"] = [
        run_sustained_load_scenario(records_per_second=300, duration_seconds=10)
    ]

    # Scenario 4: Worker saturation
    print("\n--- Worker Saturation ---")
    results["saturation"] = run_worker_saturation_scenario(
        num_records=1000, processing_time_ms=5.0, worker_counts=[2, 4, 8, 16]
    )

    # Summary
    print("\n" + "=" * 60)
    print("SCALE TEST SUMMARY")
    print("=" * 60)

    for category, category_results in results.items():
        print(f"\n{category.upper()}:")
        for r in category_results:
            print(
                f"  {r.record_count:,} records: {r.throughput:,.0f} rec/s, "
                f"p95={r.p95_latency_ms:.1f}ms"
            )

    return results


if __name__ == "__main__":
    run_all_scale_scenarios()
