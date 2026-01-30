"""
HiveFrame Streaming Core
========================
Basic stream processing using bee-inspired patterns.

Key concepts:
- Continuous foraging: Workers continuously process incoming data
- Adaptive throttling: Pheromone-based backpressure
- Quality-driven routing: Route to workers based on observed performance
"""

import asyncio
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from ..core import BeeRole, ColonyState, Pheromone, WaggleDance


@dataclass
class StreamRecord:
    """A single record in a data stream."""

    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    partition: int = 0


class StreamPartitioner:
    """
    Bee-Inspired Stream Partitioner
    --------------------------------
    Distributes incoming records across partitions using
    waggle-dance-inspired quality metrics.

    Unlike Spark's hash partitioning, this uses:
    - Probabilistic routing weighted by partition health
    - Dynamic rebalancing based on processing latency
    - Backpressure propagation through pheromone signals

    Can also operate in deterministic mode (hash-based) for
    compatibility and testing purposes.
    """

    def __init__(self, num_partitions: int = 8, deterministic: bool = True):
        """
        Initialize partitioner.

        Args:
            num_partitions: Number of partitions to distribute across
            deterministic: If True, use hash-based partitioning (same key -> same partition).
                          If False, use bee-inspired probabilistic routing.
        """
        self.num_partitions = num_partitions
        self.deterministic = deterministic
        self.partition_health: Dict[int, float] = {i: 1.0 for i in range(num_partitions)}
        self.partition_latency: Dict[int, deque] = {
            i: deque(maxlen=100) for i in range(num_partitions)
        }
        self._lock = threading.Lock()

    def partition(self, record: Any) -> int:
        """
        Select partition for a record.

        In deterministic mode: uses hash-based partitioning (same key -> same partition).
        In probabilistic mode: uses fitness-proportional selection weighted by partition health.

        Args:
            record: Either a StreamRecord or a key value (str, int, etc.)

        Returns:
            Partition index (0 to num_partitions-1)
        """
        # Extract key from record
        if isinstance(record, StreamRecord):
            key = record.key
        else:
            key = record

        # Deterministic mode: simple hash-based partitioning
        if self.deterministic:
            return hash(key) % self.num_partitions

        # Probabilistic mode: bee-inspired fitness-proportional selection
        with self._lock:
            # Calculate weights from health scores
            weights = [self.partition_health[i] for i in range(self.num_partitions)]
            total = sum(weights)

            if total == 0:
                # All partitions unhealthy - random fallback
                return hash(key) % self.num_partitions

            # Roulette wheel selection
            r = random.uniform(0, total)
            cumsum = 0
            for i, weight in enumerate(weights):
                cumsum += weight
                if cumsum >= r:
                    return i

            return self.num_partitions - 1

    def report_latency(self, partition: int, latency: float) -> None:
        """Update partition health based on observed latency."""
        with self._lock:
            self.partition_latency[partition].append(latency)

            # Calculate health from recent latencies
            latencies = list(self.partition_latency[partition])
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                # Health inversely proportional to latency
                self.partition_health[partition] = 1.0 / (1.0 + avg_latency)

    def get_health_report(self) -> Dict[int, float]:
        """Get current health status of all partitions."""
        with self._lock:
            return dict(self.partition_health)


class StreamBuffer:
    """
    Bee-Inspired Stream Buffer
    --------------------------
    Implements the "hive" where incoming nectar (records) are stored
    before processing. Uses adaptive sizing based on colony temperature.
    """

    def __init__(
        self, max_size: int = 10000, high_water_mark: float = 0.8, low_water_mark: float = 0.3
    ):
        self.max_size = max_size
        self.high_water_mark = high_water_mark
        self.low_water_mark = low_water_mark
        self.buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    def put(self, record: StreamRecord, timeout: float = 1.0) -> bool:
        """
        Add record to buffer with backpressure.
        Returns False if buffer is full and timeout expires.
        """
        with self._not_full:
            if len(self.buffer) >= self.max_size:
                # Buffer full - wait or timeout
                if not self._not_full.wait(timeout):
                    return False

            self.buffer.append(record)
            self._not_empty.notify()
            return True

    def get(self, timeout: float = 1.0) -> Optional[StreamRecord]:
        """Get next record from buffer."""
        with self._not_empty:
            if not self.buffer:
                if not self._not_empty.wait(timeout):
                    return None

            if self.buffer:
                record = self.buffer.popleft()
                self._not_full.notify()
                return record
            return None

    def get_batch(self, max_batch: int = 100, timeout: float = 0.1) -> List[StreamRecord]:
        """Get a batch of records for micro-batch processing."""
        batch = []
        deadline = time.time() + timeout

        while len(batch) < max_batch and time.time() < deadline:
            record = self.get(timeout=max(0, deadline - time.time()))
            if record:
                batch.append(record)
            else:
                break

        return batch

    @property
    def fill_level(self) -> float:
        """Current buffer fill level (0.0 to 1.0)."""
        with self._lock:
            return len(self.buffer) / self.max_size

    @property
    def is_overheating(self) -> bool:
        """Check if buffer is above high water mark."""
        return self.fill_level > self.high_water_mark

    @property
    def is_cool(self) -> bool:
        """Check if buffer is below low water mark."""
        return self.fill_level < self.low_water_mark


class StreamBee:
    """
    Streaming Bee Worker
    --------------------
    Continuously processes records from a stream buffer.
    Adapts behavior based on colony temperature (load).
    """

    def __init__(
        self,
        worker_id: str,
        role: BeeRole,
        buffer: StreamBuffer,
        process_fn: Callable[[StreamRecord], Any],
        colony: ColonyState,
    ):
        self.worker_id = worker_id
        self.role = role
        self.buffer = buffer
        self.process_fn = process_fn
        self.colony = colony
        self.running = False
        self.processed_count = 0
        self.error_count = 0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the worker thread."""
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        """Main worker loop."""
        while self.running:
            # Check pheromone signals for throttling
            throttle_level = self.colony.sense_pheromone("throttle")
            if throttle_level > 0.8:
                # Heavy throttling - pause briefly
                time.sleep(0.1 * throttle_level)
                continue

            # Get records based on role
            if self.role == BeeRole.EMPLOYED:
                # Employed bees process steadily
                batch = self.buffer.get_batch(max_batch=10, timeout=0.1)
            elif self.role == BeeRole.ONLOOKER:
                # Onlookers process when colony is busy
                if self.buffer.fill_level > 0.5:
                    batch = self.buffer.get_batch(max_batch=20, timeout=0.1)
                else:
                    time.sleep(0.05)
                    continue
            else:  # SCOUT
                # Scouts help only when overloaded
                if self.buffer.is_overheating:
                    batch = self.buffer.get_batch(max_batch=5, timeout=0.05)
                else:
                    time.sleep(0.1)
                    continue

            # Process batch
            for record in batch:
                try:
                    start = time.time()
                    result = self.process_fn(record)
                    latency = time.time() - start

                    # Perform waggle dance
                    dance = WaggleDance(
                        partition_id=str(record.partition),
                        quality_score=1.0,
                        processing_time=latency,
                        result_size=len(str(result)) if result else 0,
                        worker_id=self.worker_id,
                    )
                    self.colony.dance_floor.perform_dance(dance)
                    self.colony.update_temperature(self.worker_id, latency)
                    self.processed_count += 1

                except Exception:
                    self.error_count += 1
                    # Emit alarm pheromone
                    self.colony.emit_pheromone(
                        Pheromone(signal_type="alarm", intensity=0.5, source_worker=self.worker_id)
                    )


class HiveStream:
    """
    HiveStream: Bee-Inspired Stream Processing Engine
    =================================================

    A streaming data processing framework using bee colony patterns.

    Key features:
    - Adaptive partitioning based on partition health
    - Pheromone-based backpressure
    - Self-organizing worker pools
    - Micro-batch processing with adaptive batch sizes

    Usage:
        stream = HiveStream(num_workers=8)
        stream.process(source, transform_fn, sink_fn)
    """

    def __init__(
        self,
        num_workers: int = 8,
        buffer_size: int = 10000,
        employed_ratio: float = 0.5,
        onlooker_ratio: float = 0.3,
        scout_ratio: float = 0.2,
    ):
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.employed_ratio = employed_ratio
        self.onlooker_ratio = onlooker_ratio
        self.scout_ratio = scout_ratio

        self.colony = ColonyState()
        self.buffer = StreamBuffer(max_size=buffer_size)
        self.partitioner = StreamPartitioner(num_partitions=num_workers)
        self.workers: List[StreamBee] = []
        self.results: Queue = Queue()
        self.running = False

    def _create_workers(self, process_fn: Callable) -> List[StreamBee]:
        """Create stream workers with role distribution."""
        workers = []

        n_employed = int(self.num_workers * self.employed_ratio)
        n_onlookers = int(self.num_workers * self.onlooker_ratio)
        n_scouts = self.num_workers - n_employed - n_onlookers

        def wrapped_fn(record: StreamRecord) -> Any:
            result = process_fn(record.value)
            self.results.put((record.key, result))
            return result

        for i in range(n_employed):
            workers.append(
                StreamBee(
                    worker_id=f"stream_employed_{i}",
                    role=BeeRole.EMPLOYED,
                    buffer=self.buffer,
                    process_fn=wrapped_fn,
                    colony=self.colony,
                )
            )

        for i in range(n_onlookers):
            workers.append(
                StreamBee(
                    worker_id=f"stream_onlooker_{i}",
                    role=BeeRole.ONLOOKER,
                    buffer=self.buffer,
                    process_fn=wrapped_fn,
                    colony=self.colony,
                )
            )

        for i in range(n_scouts):
            workers.append(
                StreamBee(
                    worker_id=f"stream_scout_{i}",
                    role=BeeRole.SCOUT,
                    buffer=self.buffer,
                    process_fn=wrapped_fn,
                    colony=self.colony,
                )
            )

        return workers

    def start(self, process_fn: Callable) -> None:
        """Start the stream processing engine."""
        self.running = True
        self.workers = self._create_workers(process_fn)
        for worker in self.workers:
            worker.start()

    def stop(self) -> None:
        """Stop the stream processing engine."""
        self.running = False
        for worker in self.workers:
            worker.stop()

    def submit(self, key: str, value: Any) -> bool:
        """Submit a record for processing."""
        if not self.running:
            raise RuntimeError("Stream not started")

        record = StreamRecord(key=key, value=value)
        record.partition = self.partitioner.partition(record)

        # Check for backpressure
        if self.buffer.is_overheating:
            self.colony.emit_pheromone(
                Pheromone(
                    signal_type="throttle",
                    intensity=self.buffer.fill_level,
                    source_worker="ingestion",
                )
            )

        return self.buffer.put(record)

    def get_result(self, timeout: float = 1.0) -> Optional[Tuple[str, Any]]:
        """Get a processed result."""
        try:
            return self.results.get(timeout=timeout)
        except Empty:
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get current stream processing metrics."""
        return {
            "buffer_fill": self.buffer.fill_level,
            "colony_temperature": self.colony.get_colony_temperature(),
            "partition_health": self.partitioner.get_health_report(),
            "worker_stats": {
                w.worker_id: {"processed": w.processed_count, "errors": w.error_count}
                for w in self.workers
            },
            "throttle_level": self.colony.sense_pheromone("throttle"),
            "alarm_level": self.colony.sense_pheromone("alarm"),
        }


class AsyncHiveStream:
    """
    Async HiveStream for asyncio-based applications.
    """

    def __init__(self, num_workers: int = 8):
        self.num_workers = num_workers
        self.colony = ColonyState()
        self._queues: Dict[int, asyncio.Queue] = {}
        self._workers: List[asyncio.Task] = []

    async def process_stream(
        self, source: AsyncIterator[Tuple[str, Any]], transform_fn: Callable[[Any], Any]
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Process an async stream of (key, value) tuples.

        Args:
            source: Async iterator producing (key, value) pairs
            transform_fn: Function to apply to each value

        Yields:
            (key, result) pairs
        """
        # Create partition queues
        for i in range(self.num_workers):
            self._queues[i] = asyncio.Queue(maxsize=1000)

        results_queue: asyncio.Queue = asyncio.Queue()

        # Worker coroutine
        async def worker(partition: int):
            queue = self._queues[partition]
            while True:
                try:
                    key, value = await asyncio.wait_for(queue.get(), timeout=0.1)
                    start = time.time()
                    result = transform_fn(value)
                    latency = time.time() - start

                    # Record dance
                    dance = WaggleDance(
                        partition_id=str(partition),
                        quality_score=1.0,
                        processing_time=latency,
                        result_size=len(str(result)) if result else 0,
                        worker_id=f"async_worker_{partition}",
                    )
                    self.colony.dance_floor.perform_dance(dance)

                    await results_queue.put((key, result))
                    queue.task_done()
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break

        # Start workers
        for i in range(self.num_workers):
            task = asyncio.create_task(worker(i))
            self._workers.append(task)

        # Ingest from source
        async def ingest():
            async for key, value in source:
                partition = hash(key) % self.num_workers
                await self._queues[partition].put((key, value))

        ingest_task = asyncio.create_task(ingest())

        # Yield results
        try:
            while True:
                try:
                    result = await asyncio.wait_for(results_queue.get(), timeout=0.5)
                    yield result
                except asyncio.TimeoutError:
                    if ingest_task.done() and results_queue.empty():
                        break
        finally:
            ingest_task.cancel()
            for task in self._workers:
                task.cancel()
