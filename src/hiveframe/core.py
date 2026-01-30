"""
HiveFrame Core Engine
=====================
A bee-inspired distributed data processing framework.

Key biomimicry concepts implemented:
- Waggle Dance Protocol: Workers advertise task quality through dance-like signals
- Three-Tier Workers: Employed (exploit), Onlooker (reinforce), Scout (explore)
- Stigmergic Coordination: Indirect communication through shared colony state
- Quorum-Based Consensus: Decisions emerge from local interactions
- Adaptive Task Allocation: Self-organizing based on local stimuli
"""

import math
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class BeeRole(Enum):
    """The three fundamental bee roles in a colony."""

    EMPLOYED = auto()  # Exploit known food sources (process assigned partitions)
    ONLOOKER = auto()  # Select and reinforce good solutions
    SCOUT = auto()  # Explore new territory (discover new work)


@dataclass
class WaggleDance:
    """
    Waggle Dance Protocol
    ---------------------
    In nature: Foragers communicate food source quality through figure-8 dances.
    - Dance angle → direction relative to sun
    - Dance duration → distance to source
    - Dance vigor → nectar quality

    In HiveFrame: Workers advertise task results through dance signals.
    - partition_id → which data partition was processed
    - quality_score → how valuable/successful the processing was
    - processing_time → how long it took (latency metric)
    - result_size → output volume (throughput metric)
    """

    partition_id: str
    quality_score: float  # 0.0 to 1.0, higher is better
    processing_time: float  # seconds
    result_size: int  # bytes or record count
    worker_id: str
    timestamp: float = field(default_factory=time.time)

    @property
    def vigor(self) -> float:
        """Dance vigor: composite metric favoring quality and speed."""
        if self.processing_time == 0:
            return self.quality_score
        throughput = self.result_size / self.processing_time
        return self.quality_score * (1 + math.log1p(throughput) / 10)


@dataclass
class FoodSource:
    """
    Food Source (Data Partition)
    ----------------------------
    Represents a unit of work to be processed.
    Tracks fitness and abandonment for the ABC algorithm.
    """

    partition_id: str
    data: Any
    fitness: float = 0.5  # Initial neutral fitness
    trials: int = 0  # Consecutive failed improvements
    assigned_worker: Optional[str] = None
    last_processed: float = field(default_factory=time.time)

    def update_fitness(self, dance: WaggleDance) -> None:
        """Update fitness based on waggle dance feedback."""
        # Exponential moving average with dance vigor
        alpha = 0.3
        self.fitness = alpha * dance.vigor + (1 - alpha) * self.fitness
        self.last_processed = time.time()

    def is_abandoned(self, limit: int) -> bool:
        """Check if this source should be abandoned (ABC algorithm)."""
        return self.trials >= limit


class DanceFloor:
    """
    Dance Floor (Communication Hub)
    -------------------------------
    Stigmergic coordination space where bees share information
    through waggle dances rather than direct messaging.

    Implements the "marketplace of ideas" where multiple solutions
    compete for attention based on quality.
    """

    def __init__(self, max_dances: int = 1000):
        self.dances: Dict[str, List[WaggleDance]] = defaultdict(list)
        self.max_dances = max_dances
        self._lock = threading.Lock()

    def perform_dance(self, dance: WaggleDance) -> None:
        """A worker performs a waggle dance to advertise results."""
        with self._lock:
            partition_dances = self.dances[dance.partition_id]
            partition_dances.append(dance)
            # Keep only recent dances (natural decay)
            if len(partition_dances) > 50:
                self.dances[dance.partition_id] = partition_dances[-50:]

    def observe_dances(self, n: int = 8) -> List[WaggleDance]:
        """
        Onlooker bees observe dances to decide where to forage.
        In nature, followers average ~8 waggle runs before departing.
        """
        with self._lock:
            all_dances = []
            for partition_dances in self.dances.values():
                if partition_dances:
                    # Sample recent dances
                    recent = partition_dances[-10:]
                    all_dances.extend(recent)

            if not all_dances:
                return []

            # Probabilistic selection weighted by vigor
            weights = [d.vigor for d in all_dances]
            total = sum(weights)
            if total == 0:
                return random.sample(all_dances, min(n, len(all_dances)))

            # Roulette wheel selection
            selected = []
            for _ in range(min(n, len(all_dances))):
                r = random.uniform(0, total)
                cumsum: float = 0
                for dance, weight in zip(all_dances, weights):
                    cumsum += weight
                    if cumsum >= r:
                        selected.append(dance)
                        break
            return selected

    def get_partition_quality(self, partition_id: str) -> float:
        """Get aggregate quality signal for a partition."""
        with self._lock:
            dances = self.dances.get(partition_id, [])
            if not dances:
                return 0.5  # Neutral
            recent = dances[-5:]
            return sum(d.vigor for d in recent) / len(recent)


@dataclass
class Pheromone:
    """
    Pheromone Signal
    ----------------
    Indirect coordination through environmental markers.
    Used for rate limiting, backpressure, and alarm signals.
    """

    signal_type: str  # 'throttle', 'alarm', 'recruit', 'inhibit'
    intensity: float  # 0.0 to 1.0
    source_worker: str
    timestamp: float = field(default_factory=time.time)
    decay_rate: float = 0.1  # Per second

    def current_intensity(self) -> float:
        """Pheromones decay over time."""
        elapsed = time.time() - self.timestamp
        return self.intensity * math.exp(-self.decay_rate * elapsed)


class ColonyState:
    """
    Colony State (Shared Environment)
    ---------------------------------
    The stigmergic coordination space containing:
    - Food sources (data partitions)
    - Dance floor (communication)
    - Pheromone trails (coordination signals)
    - Temperature (load metrics for homeostasis)
    """

    def __init__(self, abandonment_limit: int = 10):
        self.food_sources: Dict[str, FoodSource] = {}
        self.dance_floor = DanceFloor()
        self.pheromones: List[Pheromone] = []
        self.abandonment_limit = abandonment_limit
        self.temperature: Dict[str, float] = defaultdict(lambda: 0.5)  # Worker load
        self._lock = threading.Lock()

    def add_food_source(self, partition_id: str, data: Any) -> FoodSource:
        """Register a new data partition as a food source."""
        source = FoodSource(partition_id=partition_id, data=data)
        with self._lock:
            self.food_sources[partition_id] = source
        return source

    def get_unassigned_sources(self) -> List[FoodSource]:
        """Get food sources available for foraging."""
        with self._lock:
            return [s for s in self.food_sources.values() if s.assigned_worker is None]

    def get_abandoned_sources(self) -> List[FoodSource]:
        """Get sources that should be abandoned and re-explored."""
        with self._lock:
            return [s for s in self.food_sources.values() if s.is_abandoned(self.abandonment_limit)]

    def emit_pheromone(self, pheromone: Pheromone) -> None:
        """Worker emits a pheromone signal."""
        with self._lock:
            self.pheromones.append(pheromone)
            # Cleanup old pheromones
            self.pheromones = [p for p in self.pheromones if p.current_intensity() > 0.01]

    def sense_pheromone(self, signal_type: str) -> float:
        """Sense aggregate pheromone intensity of a type."""
        with self._lock:
            relevant = [p for p in self.pheromones if p.signal_type == signal_type]
            if not relevant:
                return 0.0
            return sum(p.current_intensity() for p in relevant) / len(relevant)

    def update_temperature(self, worker_id: str, load: float) -> None:
        """Update local temperature (load) for homeostatic regulation."""
        with self._lock:
            self.temperature[worker_id] = load

    def get_colony_temperature(self) -> float:
        """Get average colony temperature (overall load)."""
        with self._lock:
            if not self.temperature:
                return 0.5
            return sum(self.temperature.values()) / len(self.temperature)


class Bee:
    """
    Bee Worker
    ----------
    An autonomous agent that processes data partitions.
    Follows simple local rules that produce emergent global optimization.
    """

    def __init__(
        self,
        worker_id: str,
        role: BeeRole,
        colony: ColonyState,
        process_fn: Callable[[Any], Tuple[Any, float]],
    ):
        """
        Args:
            worker_id: Unique identifier
            role: EMPLOYED, ONLOOKER, or SCOUT
            colony: Shared colony state
            process_fn: Function to process data, returns (result, quality_score)
        """
        self.worker_id = worker_id
        self.role = role
        self.colony = colony
        self.process_fn = process_fn
        self.current_source: Optional[FoodSource] = None
        self.results: List[Tuple[str, Any]] = []

    def forage(self) -> Optional[WaggleDance]:
        """
        Perform one foraging cycle based on role.
        Returns a waggle dance if successful.
        """
        if self.role == BeeRole.EMPLOYED:
            return self._employed_forage()
        elif self.role == BeeRole.ONLOOKER:
            return self._onlooker_forage()
        else:  # SCOUT
            return self._scout_forage()

    def _employed_forage(self) -> Optional[WaggleDance]:
        """
        Employed Bee: Exploit assigned food source.
        Process the partition and perform waggle dance.
        """
        if self.current_source is None:
            # Find an unassigned source
            sources = self.colony.get_unassigned_sources()
            if not sources:
                return None
            # Select probabilistically by fitness
            weights = [s.fitness for s in sources]
            total = sum(weights)
            if total == 0:
                self.current_source = random.choice(sources)
            else:
                r = random.uniform(0, total)
                cumsum: float = 0
                for source, weight in zip(sources, weights):
                    cumsum += weight
                    if cumsum >= r:
                        self.current_source = source
                        break
            if self.current_source:
                self.current_source.assigned_worker = self.worker_id

        if self.current_source is None:
            return None

        # Process the data
        start_time = time.time()
        try:
            result, quality = self.process_fn(self.current_source.data)
            processing_time = time.time() - start_time

            # Store result
            self.results.append((self.current_source.partition_id, result))

            # Create waggle dance
            dance = WaggleDance(
                partition_id=self.current_source.partition_id,
                quality_score=quality,
                processing_time=processing_time,
                result_size=len(str(result)) if result else 0,
                worker_id=self.worker_id,
            )

            # Update source fitness
            self.current_source.update_fitness(dance)
            self.current_source.trials = 0  # Reset abandonment counter

            # Perform dance on dance floor
            self.colony.dance_floor.perform_dance(dance)

            # Update local temperature
            self.colony.update_temperature(self.worker_id, processing_time)

            return dance

        except Exception:
            # Failed processing - increment abandonment counter
            self.current_source.trials += 1
            return None

    def _onlooker_forage(self) -> Optional[WaggleDance]:
        """
        Onlooker Bee: Observe dances and reinforce good solutions.
        Select work based on observed quality signals.
        """
        # Observe dances (average ~8 runs in nature)
        observed = self.colony.dance_floor.observe_dances(n=8)

        if not observed:
            return None

        # Select partition with best average quality
        partition_scores: Dict[str, List[float]] = defaultdict(list)
        for dance in observed:
            partition_scores[dance.partition_id].append(dance.vigor)

        best_partition = max(
            partition_scores.keys(),
            key=lambda p: sum(partition_scores[p]) / len(partition_scores[p]),
        )

        # Find and process this partition
        source = self.colony.food_sources.get(best_partition)
        if source is None:
            return None

        # Process with employed bee logic
        self.current_source = source
        return self._employed_forage()

    def _scout_forage(self) -> Optional[WaggleDance]:
        """
        Scout Bee: Explore new territory or replace abandoned sources.
        Maintains diversity and prevents local optima.
        """
        # Check for abandoned sources
        abandoned = self.colony.get_abandoned_sources()

        if abandoned:
            # Replace abandoned source with random exploration
            source = random.choice(abandoned)
            source.trials = 0
            source.fitness = random.uniform(0.3, 0.7)  # Reset with some randomness
            self.current_source = source
            return self._employed_forage()

        # Otherwise, just pick any unprocessed source
        unassigned = self.colony.get_unassigned_sources()
        if unassigned:
            self.current_source = random.choice(unassigned)
            self.current_source.assigned_worker = self.worker_id
            return self._employed_forage()

        return None


class HiveFrame:
    """
    HiveFrame: Bee-Inspired Data Processing Engine
    ==============================================

    A distributed data processing framework that replaces Spark's
    centralized driver model with decentralized bee colony coordination.

    Key differences from Spark:
    - No central driver/scheduler bottleneck
    - Probabilistic work distribution (not round-robin)
    - Adaptive load balancing through local stimuli
    - Self-healing through abandonment mechanism
    - Quality-weighted task reinforcement

    Usage:
        hive = HiveFrame(num_workers=8)
        results = hive.process(data_partitions, transform_fn)
    """

    def __init__(
        self,
        num_workers: int = 8,
        employed_ratio: float = 0.5,
        onlooker_ratio: float = 0.4,
        scout_ratio: float = 0.1,
        abandonment_limit: int = 10,
        max_cycles: int = 100,
    ):
        """
        Initialize the HiveFrame engine.

        Args:
            num_workers: Total number of bee workers
            employed_ratio: Fraction of workers that exploit (default 50%)
            onlooker_ratio: Fraction that reinforce (default 40%)
            scout_ratio: Fraction that explore (default 10%)
            abandonment_limit: Cycles before abandoning a source
            max_cycles: Maximum processing cycles
        """
        self.num_workers = num_workers
        self.employed_ratio = employed_ratio
        self.onlooker_ratio = onlooker_ratio
        self.scout_ratio = scout_ratio
        self.abandonment_limit = abandonment_limit
        self.max_cycles = max_cycles

        self.colony: Optional[ColonyState] = None
        self.bees: List[Bee] = []
        self.executor: Optional[ThreadPoolExecutor] = None

    def _create_bees(self, process_fn: Callable) -> List[Bee]:
        """Create the bee colony with role distribution."""
        bees = []

        n_employed = int(self.num_workers * self.employed_ratio)
        n_onlookers = int(self.num_workers * self.onlooker_ratio)
        n_scouts = self.num_workers - n_employed - n_onlookers

        assert self.colony is not None

        for i in range(n_employed):
            bees.append(
                Bee(
                    worker_id=f"employed_{i}",
                    role=BeeRole.EMPLOYED,
                    colony=self.colony,
                    process_fn=process_fn,
                )
            )

        for i in range(n_onlookers):
            bees.append(
                Bee(
                    worker_id=f"onlooker_{i}",
                    role=BeeRole.ONLOOKER,
                    colony=self.colony,
                    process_fn=process_fn,
                )
            )

        for i in range(n_scouts):
            bees.append(
                Bee(
                    worker_id=f"scout_{i}",
                    role=BeeRole.SCOUT,
                    colony=self.colony,
                    process_fn=process_fn,
                )
            )

        return bees

    def process(
        self,
        data: List[Any],
        transform_fn: Callable[[Any], Any],
        partition_fn: Optional[Callable[[Any], str]] = None,
    ) -> Dict[str, Any]:
        """
        Process data using the bee colony.

        Args:
            data: List of data items to process
            transform_fn: Function to apply to each item
            partition_fn: Optional function to create partition IDs

        Returns:
            Dictionary mapping partition_id to results
        """
        # Initialize colony state
        self.colony = ColonyState(abandonment_limit=self.abandonment_limit)

        # Create food sources (partitions)
        for i, item in enumerate(data):
            partition_id = partition_fn(item) if partition_fn else f"partition_{i}"
            self.colony.add_food_source(partition_id, item)

        # Wrap transform function to return quality score
        def process_with_quality(item: Any) -> Tuple[Any, float]:
            try:
                result = transform_fn(item)
                # Quality based on successful completion
                quality = 1.0 if result is not None else 0.0
                return result, quality
            except Exception:
                return None, 0.0

        # Create bee workers
        self.bees = self._create_bees(process_with_quality)

        # Run foraging cycles
        cycles = 0
        processed_partitions: set = set()

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            while cycles < self.max_cycles:
                # Check if all partitions processed
                if len(processed_partitions) >= len(data):
                    break

                # Each bee forages in parallel
                futures = [executor.submit(bee.forage) for bee in self.bees]

                for future in as_completed(futures):
                    dance = future.result()
                    if dance:
                        processed_partitions.add(dance.partition_id)

                # Check homeostasis (backpressure)
                colony_temp = self.colony.get_colony_temperature()
                if colony_temp > 0.8:
                    # Colony overheating - emit throttle pheromone
                    self.colony.emit_pheromone(
                        Pheromone(
                            signal_type="throttle", intensity=colony_temp, source_worker="colony"
                        )
                    )
                    time.sleep(0.01)  # Brief cooling period

                cycles += 1

        # Collect results from all bees
        results = {}
        for bee in self.bees:
            for partition_id, result in bee.results:
                if partition_id not in results:
                    results[partition_id] = result

        return results

    def map(self, data: List[T], fn: Callable[[T], R]) -> List[R]:
        """
        Map operation (like Spark RDD.map).
        Apply function to each element.
        """
        results = self.process(data, fn)
        # Preserve order
        return [results.get(f"partition_{i}") for i in range(len(data))]  # type: ignore

    def filter(self, data: List[T], predicate: Callable[[T], bool]) -> List[T]:
        """
        Filter operation (like Spark RDD.filter).
        Keep elements where predicate is True.
        """

        def filter_fn(item):
            return item if predicate(item) else None

        results = self.process(data, filter_fn)
        return [r for r in results.values() if r is not None]

    def reduce(self, data: List[T], fn: Callable[[T, T], T]) -> T:
        """
        Reduce operation (like Spark RDD.reduce).
        Aggregate elements using associative function.
        """
        if not data:
            raise ValueError("Cannot reduce empty collection")

        # Process in tree-reduction style
        current = list(data)

        while len(current) > 1:
            pairs = []
            for i in range(0, len(current) - 1, 2):
                pairs.append((current[i], current[i + 1]))
            if len(current) % 2 == 1:
                pairs.append((current[-1], current[-1]))  # Pair with itself

            def reduce_pair(pair):
                a, b = pair
                return fn(a, b)

            results = self.process(pairs, reduce_pair)
            current = list(results.values())

        return current[0]

    def group_by_key(self, data: List[Tuple[str, T]]) -> Dict[str, List[T]]:
        """
        GroupByKey operation (like Spark RDD.groupByKey).
        Group values by key.
        """
        # Local grouping first (no need for distributed shuffle)
        groups: Dict[str, List[T]] = defaultdict(list)
        for key, value in data:
            groups[key].append(value)
        return dict(groups)

    def flat_map(self, data: List[T], fn: Callable[[T], List[R]]) -> List[R]:
        """
        FlatMap operation (like Spark RDD.flatMap).
        Apply function and flatten results.
        """
        results = self.process(data, fn)
        flattened = []
        for result in results.values():
            if result:
                flattened.extend(result)
        return flattened


# Convenience functions for creating HiveFrame operations
def create_hive(num_workers: int = 8, **kwargs) -> HiveFrame:
    """Create a HiveFrame instance with specified configuration."""
    return HiveFrame(num_workers=num_workers, **kwargs)
