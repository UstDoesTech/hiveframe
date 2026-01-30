"""
Multi-Hive Federation
=====================

Connect multiple HiveFrame clusters that coordinate like allied bee colonies.
Enables cross-cluster workload distribution and collaborative processing.

Key Concepts:
- Allied Colonies: Multiple hives that share work when beneficial
- Drifting Bees: Workers that move between hives based on workload
- Inter-Colony Waggle: Cross-hive task quality signaling
- Resource Sharing: Balanced utilization across the federation
"""

import math
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set


class FederationProtocol(Enum):
    """Communication protocols between federated hives."""

    GOSSIP = auto()  # Epidemic-style information spreading
    QUORUM = auto()  # Majority-based decisions
    PHEROMONE = auto()  # Indirect signaling through shared state


class HiveHealth(Enum):
    """Health status of a hive in the federation."""

    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


@dataclass
class FederatedHive:
    """
    Represents a single hive in the federation.

    Each hive is an autonomous unit that can process tasks independently
    but also participates in federated workload distribution.
    """

    hive_id: str
    endpoint: str
    datacenter: str
    region: str
    capacity: int  # Number of workers
    current_load: float = 0.0  # 0.0 to 1.0
    health: HiveHealth = HiveHealth.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def available_capacity(self) -> float:
        """Calculate available processing capacity."""
        return max(0, self.capacity * (1 - self.current_load))

    @property
    def is_available(self) -> bool:
        """Check if hive can accept new work."""
        return (
            self.health in (HiveHealth.HEALTHY, HiveHealth.DEGRADED)
            and self.current_load < 0.95
            and time.time() - self.last_heartbeat < 30  # 30 second timeout
        )

    def update_heartbeat(self, load: float) -> None:
        """Update hive status from heartbeat."""
        self.last_heartbeat = time.time()
        self.current_load = load

        if load > 0.9:
            self.health = HiveHealth.DEGRADED
        elif load <= 0.9:
            self.health = HiveHealth.HEALTHY


@dataclass
class InterColonyDance:
    """
    Inter-colony waggle dance for cross-hive signaling.

    Similar to waggle dance but used for communicating task opportunities
    between allied hives in the federation.
    """

    source_hive: str
    task_id: str
    task_quality: float
    estimated_duration: float
    data_size: int
    locality_hints: List[str]  # Preferred datacenters/regions
    timestamp: float = field(default_factory=time.time)

    @property
    def urgency(self) -> float:
        """Calculate urgency score for task distribution."""
        # Higher quality and smaller tasks are more attractive
        size_factor = 1 / (1 + math.log1p(self.data_size / 1e6))
        return self.task_quality * size_factor


class HiveRegistry:
    """
    Registry of all hives in the federation.

    Maintains membership and health status using gossip protocol
    for eventual consistency across the federation.
    """

    def __init__(self, gossip_interval: float = 5.0):
        self._hives: Dict[str, FederatedHive] = {}
        self._lock = threading.RLock()
        self.gossip_interval = gossip_interval
        self._gossip_state: Dict[str, Dict] = {}  # Version vectors

    def register(self, hive: FederatedHive) -> None:
        """Register a new hive with the federation."""
        with self._lock:
            self._hives[hive.hive_id] = hive
            self._gossip_state[hive.hive_id] = {"version": 1, "timestamp": time.time()}

    def deregister(self, hive_id: str) -> None:
        """Remove a hive from the federation."""
        with self._lock:
            if hive_id in self._hives:
                del self._hives[hive_id]
            if hive_id in self._gossip_state:
                del self._gossip_state[hive_id]

    def get_hive(self, hive_id: str) -> Optional[FederatedHive]:
        """Get a specific hive by ID."""
        with self._lock:
            return self._hives.get(hive_id)

    def list_hives(self) -> List[FederatedHive]:
        """Get all registered hives."""
        with self._lock:
            return list(self._hives.values())

    def get_available_hives(self) -> List[FederatedHive]:
        """Get hives that can accept work."""
        with self._lock:
            return [h for h in self._hives.values() if h.is_available]

    def get_hives_by_region(self, region: str) -> List[FederatedHive]:
        """Get all hives in a specific region."""
        with self._lock:
            return [h for h in self._hives.values() if h.region == region]

    def get_hives_by_datacenter(self, datacenter: str) -> List[FederatedHive]:
        """Get all hives in a specific datacenter."""
        with self._lock:
            return [h for h in self._hives.values() if h.datacenter == datacenter]

    def gossip_merge(self, remote_state: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Merge gossip state with remote peer.

        Returns state that should be sent back (items we have that are newer).
        """
        with self._lock:
            updates_for_remote = {}

            for hive_id, remote_info in remote_state.items():
                local_info = self._gossip_state.get(hive_id)

                if local_info is None:
                    # We don't have this hive, accept remote
                    self._gossip_state[hive_id] = remote_info
                elif remote_info["version"] > local_info["version"]:
                    # Remote is newer, accept it
                    self._gossip_state[hive_id] = remote_info
                elif remote_info["version"] < local_info["version"]:
                    # We have newer, send to remote
                    updates_for_remote[hive_id] = local_info

            return updates_for_remote

    def check_health(self) -> Dict[str, HiveHealth]:
        """Check and update health status of all hives."""
        current_time = time.time()
        health_report = {}

        with self._lock:
            for hive_id, hive in self._hives.items():
                elapsed = current_time - hive.last_heartbeat

                if elapsed > 60:
                    hive.health = HiveHealth.UNHEALTHY
                elif elapsed > 30:
                    hive.health = HiveHealth.UNKNOWN

                health_report[hive_id] = hive.health

        return health_report


class FederationCoordinator:
    """
    Coordinates work distribution across the federation.

    Uses a combination of gossip protocol for state sharing and
    fitness-based selection for task assignment.
    """

    def __init__(
        self,
        local_hive: FederatedHive,
        registry: Optional[HiveRegistry] = None,
        protocol: FederationProtocol = FederationProtocol.GOSSIP,
        locality_weight: float = 0.3,
    ):
        self.local_hive = local_hive
        self.registry = registry or HiveRegistry()
        self.protocol = protocol
        self.locality_weight = locality_weight
        self._pending_dances: List[InterColonyDance] = []
        self._lock = threading.Lock()

        # Register local hive
        self.registry.register(local_hive)

    def announce_task(
        self,
        task_id: str,
        quality: float,
        estimated_duration: float,
        data_size: int,
        locality_hints: Optional[List[str]] = None,
    ) -> InterColonyDance:
        """
        Announce a task opportunity to the federation.

        Other hives can observe this dance and decide to help process.
        """
        dance = InterColonyDance(
            source_hive=self.local_hive.hive_id,
            task_id=task_id,
            task_quality=quality,
            estimated_duration=estimated_duration,
            data_size=data_size,
            locality_hints=locality_hints or [],
        )

        with self._lock:
            self._pending_dances.append(dance)
            # Keep only recent dances
            cutoff = time.time() - 60
            self._pending_dances = [d for d in self._pending_dances if d.timestamp > cutoff]

        return dance

    def observe_dances(self, n: int = 10) -> List[InterColonyDance]:
        """Observe recent inter-colony dances."""
        with self._lock:
            if not self._pending_dances:
                return []

            # Sort by urgency and return top N
            sorted_dances = sorted(self._pending_dances, key=lambda d: d.urgency, reverse=True)
            return sorted_dances[:n]

    def select_target_hive(
        self, task: InterColonyDance, exclude: Optional[Set[str]] = None
    ) -> Optional[FederatedHive]:
        """
        Select best hive to process a task.

        Uses fitness-proportional selection with locality awareness.
        """
        exclude = exclude or set()
        available = [h for h in self.registry.get_available_hives() if h.hive_id not in exclude]

        if not available:
            return None

        # Calculate fitness for each hive
        fitness_scores = []
        for hive in available:
            # Base fitness from available capacity
            capacity_fitness = hive.available_capacity / hive.capacity

            # Locality bonus
            locality_bonus = 0.0
            if hive.datacenter in task.locality_hints:
                locality_bonus = self.locality_weight
            elif hive.region in task.locality_hints:
                locality_bonus = self.locality_weight / 2

            # Health penalty
            health_factor = 1.0 if hive.health == HiveHealth.HEALTHY else 0.7

            total_fitness = (capacity_fitness + locality_bonus) * health_factor
            fitness_scores.append((hive, total_fitness))

        # Roulette wheel selection
        total = sum(f for _, f in fitness_scores)
        if total == 0:
            return random.choice(available)

        r = random.uniform(0, total)
        cumsum = 0
        for hive, fitness in fitness_scores:
            cumsum += fitness
            if cumsum >= r:
                return hive

        return available[-1]

    def balance_load(self) -> Dict[str, List[str]]:
        """
        Calculate load balancing recommendations.

        Returns mapping of overloaded hive -> list of hives to shed to.
        """
        recommendations = {}
        available = self.registry.get_available_hives()

        # Find overloaded hives
        overloaded = [h for h in available if h.current_load > 0.8]
        underloaded = [h for h in available if h.current_load < 0.5]

        for over_hive in overloaded:
            # Find underloaded hives in same region preferentially
            candidates = [h for h in underloaded if h.region == over_hive.region]
            if not candidates:
                candidates = underloaded

            if candidates:
                recommendations[over_hive.hive_id] = [c.hive_id for c in candidates[:3]]

        return recommendations


class HiveFederation:
    """
    Main interface for multi-hive federation.

    Manages federation membership, coordinates cross-hive operations,
    and provides a unified interface for federated processing.

    Example:
        # Create local hive
        local = FederatedHive(
            hive_id='hive-1',
            endpoint='http://hive1:8080',
            datacenter='us-east-1a',
            region='us-east-1',
            capacity=100
        )

        # Create federation
        federation = HiveFederation(local)

        # Join other hives
        federation.join(remote_hive)

        # Distribute task
        target = federation.route_task(task_id, data_size=1000000)
    """

    def __init__(
        self,
        local_hive: FederatedHive,
        protocol: FederationProtocol = FederationProtocol.GOSSIP,
        auto_rebalance: bool = True,
    ):
        self.local_hive = local_hive
        self.protocol = protocol
        self.auto_rebalance = auto_rebalance
        self.registry = HiveRegistry()
        self.coordinator = FederationCoordinator(
            local_hive=local_hive, registry=self.registry, protocol=protocol
        )
        self._callbacks: Dict[str, Callable] = {}
        self._running = False
        self._rebalance_thread: Optional[threading.Thread] = None

    def join(self, hive: FederatedHive) -> None:
        """Add a hive to the federation."""
        self.registry.register(hive)

        if "on_join" in self._callbacks:
            self._callbacks["on_join"](hive)

    def leave(self, hive_id: str) -> None:
        """Remove a hive from the federation."""
        hive = self.registry.get_hive(hive_id)
        self.registry.deregister(hive_id)

        if hive and "on_leave" in self._callbacks:
            self._callbacks["on_leave"](hive)

    def on_event(self, event: str, callback: Callable) -> None:
        """Register callback for federation events."""
        self._callbacks[event] = callback

    def route_task(
        self,
        task_id: str,
        data_size: int = 0,
        quality: float = 0.5,
        locality_hints: Optional[List[str]] = None,
        exclude_hives: Optional[Set[str]] = None,
    ) -> Optional[FederatedHive]:
        """
        Route a task to the best available hive.

        Uses swarm intelligence to select optimal target based on
        capacity, locality, and health.
        """
        # First check if local hive can handle it
        if (
            self.local_hive.is_available
            and self.local_hive.current_load < 0.7
            and (
                not locality_hints
                or self.local_hive.datacenter in locality_hints
                or self.local_hive.region in locality_hints
            )
        ):
            return self.local_hive

        # Announce task and select target
        dance = self.coordinator.announce_task(
            task_id=task_id,
            quality=quality,
            estimated_duration=0.0,  # Unknown initially
            data_size=data_size,
            locality_hints=locality_hints,
        )

        return self.coordinator.select_target_hive(dance, exclude_hives)

    def broadcast_status(self, load: float) -> None:
        """Broadcast local hive status to federation."""
        self.local_hive.update_heartbeat(load)

    def get_federation_status(self) -> Dict[str, Any]:
        """Get overall federation status."""
        hives = self.registry.list_hives()
        health = self.registry.check_health()

        return {
            "total_hives": len(hives),
            "healthy_hives": sum(1 for h in health.values() if h == HiveHealth.HEALTHY),
            "total_capacity": sum(h.capacity for h in hives),
            "available_capacity": sum(h.available_capacity for h in hives),
            "average_load": sum(h.current_load for h in hives) / len(hives) if hives else 0,
            "hives": [
                {
                    "id": h.hive_id,
                    "datacenter": h.datacenter,
                    "region": h.region,
                    "health": h.health.name,
                    "load": h.current_load,
                }
                for h in hives
            ],
        }

    def start(self) -> None:
        """Start federation services."""
        self._running = True

        if self.auto_rebalance:
            self._rebalance_thread = threading.Thread(target=self._rebalance_loop, daemon=True)
            self._rebalance_thread.start()

    def stop(self) -> None:
        """Stop federation services."""
        self._running = False
        if self._rebalance_thread:
            self._rebalance_thread.join(timeout=5)

    def _rebalance_loop(self) -> None:
        """Background thread for load rebalancing."""
        while self._running:
            try:
                recommendations = self.coordinator.balance_load()

                if recommendations and "on_rebalance" in self._callbacks:
                    self._callbacks["on_rebalance"](recommendations)

            except Exception:
                pass  # Continue on error

            time.sleep(10)  # Check every 10 seconds
