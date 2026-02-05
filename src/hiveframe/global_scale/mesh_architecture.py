"""
Global Mesh Architecture - Seamless operation across continents

Implements geo-distributed swarm coordination using bio-inspired principles
to enable planet-scale data processing with minimal latency.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class RegionStatus(Enum):
    """Status of a geographical region in the mesh"""

    ACTIVE = "active"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    SYNCING = "syncing"


@dataclass
class Region:
    """Represents a geographical region in the global mesh"""

    region_id: str
    location: Tuple[float, float]  # (latitude, longitude)
    status: RegionStatus = RegionStatus.ACTIVE
    capacity: int = 100
    current_load: int = 0
    latency_map: Dict[str, float] = field(default_factory=dict)
    pheromone_level: float = 1.0

    def utilization(self) -> float:
        """Calculate region utilization (0.0 to 1.0)"""
        return self.current_load / self.capacity if self.capacity > 0 else 0.0

    def fitness(self) -> float:
        """Calculate region fitness for task assignment (higher is better)"""
        # Bio-inspired: combine utilization, status, and pheromone level
        if self.status == RegionStatus.OFFLINE:
            return 0.0

        status_weight = {
            RegionStatus.ACTIVE: 1.0,
            RegionStatus.DEGRADED: 0.5,
            RegionStatus.SYNCING: 0.3,
            RegionStatus.OFFLINE: 0.0,
        }[self.status]

        # Lower utilization is better (more capacity available)
        utilization_score = 1.0 - self.utilization()

        return status_weight * utilization_score * self.pheromone_level


class GlobalMeshCoordinator:
    """
    Coordinates distributed swarm operations across multiple continents.

    Uses bio-inspired mesh topology where regions communicate like
    interconnected bee colonies, sharing workload information through
    waggle dance patterns and pheromone trails.
    """

    def __init__(self, max_regions: int = 50):
        self.regions: Dict[str, Region] = {}
        self.max_regions = max_regions
        self.routing_table: Dict[str, List[str]] = {}  # region -> [reachable regions]
        self.total_tasks_routed = 0
        self.pheromone_decay_rate = 0.95

    def register_region(
        self,
        region_id: str,
        location: Tuple[float, float],
        capacity: int = 100,
    ) -> bool:
        """Register a new region in the global mesh"""
        if len(self.regions) >= self.max_regions:
            return False

        if region_id in self.regions:
            return False

        region = Region(
            region_id=region_id,
            location=location,
            capacity=capacity,
        )

        self.regions[region_id] = region
        self.routing_table[region_id] = []

        # Calculate latencies to all other regions (simplified model)
        for other_id, other_region in self.regions.items():
            if other_id != region_id:
                latency = self._estimate_latency(location, other_region.location)
                region.latency_map[other_id] = latency
                other_region.latency_map[region_id] = latency

                # Add to routing table if latency is acceptable
                if latency < 500:  # ms
                    self.routing_table[region_id].append(other_id)
                    self.routing_table[other_id].append(region_id)

        return True

    def _estimate_latency(
        self,
        loc1: Tuple[float, float],
        loc2: Tuple[float, float],
    ) -> float:
        """
        Estimate network latency between two locations.

        Simplified model based on great circle distance.
        Real implementation would use actual network measurements.
        """
        lat1, lon1 = loc1
        lat2, lon2 = loc2

        # Haversine formula for great circle distance
        import math

        earth_radius = 6371  # Earth radius in km

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_km = earth_radius * c

        # Approximate latency: ~20ms per 1000km + base latency
        latency = (distance_km / 1000) * 20 + 10 + random.uniform(-5, 5)
        return max(1.0, latency)

    def route_task(
        self,
        task_id: str,
        preferred_region: Optional[str] = None,
        data_locality: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Route a task to the optimal region using swarm intelligence.

        Uses bee-inspired decision making:
        - Scout bees explore available regions
        - Waggle dance communicates region fitness
        - Pheromone trails guide future routing decisions
        """
        if not self.regions:
            return None

        # Filter available regions
        available_regions = [
            r
            for r in self.regions.values()
            if r.status in (RegionStatus.ACTIVE, RegionStatus.DEGRADED) and r.utilization() < 0.95
        ]

        if not available_regions:
            return None

        # If preferred region is available and has capacity, use it
        if preferred_region and preferred_region in self.regions:
            region = self.regions[preferred_region]
            if region in available_regions:
                self._assign_task(region.region_id)
                return region.region_id

        # Data locality optimization: prefer regions with data
        if data_locality:
            local_regions = [r for r in available_regions if r.region_id in data_locality]
            if local_regions:
                available_regions = local_regions

        # Calculate weighted probabilities based on fitness (waggle dance)
        fitness_scores = [r.fitness() for r in available_regions]
        total_fitness = sum(fitness_scores)

        if total_fitness == 0:
            # Fallback: random selection
            selected = random.choice(available_regions)
        else:
            # Probabilistic selection weighted by fitness
            probabilities = [f / total_fitness for f in fitness_scores]
            selected = random.choices(available_regions, weights=probabilities)[0]

        # Assign task and update pheromone
        self._assign_task(selected.region_id)
        selected.pheromone_level = min(2.0, selected.pheromone_level * 1.1)

        return selected.region_id

    def _assign_task(self, region_id: str) -> None:
        """Assign a task to a region"""
        region = self.regions[region_id]
        region.current_load += 1
        self.total_tasks_routed += 1

    def complete_task(self, region_id: str, success: bool = True) -> None:
        """Mark a task as complete and update region state"""
        if region_id not in self.regions:
            return

        region = self.regions[region_id]
        region.current_load = max(0, region.current_load - 1)

        # Update pheromone based on success
        if success:
            region.pheromone_level = min(2.0, region.pheromone_level * 1.05)
        else:
            region.pheromone_level = max(0.1, region.pheromone_level * 0.9)

    def evaporate_pheromones(self) -> None:
        """
        Evaporate pheromones over time (bee colony behavior).

        Should be called periodically to prevent stale routing decisions.
        """
        for region in self.regions.values():
            region.pheromone_level = max(0.5, region.pheromone_level * self.pheromone_decay_rate)

    def get_region_status(self, region_id: str) -> Optional[Dict]:
        """Get detailed status of a region"""
        if region_id not in self.regions:
            return None

        region = self.regions[region_id]
        return {
            "region_id": region.region_id,
            "location": region.location,
            "status": region.status.value,
            "capacity": region.capacity,
            "current_load": region.current_load,
            "utilization": region.utilization(),
            "fitness": region.fitness(),
            "pheromone_level": region.pheromone_level,
            "connected_regions": len(self.routing_table.get(region_id, [])),
        }

    def get_mesh_stats(self) -> Dict:
        """Get overall mesh statistics"""
        active_regions = sum(1 for r in self.regions.values() if r.status == RegionStatus.ACTIVE)

        total_capacity = sum(r.capacity for r in self.regions.values())
        total_load = sum(r.current_load for r in self.regions.values())
        avg_utilization = total_load / total_capacity if total_capacity > 0 else 0.0

        return {
            "total_regions": len(self.regions),
            "active_regions": active_regions,
            "total_capacity": total_capacity,
            "current_load": total_load,
            "average_utilization": avg_utilization,
            "total_tasks_routed": self.total_tasks_routed,
        }


class CrossRegionReplicator:
    """
    Handles data replication across regions for fault tolerance and locality.

    Uses swarm intelligence to determine optimal replication strategies,
    similar to how bees distribute food storage across the hive.
    """

    def __init__(self, replication_factor: int = 3):
        self.replication_factor = replication_factor
        self.replica_map: Dict[str, List[str]] = {}  # data_id -> [region_ids]
        self.replication_queue: List[Tuple[str, str]] = []

    def replicate_data(
        self,
        data_id: str,
        source_region: str,
        coordinator: GlobalMeshCoordinator,
    ) -> List[str]:
        """
        Replicate data to optimal regions using swarm intelligence.

        Returns list of target regions for replication.
        """
        if data_id in self.replica_map:
            return self.replica_map[data_id]

        # Select target regions based on:
        # 1. Geographic diversity (avoid single points of failure)
        # 2. Available capacity
        # 3. Network latency from source

        target_regions = [source_region]
        remaining_needed = self.replication_factor - 1

        if source_region not in coordinator.regions:
            return target_regions

        source = coordinator.regions[source_region]

        # Get candidate regions sorted by latency
        candidates = [
            (region_id, source.latency_map.get(region_id, float("inf")))
            for region_id in coordinator.regions.keys()
            if region_id != source_region
            and coordinator.regions[region_id].status == RegionStatus.ACTIVE
            and coordinator.regions[region_id].utilization() < 0.8
        ]

        candidates.sort(key=lambda x: x[1])

        # Select replicas with geographic diversity
        for region_id, latency in candidates[:remaining_needed]:
            target_regions.append(region_id)

        self.replica_map[data_id] = target_regions
        return target_regions

    def get_nearest_replica(
        self,
        data_id: str,
        requesting_region: str,
        coordinator: GlobalMeshCoordinator,
    ) -> Optional[str]:
        """Find nearest replica of data for a requesting region"""
        if data_id not in self.replica_map:
            return None

        replicas = self.replica_map[data_id]
        if requesting_region in replicas:
            return requesting_region

        if requesting_region not in coordinator.regions:
            return replicas[0] if replicas else None

        source = coordinator.regions[requesting_region]

        # Find replica with lowest latency
        best_replica = None
        best_latency = float("inf")

        for replica_region in replicas:
            latency = source.latency_map.get(replica_region, float("inf"))
            if latency < best_latency:
                best_latency = latency
                best_replica = replica_region

        return best_replica


class LatencyAwareRouter:
    """
    Routes requests to minimize latency using bio-inspired pathfinding.

    Similar to how bees optimize flight paths to flowers, this router
    learns optimal paths through the global mesh network.
    """

    def __init__(self):
        self.path_cache: Dict[Tuple[str, str], List[str]] = {}
        self.path_latencies: Dict[Tuple[str, str], float] = {}
        self.path_pheromones: Dict[Tuple[str, str], float] = {}

    def find_optimal_path(
        self,
        source: str,
        destination: str,
        coordinator: GlobalMeshCoordinator,
    ) -> Optional[List[str]]:
        """
        Find optimal path between regions using ant colony optimization.

        Returns list of region IDs representing the path.
        """
        cache_key = (source, destination)

        # Check cache first
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]

        # Direct connection
        if destination in coordinator.routing_table.get(source, []):
            path = [source, destination]
            self.path_cache[cache_key] = path
            return path

        # BFS to find shortest path
        visited = {source}
        queue = [(source, [source])]

        while queue:
            current, path = queue.pop(0)

            if current == destination:
                self.path_cache[cache_key] = path
                self._update_path_pheromone(cache_key, path, coordinator)
                return path

            for neighbor in coordinator.routing_table.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def _update_path_pheromone(
        self,
        cache_key: Tuple[str, str],
        path: List[str],
        coordinator: GlobalMeshCoordinator,
    ) -> None:
        """Update pheromone for a discovered path"""
        # Calculate total path latency
        total_latency = 0.0
        for i in range(len(path) - 1):
            source_region = coordinator.regions.get(path[i])
            if source_region:
                total_latency += source_region.latency_map.get(path[i + 1], 0)

        self.path_latencies[cache_key] = total_latency

        # Higher pheromone for lower latency paths
        self.path_pheromones[cache_key] = 1.0 / (1.0 + total_latency / 100)

    def get_path_latency(self, source: str, destination: str) -> Optional[float]:
        """Get estimated latency for a path"""
        cache_key = (source, destination)
        return self.path_latencies.get(cache_key)

    def evaporate_paths(self) -> None:
        """Evaporate pheromones on cached paths"""
        keys_to_remove = []

        for key, pheromone in self.path_pheromones.items():
            new_pheromone = pheromone * 0.95
            if new_pheromone < 0.1:
                keys_to_remove.append(key)
            else:
                self.path_pheromones[key] = new_pheromone

        for key in keys_to_remove:
            self.path_cache.pop(key, None)
            self.path_latencies.pop(key, None)
            self.path_pheromones.pop(key, None)
