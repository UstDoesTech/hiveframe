"""
Locality-Aware Scheduling
=========================

Cross-datacenter swarm coordination with data locality awareness.
Enables efficient global task distribution while minimizing data movement.

Key Concepts:
- Data Locality Levels: Process data where it lives when possible
- Cross-Datacenter Coordination: Federated execution with locality hints
- Network Topology Awareness: Understand datacenter distances
- Adaptive Placement: Learn optimal placement over time
"""

import math
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class LocalityLevel(Enum):
    """Levels of data locality preference."""

    PROCESS_LOCAL = auto()  # Data in same process/memory
    NODE_LOCAL = auto()  # Data on same node
    RACK_LOCAL = auto()  # Data in same rack
    DATACENTER_LOCAL = auto()  # Data in same datacenter
    REGION_LOCAL = auto()  # Data in same region
    ANY = auto()  # No locality preference


@dataclass
class LocalityHint:
    """
    Hints about data location for locality-aware scheduling.
    """

    data_id: str
    preferred_nodes: List[str] = field(default_factory=list)
    preferred_racks: List[str] = field(default_factory=list)
    preferred_datacenters: List[str] = field(default_factory=list)
    preferred_regions: List[str] = field(default_factory=list)
    data_size_bytes: int = 0
    replication_factor: int = 1

    def matches_locality(
        self,
        node: Optional[str] = None,
        rack: Optional[str] = None,
        datacenter: Optional[str] = None,
        region: Optional[str] = None,
    ) -> LocalityLevel:
        """Determine locality level for a given placement."""
        if node and node in self.preferred_nodes:
            return LocalityLevel.NODE_LOCAL
        if rack and rack in self.preferred_racks:
            return LocalityLevel.RACK_LOCAL
        if datacenter and datacenter in self.preferred_datacenters:
            return LocalityLevel.DATACENTER_LOCAL
        if region and region in self.preferred_regions:
            return LocalityLevel.REGION_LOCAL
        return LocalityLevel.ANY


@dataclass
class DataLocality:
    """
    Information about where data is located.
    """

    data_id: str
    locations: List[Dict[str, str]]  # List of {node, rack, datacenter, region}
    size_bytes: int = 0
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def get_best_location(
        self, target_datacenter: Optional[str] = None, target_region: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """Get best location for processing, preferring local."""
        if not self.locations:
            return None

        # Prefer same datacenter
        if target_datacenter:
            for loc in self.locations:
                if loc.get("datacenter") == target_datacenter:
                    return loc

        # Then same region
        if target_region:
            for loc in self.locations:
                if loc.get("region") == target_region:
                    return loc

        # Otherwise first location
        return self.locations[0]


@dataclass
class NetworkTopology:
    """
    Network topology information for locality decisions.
    """

    # Latency matrix between datacenters (ms)
    datacenter_latencies: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Bandwidth between datacenters (MB/s)
    datacenter_bandwidth: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Region to datacenter mapping
    region_datacenters: Dict[str, List[str]] = field(default_factory=dict)

    def get_latency(self, dc1: str, dc2: str) -> float:
        """Get latency between two datacenters."""
        if dc1 == dc2:
            return 0.5  # Same DC latency
        key = (min(dc1, dc2), max(dc1, dc2))
        return self.datacenter_latencies.get(key, 100.0)  # Default 100ms

    def get_bandwidth(self, dc1: str, dc2: str) -> float:
        """Get bandwidth between two datacenters."""
        if dc1 == dc2:
            return 10000.0  # Same DC: 10GB/s
        key = (min(dc1, dc2), max(dc1, dc2))
        return self.datacenter_bandwidth.get(key, 100.0)  # Default 100MB/s

    def get_transfer_time(self, dc1: str, dc2: str, size_bytes: int) -> float:
        """Estimate data transfer time in seconds."""
        latency_s = self.get_latency(dc1, dc2) / 1000
        bandwidth_bps = self.get_bandwidth(dc1, dc2) * 1024 * 1024
        transfer_s = size_bytes / bandwidth_bps if bandwidth_bps > 0 else float("inf")
        return latency_s + transfer_s

    def add_datacenter(
        self,
        datacenter: str,
        region: str,
        latencies: Optional[Dict[str, float]] = None,
        bandwidths: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add a datacenter to the topology."""
        # Update region mapping
        if region not in self.region_datacenters:
            self.region_datacenters[region] = []
        if datacenter not in self.region_datacenters[region]:
            self.region_datacenters[region].append(datacenter)

        # Update latencies
        if latencies:
            for other_dc, latency in latencies.items():
                key = (min(datacenter, other_dc), max(datacenter, other_dc))
                self.datacenter_latencies[key] = latency

        # Update bandwidths
        if bandwidths:
            for other_dc, bandwidth in bandwidths.items():
                key = (min(datacenter, other_dc), max(datacenter, other_dc))
                self.datacenter_bandwidth[key] = bandwidth


@dataclass
class SchedulingDecision:
    """Result of a scheduling decision."""

    task_id: str
    target_datacenter: str
    target_node: Optional[str] = None
    locality_level: LocalityLevel = LocalityLevel.ANY
    estimated_data_transfer_time: float = 0.0
    score: float = 0.0


class LocalityAwareScheduler:
    """
    Schedules tasks with awareness of data locality.

    Minimizes data movement by preferring to execute tasks
    where their input data is located.
    """

    def __init__(
        self,
        topology: Optional[NetworkTopology] = None,
        locality_weights: Optional[Dict[LocalityLevel, float]] = None,
    ):
        self.topology = topology or NetworkTopology()

        # Weights for different locality levels (higher = more preferred)
        self.locality_weights = locality_weights or {
            LocalityLevel.PROCESS_LOCAL: 1.0,
            LocalityLevel.NODE_LOCAL: 0.9,
            LocalityLevel.RACK_LOCAL: 0.7,
            LocalityLevel.DATACENTER_LOCAL: 0.5,
            LocalityLevel.REGION_LOCAL: 0.3,
            LocalityLevel.ANY: 0.1,
        }

        # Data location registry
        self._data_locations: Dict[str, DataLocality] = {}
        self._lock = threading.Lock()

        # Scheduling statistics
        self._stats: Dict[str, int] = defaultdict(int)

    def register_data(
        self, data_id: str, locations: List[Dict[str, str]], size_bytes: int = 0
    ) -> DataLocality:
        """Register data location for scheduling decisions."""
        with self._lock:
            locality = DataLocality(data_id=data_id, locations=locations, size_bytes=size_bytes)
            self._data_locations[data_id] = locality
            return locality

    def update_data_location(self, data_id: str, new_location: Dict[str, str]) -> None:
        """Update data location (e.g., after replication or migration)."""
        with self._lock:
            if data_id in self._data_locations:
                self._data_locations[data_id].locations.append(new_location)

    def schedule(
        self,
        task_id: str,
        data_ids: List[str],
        available_datacenters: List[str],
        available_nodes: Optional[Dict[str, List[str]]] = None,
    ) -> SchedulingDecision:
        """
        Schedule a task based on data locality.

        Args:
            task_id: Unique task identifier
            data_ids: IDs of data required by the task
            available_datacenters: List of datacenters that can execute
            available_nodes: Optional mapping of datacenter -> available nodes

        Returns:
            SchedulingDecision with target datacenter and locality level
        """
        with self._lock:
            # Gather locality information for all required data
            hints = []
            total_size = 0

            for data_id in data_ids:
                data_locality = self._data_locations.get(data_id)
                if data_locality:
                    hints.append(data_locality)
                    total_size += data_locality.size_bytes

            if not hints:
                # No locality info, pick random available DC
                dc = random.choice(available_datacenters)
                self._stats["no_locality"] += 1
                return SchedulingDecision(
                    task_id=task_id,
                    target_datacenter=dc,
                    locality_level=LocalityLevel.ANY,
                    score=self.locality_weights[LocalityLevel.ANY],
                )

            # Score each datacenter
            dc_scores: Dict[str, Tuple[float, LocalityLevel]] = {}

            for dc in available_datacenters:
                score = 0.0
                best_locality = LocalityLevel.ANY

                for hint in hints:
                    for loc in hint.locations:
                        if loc.get("datacenter") == dc:
                            # Data is in this DC
                            loc_locality = LocalityLevel.DATACENTER_LOCAL

                            # Check for node locality
                            if available_nodes and dc in available_nodes:
                                if loc.get("node") in available_nodes[dc]:
                                    loc_locality = LocalityLevel.NODE_LOCAL

                            loc_score = self.locality_weights[loc_locality]

                            # Weight by data size
                            size_weight = hint.size_bytes / total_size if total_size > 0 else 1.0
                            score += loc_score * size_weight

                            if loc_locality.value < best_locality.value:
                                best_locality = loc_locality
                            break
                    else:
                        # Data not in this DC, calculate transfer cost
                        best_loc = hint.get_best_location()
                        if best_loc:
                            src_dc = best_loc.get("datacenter", "")
                            transfer_time = self.topology.get_transfer_time(
                                src_dc, dc, hint.size_bytes
                            )
                            # Penalize based on transfer time
                            transfer_penalty = 1.0 / (1.0 + transfer_time)
                            score += self.locality_weights[LocalityLevel.ANY] * transfer_penalty

                dc_scores[dc] = (score, best_locality)

            # Select best datacenter
            best_dc = max(dc_scores.keys(), key=lambda d: dc_scores[d][0])
            score, locality_level = dc_scores[best_dc]

            # Calculate estimated transfer time for remote data
            transfer_time = 0.0
            for hint in hints:
                best_loc = hint.get_best_location(target_datacenter=best_dc)
                if best_loc and best_loc.get("datacenter") != best_dc:
                    transfer_time += self.topology.get_transfer_time(
                        best_loc.get("datacenter", ""), best_dc, hint.size_bytes
                    )

            # Update statistics
            self._stats[f"locality_{locality_level.name}"] += 1

            # Select node if available
            target_node = None
            if available_nodes and best_dc in available_nodes:
                # Prefer node with local data
                for hint in hints:
                    for loc in hint.locations:
                        if (
                            loc.get("datacenter") == best_dc
                            and loc.get("node") in available_nodes[best_dc]
                        ):
                            target_node = loc.get("node")
                            break
                    if target_node:
                        break

                if not target_node:
                    target_node = random.choice(available_nodes[best_dc])

            return SchedulingDecision(
                task_id=task_id,
                target_datacenter=best_dc,
                target_node=target_node,
                locality_level=locality_level,
                estimated_data_transfer_time=transfer_time,
                score=score,
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        with self._lock:
            total = sum(self._stats.values())
            return {
                "total_decisions": total,
                "by_locality": dict(self._stats),
                "locality_rate": {k: v / total if total > 0 else 0 for k, v in self._stats.items()},
            }


class CrossDatacenterManager:
    """
    Manages cross-datacenter coordination for the swarm.

    Handles data placement, replication, and migration
    to optimize locality for workloads.
    """

    def __init__(self, topology: NetworkTopology, scheduler: LocalityAwareScheduler):
        self.topology = topology
        self.scheduler = scheduler

        # Track data access patterns
        self._access_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()

        # Replication recommendations
        self._replication_queue: List[Dict[str, Any]] = []

    def record_access(
        self, data_id: str, accessing_datacenter: str, access_type: str = "read"
    ) -> None:
        """Record a data access for pattern learning."""
        with self._lock:
            self._access_history[data_id].append(
                {"datacenter": accessing_datacenter, "timestamp": time.time(), "type": access_type}
            )

            # Keep only recent history (last 1000 accesses per data item)
            if len(self._access_history[data_id]) > 1000:
                self._access_history[data_id] = self._access_history[data_id][-1000:]

    def analyze_access_patterns(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze access patterns to identify hot data and access localities.

        Returns mapping of data_id -> datacenter -> access_frequency.
        """
        with self._lock:
            patterns = {}

            for data_id, accesses in self._access_history.items():
                dc_counts: Dict[str, float] = defaultdict(float)

                # Weight recent accesses more heavily
                current_time = time.time()
                for access in accesses:
                    age = current_time - access["timestamp"]
                    weight = math.exp(-age / 3600)  # Exponential decay over 1 hour
                    dc_counts[access["datacenter"]] += weight

                total = sum(dc_counts.values())
                if total > 0:
                    patterns[data_id] = {dc: count / total for dc, count in dc_counts.items()}

            return patterns

    def suggest_replications(self, replication_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Suggest data replications based on access patterns.

        Returns list of suggested replications.
        """
        patterns = self.analyze_access_patterns()
        suggestions = []

        for data_id, dc_frequencies in patterns.items():
            # Get current locations
            locality = self.scheduler._data_locations.get(data_id)
            if not locality:
                continue

            current_dcs = {loc.get("datacenter") for loc in locality.locations}

            # Find datacenters with significant access but no local copy
            for dc, freq in dc_frequencies.items():
                if freq >= replication_threshold and dc not in current_dcs:
                    suggestions.append(
                        {
                            "data_id": data_id,
                            "target_datacenter": dc,
                            "access_frequency": freq,
                            "size_bytes": locality.size_bytes,
                            "priority": freq * locality.access_count,
                        }
                    )

        # Sort by priority
        suggestions.sort(key=lambda s: -float(s.get("priority", 0)))

        return suggestions

    def suggest_migrations(self, migration_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Suggest data migrations when access is heavily skewed.

        Returns list of suggested migrations.
        """
        patterns = self.analyze_access_patterns()
        suggestions = []

        for data_id, dc_frequencies in patterns.items():
            # Get current primary location
            locality = self.scheduler._data_locations.get(data_id)
            if not locality or not locality.locations:
                continue

            primary_dc = locality.locations[0].get("datacenter")
            if primary_dc is None:
                continue
            primary_freq = dc_frequencies.get(primary_dc, 0)

            # Check if another DC has dominant access
            for dc, freq in dc_frequencies.items():
                if dc != primary_dc and freq >= migration_threshold:
                    # This DC has dominant access - suggest migration
                    suggestions.append(
                        {
                            "data_id": data_id,
                            "source_datacenter": primary_dc,
                            "target_datacenter": dc,
                            "access_frequency_source": primary_freq,
                            "access_frequency_target": freq,
                            "size_bytes": locality.size_bytes,
                        }
                    )
                    break

        return suggestions

    def optimize_placement(self) -> Dict[str, Any]:
        """
        Run optimization to improve data placement.

        Returns optimization report with actions taken.
        """
        replications = self.suggest_replications()
        migrations = self.suggest_migrations()

        report = {
            "timestamp": time.time(),
            "suggested_replications": len(replications),
            "suggested_migrations": len(migrations),
            "replications": replications[:10],  # Top 10
            "migrations": migrations[:10],  # Top 10
            "access_patterns": self.analyze_access_patterns(),
        }

        return report
