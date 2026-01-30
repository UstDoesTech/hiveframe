"""
Caching Swarm (Phase 2)
=======================

Distributed intelligent caching based on pheromone trails.
Enables swarm-coordinated cache management for optimal data locality.

Key Concepts:
- Pheromone-based Cache Eviction: Frequently accessed data leaves strong trails
- Distributed Cache Coordination: Swarm-wide cache coherence
- Intelligent Prefetching: Scout bees predict and prefetch data
- Adaptive Cache Sizing: Cache sizes adjust based on workload
"""

import hashlib
import math
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class EvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = auto()  # Least Recently Used
    LFU = auto()  # Least Frequently Used
    PHEROMONE = auto()  # Pheromone-based (swarm intelligence)
    ARC = auto()  # Adaptive Replacement Cache
    LIRS = auto()  # Low Inter-reference Recency Set


class CacheLevel(Enum):
    """Cache hierarchy levels."""

    L1_PROCESS = auto()  # In-process cache (fastest)
    L2_NODE = auto()  # Node-level shared cache
    L3_CLUSTER = auto()  # Cluster-wide distributed cache


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    key: str
    value: Any
    size_bytes: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    pheromone_level: float = 1.0  # Swarm fitness signal
    ttl: Optional[float] = None  # Time to live in seconds
    tags: Set[str] = field(default_factory=set)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    @property
    def fitness(self) -> float:
        """Calculate fitness score for eviction decisions."""
        recency_seconds = time.time() - self.last_accessed

        # Fitness combines frequency, recency, and pheromone
        frequency_score = math.log1p(self.access_count)
        recency_score = 1.0 / (1.0 + recency_seconds / 60)  # Decay over minutes

        return frequency_score * 0.3 + recency_score * 0.3 + self.pheromone_level * 0.4

    def access(self) -> None:
        """Record an access to this entry."""
        self.last_accessed = time.time()
        self.access_count += 1
        # Increase pheromone on access
        self.pheromone_level = min(10.0, self.pheromone_level * 1.1)

    def decay_pheromone(self, rate: float = 0.01) -> None:
        """Apply pheromone decay."""
        self.pheromone_level *= 1.0 - rate
        self.pheromone_level = max(0.1, self.pheromone_level)


@dataclass
class PheromoneTrail:
    """
    Pheromone trail for cache coordination.

    Represents the "scent" left by data access patterns,
    guiding cache decisions across the swarm.
    """

    key: str
    intensity: float
    source_node: str
    timestamp: float = field(default_factory=time.time)
    decay_rate: float = 0.1  # Per second
    propagation_count: int = 0

    def current_intensity(self) -> float:
        """Get current intensity after decay."""
        elapsed = time.time() - self.timestamp
        return self.intensity * math.exp(-self.decay_rate * elapsed)

    def reinforce(self, amount: float = 1.0) -> None:
        """Reinforce trail (data was accessed again)."""
        self.intensity = min(100.0, self.intensity + amount)
        self.timestamp = time.time()


class CacheStatistics:
    """Statistics for cache performance monitoring."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.inserts = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self._lock = threading.Lock()

    def record_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self.hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self.misses += 1

    def record_eviction(self) -> None:
        """Record an eviction."""
        with self._lock:
            self.evictions += 1

    def record_insert(self) -> None:
        """Record an insert."""
        with self._lock:
            self.inserts += 1

    def record_prefetch_hit(self) -> None:
        """Record a prefetch that was used."""
        with self._lock:
            self.prefetch_hits += 1

    def record_prefetch_miss(self) -> None:
        """Record a prefetch that wasn't used."""
        with self._lock:
            self.prefetch_misses += 1

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def prefetch_accuracy(self) -> float:
        """Calculate prefetch accuracy."""
        total = self.prefetch_hits + self.prefetch_misses
        return self.prefetch_hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Export statistics as dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
            "inserts": self.inserts,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "prefetch_accuracy": self.prefetch_accuracy,
        }


class PheromoneCache:
    """
    Pheromone-based cache with swarm intelligence eviction.

    Uses pheromone trails to make eviction decisions,
    similar to how ants mark paths to food sources.
    """

    def __init__(
        self, max_size_bytes: int = 100 * 1024 * 1024, decay_interval: float = 1.0  # 100MB default
    ):
        self.max_size_bytes = max_size_bytes
        self.decay_interval = decay_interval

        self._entries: Dict[str, CacheEntry] = {}
        self._current_size = 0
        self._lock = threading.RLock()
        self.stats = CacheStatistics()

        # Start decay thread
        self._decay_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start background decay thread."""
        self._running = True
        self._decay_thread = threading.Thread(target=self._decay_loop, daemon=True)
        self._decay_thread.start()

    def stop(self) -> None:
        """Stop background decay thread."""
        self._running = False
        if self._decay_thread:
            self._decay_thread.join(timeout=5.0)

    def _decay_loop(self) -> None:
        """Background loop for pheromone decay."""
        while self._running:
            time.sleep(self.decay_interval)
            self._apply_decay()

    def _apply_decay(self) -> None:
        """Apply pheromone decay to all entries."""
        with self._lock:
            for entry in self._entries.values():
                entry.decay_pheromone()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Returns None if not found or expired.
        """
        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                self.stats.record_miss()
                return None

            if entry.is_expired:
                self._remove_entry(key)
                self.stats.record_miss()
                return None

            entry.access()
            self.stats.record_hit()
            return entry.value

    def put(
        self,
        key: str,
        value: Any,
        size_bytes: Optional[int] = None,
        ttl: Optional[float] = None,
        tags: Optional[Set[str]] = None,
    ) -> bool:
        """
        Put value into cache.

        May evict other entries to make room.
        """
        # Estimate size if not provided
        if size_bytes is None:
            size_bytes = len(str(value).encode("utf-8"))

        # Don't cache if larger than max
        if size_bytes > self.max_size_bytes:
            return False

        with self._lock:
            # Remove existing entry if present
            if key in self._entries:
                self._remove_entry(key)

            # Evict until we have room
            while self._current_size + size_bytes > self.max_size_bytes:
                if not self._evict_one():
                    return False

            # Add new entry
            entry = CacheEntry(
                key=key, value=value, size_bytes=size_bytes, ttl=ttl, tags=tags or set()
            )
            self._entries[key] = entry
            self._current_size += size_bytes
            self.stats.record_insert()

            return True

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache."""
        if key in self._entries:
            entry = self._entries[key]
            self._current_size -= entry.size_bytes
            del self._entries[key]

    def _evict_one(self) -> bool:
        """Evict one entry using pheromone-based selection."""
        if not self._entries:
            return False

        # Find entry with lowest fitness
        victim = min(self._entries.values(), key=lambda e: e.fitness)

        self._remove_entry(victim.key)
        self.stats.record_eviction()
        return True

    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache."""
        with self._lock:
            if key in self._entries:
                self._remove_entry(key)
                return True
            return False

    def invalidate_by_tag(self, tag: str) -> int:
        """Remove all entries with a specific tag."""
        with self._lock:
            to_remove = [key for key, entry in self._entries.items() if tag in entry.tags]
            for key in to_remove:
                self._remove_entry(key)
            return len(to_remove)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()
            self._current_size = 0

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._entries)

    @property
    def size_bytes(self) -> int:
        """Current size in bytes."""
        return self._current_size

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.to_dict()
        stats["entries"] = self.size
        stats["size_bytes"] = self.size_bytes
        stats["max_size_bytes"] = self.max_size_bytes
        stats["utilization"] = self.size_bytes / self.max_size_bytes
        return stats


class SwarmPrefetcher:
    """
    Intelligent prefetcher using swarm coordination.

    Scout bees analyze access patterns and predict future
    data needs, prefetching to minimize latency.
    """

    def __init__(self, cache: PheromoneCache, fetch_fn: Callable[[str], Any], lookahead: int = 3):
        self.cache = cache
        self.fetch_fn = fetch_fn
        self.lookahead = lookahead

        # Access pattern tracking
        self._access_sequences: Dict[str, List[str]] = defaultdict(list)
        self._pattern_counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._lock = threading.Lock()

        # Background prefetch queue
        self._prefetch_queue: List[str] = []
        self._prefetch_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start prefetch background thread."""
        self._running = True
        self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._prefetch_thread.start()

    def stop(self) -> None:
        """Stop prefetch background thread."""
        self._running = False
        if self._prefetch_thread:
            self._prefetch_thread.join(timeout=5.0)

    def record_access(self, key: str, context: Optional[str] = None) -> None:
        """
        Record an access for pattern learning.

        Args:
            key: The accessed key
            context: Optional context identifier (e.g., query ID)
        """
        context = context or "default"

        with self._lock:
            sequence = self._access_sequences[context]
            sequence.append(key)

            # Keep only recent history
            if len(sequence) > 100:
                sequence.pop(0)

            # Update pattern counts
            for window_size in range(1, min(self.lookahead + 1, len(sequence))):
                pattern = tuple(sequence[-window_size - 1 : -1])
                if pattern:
                    self._pattern_counts[pattern][key] += 1

            # Trigger prefetch predictions
            predictions = self._predict_next(sequence)
            for pred_key in predictions:
                if pred_key not in self._prefetch_queue:
                    self._prefetch_queue.append(pred_key)

    def _predict_next(self, sequence: List[str]) -> List[str]:
        """Predict next keys based on access pattern."""
        predictions = []

        # Try patterns of decreasing length
        for window_size in range(min(self.lookahead, len(sequence)), 0, -1):
            pattern = tuple(sequence[-window_size:])

            if pattern in self._pattern_counts:
                counts = self._pattern_counts[pattern]

                # Get top predictions
                sorted_preds = sorted(counts.items(), key=lambda x: x[1], reverse=True)[
                    : self.lookahead
                ]

                for key, count in sorted_preds:
                    if key not in predictions and count >= 2:
                        predictions.append(key)

                if predictions:
                    break

        return predictions

    def _prefetch_loop(self) -> None:
        """Background loop for prefetching."""
        while self._running:
            # Get next item to prefetch
            key = None
            with self._lock:
                if self._prefetch_queue:
                    key = self._prefetch_queue.pop(0)

            if key:
                # Check if already cached
                if self.cache.get(key) is not None:
                    self.cache.stats.record_prefetch_hit()
                else:
                    try:
                        # Fetch and cache
                        value = self.fetch_fn(key)
                        self.cache.put(key, value, tags={"prefetch"})
                    except Exception:
                        self.cache.stats.record_prefetch_miss()

            time.sleep(0.01)  # Small delay to avoid busy-waiting

    def get_predictions(self, recent_keys: List[str]) -> List[str]:
        """Get predictions for explicit prefetching."""
        with self._lock:
            return self._predict_next(recent_keys)


class DistributedCacheNode:
    """
    A node in the distributed caching swarm.

    Coordinates with other nodes to provide cluster-wide caching
    with pheromone-based data placement.
    """

    def __init__(self, node_id: str, local_cache: PheromoneCache, cluster_size: int = 8):
        self.node_id = node_id
        self.local_cache = local_cache
        self.cluster_size = cluster_size

        # Pheromone trails for cluster coordination
        self._pheromone_trails: Dict[str, PheromoneTrail] = {}
        self._peer_nodes: Dict[str, "DistributedCacheNode"] = {}
        self._lock = threading.Lock()

    def register_peer(self, node: "DistributedCacheNode") -> None:
        """Register a peer node in the cluster."""
        with self._lock:
            self._peer_nodes[node.node_id] = node

    def unregister_peer(self, node_id: str) -> None:
        """Unregister a peer node."""
        with self._lock:
            if node_id in self._peer_nodes:
                del self._peer_nodes[node_id]

    def get(self, key: str) -> Optional[Any]:
        """
        Get value, checking local then peers.
        """
        # Try local cache first
        value = self.local_cache.get(key)
        if value is not None:
            self._emit_pheromone(key, 1.0)
            return value

        # Check pheromone trails for hints
        trail = self._pheromone_trails.get(key)
        if trail and trail.current_intensity() > 0.1:
            # Try the node that left the trail
            peer = self._peer_nodes.get(trail.source_node)
            if peer:
                value = peer.local_cache.get(key)
                if value is not None:
                    # Cache locally for future access
                    self.local_cache.put(key, value)
                    self._emit_pheromone(key, 0.5)
                    return value

        # Random walk to other peers
        peers = list(self._peer_nodes.values())
        random.shuffle(peers)

        for peer in peers[:3]:  # Check up to 3 peers
            value = peer.local_cache.get(key)
            if value is not None:
                # Cache locally
                self.local_cache.put(key, value)
                self._emit_pheromone(key, 0.3)
                return value

        return None

    def put(self, key: str, value: Any, **kwargs) -> bool:
        """
        Put value into distributed cache.

        May replicate to peers based on pheromone signals.
        """
        # Store locally
        success = self.local_cache.put(key, value, **kwargs)
        if not success:
            return False

        # Emit pheromone
        self._emit_pheromone(key, 1.0)

        # Check if should replicate based on pheromone intensity
        trail = self._pheromone_trails.get(key)
        if trail and trail.current_intensity() > 5.0:
            # High demand - replicate to some peers
            peers = list(self._peer_nodes.values())
            random.shuffle(peers)

            for peer in peers[:2]:
                peer.local_cache.put(key, value, **kwargs)

        return True

    def invalidate(self, key: str) -> None:
        """Invalidate key across cluster."""
        self.local_cache.invalidate(key)

        # Propagate to peers
        for peer in self._peer_nodes.values():
            peer.local_cache.invalidate(key)

        # Clear pheromone trail
        with self._lock:
            if key in self._pheromone_trails:
                del self._pheromone_trails[key]

    def _emit_pheromone(self, key: str, intensity: float) -> None:
        """Emit pheromone signal for a key."""
        with self._lock:
            if key in self._pheromone_trails:
                self._pheromone_trails[key].reinforce(intensity)
            else:
                self._pheromone_trails[key] = PheromoneTrail(
                    key=key, intensity=intensity, source_node=self.node_id
                )

        # Propagate to peers (gossip protocol)
        self._propagate_pheromone(key, intensity)

    def _propagate_pheromone(self, key: str, intensity: float) -> None:
        """Propagate pheromone to peer nodes."""
        trail = self._pheromone_trails.get(key)
        if trail and trail.propagation_count < 3:
            # Select random subset of peers
            peers = list(self._peer_nodes.values())
            random.shuffle(peers)

            for peer in peers[:2]:
                peer._receive_pheromone(key, intensity * 0.7, self.node_id)

            trail.propagation_count += 1

    def _receive_pheromone(self, key: str, intensity: float, source_node: str) -> None:
        """Receive pheromone signal from peer."""
        with self._lock:
            if key in self._pheromone_trails:
                existing = self._pheromone_trails[key]
                # Only update if received intensity is higher
                if intensity > existing.current_intensity():
                    self._pheromone_trails[key] = PheromoneTrail(
                        key=key, intensity=intensity, source_node=source_node
                    )
            else:
                self._pheromone_trails[key] = PheromoneTrail(
                    key=key, intensity=intensity, source_node=source_node
                )


class CachingSwarm:
    """
    Main interface for the distributed caching swarm.

    Coordinates multiple cache nodes using pheromone-based
    communication for optimal data placement and retrieval.

    Example:
        swarm = CachingSwarm(num_nodes=8, max_size_per_node=100*1024*1024)
        swarm.start()

        # Store data
        swarm.put('key1', {'data': 'value'})

        # Retrieve data (from any node)
        value = swarm.get('key1')

        swarm.stop()
    """

    def __init__(self, num_nodes: int = 8, max_size_per_node: int = 100 * 1024 * 1024):
        self.num_nodes = num_nodes
        self.max_size_per_node = max_size_per_node

        # Create nodes
        self.nodes: List[DistributedCacheNode] = []
        for i in range(num_nodes):
            cache = PheromoneCache(max_size_bytes=max_size_per_node)
            node = DistributedCacheNode(
                node_id=f"node_{i}", local_cache=cache, cluster_size=num_nodes
            )
            self.nodes.append(node)

        # Register peers
        for node in self.nodes:
            for peer in self.nodes:
                if peer.node_id != node.node_id:
                    node.register_peer(peer)

    def start(self) -> None:
        """Start all cache nodes."""
        for node in self.nodes:
            node.local_cache.start()

    def stop(self) -> None:
        """Stop all cache nodes."""
        for node in self.nodes:
            node.local_cache.stop()

    def _select_node(self, key: str) -> DistributedCacheNode:
        """Select node for a key using consistent hashing."""
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return self.nodes[hash_val % len(self.nodes)]

    def get(self, key: str) -> Optional[Any]:
        """Get value from swarm."""
        # Start at the responsible node
        node = self._select_node(key)
        return node.get(key)

    def put(self, key: str, value: Any, **kwargs) -> bool:
        """Put value into swarm."""
        node = self._select_node(key)
        return node.put(key, value, **kwargs)

    def invalidate(self, key: str) -> None:
        """Invalidate key across swarm."""
        node = self._select_node(key)
        node.invalidate(key)

    def clear(self) -> None:
        """Clear all caches."""
        for node in self.nodes:
            node.local_cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        total_hits = 0
        total_misses = 0
        total_size = 0
        total_entries = 0

        for node in self.nodes:
            stats = node.local_cache.get_statistics()
            total_hits += stats["hits"]
            total_misses += stats["misses"]
            total_size += stats["size_bytes"]
            total_entries += stats["entries"]

        return {
            "num_nodes": self.num_nodes,
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": (
                total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
            ),
        }
