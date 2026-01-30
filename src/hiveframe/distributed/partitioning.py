"""
Adaptive Partitioning
=====================

Dynamic partition splitting and merging based on swarm fitness signals.
Partitions adjust their size automatically based on processing characteristics.

Key Concepts:
- Fitness-based Splitting: Large partitions with low fitness are split
- Swarm Merging: Small partitions with similar data are merged
- Dynamic Sizing: Partition sizes adapt to workload patterns
- Honeycomb Structure: Natural hexagonal distribution for efficiency
"""

import time
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto
from collections import defaultdict
import math
import random


class PartitionStrategy(Enum):
    """Partitioning strategies."""
    HASH = auto()       # Hash-based distribution
    RANGE = auto()      # Range-based partitioning
    ROUND_ROBIN = auto()  # Round-robin assignment
    FITNESS = auto()    # Fitness-based adaptive
    LOCALITY = auto()   # Data locality aware


class PartitionState(Enum):
    """Current state of a partition."""
    ACTIVE = auto()
    SPLITTING = auto()
    MERGING = auto()
    PENDING = auto()
    COMPLETED = auto()


@dataclass
class PartitionMetrics:
    """Metrics tracked for each partition."""
    records_processed: int = 0
    bytes_processed: int = 0
    processing_time_ms: float = 0.0
    error_count: int = 0
    last_processed: float = field(default_factory=time.time)
    fitness_history: List[float] = field(default_factory=list)
    
    @property
    def throughput(self) -> float:
        """Records per second."""
        if self.processing_time_ms == 0:
            return 0.0
        return self.records_processed / (self.processing_time_ms / 1000)
    
    @property
    def average_fitness(self) -> float:
        """Rolling average fitness score."""
        if not self.fitness_history:
            return 0.5
        recent = self.fitness_history[-10:]  # Last 10 readings
        return sum(recent) / len(recent)
    
    def record_fitness(self, fitness: float) -> None:
        """Record a fitness measurement."""
        self.fitness_history.append(fitness)
        # Keep only recent history
        if len(self.fitness_history) > 100:
            self.fitness_history = self.fitness_history[-100:]


@dataclass
class Partition:
    """
    A data partition with adaptive sizing capabilities.
    
    Partitions can split or merge based on fitness feedback,
    similar to how bee colonies adjust comb cell sizes.
    """
    partition_id: str
    data: List[Any]
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    state: PartitionState = PartitionState.ACTIVE
    metrics: PartitionMetrics = field(default_factory=PartitionMetrics)
    created_at: float = field(default_factory=time.time)
    split_threshold: float = 0.3  # Split if fitness below this
    merge_threshold: float = 0.9  # Merge if fitness above this
    min_size: int = 10
    max_size: int = 100000
    
    @property
    def size(self) -> int:
        """Number of records in partition."""
        return len(self.data)
    
    @property
    def should_split(self) -> bool:
        """Check if partition should be split."""
        return (
            self.state == PartitionState.ACTIVE and
            self.size > self.min_size * 2 and
            self.metrics.average_fitness < self.split_threshold and
            self.metrics.records_processed > 0
        )
    
    @property
    def should_merge(self) -> bool:
        """Check if partition could merge with siblings."""
        return (
            self.state == PartitionState.ACTIVE and
            self.size < self.max_size / 2 and
            self.metrics.average_fitness > self.merge_threshold
        )
    
    def split(self) -> Tuple['Partition', 'Partition']:
        """
        Split partition into two child partitions.
        
        Uses midpoint splitting for even distribution.
        """
        self.state = PartitionState.SPLITTING
        mid = len(self.data) // 2
        
        left_data = self.data[:mid]
        right_data = self.data[mid:]
        
        left = Partition(
            partition_id=f"{self.partition_id}_L",
            data=left_data,
            parent_id=self.partition_id,
            min_size=self.min_size,
            max_size=self.max_size
        )
        
        right = Partition(
            partition_id=f"{self.partition_id}_R",
            data=right_data,
            parent_id=self.partition_id,
            min_size=self.min_size,
            max_size=self.max_size
        )
        
        self.children = [left.partition_id, right.partition_id]
        self.state = PartitionState.COMPLETED
        self.data = []  # Clear data from parent
        
        return left, right
    
    @staticmethod
    def merge(p1: 'Partition', p2: 'Partition') -> 'Partition':
        """
        Merge two partitions into one.
        
        Combines data and metrics from both partitions.
        """
        p1.state = PartitionState.MERGING
        p2.state = PartitionState.MERGING
        
        merged_id = f"{p1.partition_id}+{p2.partition_id}"
        merged_data = p1.data + p2.data
        
        merged = Partition(
            partition_id=merged_id,
            data=merged_data,
            min_size=p1.min_size,
            max_size=p1.max_size
        )
        
        # Combine metrics
        merged.metrics.records_processed = (
            p1.metrics.records_processed + p2.metrics.records_processed
        )
        merged.metrics.bytes_processed = (
            p1.metrics.bytes_processed + p2.metrics.bytes_processed
        )
        
        p1.state = PartitionState.COMPLETED
        p2.state = PartitionState.COMPLETED
        p1.data = []
        p2.data = []
        
        return merged


class PartitionSplitter:
    """
    Handles partition splitting logic.
    
    Uses multiple strategies for determining split points
    based on data characteristics.
    """
    
    def __init__(self, min_partition_size: int = 10):
        self.min_partition_size = min_partition_size
        
    def can_split(self, partition: Partition) -> bool:
        """Check if partition can be split."""
        return partition.size >= self.min_partition_size * 2
        
    def split_by_fitness(
        self,
        partition: Partition,
        fitness_fn: Callable[[Any], float]
    ) -> Tuple[Partition, Partition]:
        """
        Split partition by fitness scoring.
        
        Items with similar fitness scores are grouped together.
        """
        if not self.can_split(partition):
            raise ValueError("Partition too small to split")
            
        # Score each item
        scored = [(fitness_fn(item), item) for item in partition.data]
        scored.sort(key=lambda x: x[0])
        
        # Split at midpoint
        mid = len(scored) // 2
        left_data = [item for _, item in scored[:mid]]
        right_data = [item for _, item in scored[mid:]]
        
        left = Partition(
            partition_id=f"{partition.partition_id}_L",
            data=left_data,
            parent_id=partition.partition_id
        )
        
        right = Partition(
            partition_id=f"{partition.partition_id}_H",  # High fitness
            data=right_data,
            parent_id=partition.partition_id
        )
        
        partition.children = [left.partition_id, right.partition_id]
        partition.state = PartitionState.COMPLETED
        
        return left, right
        
    def split_by_key(
        self,
        partition: Partition,
        key_fn: Callable[[Any], Any]
    ) -> List[Partition]:
        """
        Split partition by key grouping.
        
        Items with same key go to same child partition.
        """
        groups: Dict[Any, List[Any]] = defaultdict(list)
        
        for item in partition.data:
            key = key_fn(item)
            groups[key].append(item)
            
        children = []
        for key, items in groups.items():
            child = Partition(
                partition_id=f"{partition.partition_id}_{hash(key) % 1000}",
                data=items,
                parent_id=partition.partition_id
            )
            children.append(child)
            
        partition.children = [c.partition_id for c in children]
        partition.state = PartitionState.COMPLETED
        
        return children


class PartitionMerger:
    """
    Handles partition merging logic.
    
    Identifies opportunities to combine small partitions
    for more efficient processing.
    """
    
    def __init__(self, max_partition_size: int = 100000):
        self.max_partition_size = max_partition_size
        
    def can_merge(self, p1: Partition, p2: Partition) -> bool:
        """Check if two partitions can be merged."""
        combined_size = p1.size + p2.size
        return combined_size <= self.max_partition_size
        
    def find_merge_candidates(
        self,
        partitions: List[Partition]
    ) -> List[Tuple[Partition, Partition]]:
        """
        Find pairs of partitions that could benefit from merging.
        
        Prefers merging siblings and partitions with similar fitness.
        """
        candidates = []
        
        # Group by parent for sibling merging
        siblings: Dict[Optional[str], List[Partition]] = defaultdict(list)
        for p in partitions:
            if p.state == PartitionState.ACTIVE and p.should_merge:
                siblings[p.parent_id].append(p)
                
        for parent_id, sibs in siblings.items():
            if len(sibs) >= 2:
                # Sort by fitness similarity
                sibs.sort(key=lambda p: p.metrics.average_fitness)
                for i in range(0, len(sibs) - 1, 2):
                    if self.can_merge(sibs[i], sibs[i+1]):
                        candidates.append((sibs[i], sibs[i+1]))
                        
        return candidates
        
    def merge_all(
        self,
        candidates: List[Tuple[Partition, Partition]]
    ) -> List[Partition]:
        """Merge all candidate pairs."""
        merged = []
        for p1, p2 in candidates:
            merged_partition = Partition.merge(p1, p2)
            merged.append(merged_partition)
        return merged


class FitnessPartitioner:
    """
    Assigns data to partitions based on fitness scoring.
    
    Similar to how bees distribute work based on food source quality.
    """
    
    def __init__(
        self,
        num_partitions: int = 8,
        fitness_fn: Optional[Callable[[Any], float]] = None
    ):
        self.num_partitions = num_partitions
        self.fitness_fn = fitness_fn or (lambda x: 0.5)
        self._partitions: Dict[str, Partition] = {}
        
    def partition(self, data: List[Any]) -> List[Partition]:
        """
        Partition data based on fitness scores.
        
        High-fitness items are grouped together for efficient processing.
        """
        if not data:
            return []
            
        # Score all items
        scored = [(self.fitness_fn(item), item) for item in data]
        scored.sort(key=lambda x: x[0])
        
        # Create partitions with roughly equal sizes
        partition_size = max(1, len(data) // self.num_partitions)
        partitions = []
        
        for i in range(self.num_partitions):
            start = i * partition_size
            end = start + partition_size if i < self.num_partitions - 1 else len(data)
            
            if start >= len(data):
                break
                
            partition_data = [item for _, item in scored[start:end]]
            partition = Partition(
                partition_id=f"fitness_partition_{i}",
                data=partition_data
            )
            partitions.append(partition)
            self._partitions[partition.partition_id] = partition
            
        return partitions
        
    def get_partition(self, partition_id: str) -> Optional[Partition]:
        """Get partition by ID."""
        return self._partitions.get(partition_id)
        
    def update_fitness(self, partition_id: str, fitness: float) -> None:
        """Update fitness for a partition."""
        if partition_id in self._partitions:
            self._partitions[partition_id].metrics.record_fitness(fitness)


class AdaptivePartitioner:
    """
    Main adaptive partitioning interface.
    
    Automatically adjusts partition sizes based on runtime feedback,
    splitting overloaded partitions and merging underutilized ones.
    
    Example:
        partitioner = AdaptivePartitioner(
            initial_partitions=8,
            min_partition_size=100,
            max_partition_size=10000
        )
        
        # Initial partitioning
        partitions = partitioner.partition(data)
        
        # Process and provide feedback
        for partition in partitions:
            result = process(partition)
            partitioner.record_fitness(partition.partition_id, result.quality)
        
        # Adapt based on feedback
        partitions = partitioner.adapt()
    """
    
    def __init__(
        self,
        initial_partitions: int = 8,
        min_partition_size: int = 10,
        max_partition_size: int = 100000,
        split_threshold: float = 0.3,
        merge_threshold: float = 0.9,
        strategy: PartitionStrategy = PartitionStrategy.FITNESS
    ):
        self.initial_partitions = initial_partitions
        self.min_partition_size = min_partition_size
        self.max_partition_size = max_partition_size
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        self.strategy = strategy
        
        self._partitions: Dict[str, Partition] = {}
        self._splitter = PartitionSplitter(min_partition_size)
        self._merger = PartitionMerger(max_partition_size)
        self._lock = threading.Lock()
        
    def partition(
        self,
        data: List[Any],
        key_fn: Optional[Callable[[Any], str]] = None
    ) -> List[Partition]:
        """
        Create initial partitions for data.
        
        Args:
            data: Data to partition
            key_fn: Optional key function for grouping
            
        Returns:
            List of partitions
        """
        with self._lock:
            if self.strategy == PartitionStrategy.HASH:
                return self._hash_partition(data, key_fn)
            elif self.strategy == PartitionStrategy.RANGE:
                return self._range_partition(data)
            elif self.strategy == PartitionStrategy.ROUND_ROBIN:
                return self._round_robin_partition(data)
            else:  # FITNESS or default
                return self._fitness_partition(data)
                
    def _hash_partition(
        self,
        data: List[Any],
        key_fn: Optional[Callable[[Any], str]] = None
    ) -> List[Partition]:
        """Hash-based partitioning."""
        buckets: Dict[int, List[Any]] = defaultdict(list)
        key_fn = key_fn or (lambda x: str(x))
        
        for item in data:
            key = key_fn(item)
            bucket = int(hashlib.md5(str(key).encode()).hexdigest(), 16) % self.initial_partitions
            buckets[bucket].append(item)
            
        partitions = []
        for bucket_id, items in buckets.items():
            partition = Partition(
                partition_id=f"hash_{bucket_id}",
                data=items,
                min_size=self.min_partition_size,
                max_size=self.max_partition_size,
                split_threshold=self.split_threshold,
                merge_threshold=self.merge_threshold
            )
            partitions.append(partition)
            self._partitions[partition.partition_id] = partition
            
        return partitions
        
    def _range_partition(self, data: List[Any]) -> List[Partition]:
        """Range-based partitioning."""
        sorted_data = sorted(data, key=str)
        partition_size = max(1, len(data) // self.initial_partitions)
        
        partitions = []
        for i in range(self.initial_partitions):
            start = i * partition_size
            end = start + partition_size if i < self.initial_partitions - 1 else len(data)
            
            if start >= len(data):
                break
                
            partition = Partition(
                partition_id=f"range_{i}",
                data=sorted_data[start:end],
                min_size=self.min_partition_size,
                max_size=self.max_partition_size,
                split_threshold=self.split_threshold,
                merge_threshold=self.merge_threshold
            )
            partitions.append(partition)
            self._partitions[partition.partition_id] = partition
            
        return partitions
        
    def _round_robin_partition(self, data: List[Any]) -> List[Partition]:
        """Round-robin partitioning."""
        buckets: Dict[int, List[Any]] = defaultdict(list)
        
        for i, item in enumerate(data):
            bucket = i % self.initial_partitions
            buckets[bucket].append(item)
            
        partitions = []
        for bucket_id, items in buckets.items():
            partition = Partition(
                partition_id=f"rr_{bucket_id}",
                data=items,
                min_size=self.min_partition_size,
                max_size=self.max_partition_size,
                split_threshold=self.split_threshold,
                merge_threshold=self.merge_threshold
            )
            partitions.append(partition)
            self._partitions[partition.partition_id] = partition
            
        return partitions
        
    def _fitness_partition(self, data: List[Any]) -> List[Partition]:
        """Fitness-based adaptive partitioning."""
        partition_size = max(1, len(data) // self.initial_partitions)
        
        partitions = []
        for i in range(self.initial_partitions):
            start = i * partition_size
            end = start + partition_size if i < self.initial_partitions - 1 else len(data)
            
            if start >= len(data):
                break
                
            partition = Partition(
                partition_id=f"adaptive_{i}",
                data=data[start:end],
                min_size=self.min_partition_size,
                max_size=self.max_partition_size,
                split_threshold=self.split_threshold,
                merge_threshold=self.merge_threshold
            )
            partitions.append(partition)
            self._partitions[partition.partition_id] = partition
            
        return partitions
        
    def record_fitness(
        self,
        partition_id: str,
        fitness: float,
        records_processed: int = 0,
        processing_time_ms: float = 0.0
    ) -> None:
        """
        Record fitness feedback for a partition.
        
        This drives the adaptive behavior.
        """
        with self._lock:
            partition = self._partitions.get(partition_id)
            if partition:
                partition.metrics.record_fitness(fitness)
                partition.metrics.records_processed += records_processed
                partition.metrics.processing_time_ms += processing_time_ms
                partition.metrics.last_processed = time.time()
                
    def adapt(self) -> List[Partition]:
        """
        Adapt partitions based on fitness feedback.
        
        Splits low-fitness partitions and merges high-fitness ones.
        Returns the new set of active partitions.
        """
        with self._lock:
            new_partitions = []
            partitions_to_remove = []
            
            active = [p for p in self._partitions.values() 
                     if p.state == PartitionState.ACTIVE]
            
            # Check for splits
            for partition in active:
                if partition.should_split and self._splitter.can_split(partition):
                    left, right = partition.split()
                    self._partitions[left.partition_id] = left
                    self._partitions[right.partition_id] = right
                    new_partitions.extend([left, right])
                    partitions_to_remove.append(partition.partition_id)
                else:
                    new_partitions.append(partition)
                    
            # Check for merges
            merge_candidates = self._merger.find_merge_candidates(new_partitions)
            merged_ids = set()
            
            for p1, p2 in merge_candidates:
                if p1.partition_id not in merged_ids and p2.partition_id not in merged_ids:
                    merged = Partition.merge(p1, p2)
                    self._partitions[merged.partition_id] = merged
                    merged_ids.add(p1.partition_id)
                    merged_ids.add(p2.partition_id)
                    
            # Update active partition list
            final_partitions = []
            for p in new_partitions:
                if p.partition_id not in merged_ids:
                    final_partitions.append(p)
                    
            # Add merged partitions
            for p in self._partitions.values():
                if p.state == PartitionState.ACTIVE and p.partition_id not in [fp.partition_id for fp in final_partitions]:
                    # This is a newly merged partition
                    if '+' in p.partition_id:  # Merged partitions have + in ID
                        final_partitions.append(p)
                        
            return final_partitions
            
    def get_partition(self, partition_id: str) -> Optional[Partition]:
        """Get partition by ID."""
        with self._lock:
            return self._partitions.get(partition_id)
            
    def get_active_partitions(self) -> List[Partition]:
        """Get all active partitions."""
        with self._lock:
            return [p for p in self._partitions.values() 
                   if p.state == PartitionState.ACTIVE]
                   
    def get_statistics(self) -> Dict[str, Any]:
        """Get partitioning statistics."""
        with self._lock:
            active = [p for p in self._partitions.values() 
                     if p.state == PartitionState.ACTIVE]
            
            return {
                'total_partitions': len(active),
                'total_records': sum(p.size for p in active),
                'average_fitness': sum(p.metrics.average_fitness for p in active) / len(active) if active else 0,
                'partitions_to_split': sum(1 for p in active if p.should_split),
                'partitions_to_merge': sum(1 for p in active if p.should_merge),
                'size_distribution': {
                    'min': min(p.size for p in active) if active else 0,
                    'max': max(p.size for p in active) if active else 0,
                    'avg': sum(p.size for p in active) / len(active) if active else 0
                }
            }
