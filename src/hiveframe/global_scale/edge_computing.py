"""
Edge Computing - Process data at the source with local swarms

Enables HiveFrame to run on edge devices with intelligent synchronization
to cloud clusters, similar to how bee scouts operate independently while
staying connected to the main colony.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import hashlib


class EdgeNodeType(Enum):
    """Type of edge device"""
    GATEWAY = "gateway"  # Aggregation point
    SENSOR = "sensor"    # Data source
    COMPUTE = "compute"  # Processing node
    HYBRID = "hybrid"    # Multi-purpose


class SyncStrategy(Enum):
    """Data synchronization strategy"""
    IMMEDIATE = "immediate"      # Sync every change
    PERIODIC = "periodic"        # Sync on schedule
    THRESHOLD = "threshold"      # Sync when buffer fills
    ADAPTIVE = "adaptive"        # AI-driven sync timing


@dataclass
class EdgeNode:
    """Represents an edge computing node"""
    node_id: str
    node_type: EdgeNodeType
    location: str
    capacity: int = 10  # Lower than cloud nodes
    current_load: int = 0
    is_online: bool = True
    battery_level: float = 1.0  # For battery-powered devices
    data_buffer: List[Any] = field(default_factory=list)
    last_sync: float = field(default_factory=time.time)
    cloud_affinity: Optional[str] = None  # Preferred cloud region
    
    def can_accept_task(self) -> bool:
        """Check if node can accept more work"""
        return (
            self.is_online
            and self.battery_level > 0.2
            and self.current_load < self.capacity
        )
    
    def fitness(self) -> float:
        """Calculate node fitness for task assignment"""
        if not self.is_online or self.battery_level < 0.2:
            return 0.0
        
        utilization = self.current_load / self.capacity if self.capacity > 0 else 1.0
        capacity_score = 1.0 - utilization
        
        # Factor in battery for mobile devices
        battery_score = self.battery_level
        
        return capacity_score * 0.7 + battery_score * 0.3


class EdgeNodeManager:
    """
    Manages a swarm of edge computing nodes.
    
    Edge nodes operate like scout bees - they can work independently
    but coordinate with the main colony when beneficial.
    """
    
    def __init__(self, max_nodes: int = 1000):
        self.nodes: Dict[str, EdgeNode] = {}
        self.max_nodes = max_nodes
        self.node_groups: Dict[str, List[str]] = {}  # Group nodes by location
        
    def register_edge_node(
        self,
        node_id: str,
        node_type: EdgeNodeType,
        location: str,
        capacity: int = 10,
        cloud_affinity: Optional[str] = None,
    ) -> bool:
        """Register a new edge node"""
        if len(self.nodes) >= self.max_nodes:
            return False
        
        if node_id in self.nodes:
            return False
        
        node = EdgeNode(
            node_id=node_id,
            node_type=node_type,
            location=location,
            capacity=capacity,
            cloud_affinity=cloud_affinity,
        )
        
        self.nodes[node_id] = node
        
        # Add to location group
        if location not in self.node_groups:
            self.node_groups[location] = []
        self.node_groups[location].append(node_id)
        
        return True
    
    def unregister_edge_node(self, node_id: str) -> bool:
        """Remove an edge node from management"""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        
        # Remove from location group
        if node.location in self.node_groups:
            self.node_groups[node.location].remove(node_id)
            if not self.node_groups[node.location]:
                del self.node_groups[node.location]
        
        del self.nodes[node_id]
        return True
    
    def assign_task_to_edge(
        self,
        task_id: str,
        preferred_location: Optional[str] = None,
        node_type: Optional[EdgeNodeType] = None,
    ) -> Optional[str]:
        """
        Assign a task to an optimal edge node using swarm intelligence.
        
        Returns the selected node ID or None if no suitable node found.
        """
        # Filter available nodes
        candidates = [
            node for node in self.nodes.values()
            if node.can_accept_task()
        ]
        
        if not candidates:
            return None
        
        # Apply filters
        if preferred_location:
            location_candidates = [
                n for n in candidates
                if n.location == preferred_location
            ]
            if location_candidates:
                candidates = location_candidates
        
        if node_type:
            type_candidates = [
                n for n in candidates
                if n.node_type == node_type or n.node_type == EdgeNodeType.HYBRID
            ]
            if type_candidates:
                candidates = type_candidates
        
        # Select based on fitness
        import random
        fitness_scores = [n.fitness() for n in candidates]
        total_fitness = sum(fitness_scores)
        
        if total_fitness == 0:
            selected = random.choice(candidates)
        else:
            probabilities = [f / total_fitness for f in fitness_scores]
            selected = random.choices(candidates, weights=probabilities)[0]
        
        # Assign task
        selected.current_load += 1
        return selected.node_id
    
    def complete_task(self, node_id: str, data: Optional[Any] = None) -> None:
        """Mark task complete and optionally buffer data for sync"""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.current_load = max(0, node.current_load - 1)
        
        if data is not None:
            node.data_buffer.append(data)
    
    def update_node_status(
        self,
        node_id: str,
        is_online: Optional[bool] = None,
        battery_level: Optional[float] = None,
    ) -> bool:
        """Update node operational status"""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        
        if is_online is not None:
            node.is_online = is_online
        
        if battery_level is not None:
            node.battery_level = max(0.0, min(1.0, battery_level))
        
        return True
    
    def get_nodes_by_location(self, location: str) -> List[str]:
        """Get all nodes at a specific location"""
        return self.node_groups.get(location, [])
    
    def get_edge_stats(self) -> Dict:
        """Get statistics about edge nodes"""
        online_nodes = sum(1 for n in self.nodes.values() if n.is_online)
        total_capacity = sum(n.capacity for n in self.nodes.values())
        total_load = sum(n.current_load for n in self.nodes.values())
        
        avg_battery = (
            sum(n.battery_level for n in self.nodes.values()) / len(self.nodes)
            if self.nodes else 0.0
        )
        
        total_buffered = sum(len(n.data_buffer) for n in self.nodes.values())
        
        return {
            "total_nodes": len(self.nodes),
            "online_nodes": online_nodes,
            "total_capacity": total_capacity,
            "current_load": total_load,
            "average_battery": avg_battery,
            "buffered_items": total_buffered,
            "locations": len(self.node_groups),
        }


class EdgeCloudSync:
    """
    Synchronizes data between edge nodes and cloud clusters.
    
    Uses bee-inspired communication patterns where edge nodes
    act like foragers bringing data back to the hive.
    """
    
    def __init__(self, default_strategy: SyncStrategy = SyncStrategy.ADAPTIVE):
        self.default_strategy = default_strategy
        self.sync_history: List[Dict] = []
        self.node_strategies: Dict[str, SyncStrategy] = {}
        self.sync_thresholds: Dict[str, int] = {}  # Buffer size thresholds
        self.sync_intervals: Dict[str, float] = {}  # Time-based intervals
        
    def configure_node_sync(
        self,
        node_id: str,
        strategy: SyncStrategy,
        threshold: Optional[int] = None,
        interval: Optional[float] = None,
    ) -> None:
        """Configure sync strategy for a specific node"""
        self.node_strategies[node_id] = strategy
        
        if threshold is not None:
            self.sync_thresholds[node_id] = threshold
        
        if interval is not None:
            self.sync_intervals[node_id] = interval
    
    def should_sync(self, node: EdgeNode) -> bool:
        """
        Determine if a node should sync now using swarm intelligence.
        
        Adaptive strategy learns optimal sync patterns like bees
        learn optimal foraging times.
        """
        strategy = self.node_strategies.get(node.node_id, self.default_strategy)
        
        if strategy == SyncStrategy.IMMEDIATE:
            return len(node.data_buffer) > 0
        
        elif strategy == SyncStrategy.PERIODIC:
            interval = self.sync_intervals.get(node.node_id, 300)  # 5 min default
            return time.time() - node.last_sync >= interval
        
        elif strategy == SyncStrategy.THRESHOLD:
            threshold = self.sync_thresholds.get(node.node_id, 100)
            return len(node.data_buffer) >= threshold
        
        elif strategy == SyncStrategy.ADAPTIVE:
            # Adaptive: combine multiple factors
            time_since_sync = time.time() - node.last_sync
            buffer_ratio = len(node.data_buffer) / 100
            battery_ok = node.battery_level > 0.5
            
            # Score: higher means should sync
            score = buffer_ratio + (time_since_sync / 600)
            
            # Only sync if battery is sufficient
            return score > 0.7 and battery_ok
        
        return False
    
    def sync_node(
        self,
        node: EdgeNode,
        cloud_region: Optional[str] = None,
    ) -> Dict:
        """
        Synchronize node data to cloud.
        
        Returns sync result with statistics.
        """
        if not node.data_buffer:
            return {
                "success": True,
                "items_synced": 0,
                "node_id": node.node_id,
                "target_region": cloud_region or node.cloud_affinity,
            }
        
        target = cloud_region or node.cloud_affinity or "default"
        
        # Simulate sync (in real implementation, would send to cloud)
        items_synced = len(node.data_buffer)
        
        # Record sync event
        sync_event = {
            "timestamp": time.time(),
            "node_id": node.node_id,
            "target_region": target,
            "items_synced": items_synced,
            "battery_before": node.battery_level,
        }
        
        # Clear buffer and update last sync time
        node.data_buffer.clear()
        node.last_sync = time.time()
        
        # Simulate battery drain from sync
        battery_drain = min(0.01 * items_synced, 0.1)
        node.battery_level = max(0.0, node.battery_level - battery_drain)
        
        sync_event["battery_after"] = node.battery_level
        self.sync_history.append(sync_event)
        
        return {
            "success": True,
            "items_synced": items_synced,
            "node_id": node.node_id,
            "target_region": target,
            "battery_remaining": node.battery_level,
        }
    
    def get_sync_stats(self) -> Dict:
        """Get synchronization statistics"""
        if not self.sync_history:
            return {
                "total_syncs": 0,
                "total_items": 0,
                "average_batch_size": 0,
            }
        
        total_items = sum(s["items_synced"] for s in self.sync_history)
        avg_batch = total_items / len(self.sync_history)
        
        return {
            "total_syncs": len(self.sync_history),
            "total_items": total_items,
            "average_batch_size": avg_batch,
        }


class OfflineOperationSupport:
    """
    Enables edge nodes to operate offline and sync when reconnected.
    
    Similar to how bees can operate independently when separated
    from the colony and rejoin later.
    """
    
    def __init__(self, max_offline_buffer: int = 10000):
        self.max_offline_buffer = max_offline_buffer
        self.offline_nodes: Set[str] = set()
        self.conflict_resolution_strategy = "last_write_wins"
        self.pending_operations: Dict[str, List[Dict]] = {}
        
    def node_went_offline(self, node_id: str) -> None:
        """Mark a node as offline"""
        self.offline_nodes.add(node_id)
        if node_id not in self.pending_operations:
            self.pending_operations[node_id] = []
    
    def node_came_online(self, node_id: str) -> bool:
        """Mark a node as back online"""
        if node_id in self.offline_nodes:
            self.offline_nodes.remove(node_id)
            return True
        return False
    
    def buffer_offline_operation(
        self,
        node_id: str,
        operation: Dict,
    ) -> bool:
        """
        Buffer an operation performed while offline.
        
        Returns True if buffered, False if buffer is full.
        """
        if node_id not in self.pending_operations:
            self.pending_operations[node_id] = []
        
        buffer = self.pending_operations[node_id]
        
        if len(buffer) >= self.max_offline_buffer:
            return False
        
        operation["timestamp"] = time.time()
        buffer.append(operation)
        return True
    
    def reconcile_offline_operations(
        self,
        node_id: str,
    ) -> Dict:
        """
        Reconcile operations performed while offline.
        
        Returns reconciliation results.
        """
        if node_id not in self.pending_operations:
            return {
                "node_id": node_id,
                "operations_reconciled": 0,
                "conflicts_resolved": 0,
            }
        
        operations = self.pending_operations[node_id]
        conflicts_resolved = 0
        
        # Group operations by data_id to detect conflicts
        by_data_id: Dict[str, List[Dict]] = {}
        for op in operations:
            data_id = op.get("data_id")
            if data_id:
                if data_id not in by_data_id:
                    by_data_id[data_id] = []
                by_data_id[data_id].append(op)
        
        # Detect and resolve conflicts
        for data_id, ops in by_data_id.items():
            if len(ops) > 1:
                conflicts_resolved += 1
                # Apply conflict resolution strategy
                # (simplified - real implementation would be more sophisticated)
        
        # Clear pending operations
        operations_count = len(operations)
        self.pending_operations[node_id] = []
        
        return {
            "node_id": node_id,
            "operations_reconciled": operations_count,
            "conflicts_resolved": conflicts_resolved,
        }
    
    def get_offline_stats(self) -> Dict:
        """Get statistics about offline operations"""
        total_pending = sum(
            len(ops) for ops in self.pending_operations.values()
        )
        
        return {
            "offline_nodes": len(self.offline_nodes),
            "total_pending_operations": total_pending,
            "nodes_with_pending": len(self.pending_operations),
        }
