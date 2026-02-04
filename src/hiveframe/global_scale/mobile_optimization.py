"""
5G/6G Optimization - Ultra-low latency mobile data processing

Optimizes HiveFrame for next-generation mobile networks with
network slicing and handoff handling, inspired by how bees
maintain coordination while in flight.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import random


class NetworkSliceType(Enum):
    """Types of 5G/6G network slices"""
    URLLC = "urllc"  # Ultra-Reliable Low-Latency Communications
    EMBB = "embb"    # Enhanced Mobile Broadband
    MMTC = "mmtc"    # Massive Machine Type Communications


class MobilityState(Enum):
    """Mobile device mobility state"""
    STATIONARY = "stationary"
    WALKING = "walking"
    DRIVING = "driving"
    HIGH_SPEED = "high_speed"  # Train, plane


@dataclass
class MobileDevice:
    """Represents a mobile device in the swarm"""
    device_id: str
    current_cell: str
    mobility_state: MobilityState = MobilityState.STATIONARY
    signal_strength: float = 1.0  # 0.0 to 1.0
    slice_type: NetworkSliceType = NetworkSliceType.EMBB
    active_tasks: List[str] = field(default_factory=list)
    handoff_count: int = 0
    
    def can_handle_latency_critical(self) -> bool:
        """Check if device can handle latency-critical tasks"""
        return (
            self.signal_strength > 0.7
            and self.mobility_state in (MobilityState.STATIONARY, MobilityState.WALKING)
            and self.slice_type == NetworkSliceType.URLLC
        )


class MobileAwareScheduler:
    """
    Schedules tasks with awareness of mobile network conditions.
    
    Like how bees adjust foraging patterns based on weather and
    daylight, this scheduler adapts to network conditions.
    """
    
    def __init__(self):
        self.devices: Dict[str, MobileDevice] = {}
        self.task_requirements: Dict[str, Dict] = {}
        self.scheduled_tasks = 0
        
    def register_device(
        self,
        device_id: str,
        current_cell: str,
        slice_type: NetworkSliceType = NetworkSliceType.EMBB,
    ) -> bool:
        """Register a mobile device"""
        if device_id in self.devices:
            return False
        
        device = MobileDevice(
            device_id=device_id,
            current_cell=current_cell,
            slice_type=slice_type,
        )
        
        self.devices[device_id] = device
        return True
    
    def schedule_task(
        self,
        task_id: str,
        latency_sensitive: bool = False,
        bandwidth_heavy: bool = False,
        preferred_device: Optional[str] = None,
    ) -> Optional[str]:
        """
        Schedule a task to an appropriate mobile device.
        
        Returns assigned device ID or None if no suitable device.
        """
        # Store task requirements
        self.task_requirements[task_id] = {
            "latency_sensitive": latency_sensitive,
            "bandwidth_heavy": bandwidth_heavy,
        }
        
        # Filter suitable devices
        candidates = []
        
        for device in self.devices.values():
            # Check basic requirements
            if latency_sensitive and not device.can_handle_latency_critical():
                continue
            
            if bandwidth_heavy and device.slice_type != NetworkSliceType.EMBB:
                continue
            
            # Check capacity (max 5 concurrent tasks per device)
            if len(device.active_tasks) >= 5:
                continue
            
            candidates.append(device)
        
        if not candidates:
            return None
        
        # Prefer specified device if available
        if preferred_device and preferred_device in [d.device_id for d in candidates]:
            selected = self.devices[preferred_device]
        else:
            # Select based on signal strength and current load
            selected = max(
                candidates,
                key=lambda d: d.signal_strength * (1.0 - len(d.active_tasks) / 5.0)
            )
        
        # Assign task
        selected.active_tasks.append(task_id)
        self.scheduled_tasks += 1
        
        return selected.device_id
    
    def complete_task(self, device_id: str, task_id: str) -> bool:
        """Mark a task as complete"""
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        if task_id in device.active_tasks:
            device.active_tasks.remove(task_id)
            return True
        
        return False
    
    def update_device_state(
        self,
        device_id: str,
        mobility_state: Optional[MobilityState] = None,
        signal_strength: Optional[float] = None,
        current_cell: Optional[str] = None,
    ) -> bool:
        """Update device state"""
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        
        if mobility_state is not None:
            device.mobility_state = mobility_state
        
        if signal_strength is not None:
            device.signal_strength = max(0.0, min(1.0, signal_strength))
        
        if current_cell is not None:
            if current_cell != device.current_cell:
                device.handoff_count += 1
            device.current_cell = current_cell
        
        return True
    
    def get_scheduler_stats(self) -> Dict:
        """Get scheduler statistics"""
        total_tasks = sum(len(d.active_tasks) for d in self.devices.values())
        avg_signal = (
            sum(d.signal_strength for d in self.devices.values()) / len(self.devices)
            if self.devices else 0.0
        )
        
        return {
            "total_devices": len(self.devices),
            "active_tasks": total_tasks,
            "average_signal_strength": avg_signal,
            "total_scheduled": self.scheduled_tasks,
        }


class NetworkSliceIntegration:
    """
    Integrates with 5G/6G network slicing for QoS guarantees.
    
    Network slices are like specialized bee roles - each optimized
    for specific types of tasks.
    """
    
    def __init__(self):
        self.slices: Dict[NetworkSliceType, Dict] = {
            NetworkSliceType.URLLC: {
                "max_latency_ms": 1,
                "reliability": 0.999999,
                "bandwidth_mbps": 10,
                "allocated_devices": set(),
            },
            NetworkSliceType.EMBB: {
                "max_latency_ms": 50,
                "reliability": 0.999,
                "bandwidth_mbps": 1000,
                "allocated_devices": set(),
            },
            NetworkSliceType.MMTC: {
                "max_latency_ms": 1000,
                "reliability": 0.99,
                "bandwidth_mbps": 1,
                "allocated_devices": set(),
            },
        }
        self.slice_usage: Dict[NetworkSliceType, int] = {
            slice_type: 0 for slice_type in NetworkSliceType
        }
    
    def allocate_slice(
        self,
        device_id: str,
        slice_type: NetworkSliceType,
    ) -> bool:
        """Allocate a network slice to a device"""
        if slice_type not in self.slices:
            return False
        
        self.slices[slice_type]["allocated_devices"].add(device_id)
        return True
    
    def deallocate_slice(
        self,
        device_id: str,
        slice_type: NetworkSliceType,
    ) -> bool:
        """Deallocate a network slice from a device"""
        if slice_type not in self.slices:
            return False
        
        devices = self.slices[slice_type]["allocated_devices"]
        if device_id in devices:
            devices.remove(device_id)
            return True
        
        return False
    
    def get_slice_requirements(
        self,
        slice_type: NetworkSliceType,
    ) -> Optional[Dict]:
        """Get QoS requirements for a slice type"""
        return self.slices.get(slice_type)
    
    def record_slice_usage(
        self,
        slice_type: NetworkSliceType,
        data_mb: float,
    ) -> None:
        """Record data usage for a slice"""
        if slice_type in self.slice_usage:
            self.slice_usage[slice_type] += int(data_mb)
    
    def get_slice_stats(self) -> Dict:
        """Get statistics for all network slices"""
        stats = {}
        
        for slice_type, config in self.slices.items():
            stats[slice_type.value] = {
                "allocated_devices": len(config["allocated_devices"]),
                "max_latency_ms": config["max_latency_ms"],
                "bandwidth_mbps": config["bandwidth_mbps"],
                "data_usage_mb": self.slice_usage[slice_type],
            }
        
        return stats


class HandoffHandler:
    """
    Handles device handoffs between cells with minimal disruption.
    
    Like how bees maintain awareness of their location when moving
    between different areas of their foraging range.
    """
    
    def __init__(self):
        self.cells: Dict[str, Set[str]] = {}  # cell_id -> device_ids
        self.handoff_history: List[Dict] = []
        self.migration_buffer: Dict[str, Dict] = {}  # device_id -> migration state
        
    def register_cell(self, cell_id: str) -> bool:
        """Register a new cell tower"""
        if cell_id in self.cells:
            return False
        
        self.cells[cell_id] = set()
        return True
    
    def initiate_handoff(
        self,
        device_id: str,
        source_cell: str,
        target_cell: str,
        active_tasks: List[str],
    ) -> Dict:
        """
        Initiate handoff of a device from one cell to another.
        
        Uses make-before-break strategy to minimize disruption.
        """
        if source_cell not in self.cells or target_cell not in self.cells:
            return {"success": False, "reason": "cell_not_found"}
        
        # Start migration
        migration_id = f"mig_{device_id}_{int(time.time())}"
        
        migration_state = {
            "migration_id": migration_id,
            "device_id": device_id,
            "source_cell": source_cell,
            "target_cell": target_cell,
            "active_tasks": active_tasks.copy(),
            "start_time": time.time(),
            "phase": "preparation",
        }
        
        self.migration_buffer[device_id] = migration_state
        
        # Phase 1: Establish connection with target cell
        migration_state["phase"] = "connecting"
        
        # Phase 2: Transfer state
        migration_state["phase"] = "transferring"
        
        # Phase 3: Switch primary connection
        if device_id in self.cells[source_cell]:
            self.cells[source_cell].remove(device_id)
        self.cells[target_cell].add(device_id)
        
        migration_state["phase"] = "completed"
        migration_state["end_time"] = time.time()
        migration_state["duration_ms"] = (
            migration_state["end_time"] - migration_state["start_time"]
        ) * 1000
        
        # Record handoff
        self.handoff_history.append({
            "device_id": device_id,
            "source_cell": source_cell,
            "target_cell": target_cell,
            "duration_ms": migration_state["duration_ms"],
            "tasks_migrated": len(active_tasks),
            "timestamp": time.time(),
        })
        
        # Clean up migration state
        del self.migration_buffer[device_id]
        
        return {
            "success": True,
            "migration_id": migration_id,
            "duration_ms": migration_state["duration_ms"],
            "tasks_migrated": len(active_tasks),
        }
    
    def get_cell_devices(self, cell_id: str) -> Set[str]:
        """Get all devices currently connected to a cell"""
        return self.cells.get(cell_id, set())
    
    def get_handoff_stats(self) -> Dict:
        """Get handoff statistics"""
        if not self.handoff_history:
            return {
                "total_handoffs": 0,
                "average_duration_ms": 0,
                "active_migrations": 0,
            }
        
        total_duration = sum(h["duration_ms"] for h in self.handoff_history)
        avg_duration = total_duration / len(self.handoff_history)
        
        return {
            "total_handoffs": len(self.handoff_history),
            "average_duration_ms": avg_duration,
            "active_migrations": len(self.migration_buffer),
        }
    
    def predict_handoff(
        self,
        device_id: str,
        current_cell: str,
        signal_strength: float,
        mobility_state: MobilityState,
    ) -> Optional[str]:
        """
        Predict if handoff will be needed soon.
        
        Returns likely target cell ID or None.
        """
        # Simple prediction based on signal strength
        if signal_strength < 0.3:
            # Signal is weak, handoff likely needed
            # In real implementation, would use network topology
            # and device movement patterns
            return f"cell_{random.randint(1, 10)}"
        
        if mobility_state == MobilityState.HIGH_SPEED and signal_strength < 0.5:
            # High speed + moderate signal = likely handoff
            return f"cell_{random.randint(1, 10)}"
        
        return None
