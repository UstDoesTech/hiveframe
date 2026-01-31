"""
GPU Cell - GPU-accelerated Computation
======================================

Support for GPU-accelerated notebook cells using bee-inspired task allocation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class GPUDeviceType(Enum):
    """GPU device types."""
    CUDA = "cuda"
    ROCm = "rocm"
    Metal = "metal"
    CPU_FALLBACK = "cpu"


@dataclass
class GPUDevice:
    """GPU device information."""
    device_id: int
    device_type: GPUDeviceType
    name: str
    memory_total: int  # bytes
    memory_available: int  # bytes
    compute_capability: Optional[str] = None


@dataclass
class GPUTask:
    """A GPU computation task."""
    task_id: str
    code: str
    device: GPUDevice
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None


class GPUCell:
    """
    GPU-accelerated notebook cell.
    
    Manages GPU resources using bee-inspired allocation where GPUs
    are treated like flower patches - bees (tasks) are assigned based
    on availability and workload.
    
    Example:
        gpu_cell = GPUCell()
        
        # List available GPUs
        devices = gpu_cell.list_devices()
        print(f"Available GPUs: {len(devices)}")
        
        # Execute GPU-accelerated code
        result = gpu_cell.execute(\"\"\"
        import numpy as np
        # GPU-accelerated matrix multiplication
        a = np.random.rand(1000, 1000)
        b = np.random.rand(1000, 1000)
        c = np.dot(a, b)
        c.sum()
        \"\"\")
    """
    
    def __init__(self, auto_detect: bool = True):
        """
        Initialize GPU cell manager.
        
        Args:
            auto_detect: Automatically detect available GPUs
        """
        self.devices: List[GPUDevice] = []
        self.tasks: Dict[str, GPUTask] = {}
        
        if auto_detect:
            self._detect_devices()
    
    def _detect_devices(self) -> None:
        """Detect available GPU devices."""
        # Try to detect CUDA devices
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    device = GPUDevice(
                        device_id=i,
                        device_type=GPUDeviceType.CUDA,
                        name=props.name,
                        memory_total=props.total_memory,
                        memory_available=props.total_memory,
                        compute_capability=f"{props.major}.{props.minor}"
                    )
                    self.devices.append(device)
        except ImportError:
            pass
        
        # Fallback to CPU
        if not self.devices:
            device = GPUDevice(
                device_id=0,
                device_type=GPUDeviceType.CPU_FALLBACK,
                name="CPU (Fallback)",
                memory_total=0,
                memory_available=0
            )
            self.devices.append(device)
    
    def list_devices(self) -> List[GPUDevice]:
        """List available GPU devices."""
        return self.devices.copy()
    
    def select_device(self) -> GPUDevice:
        """
        Select a GPU device using bee-inspired load balancing.
        
        Similar to how bees select flower patches based on quality,
        this selects GPUs based on available memory.
        
        Returns:
            Selected GPUDevice
        """
        if not self.devices:
            raise RuntimeError("No GPU devices available")
        
        # Select device with most available memory
        return max(self.devices, key=lambda d: d.memory_available)
    
    def execute(
        self,
        code: str,
        device: Optional[GPUDevice] = None,
        task_id: Optional[str] = None
    ) -> GPUTask:
        """
        Execute GPU-accelerated code.
        
        WARNING: This is a simplified implementation for Phase 3.
        Production use requires proper sandboxing and security measures.
        
        Args:
            code: Code to execute
            device: Optional specific device (auto-selected if None)
            task_id: Optional task identifier
            
        Returns:
            GPUTask with execution results
        """
        if device is None:
            device = self.select_device()
        
        if task_id is None:
            task_id = f"gpu_task_{len(self.tasks)}"
        
        task = GPUTask(
            task_id=task_id,
            code=code,
            device=device,
            start_time=datetime.now()
        )
        
        try:
            # Execute code (simplified - in reality would set GPU context)
            task.status = "running"
            
            # For now, execute as regular Python
            # In real implementation, would set CUDA device context
            # NOTE: exec on user code is a security risk - needs sandboxing for production
            context = {}
            exec(code, context)
            
            # Get result (last expression or variable named 'result')
            result = context.get('result', None)
            
            task.status = "success"
            task.result = result
        
        except Exception as e:
            task.status = "error"
            task.error = str(e)
        
        finally:
            task.end_time = datetime.now()
        
        self.tasks[task_id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[GPUTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def list_tasks(self) -> List[GPUTask]:
        """List all tasks."""
        return list(self.tasks.values())
    
    def get_device_info(self, device_id: int) -> Optional[GPUDevice]:
        """Get information about a specific device."""
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None
