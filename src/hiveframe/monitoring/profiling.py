"""
HiveFrame Performance Profiling
================================
Profile performance of colony operations.

Tracks timing, resource usage, and bottlenecks.
"""

import time
import threading
import statistics
from collections import defaultdict
from typing import Dict, List
from contextlib import contextmanager


class PerformanceProfiler:
    """
    Profile performance of colony operations.
    
    Tracks timing, resource usage, and bottlenecks.
    """
    
    def __init__(self):
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
    @contextmanager
    def profile(self, operation: str):
        """Profile a block of code."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                self._timings[operation].append(elapsed)
                # Keep only recent timings
                if len(self._timings[operation]) > 1000:
                    self._timings[operation] = self._timings[operation][-1000:]
                    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get timing statistics for an operation."""
        with self._lock:
            timings = self._timings.get(operation, [])
            
        if not timings:
            return {
                'count': 0,
                'mean': 0,
                'min': 0,
                'max': 0,
                'p50': 0,
                'p95': 0,
                'p99': 0
            }
            
        timings = sorted(timings)
        return {
            'count': len(timings),
            'mean': statistics.mean(timings),
            'min': min(timings),
            'max': max(timings),
            'p50': timings[int(len(timings) * 0.50)],
            'p95': timings[int(len(timings) * 0.95)] if len(timings) > 20 else max(timings),
            'p99': timings[int(len(timings) * 0.99)] if len(timings) > 100 else max(timings)
        }
        
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get stats for all profiled operations."""
        with self._lock:
            operations = list(self._timings.keys())
        return {op: self.get_stats(op) for op in operations}
        
    def report(self) -> str:
        """Generate human-readable performance report."""
        stats = self.get_all_stats()
        
        lines = ["Performance Report", "=" * 50]
        
        for op, s in sorted(stats.items(), key=lambda x: -x[1]['mean']):
            lines.append(f"\n{op}:")
            lines.append(f"  Count: {s['count']}")
            lines.append(f"  Mean:  {s['mean']*1000:.2f}ms")
            lines.append(f"  Min:   {s['min']*1000:.2f}ms")
            lines.append(f"  Max:   {s['max']*1000:.2f}ms")
            lines.append(f"  P95:   {s['p95']*1000:.2f}ms")
            
        return "\n".join(lines)


# Global profiler
_default_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the default profiler."""
    return _default_profiler
