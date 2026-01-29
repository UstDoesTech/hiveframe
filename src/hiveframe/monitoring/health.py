"""
HiveFrame Health Monitoring
===========================
Colony health monitoring using bee-inspired metrics.

Health indicators:
- Temperature: Overall colony load/stress
- Worker distribution: Balance across roles
- Waggle dance frequency: Communication activity
- Pheromone levels: Coordination signals
- Abandonment rate: Food source quality
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

from .metrics import get_registry


@dataclass
class WorkerHealthSnapshot:
    """Point-in-time health snapshot of a worker."""
    worker_id: str
    role: str
    processed_count: int
    error_count: int
    last_activity: float
    current_load: float
    avg_latency: float
    status: str  # 'healthy', 'degraded', 'unhealthy', 'dead'


@dataclass
class ColonyHealthReport:
    """Overall colony health report."""
    timestamp: float
    total_workers: int
    healthy_workers: int
    degraded_workers: int
    unhealthy_workers: int
    dead_workers: int
    overall_status: str
    temperature: float
    throughput: float
    error_rate: float
    worker_snapshots: List[WorkerHealthSnapshot]
    alerts: List[str]


class ColonyHealthMonitor:
    """
    Monitors colony health using bee-inspired metrics.
    
    Health indicators:
    - Temperature: Overall colony load/stress
    - Worker distribution: Balance across roles
    - Waggle dance frequency: Communication activity
    - Pheromone levels: Coordination signals
    - Abandonment rate: Food source quality
    """
    
    def __init__(
        self,
        colony,  # ColonyState from core
        check_interval: float = 5.0,
        unhealthy_threshold: float = 0.7,
        dead_threshold: float = 30.0  # Seconds without activity
    ):
        self.colony = colony
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.dead_threshold = dead_threshold
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._history: deque = deque(maxlen=100)
        self._alerts: List[str] = []
        self._lock = threading.Lock()
        
        # Metrics
        self._registry = get_registry()
        self._temperature_gauge = self._registry.gauge(
            "colony_temperature", "Current colony temperature (load)"
        )
        self._worker_gauge = self._registry.gauge(
            "workers_by_status", "Workers by health status"
        )
        self._throughput_gauge = self._registry.gauge(
            "colony_throughput", "Records processed per second"
        )
        self._error_rate_gauge = self._registry.gauge(
            "colony_error_rate", "Error rate (errors per record)"
        )
        
    def start(self) -> None:
        """Start background health monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                report = self._collect_health()
                
                with self._lock:
                    self._history.append(report)
                    
                # Update metrics
                self._temperature_gauge.set(report.temperature)
                self._worker_gauge.set(report.healthy_workers, {'status': 'healthy'})
                self._worker_gauge.set(report.degraded_workers, {'status': 'degraded'})
                self._worker_gauge.set(report.unhealthy_workers, {'status': 'unhealthy'})
                self._worker_gauge.set(report.dead_workers, {'status': 'dead'})
                self._throughput_gauge.set(report.throughput)
                self._error_rate_gauge.set(report.error_rate)
                
            except Exception:
                pass
                
            time.sleep(self.check_interval)
            
    def _collect_health(self) -> ColonyHealthReport:
        """Collect current health metrics."""
        now = time.time()
        workers = []
        alerts = []
        
        total_processed = 0
        total_errors = 0
        total_load = 0
        
        # Collect worker stats (simulated - in real implementation
        # this would query actual worker state)
        temperature = self.colony.get_colony_temperature()
        
        # Analyze temperature
        if temperature > 0.9:
            alerts.append("CRITICAL: Colony temperature > 90% - severe overload")
        elif temperature > 0.8:
            alerts.append("WARNING: Colony temperature > 80% - high load")
            
        # Analyze pheromone levels
        throttle_level = self.colony.sense_pheromone('throttle')
        if throttle_level > 0.5:
            alerts.append(f"WARNING: High throttle pheromone ({throttle_level:.2f}) - backpressure active")
            
        alarm_level = self.colony.sense_pheromone('alarm')
        if alarm_level > 0.3:
            alerts.append(f"WARNING: Alarm pheromones detected ({alarm_level:.2f}) - errors occurring")
            
        # Analyze food source health
        abandoned = self.colony.get_abandoned_sources()
        if abandoned:
            alerts.append(f"INFO: {len(abandoned)} food sources abandoned - scout bees reassigning")
            
        # Calculate overall status
        if temperature > 0.9 or alarm_level > 0.5:
            overall_status = 'unhealthy'
        elif temperature > 0.7 or throttle_level > 0.5:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
            
        return ColonyHealthReport(
            timestamp=now,
            total_workers=len(self.colony.temperature),
            healthy_workers=sum(1 for t in self.colony.temperature.values() if t < 0.5),
            degraded_workers=sum(1 for t in self.colony.temperature.values() if 0.5 <= t < 0.8),
            unhealthy_workers=sum(1 for t in self.colony.temperature.values() if t >= 0.8),
            dead_workers=0,  # Would need to track last activity times
            overall_status=overall_status,
            temperature=temperature,
            throughput=0,  # Would need to calculate from time series
            error_rate=0,  # Would need to track errors
            worker_snapshots=workers,
            alerts=alerts
        )
        
    def get_current_health(self) -> ColonyHealthReport:
        """Get latest health report."""
        with self._lock:
            if self._history:
                return self._history[-1]
        return self._collect_health()
        
    def get_health_history(self, limit: int = 20) -> List[ColonyHealthReport]:
        """Get historical health reports."""
        with self._lock:
            return list(self._history)[-limit:]
