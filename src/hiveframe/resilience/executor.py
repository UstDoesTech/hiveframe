"""
HiveFrame Resilience - Resilient Executor
=========================================
Combined resilience patterns for robust execution.
"""

import time
import threading
from typing import Any, Callable, Dict, Optional, TypeVar

from ..exceptions import CircuitOpenError, WorkerExhausted

from .retry import RetryPolicy, RetryState
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .bulkhead import Bulkhead
from .timeout import TimeoutWrapper


T = TypeVar('T')


class ResilientExecutor:
    """
    Combined resilience patterns for robust execution.
    
    Combines retry, circuit breaker, bulkhead, and timeout
    for comprehensive fault tolerance.
    """
    
    def __init__(
        self,
        name: str = "default",
        retry_policy: Optional[RetryPolicy] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        max_concurrent: int = 10,
        timeout: float = 30.0,
        *,
        circuit_breaker: Optional[CircuitBreaker] = None  # Alternate API
    ):
        self.name = name
        self.retry_policy = retry_policy or RetryPolicy()
        # Support both circuit_config (creates new) and circuit_breaker (existing)
        if circuit_breaker is not None:
            self.circuit = circuit_breaker
        else:
            self.circuit = CircuitBreaker(f"{name}_circuit", circuit_config)
        self.bulkhead = Bulkhead(f"{name}_bulkhead", max_concurrent)
        self.timeout = TimeoutWrapper(timeout, name)
        
        self._call_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._lock = threading.Lock()
        
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with full resilience stack.
        
        Order: Bulkhead -> Circuit Breaker -> Timeout -> Retry
        """
        with self._lock:
            self._call_count += 1
            
        try:
            # First, check bulkhead
            if not self.bulkhead.acquire():
                raise WorkerExhausted(f"Executor '{self.name}' is at capacity")
                
            try:
                # Then, check circuit breaker
                if not self.circuit.allow_request():
                    raise CircuitOpenError(
                        f"Circuit for '{self.name}' is open",
                        service_name=self.name
                    )
                    
                # Finally, execute with timeout and retry
                state = RetryState()
                
                while True:
                    try:
                        state.attempt += 1
                        result = self.timeout.call(func, *args, **kwargs)
                        self.circuit.record_success()
                        
                        with self._lock:
                            self._success_count += 1
                            
                        return result
                        
                    except Exception as e:
                        state.last_exception = e
                        
                        if not self.retry_policy.should_retry(e, state.attempt):
                            self.circuit.record_failure()
                            
                            with self._lock:
                                self._failure_count += 1
                                
                            raise
                            
                        delay = self.retry_policy.calculate_delay(
                            state.attempt, state.last_delay
                        )
                        state.last_delay = delay
                        time.sleep(delay)
                        
            finally:
                self.bulkhead.release()
                
        except Exception:
            with self._lock:
                self._failure_count += 1
            raise
            
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        with self._lock:
            return {
                'name': self.name,
                'calls': self._call_count,
                'successes': self._success_count,
                'failures': self._failure_count,
                'success_rate': (
                    self._success_count / self._call_count
                    if self._call_count > 0 else 0
                ),
                'circuit': self.circuit.get_stats(),
                'bulkhead': self.bulkhead.get_stats()
            }
