"""
HiveFrame Resilience - Circuit Breaker
======================================
Circuit breaker pattern for cascading failure prevention.
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

from ..exceptions import CircuitOpenError

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failures exceeded threshold, rejecting calls
    HALF_OPEN = auto()  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 30.0  # Seconds before trying half-open
    half_open_max_calls: int = 3  # Max concurrent calls in half-open


class CircuitBreaker:
    """
    Circuit Breaker Pattern
    -----------------------
    Prevents cascading failures by detecting repeated errors
    and temporarily blocking requests to a failing service.

    Inspired by bee alarm pheromones - when danger is detected,
    bees emit pheromones that warn others to stay away.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests rejected immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()
        self._listeners: List[Callable[[CircuitState, CircuitState], None]] = []

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            # Check if we should transition to half-open
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.config.timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state (called under lock)."""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitState.OPEN:
            self._last_failure_time = time.time()

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(old_state, new_state)
            except:
                pass

    def add_state_listener(self, listener: Callable[[CircuitState, CircuitState], None]) -> None:
        """Add listener for state transitions."""
        self._listeners.append(listener)

    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        state = self.state  # This may trigger half-open transition

        with self._lock:
            if state == CircuitState.CLOSED:
                return True
            elif state == CircuitState.OPEN:
                return False
            else:  # HALF_OPEN
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

    def record_success(self) -> None:
        """Record successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record failed call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens circuit
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.

        Raises CircuitOpenError if circuit is open.
        """
        if not self.allow_request():
            raise CircuitOpenError(
                f"Circuit '{self.name}' is open",
                service_name=self.name,
                reset_time=self._last_failure_time + self.config.timeout,
            )

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.name,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "half_open_calls": self._half_open_calls,
                "last_failure_time": self._last_failure_time,
                "time_until_half_open": (
                    max(0, (self._last_failure_time + self.config.timeout) - time.time())
                    if self._state == CircuitState.OPEN
                    else 0
                ),
            }


def with_circuit_breaker(circuit: CircuitBreaker):
    """
    Decorator to wrap function with circuit breaker.

    Example:
        db_circuit = CircuitBreaker("database")

        @with_circuit_breaker(db_circuit)
        def query_database(sql):
            return db.execute(sql)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return circuit.call(func, *args, **kwargs)

        return wrapper

    return decorator
