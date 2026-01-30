"""
HiveFrame Resilience Package
============================
Retry policies, circuit breakers, and fault tolerance mechanisms.

Provides:
- Retry policies with various backoff strategies
- Circuit breaker pattern for cascading failure prevention
- Bulkhead pattern for resource isolation
- Timeout enforcement
- Combined resilient executor
"""

from .bulkhead import Bulkhead
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    with_circuit_breaker,
)
from .executor import ResilientExecutor
from .retry import (
    BackoffStrategy,
    RetryContext,
    RetryPolicy,
    RetryState,
    with_retry,
)
from .timeout import (
    TimeoutWrapper,
    with_timeout,
)

__all__ = [
    # Retry
    "BackoffStrategy",
    "RetryPolicy",
    "RetryState",
    "RetryContext",
    "with_retry",
    # Circuit Breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "with_circuit_breaker",
    # Bulkhead
    "Bulkhead",
    # Timeout
    "TimeoutWrapper",
    "with_timeout",
    # Executor
    "ResilientExecutor",
]
