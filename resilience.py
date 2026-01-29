"""
HiveFrame Resilience Patterns
=============================
Retry policies, circuit breakers, and fault tolerance mechanisms.
"""

import time
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from enum import Enum, auto
from functools import wraps
import math

from .exceptions import (
    HiveFrameError, TransientError, CircuitOpenError,
    TimeoutError as HiveTimeoutError, ErrorSeverity
)


T = TypeVar('T')


# ============================================================================
# Retry Policies
# ============================================================================

class BackoffStrategy(Enum):
    """Backoff strategies for retry delays."""
    FIXED = auto()           # Same delay each time
    LINEAR = auto()          # Linearly increasing delay
    EXPONENTIAL = auto()     # Exponentially increasing delay
    DECORRELATED_JITTER = auto()  # AWS-style decorrelated jitter


@dataclass
class RetryPolicy:
    """
    Configurable retry policy with multiple backoff strategies.
    
    Inspired by bee foraging behavior - bees don't give up immediately
    when a flower is empty, but they don't persist forever either.
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter: bool = True
    jitter_factor: float = 0.25
    retryable_exceptions: tuple = (TransientError,)
    
    def calculate_delay(self, attempt: int, last_delay: float = 0) -> float:
        """Calculate delay for next retry attempt."""
        if self.strategy == BackoffStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay * attempt
        elif self.strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** (attempt - 1))
        elif self.strategy == BackoffStrategy.DECORRELATED_JITTER:
            # AWS-style: delay = min(cap, random(base, previous_delay * 3))
            delay = random.uniform(self.base_delay, max(self.base_delay, last_delay * 3))
        else:
            delay = self.base_delay
            
        # Apply jitter
        if self.jitter and self.strategy != BackoffStrategy.DECORRELATED_JITTER:
            jitter_range = delay * self.jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)
            
        return min(max(0, delay), self.max_delay)
        
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.max_retries:
            return False
            
        if isinstance(exception, self.retryable_exceptions):
            return True
            
        if isinstance(exception, HiveFrameError):
            return exception.retryable
            
        return False


@dataclass
class RetryState:
    """Tracks state across retry attempts."""
    attempt: int = 0
    total_delay: float = 0.0
    last_delay: float = 0.0
    last_exception: Optional[Exception] = None
    start_time: float = field(default_factory=time.time)
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


def with_retry(
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[Callable[[RetryState, Exception], None]] = None,
    on_failure: Optional[Callable[[RetryState, Exception], None]] = None
):
    """
    Decorator to add retry logic to functions.
    
    Example:
        @with_retry(RetryPolicy(max_retries=5, strategy=BackoffStrategy.EXPONENTIAL))
        def fetch_data(url):
            return requests.get(url).json()
    """
    if policy is None:
        policy = RetryPolicy()
        
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            state = RetryState()
            
            while True:
                try:
                    state.attempt += 1
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    state.last_exception = e
                    
                    if not policy.should_retry(e, state.attempt):
                        if on_failure:
                            on_failure(state, e)
                        raise
                        
                    delay = policy.calculate_delay(state.attempt, state.last_delay)
                    state.last_delay = delay
                    state.total_delay += delay
                    
                    if on_retry:
                        on_retry(state, e)
                        
                    time.sleep(delay)
                    
        return wrapper
    return decorator


class RetryContext:
    """
    Context manager for retry operations with more control.
    
    Example:
        with RetryContext(policy) as retry:
            for attempt in retry:
                try:
                    result = risky_operation()
                    break
                except TransientError as e:
                    retry.record_failure(e)
    """
    
    def __init__(self, policy: Optional[RetryPolicy] = None):
        self.policy = policy or RetryPolicy()
        self.state = RetryState()
        self._should_continue = True
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
        
    def __iter__(self):
        while self._should_continue and self.state.attempt < self.policy.max_retries:
            yield self.state.attempt
            self.state.attempt += 1
            
    def record_failure(self, exception: Exception) -> None:
        """Record failure and prepare for potential retry."""
        self.state.last_exception = exception
        
        if not self.policy.should_retry(exception, self.state.attempt):
            self._should_continue = False
            raise exception
            
        delay = self.policy.calculate_delay(self.state.attempt, self.state.last_delay)
        self.state.last_delay = delay
        self.state.total_delay += delay
        time.sleep(delay)
        
    def success(self) -> None:
        """Mark operation as successful, stopping retries."""
        self._should_continue = False


# ============================================================================
# Circuit Breaker
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failures exceeded threshold, rejecting calls
    HALF_OPEN = auto()   # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes to close from half-open
    timeout: float = 30.0               # Seconds before trying half-open
    half_open_max_calls: int = 3        # Max concurrent calls in half-open


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
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
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
                
    def add_state_listener(
        self,
        listener: Callable[[CircuitState, CircuitState], None]
    ) -> None:
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
                reset_time=self._last_failure_time + self.config.timeout
            )
            
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
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
                'name': self.name,
                'state': self._state.name,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'half_open_calls': self._half_open_calls,
                'last_failure_time': self._last_failure_time,
                'time_until_half_open': max(
                    0,
                    (self._last_failure_time + self.config.timeout) - time.time()
                ) if self._state == CircuitState.OPEN else 0
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


# ============================================================================
# Bulkhead Pattern
# ============================================================================

class Bulkhead:
    """
    Bulkhead Pattern
    ----------------
    Isolates failures by limiting concurrent access to a resource.
    
    Like compartments in a ship's hull, if one area fails,
    others remain unaffected.
    
    Inspired by bee hive structure - separate cells for
    different functions prevent colony-wide failures.
    """
    
    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_queue: int = 100,
        timeout: float = 30.0
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.timeout = timeout
        
        self._semaphore = threading.Semaphore(max_concurrent)
        self._queue_count = 0
        self._lock = threading.Lock()
        self._active_count = 0
        self._rejected_count = 0
        
    def acquire(self) -> bool:
        """
        Acquire slot in bulkhead.
        Returns False if would exceed limits.
        """
        with self._lock:
            if self._active_count >= self.max_concurrent:
                if self._queue_count >= self.max_queue:
                    self._rejected_count += 1
                    return False
                self._queue_count += 1
                
        acquired = self._semaphore.acquire(timeout=self.timeout)
        
        with self._lock:
            if self._queue_count > 0:
                self._queue_count -= 1
            if acquired:
                self._active_count += 1
            else:
                self._rejected_count += 1
                
        return acquired
        
    def release(self) -> None:
        """Release slot in bulkhead."""
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
        self._semaphore.release()
        
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function within bulkhead."""
        if not self.acquire():
            raise WorkerExhausted(
                f"Bulkhead '{self.name}' is full",
                active_workers=self._active_count,
                total_workers=self.max_concurrent
            )
            
        try:
            return func(*args, **kwargs)
        finally:
            self.release()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        with self._lock:
            return {
                'name': self.name,
                'active': self._active_count,
                'queued': self._queue_count,
                'max_concurrent': self.max_concurrent,
                'max_queue': self.max_queue,
                'rejected_count': self._rejected_count,
                'utilization': self._active_count / self.max_concurrent
            }


# ============================================================================
# Timeout Wrapper
# ============================================================================

class TimeoutWrapper:
    """
    Timeout enforcement for operations.
    
    Uses threading to enforce timeouts on synchronous operations.
    """
    
    def __init__(self, timeout: float, operation_name: str = "operation"):
        self.timeout = timeout
        self.operation_name = operation_name
        
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with timeout."""
        result = [None]
        exception = [None]
        completed = threading.Event()
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
            finally:
                completed.set()
                
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        
        if not completed.wait(timeout=self.timeout):
            raise HiveTimeoutError(
                f"Operation '{self.operation_name}' timed out after {self.timeout}s",
                timeout_seconds=self.timeout,
                operation=self.operation_name
            )
            
        if exception[0]:
            raise exception[0]
            
        return result[0]


def with_timeout(timeout: float, operation_name: str = "operation"):
    """
    Decorator to add timeout to function.
    
    Example:
        @with_timeout(30.0, "database_query")
        def slow_query():
            ...
    """
    wrapper = TimeoutWrapper(timeout, operation_name)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapped(*args, **kwargs) -> T:
            return wrapper.call(func, *args, **kwargs)
        return wrapped
    return decorator


# ============================================================================
# Combined Resilience
# ============================================================================

class ResilientExecutor:
    """
    Combined resilience patterns for robust execution.
    
    Combines retry, circuit breaker, bulkhead, and timeout
    for comprehensive fault tolerance.
    """
    
    def __init__(
        self,
        name: str,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        max_concurrent: int = 10,
        timeout: float = 30.0
    ):
        self.name = name
        self.retry_policy = retry_policy or RetryPolicy()
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


# Import WorkerExhausted for bulkhead
from .exceptions import WorkerExhausted
