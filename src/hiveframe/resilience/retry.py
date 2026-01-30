"""
HiveFrame Resilience - Retry Policies
=====================================
Configurable retry logic with multiple backoff strategies.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import Callable, Optional, TypeVar

from ..exceptions import HiveFrameError, TransientError

T = TypeVar("T")


class BackoffStrategy(Enum):
    """Backoff strategies for retry delays."""

    FIXED = auto()  # Same delay each time
    LINEAR = auto()  # Linearly increasing delay
    EXPONENTIAL = auto()  # Exponentially increasing delay
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
        """Calculate delay for next retry attempt (0-based or 1-based indexing)."""
        # Support both 0-based (test) and 1-based (internal) indexing
        # For exponential: attempt 0 -> 0.1, attempt 1 -> 0.2, attempt 2 -> 0.4
        if self.strategy == BackoffStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (2**attempt)
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

    # Alias for backward compatibility with tests
    def _calculate_delay(self, attempt: int, last_delay: float = 0) -> float:
        """Alias for calculate_delay (backward compatibility)."""
        return self.calculate_delay(attempt, last_delay)

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
    on_failure: Optional[Callable[[RetryState, Exception], None]] = None,
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
