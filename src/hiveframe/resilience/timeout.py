"""
HiveFrame Resilience - Timeout
==============================
Timeout enforcement for operations.
"""

import threading
from functools import wraps
from typing import Callable, TypeVar

from ..exceptions import TimeoutError as HiveTimeoutError

T = TypeVar("T")


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
                operation=self.operation_name,
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
