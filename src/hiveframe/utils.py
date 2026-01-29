"""
HiveFrame Utilities
===================
Common utilities, mixins, and base classes for cross-cutting concerns.
"""

import threading
import time
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar, Protocol, runtime_checkable


T = TypeVar('T')
R = TypeVar('R')


# ============================================================================
# Thread Safety
# ============================================================================

class ThreadSafeMixin:
    """
    Mixin providing thread-safe operations via a reentrant lock.
    
    Usage:
        class MyClass(ThreadSafeMixin):
            def __init__(self):
                super().__init__()
                self._data = []
                
            def add(self, item):
                with self._lock:
                    self._data.append(item)
    """
    _lock: threading.RLock
    
    def __init__(self, *args, **kwargs):
        # Use RLock for reentrant safety (allows nested locking)
        self._lock = threading.RLock()
        super().__init__(*args, **kwargs)
    
    @contextmanager
    def locked(self):
        """Context manager for explicit lock acquisition."""
        with self._lock:
            yield


# ============================================================================
# Resource Management
# ============================================================================

class ManagedResource(ABC):
    """
    Base class for resources that require explicit lifecycle management.
    
    Provides common patterns for:
    - Open/close lifecycle
    - Context manager support
    - Metrics tracking (records processed, errors, timing)
    - State management
    
    Subclasses should implement _do_open() and _do_close().
    """
    
    def __init__(self, name: str):
        self.name = name
        self._is_open = False
        self._start_time: Optional[float] = None
        self._records_processed = 0
        self._errors = 0
    
    @abstractmethod
    def _do_open(self) -> None:
        """Implementation-specific open logic."""
        pass
    
    @abstractmethod
    def _do_close(self) -> None:
        """Implementation-specific close logic."""
        pass
    
    def open(self) -> None:
        """Open the resource."""
        if self._is_open:
            return
        self._start_time = time.time()
        self._do_open()
        self._is_open = True
    
    def close(self) -> None:
        """Close the resource."""
        if not self._is_open:
            return
        self._do_close()
        self._is_open = False
    
    def __enter__(self) -> 'ManagedResource':
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
    
    @property
    def is_open(self) -> bool:
        """Check if resource is currently open."""
        return self._is_open
    
    @property
    def elapsed_seconds(self) -> float:
        """Seconds since resource was opened."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource statistics."""
        return {
            'name': self.name,
            'is_open': self._is_open,
            'records_processed': self._records_processed,
            'errors': self._errors,
            'elapsed_seconds': self.elapsed_seconds,
            'records_per_second': (
                self._records_processed / self.elapsed_seconds 
                if self.elapsed_seconds > 0 else 0
            ),
        }


# ============================================================================
# Protocols
# ============================================================================

@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized to a dictionary."""
    def to_dict(self) -> Dict[str, Any]: ...


@runtime_checkable  
class Monitorable(Protocol):
    """Protocol for objects that expose statistics/metrics."""
    def get_stats(self) -> Dict[str, Any]: ...


# ============================================================================
# Deprecation Helpers
# ============================================================================

def deprecated_param(
    old_name: str,
    new_name: str,
    old_value: Any,
    new_value: Any,
    version: str = "0.3.0"
) -> Any:
    """
    Handle deprecated parameter names with warnings.
    
    Usage:
        def __init__(self, size_seconds=None, *, window_size=None):
            self.size_seconds = deprecated_param(
                'window_size', 'size_seconds', 
                window_size, size_seconds
            )
    
    Returns the value to use (prefers new_name, falls back to old_name).
    """
    if old_value is not None:
        warnings.warn(
            f"Parameter '{old_name}' is deprecated and will be removed in v{version}. "
            f"Use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=3
        )
        if new_value is None:
            return old_value
    return new_value


def require_one_of(*params: tuple[str, Any], param_names: str = "") -> Any:
    """
    Require exactly one of the given parameters to be provided.
    
    Usage:
        value = require_one_of(
            ('size_seconds', size_seconds),
            ('window_size', window_size),
        )
    """
    provided = [(name, val) for name, val in params if val is not None]
    
    if len(provided) == 0:
        names = " or ".join(name for name, _ in params)
        raise ValueError(f"Must provide one of: {names}")
    
    if len(provided) > 1:
        names = ", ".join(name for name, _ in provided)
        raise ValueError(f"Cannot provide multiple values for: {names}")
    
    return provided[0][1]
