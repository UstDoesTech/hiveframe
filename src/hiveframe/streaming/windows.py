"""
HiveFrame Streaming Windows
===========================
Windowing functions for stream processing.

Supports:
- Tumbling windows (fixed, non-overlapping)
- Sliding windows (fixed, overlapping)
- Session windows (gap-based dynamic)
"""

import math
import threading
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Generic, List, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class WindowType(Enum):
    """Types of time windows."""

    TUMBLING = auto()  # Fixed, non-overlapping windows
    SLIDING = auto()  # Fixed, overlapping windows
    SESSION = auto()  # Gap-based dynamic windows


@dataclass
class Window:
    """Represents a time window."""

    start: float  # Window start timestamp
    end: float  # Window end timestamp
    window_type: WindowType

    @property
    def duration(self) -> float:
        return self.end - self.start

    def contains(self, timestamp: float) -> bool:
        return self.start <= timestamp < self.end

    def __hash__(self):
        return hash((self.start, self.end))

    def __eq__(self, other):
        if not isinstance(other, Window):
            return False
        return self.start == other.start and self.end == other.end


@dataclass
class WindowedValue(Generic[K, V]):
    """A value associated with a window."""

    key: K
    value: V
    window: Window
    timestamp: float
    is_late: bool = False


class WindowAssigner(ABC):
    """Base class for window assignment strategies."""

    @abstractmethod
    def assign_windows(self, timestamp: float) -> List[Window]:
        """Assign windows for a given event timestamp."""
        pass

    @abstractmethod
    def get_next_window_end(self, current_time: float) -> float:
        """Get the end time of the next window to trigger."""
        pass


class TumblingWindowAssigner(WindowAssigner):
    """
    Tumbling Window Assigner

    Creates fixed-size, non-overlapping windows.

    Example: 5-minute tumbling windows
    [00:00-00:05], [00:05-00:10], [00:10-00:15], ...
    """

    def __init__(self, size_seconds: float = None, *, window_size: float = None):
        # Support both parameter names - window_size is deprecated
        if window_size is not None:
            warnings.warn(
                "Parameter 'window_size' is deprecated and will be removed in v0.3.0. "
                "Use 'size_seconds' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if size_seconds is None:
                size_seconds = window_size

        if size_seconds is None:
            raise ValueError("Must provide size_seconds")
        self.size_seconds = size_seconds

    def assign_windows(self, timestamp: float) -> List[Window]:
        window_start = math.floor(timestamp / self.size_seconds) * self.size_seconds
        return [
            Window(
                start=window_start,
                end=window_start + self.size_seconds,
                window_type=WindowType.TUMBLING,
            )
        ]

    def get_next_window_end(self, current_time: float) -> float:
        window_start = math.floor(current_time / self.size_seconds) * self.size_seconds
        return window_start + self.size_seconds


class SlidingWindowAssigner(WindowAssigner):
    """
    Sliding Window Assigner

    Creates overlapping windows with fixed size and slide interval.

    Example: 10-minute windows sliding every 5 minutes
    [00:00-00:10], [00:05-00:15], [00:10-00:20], ...
    """

    def __init__(
        self,
        size_seconds: float = None,
        slide_seconds: float = None,
        *,
        window_size: float = None,
        slide_interval: float = None,
    ):
        # Support both parameter names - window_size and slide_interval are deprecated
        if window_size is not None:
            warnings.warn(
                "Parameter 'window_size' is deprecated and will be removed in v0.3.0. "
                "Use 'size_seconds' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if size_seconds is None:
                size_seconds = window_size

        if slide_interval is not None:
            warnings.warn(
                "Parameter 'slide_interval' is deprecated and will be removed in v0.3.0. "
                "Use 'slide_seconds' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if slide_seconds is None:
                slide_seconds = slide_interval

        if size_seconds is None:
            raise ValueError("Must provide size_seconds")
        if slide_seconds is None:
            raise ValueError("Must provide slide_seconds")

        self.size_seconds = size_seconds
        self.slide_seconds = slide_seconds

    def assign_windows(self, timestamp: float) -> List[Window]:
        windows = []

        # Find all windows that contain this timestamp
        window_start = math.floor(timestamp / self.slide_seconds) * self.slide_seconds

        while window_start + self.size_seconds > timestamp:
            windows.append(
                Window(
                    start=window_start,
                    end=window_start + self.size_seconds,
                    window_type=WindowType.SLIDING,
                )
            )
            window_start -= self.slide_seconds

            # Safety limit
            if len(windows) > 100:
                break

        return windows

    def get_next_window_end(self, current_time: float) -> float:
        slot = math.floor(current_time / self.slide_seconds)
        return (slot + 1) * self.slide_seconds


class SessionWindowAssigner(WindowAssigner):
    """
    Session Window Assigner

    Creates windows based on activity gaps. A new window starts
    after a period of inactivity.

    Particularly useful for user session analysis.
    """

    def __init__(self, gap_seconds: float = None, *, gap: float = None):
        # Support both parameter names - gap is deprecated
        if gap is not None:
            warnings.warn(
                "Parameter 'gap' is deprecated and will be removed in v0.3.0. "
                "Use 'gap_seconds' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if gap_seconds is None:
                gap_seconds = gap

        if gap_seconds is None:
            raise ValueError("Must provide gap_seconds")
        self.gap_seconds = gap_seconds
        self._sessions: Dict[Any, Window] = {}  # key -> current session window
        self._lock = threading.Lock()

    def assign_windows(self, timestamp: float, key: Any = None) -> List[Window]:
        with self._lock:
            if key in self._sessions:
                session = self._sessions[key]

                # Check if within gap
                if timestamp < session.end + self.gap_seconds:
                    # Extend session
                    session.end = timestamp + self.gap_seconds
                    return [session]

            # Start new session
            new_session = Window(
                start=timestamp, end=timestamp + self.gap_seconds, window_type=WindowType.SESSION
            )
            self._sessions[key] = new_session
            return [new_session]

    def get_next_window_end(self, current_time: float) -> float:
        # Sessions don't have fixed ends
        return current_time + self.gap_seconds


# Convenience functions


def tumbling_window(seconds: float) -> TumblingWindowAssigner:
    """Create a tumbling window assigner."""
    return TumblingWindowAssigner(seconds)


def sliding_window(size_seconds: float, slide_seconds: float) -> SlidingWindowAssigner:
    """Create a sliding window assigner."""
    return SlidingWindowAssigner(size_seconds, slide_seconds)


def session_window(gap_seconds: float) -> SessionWindowAssigner:
    """Create a session window assigner."""
    return SessionWindowAssigner(gap_seconds)
