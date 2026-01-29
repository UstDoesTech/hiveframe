"""
HiveFrame Streaming Watermarks
==============================
Watermark support for tracking event-time progress and late data handling.
"""

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Watermark:
    """
    Watermark for tracking event-time progress.
    
    A watermark with timestamp T asserts that no more events
    with timestamp < T will arrive.
    """
    timestamp: float
    source: str = "default"
    
    def is_late(self, event_timestamp: float) -> bool:
        """Check if an event is late relative to this watermark."""
        return event_timestamp < self.timestamp


class WatermarkGenerator(ABC):
    """Base class for watermark generation strategies."""
    
    @abstractmethod
    def on_event(self, timestamp: float) -> Optional[Watermark]:
        """Called for each event. Returns watermark if one should be emitted."""
        pass
        
    @abstractmethod
    def on_periodic_emit(self) -> Optional[Watermark]:
        """Called periodically to emit watermarks even without events."""
        pass


class BoundedOutOfOrdernessWatermarkGenerator(WatermarkGenerator):
    """
    Generates watermarks allowing for bounded out-of-order events.
    
    The watermark is always (max_seen_timestamp - max_out_of_orderness).
    """
    
    def __init__(
        self,
        max_out_of_orderness_seconds: float = None,
        emit_interval_seconds: float = 1.0,
        *,
        max_out_of_orderness: float = None
    ):
        # Support both parameter names for backward compatibility
        if max_out_of_orderness_seconds is not None:
            self.max_out_of_orderness = max_out_of_orderness_seconds
        elif max_out_of_orderness is not None:
            self.max_out_of_orderness = max_out_of_orderness
        else:
            self.max_out_of_orderness = 5.0  # default
        self.emit_interval = emit_interval_seconds
        self._max_timestamp = 0.0
        self._last_emit = 0.0
        self._lock = threading.Lock()
        
    def on_event(self, timestamp: float) -> Optional[Watermark]:
        with self._lock:
            self._max_timestamp = max(self._max_timestamp, timestamp)
            
            current_time = time.time()
            if current_time - self._last_emit >= self.emit_interval:
                self._last_emit = current_time
                return Watermark(
                    timestamp=self._max_timestamp - self.max_out_of_orderness
                )
        return None
        
    def on_periodic_emit(self) -> Optional[Watermark]:
        with self._lock:
            self._last_emit = time.time()
            return Watermark(
                timestamp=self._max_timestamp - self.max_out_of_orderness
            )
    
    def advance(self, timestamp: float) -> Watermark:
        """Advance watermark with new timestamp (test-friendly API)."""
        with self._lock:
            self._max_timestamp = max(self._max_timestamp, timestamp)
            return Watermark(
                timestamp=self._max_timestamp - self.max_out_of_orderness
            )


class PunctuatedWatermarkGenerator(WatermarkGenerator):
    """
    Generates watermarks based on special marker events.
    
    Some data sources include watermark markers in the stream itself.
    """
    
    def __init__(self, is_watermark_event: Callable[[Any], Optional[float]]):
        self.is_watermark_event = is_watermark_event
        self._current_watermark = 0.0
        
    def on_event(self, timestamp: float, event: Any = None) -> Optional[Watermark]:
        if event is not None:
            wm_timestamp = self.is_watermark_event(event)
            if wm_timestamp is not None:
                self._current_watermark = max(self._current_watermark, wm_timestamp)
                return Watermark(timestamp=wm_timestamp)
        return None
        
    def on_periodic_emit(self) -> Optional[Watermark]:
        return Watermark(timestamp=self._current_watermark)


# Convenience function

def bounded_watermark(
    max_out_of_orderness: float = 5.0
) -> BoundedOutOfOrdernessWatermarkGenerator:
    """Create a bounded out-of-orderness watermark generator."""
    return BoundedOutOfOrdernessWatermarkGenerator(max_out_of_orderness)
