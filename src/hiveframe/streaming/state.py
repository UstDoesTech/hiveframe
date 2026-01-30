"""
HiveFrame Streaming State Management
====================================
Checkpointing and state backend support for fault tolerance.
"""

import json
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Checkpoint:
    """Represents a point-in-time snapshot of processing state."""

    checkpoint_id: Any  # Can be str or int
    timestamp: float
    watermark: float = 0.0
    offsets: Dict[int, int] = field(default_factory=dict)
    window_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        checkpoint_id: Any,
        timestamp: float,
        watermark: float = 0.0,
        offsets: Optional[Dict[int, int]] = None,
        window_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        state: Optional[Dict[str, Any]] = None,  # Alias for window_state (backward compat)
    ):
        self.checkpoint_id = checkpoint_id
        self.timestamp = timestamp
        self.watermark = watermark
        self.offsets = offsets if offsets is not None else {}
        # Support both 'window_state' and 'state' parameter names
        self.window_state = (
            window_state if window_state is not None else (state if state is not None else {})
        )
        # Also expose as 'state' for backward compatibility
        self.state = self.window_state
        self.metadata = metadata if metadata is not None else {}

    def to_json(self) -> str:
        return json.dumps(
            {
                "checkpoint_id": self.checkpoint_id,
                "timestamp": self.timestamp,
                "watermark": self.watermark,
                "offsets": self.offsets,
                "window_state": {k: str(v)[:1000] for k, v in self.window_state.items()},
                "metadata": self.metadata,
            }
        )


class StateBackend(ABC):
    """Abstract state storage backend."""

    @abstractmethod
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a specific checkpoint."""
        pass

    @abstractmethod
    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get the most recent checkpoint."""
        pass


class InMemoryStateBackend(StateBackend):
    """In-memory state backend for testing."""

    def __init__(self, max_checkpoints: int = 10):
        self.max_checkpoints = max_checkpoints
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._ordered: deque = deque(maxlen=max_checkpoints)
        self._lock = threading.Lock()

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            self._checkpoints[checkpoint.checkpoint_id] = checkpoint
            self._ordered.append(checkpoint.checkpoint_id)

            # Cleanup old checkpoints
            while len(self._checkpoints) > self.max_checkpoints:
                old_id = self._ordered.popleft()
                if old_id in self._checkpoints:
                    del self._checkpoints[old_id]

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        with self._lock:
            return self._checkpoints.get(checkpoint_id)

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        with self._lock:
            if self._ordered:
                return self._checkpoints.get(self._ordered[-1])
            return None
