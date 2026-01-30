"""
HiveFrame Logging
=================
Structured logging for monitoring and debugging.

Features:
- Structured key-value logging
- Context propagation
- Multiple handlers
- Level filtering
"""

import time
import threading
import json
import traceback
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import sys


class LogLevel(Enum):
    """Log levels."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogRecord:
    """Structured log record."""

    timestamp: float
    level: LogLevel
    message: str
    logger_name: str
    extra: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None

    def to_json(self) -> str:
        """Format as JSON."""
        data = {
            "timestamp": self.timestamp,
            "level": self.level.name,
            "message": self.message,
            "logger": self.logger_name,
            **self.extra,
        }
        if self.exception:
            data["exception"] = self.exception
        return json.dumps(data)

    def to_text(self) -> str:
        """Format as human-readable text."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        extra_str = " ".join(f"{k}={v}" for k, v in self.extra.items())
        line = f"[{ts}] {self.level.name:8} {self.logger_name}: {self.message}"
        if extra_str:
            line += f" | {extra_str}"
        if self.exception:
            line += f"\n{self.exception}"
        return line


class LogHandler(ABC):
    """Base class for log handlers."""

    @abstractmethod
    def handle(self, record: LogRecord) -> None:
        """Process a log record."""
        pass


class ConsoleHandler(LogHandler):
    """Write logs to console."""

    def __init__(self, format: str = "text", stream=None):
        self.format = format
        self.stream = stream or sys.stderr
        self._lock = threading.Lock()

    def handle(self, record: LogRecord) -> None:
        with self._lock:
            if self.format == "json":
                self.stream.write(record.to_json() + "\n")
            else:
                self.stream.write(record.to_text() + "\n")
            self.stream.flush()


class BufferedHandler(LogHandler):
    """Buffer logs in memory for testing/inspection."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def handle(self, record: LogRecord) -> None:
        with self._lock:
            self._buffer.append(record)

    def get_logs(self, level: Optional[LogLevel] = None, limit: int = 100) -> List[LogRecord]:
        """Get buffered logs."""
        with self._lock:
            logs = list(self._buffer)

        if level:
            logs = [l for l in logs if l.level.value >= level.value]

        return logs[-limit:]

    def clear(self) -> None:
        """Clear buffer."""
        with self._lock:
            self._buffer.clear()


class Logger:
    """
    Structured logger with context support.

    Features:
    - Structured key-value logging
    - Context propagation
    - Multiple handlers
    - Level filtering
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        handlers: Optional[List[LogHandler]] = None,
    ):
        self.name = name
        self.level = level
        self.handlers = handlers or [ConsoleHandler()]
        self._context: Dict[str, Any] = {}

    def with_context(self, **kwargs) -> "Logger":
        """Create child logger with additional context."""
        child = Logger(self.name, self.level, self.handlers)
        child._context = {**self._context, **kwargs}
        return child

    def _log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Internal logging method."""
        if level.value < self.level.value:
            return

        exc_info = kwargs.pop("exc_info", False)
        exception = None
        if exc_info:
            exception = traceback.format_exc()

        record = LogRecord(
            timestamp=time.time(),
            level=level,
            message=message,
            logger_name=self.name,
            extra={**self._context, **kwargs},
            exception=exception,
        )

        for handler in self.handlers:
            try:
                handler.handle(record)
            except Exception:
                pass  # Don't let logging failures break the app

    def debug(self, message: str, **kwargs) -> None:
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log error with exception traceback."""
        self._log(LogLevel.ERROR, message, exc_info=True, **kwargs)


def get_logger(name: str) -> Logger:
    """Get a logger instance."""
    return Logger(name)
