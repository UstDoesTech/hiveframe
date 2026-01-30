"""
HiveFrame Connectors
====================
Real data source integrations for production use cases.

Supports:
- File sources (CSV, JSON, JSONL) with file watching
- HTTP APIs with rate limiting
- Simulated Kafka-like streaming
- Database connections (when drivers available)
"""

# Import base classes
# Import messaging components
from .messaging import (
    FileEvent,
    FileWatcher,
    Message,
    MessageBroker,
    MessageQueueSink,
    MessageQueueSource,
    Topic,
)
from .sinks import CSVSink, DataSink, JSONLSink

# Import file sources and sinks
# Import HTTP source
# Import data generator
from .sources import (
    CSVSource,
    DataGenerator,
    DataSource,
    HTTPConfig,
    HTTPSource,
    JSONLSource,
    JSONSource,
)

__all__ = [
    # Base classes
    "DataSource",
    "DataSink",
    # File sources
    "CSVSource",
    "JSONLSource",
    "JSONSource",
    # File sinks
    "JSONLSink",
    "CSVSink",
    # HTTP
    "HTTPSource",
    "HTTPConfig",
    # File watching
    "FileEvent",
    "FileWatcher",
    # Message queue
    "Message",
    "Topic",
    "MessageBroker",
    "MessageQueueSource",
    "MessageQueueSink",
    # Testing
    "DataGenerator",
]
