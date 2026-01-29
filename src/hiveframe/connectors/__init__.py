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
from .sources import DataSource
from .sinks import DataSink

# Import file sources and sinks
from .sources import CSVSource, JSONLSource, JSONSource
from .sinks import JSONLSink, CSVSink

# Import HTTP source
from .sources import HTTPSource, HTTPConfig

# Import messaging components
from .messaging import (
    FileEvent, FileWatcher,
    Message, Topic, MessageBroker,
    MessageQueueSource, MessageQueueSink
)

# Import data generator
from .sources import DataGenerator


__all__ = [
    # Base classes
    'DataSource',
    'DataSink',
    
    # File sources
    'CSVSource',
    'JSONLSource',
    'JSONSource',
    
    # File sinks
    'JSONLSink',
    'CSVSink',
    
    # HTTP
    'HTTPSource',
    'HTTPConfig',
    
    # File watching
    'FileEvent',
    'FileWatcher',
    
    # Message queue
    'Message',
    'Topic',
    'MessageBroker',
    'MessageQueueSource',
    'MessageQueueSink',
    
    # Testing
    'DataGenerator',
]
