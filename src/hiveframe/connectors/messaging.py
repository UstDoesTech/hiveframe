"""
HiveFrame Messaging
===================
File watching and message queue implementations.
"""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Generator, List, Optional, Union

from ..exceptions import ConfigurationError
from .sinks import DataSink
from .sources import DataSource

# ============================================================================
# File Watcher for Incremental Processing
# ============================================================================


@dataclass
class FileEvent:
    """Event from file watcher."""

    event_type: str  # 'created', 'modified', 'deleted'
    path: Path
    timestamp: float = field(default_factory=time.time)


class FileWatcher:
    """
    Watch directory for file changes.

    Enables incremental processing of new files as they arrive.
    Bee-inspired: like scout bees discovering new food sources.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        poll_interval: float = 1.0,
        process_existing: bool = True,
    ):
        self.directory = Path(directory)
        self.pattern = pattern
        self.poll_interval = poll_interval
        self.process_existing = process_existing

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._events: Queue = Queue()
        self._known_files: Dict[Path, float] = {}  # path -> mtime

    def start(self) -> None:
        """Start watching for file changes."""
        if not self.directory.exists():
            raise ConfigurationError(f"Watch directory not found: {self.directory}")

        self._running = True

        # Process existing files if requested
        if self.process_existing:
            for path in self.directory.glob(self.pattern):
                if path.is_file():
                    self._events.put(FileEvent("created", path))
                    self._known_files[path] = path.stat().st_mtime

        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def get_event(self, timeout: float = 1.0) -> Optional[FileEvent]:
        """Get next file event."""
        try:
            event = self._events.get(timeout=timeout)
            return event  # type: ignore[no-any-return, return-value]
        except Empty:
            return None

    def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._running:
            try:
                current_files = {
                    p: p.stat().st_mtime for p in self.directory.glob(self.pattern) if p.is_file()
                }

                # Check for new files
                for path, mtime in current_files.items():
                    if path not in self._known_files:
                        self._events.put(FileEvent("created", path))
                    elif mtime > self._known_files[path]:
                        self._events.put(FileEvent("modified", path))

                # Check for deleted files
                for path in list(self._known_files.keys()):
                    if path not in current_files:
                        self._events.put(FileEvent("deleted", path))
                        del self._known_files[path]

                self._known_files = current_files

            except Exception:
                pass  # Ignore errors in watch loop

            time.sleep(self.poll_interval)


# ============================================================================
# Simulated Message Queue (Kafka-like)
# ============================================================================


@dataclass
class Message:
    """Message in a topic partition."""

    key: Optional[str]
    value: Any
    timestamp: float = field(default_factory=time.time)
    partition: int = 0
    offset: int = 0
    headers: Dict[str, str] = field(default_factory=dict)


class Topic:
    """
    In-memory topic with partitions.

    Simulates Kafka-like semantics for local development and testing.
    """

    def __init__(self, name: str, num_partitions: int = 4):
        self.name = name
        self.num_partitions = num_partitions
        self._partitions: List[List[Message]] = [[] for _ in range(num_partitions)]
        self._locks = [threading.Lock() for _ in range(num_partitions)]
        self._offsets: Dict[str, Dict[int, int]] = {}  # consumer_group -> partition -> offset

    def _get_partition(self, key: Optional[str]) -> int:
        """Determine partition for a key."""
        if key is None:
            return 0  # Round-robin would be better but this is simple
        return hash(key) % self.num_partitions

    def produce(
        self, key: Optional[str], value: Any, headers: Optional[Dict[str, str]] = None
    ) -> Message:
        """Produce message to topic."""
        partition = self._get_partition(key)

        with self._locks[partition]:
            offset = len(self._partitions[partition])
            message = Message(
                key=key, value=value, partition=partition, offset=offset, headers=headers or {}
            )
            self._partitions[partition].append(message)
            return message

    def consume(
        self, consumer_group: str, partition: int, max_messages: int = 100
    ) -> List[Message]:
        """Consume messages from a partition."""
        if partition >= self.num_partitions:
            raise ValueError(f"Partition {partition} does not exist")

        # Get current offset for this consumer group
        if consumer_group not in self._offsets:
            self._offsets[consumer_group] = {i: 0 for i in range(self.num_partitions)}

        current_offset = self._offsets[consumer_group].get(partition, 0)

        with self._locks[partition]:
            messages = self._partitions[partition][current_offset : current_offset + max_messages]

        if messages:
            self._offsets[consumer_group][partition] = current_offset + len(messages)

        return messages

    def commit(self, consumer_group: str, partition: int, offset: int) -> None:
        """Commit offset for consumer group."""
        if consumer_group not in self._offsets:
            self._offsets[consumer_group] = {}
        self._offsets[consumer_group][partition] = offset

    def get_lag(self, consumer_group: str) -> Dict[int, int]:
        """Get consumer lag per partition."""
        if consumer_group not in self._offsets:
            return {i: len(self._partitions[i]) for i in range(self.num_partitions)}

        lag = {}
        for i in range(self.num_partitions):
            current = self._offsets[consumer_group].get(i, 0)
            lag[i] = len(self._partitions[i]) - current

        return lag


class Producer:
    """Simple producer wrapper for MessageBroker."""

    def __init__(self, broker: "MessageBroker"):
        self.broker = broker

    def send(self, topic_name: str, value: Any, key: Optional[str] = None) -> Message:
        """Send a message to a topic."""
        topic = self.broker.get_topic(topic_name)
        if topic is None:
            raise ValueError(f"Topic {topic_name} does not exist")
        return topic.produce(key, value)


class Consumer:
    """Simple consumer wrapper for MessageBroker."""

    def __init__(self, broker: "MessageBroker", topic_name: str, consumer_group: str):
        self.broker = broker
        self.topic_name = topic_name
        self.consumer_group = consumer_group
        self.topic = broker.get_topic(topic_name)
        if self.topic is None:
            raise ValueError(f"Topic {topic_name} does not exist")

    def poll(self, timeout: float = 1.0, max_messages: int = 100) -> List[Message]:
        """Poll for messages from all partitions."""
        messages: List[Message] = []
        assert self.topic is not None  # Already checked in __init__
        for partition in range(self.topic.num_partitions):
            partition_messages = self.topic.consume(
                self.consumer_group, partition, max_messages
            )
            messages.extend(partition_messages)
        return messages


class MessageBroker:
    """
    In-memory message broker for testing.

    Provides Kafka-like semantics without external dependencies.
    """

    def __init__(self):
        self._topics: Dict[str, Topic] = {}
        self._lock = threading.Lock()

    def create_topic(self, name: str, num_partitions: int = 4) -> Topic:
        """Create a new topic."""
        with self._lock:
            if name in self._topics:
                raise ValueError(f"Topic {name} already exists")
            topic = Topic(name, num_partitions)
            self._topics[name] = topic
            return topic

    def get_topic(self, name: str) -> Optional[Topic]:
        """Get topic by name."""
        return self._topics.get(name)

    def list_topics(self) -> List[str]:
        """List all topics."""
        return list(self._topics.keys())

    def create_producer(self) -> Producer:
        """Create a producer for this broker."""
        return Producer(self)

    def create_consumer(self, topic_name: str, consumer_group: str) -> Consumer:
        """Create a consumer for a specific topic."""
        return Consumer(self, topic_name, consumer_group)


class MessageQueueSource(DataSource[Message]):
    """
    Message queue source for stream processing.

    Consumes from topic partitions with automatic offset management.
    """

    def __init__(
        self,
        topic: Topic,
        consumer_group: str,
        partitions: Optional[List[int]] = None,
        auto_commit: bool = True,
        poll_timeout: float = 1.0,
    ):
        super().__init__(f"mq:{topic.name}")
        self.topic = topic
        self.consumer_group = consumer_group
        self.partitions = partitions or list(range(topic.num_partitions))
        self.auto_commit = auto_commit
        self.poll_timeout = poll_timeout
        self._running = False

    def open(self) -> None:
        self._running = True
        self._start_time = time.time()
        self._is_open = True

    def close(self) -> None:
        self._running = False
        self._is_open = False

    def read(self) -> Generator[Message, None, None]:
        """Poll for messages from assigned partitions."""
        if not self._is_open:
            raise RuntimeError("Source not open")

        while self._running:
            got_messages = False

            for partition in self.partitions:
                messages = self.topic.consume(self.consumer_group, partition, max_messages=100)

                for msg in messages:
                    self._records_read += 1
                    yield msg

                    if self.auto_commit:
                        self.topic.commit(self.consumer_group, partition, msg.offset + 1)

                if messages:
                    got_messages = True

            if not got_messages:
                time.sleep(self.poll_timeout)


class MessageQueueSink(DataSink[Dict[str, Any]]):
    """
    Message queue sink for producing records.
    """

    def __init__(self, topic: Topic, key_field: Optional[str] = None):
        super().__init__(f"mq:{topic.name}")
        self.topic = topic
        self.key_field = key_field

    def open(self) -> None:
        self._start_time = time.time()
        self._is_open = True

    def close(self) -> None:
        self._is_open = False

    def write(self, record: Dict[str, Any]) -> None:
        if not self._is_open:
            raise RuntimeError("Sink not open")

        key = record.get(self.key_field) if self.key_field else None
        self.topic.produce(str(key) if key else None, record)
        self._records_written += 1
