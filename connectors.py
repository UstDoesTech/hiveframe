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

import json
import csv
import time
import threading
import hashlib
import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, Generic, Iterator, 
    List, Optional, Tuple, TypeVar, Union
)
from queue import Queue, Empty
import urllib.request
import urllib.error
import urllib.parse

from .exceptions import (
    HiveFrameError, TransientError, NetworkError, ParseError,
    ValidationError, RateLimitError, ConnectionError as HiveConnectionError,
    TimeoutError as HiveTimeoutError, ConfigurationError, DataError
)
from .resilience import (
    RetryPolicy, RetryContext, CircuitBreaker, CircuitBreakerConfig,
    BackoffStrategy, with_retry, with_timeout
)


T = TypeVar('T')


# ============================================================================
# Base Connector Interface
# ============================================================================

class DataSource(ABC, Generic[T]):
    """
    Abstract base for all data sources.
    
    Provides a uniform interface for reading data from various sources
    with built-in support for:
    - Lazy iteration (generators)
    - Error handling and recovery
    - Metrics collection
    - Resource cleanup
    """
    
    def __init__(self, name: str):
        self.name = name
        self._records_read = 0
        self._errors = 0
        self._start_time: Optional[float] = None
        self._is_open = False
        
    @abstractmethod
    def open(self) -> None:
        """Open the data source for reading."""
        pass
        
    @abstractmethod
    def close(self) -> None:
        """Close the data source and release resources."""
        pass
        
    @abstractmethod
    def read(self) -> Generator[T, None, None]:
        """Read records from the source as a generator."""
        pass
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get source metrics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            'name': self.name,
            'records_read': self._records_read,
            'errors': self._errors,
            'elapsed_seconds': elapsed,
            'records_per_second': self._records_read / elapsed if elapsed > 0 else 0,
            'is_open': self._is_open
        }


class DataSink(ABC, Generic[T]):
    """
    Abstract base for all data sinks.
    
    Provides a uniform interface for writing data to various destinations.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._records_written = 0
        self._errors = 0
        self._start_time: Optional[float] = None
        self._is_open = False
        
    @abstractmethod
    def open(self) -> None:
        """Open the sink for writing."""
        pass
        
    @abstractmethod
    def close(self) -> None:
        """Close the sink and flush any buffers."""
        pass
        
    @abstractmethod
    def write(self, record: T) -> None:
        """Write a single record."""
        pass
        
    def write_batch(self, records: List[T]) -> int:
        """Write multiple records. Returns count written."""
        written = 0
        for record in records:
            try:
                self.write(record)
                written += 1
            except Exception:
                self._errors += 1
        return written
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get sink metrics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            'name': self.name,
            'records_written': self._records_written,
            'errors': self._errors,
            'elapsed_seconds': elapsed,
            'records_per_second': self._records_written / elapsed if elapsed > 0 else 0,
            'is_open': self._is_open
        }


# ============================================================================
# File Sources
# ============================================================================

class CSVSource(DataSource[Dict[str, str]]):
    """
    CSV file data source with error handling.
    
    Features:
    - Configurable delimiter and quote char
    - Header detection or explicit column names
    - Malformed row handling
    - Large file support (streaming)
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        delimiter: str = ',',
        quotechar: str = '"',
        has_header: bool = True,
        columns: Optional[List[str]] = None,
        encoding: str = 'utf-8',
        skip_malformed: bool = True,
        max_field_size: int = 131072
    ):
        super().__init__(f"csv:{path}")
        self.path = Path(path)
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.has_header = has_header
        self.columns = columns
        self.encoding = encoding
        self.skip_malformed = skip_malformed
        self.max_field_size = max_field_size
        self._file = None
        self._reader = None
        
    def open(self) -> None:
        if not self.path.exists():
            raise ConfigurationError(f"CSV file not found: {self.path}")
            
        csv.field_size_limit(self.max_field_size)
        self._file = open(self.path, 'r', encoding=self.encoding, newline='')
        self._reader = csv.reader(
            self._file, 
            delimiter=self.delimiter,
            quotechar=self.quotechar
        )
        
        # Read header if present
        if self.has_header and self.columns is None:
            try:
                self.columns = next(self._reader)
            except StopIteration:
                raise DataError("CSV file is empty")
                
        self._start_time = time.time()
        self._is_open = True
        
    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
        self._reader = None
        self._is_open = False
        
    def read(self) -> Generator[Dict[str, str], None, None]:
        if not self._is_open:
            raise RuntimeError("Source not open")
            
        line_num = 1 if self.has_header else 0
        
        for row in self._reader:
            line_num += 1
            
            try:
                if self.columns and len(row) != len(self.columns):
                    if self.skip_malformed:
                        self._errors += 1
                        continue
                    raise ParseError(
                        f"Row {line_num} has {len(row)} columns, expected {len(self.columns)}",
                        position=line_num
                    )
                    
                record = dict(zip(self.columns or [f"col_{i}" for i in range(len(row))], row))
                self._records_read += 1
                yield record
                
            except ParseError:
                raise
            except Exception as e:
                self._errors += 1
                if not self.skip_malformed:
                    raise ParseError(f"Error parsing row {line_num}: {e}", position=line_num)


class JSONLSource(DataSource[Dict[str, Any]]):
    """
    JSON Lines (newline-delimited JSON) source.
    
    Each line is a complete JSON object. More streaming-friendly
    than regular JSON arrays.
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        encoding: str = 'utf-8',
        skip_malformed: bool = True
    ):
        super().__init__(f"jsonl:{path}")
        self.path = Path(path)
        self.encoding = encoding
        self.skip_malformed = skip_malformed
        self._file = None
        
    def open(self) -> None:
        if not self.path.exists():
            raise ConfigurationError(f"JSONL file not found: {self.path}")
            
        self._file = open(self.path, 'r', encoding=self.encoding)
        self._start_time = time.time()
        self._is_open = True
        
    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
        self._is_open = False
        
    def read(self) -> Generator[Dict[str, Any], None, None]:
        if not self._is_open:
            raise RuntimeError("Source not open")
            
        for line_num, line in enumerate(self._file, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                record = json.loads(line)
                self._records_read += 1
                yield record
                
            except json.JSONDecodeError as e:
                self._errors += 1
                if not self.skip_malformed:
                    raise ParseError(
                        f"Invalid JSON on line {line_num}: {e}",
                        raw_data=line[:100],
                        position=line_num,
                        format_type='json'
                    )


class JSONSource(DataSource[Dict[str, Any]]):
    """
    Regular JSON file source.
    
    Expects either a JSON array or object with array field.
    Loads entire file into memory - use JSONLSource for large files.
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        array_field: Optional[str] = None,
        encoding: str = 'utf-8'
    ):
        super().__init__(f"json:{path}")
        self.path = Path(path)
        self.array_field = array_field
        self.encoding = encoding
        self._data: List[Dict[str, Any]] = []
        
    def open(self) -> None:
        if not self.path.exists():
            raise ConfigurationError(f"JSON file not found: {self.path}")
            
        with open(self.path, 'r', encoding=self.encoding) as f:
            content = json.load(f)
            
        if self.array_field:
            if not isinstance(content, dict) or self.array_field not in content:
                raise DataError(f"Field '{self.array_field}' not found in JSON")
            self._data = content[self.array_field]
        elif isinstance(content, list):
            self._data = content
        else:
            raise DataError("JSON must be array or object with array field specified")
            
        self._start_time = time.time()
        self._is_open = True
        
    def close(self) -> None:
        self._data = []
        self._is_open = False
        
    def read(self) -> Generator[Dict[str, Any], None, None]:
        if not self._is_open:
            raise RuntimeError("Source not open")
            
        for record in self._data:
            self._records_read += 1
            yield record


# ============================================================================
# File Sinks
# ============================================================================

class JSONLSink(DataSink[Dict[str, Any]]):
    """JSON Lines output sink."""
    
    def __init__(
        self,
        path: Union[str, Path],
        encoding: str = 'utf-8',
        append: bool = False
    ):
        super().__init__(f"jsonl:{path}")
        self.path = Path(path)
        self.encoding = encoding
        self.append = append
        self._file = None
        
    def open(self) -> None:
        mode = 'a' if self.append else 'w'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, mode, encoding=self.encoding)
        self._start_time = time.time()
        self._is_open = True
        
    def close(self) -> None:
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
        self._is_open = False
        
    def write(self, record: Dict[str, Any]) -> None:
        if not self._is_open:
            raise RuntimeError("Sink not open")
            
        self._file.write(json.dumps(record) + '\n')
        self._records_written += 1


class CSVSink(DataSink[Dict[str, str]]):
    """CSV output sink."""
    
    def __init__(
        self,
        path: Union[str, Path],
        columns: Optional[List[str]] = None,
        delimiter: str = ',',
        write_header: bool = True,
        encoding: str = 'utf-8',
        append: bool = False
    ):
        super().__init__(f"csv:{path}")
        self.path = Path(path)
        self.columns = columns
        self.delimiter = delimiter
        self.write_header = write_header
        self.encoding = encoding
        self.append = append
        self._file = None
        self._writer = None
        self._header_written = False
        
    def open(self) -> None:
        mode = 'a' if self.append else 'w'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, mode, encoding=self.encoding, newline='')
        self._writer = csv.writer(self._file, delimiter=self.delimiter)
        self._start_time = time.time()
        self._is_open = True
        
        # Don't write header if appending to existing file
        if self.append and self.path.stat().st_size > 0:
            self._header_written = True
            
    def close(self) -> None:
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
        self._writer = None
        self._is_open = False
        
    def write(self, record: Dict[str, str]) -> None:
        if not self._is_open:
            raise RuntimeError("Sink not open")
            
        # Infer columns from first record if not specified
        if self.columns is None:
            self.columns = list(record.keys())
            
        # Write header if needed
        if self.write_header and not self._header_written:
            self._writer.writerow(self.columns)
            self._header_written = True
            
        row = [record.get(col, '') for col in self.columns]
        self._writer.writerow(row)
        self._records_written += 1


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
        process_existing: bool = True
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
                    self._events.put(FileEvent('created', path))
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
            return self._events.get(timeout=timeout)
        except Empty:
            return None
            
    def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._running:
            try:
                current_files = {
                    p: p.stat().st_mtime 
                    for p in self.directory.glob(self.pattern)
                    if p.is_file()
                }
                
                # Check for new files
                for path, mtime in current_files.items():
                    if path not in self._known_files:
                        self._events.put(FileEvent('created', path))
                    elif mtime > self._known_files[path]:
                        self._events.put(FileEvent('modified', path))
                        
                # Check for deleted files
                for path in list(self._known_files.keys()):
                    if path not in current_files:
                        self._events.put(FileEvent('deleted', path))
                        del self._known_files[path]
                        
                self._known_files = current_files
                
            except Exception:
                pass  # Ignore errors in watch loop
                
            time.sleep(self.poll_interval)


# ============================================================================
# HTTP API Source
# ============================================================================

@dataclass
class HTTPConfig:
    """Configuration for HTTP requests."""
    timeout: float = 30.0
    max_retries: int = 3
    headers: Dict[str, str] = field(default_factory=dict)
    rate_limit_per_second: float = 10.0
    circuit_failure_threshold: int = 5


class HTTPSource(DataSource[Dict[str, Any]]):
    """
    HTTP API data source with resilience patterns.
    
    Features:
    - Rate limiting
    - Automatic retries with backoff
    - Circuit breaker protection
    - Pagination support
    """
    
    def __init__(
        self,
        base_url: str,
        config: Optional[HTTPConfig] = None,
        paginator: Optional[Callable[[Dict, int], Optional[str]]] = None
    ):
        super().__init__(f"http:{base_url}")
        self.base_url = base_url.rstrip('/')
        self.config = config or HTTPConfig()
        self.paginator = paginator  # Returns next URL or None
        
        self._circuit = CircuitBreaker(
            f"http_{hashlib.md5(base_url.encode()).hexdigest()[:8]}",
            CircuitBreakerConfig(failure_threshold=self.config.circuit_failure_threshold)
        )
        self._last_request_time = 0.0
        self._lock = threading.Lock()
        
    def open(self) -> None:
        self._start_time = time.time()
        self._is_open = True
        
    def close(self) -> None:
        self._is_open = False
        
    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        with self._lock:
            min_interval = 1.0 / self.config.rate_limit_per_second
            elapsed = time.time() - self._last_request_time
            
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
                
            self._last_request_time = time.time()
            
    def _make_request(self, url: str) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        self._rate_limit()
        
        request = urllib.request.Request(url)
        for key, value in self.config.headers.items():
            request.add_header(key, value)
            
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                data = response.read().decode('utf-8')
                return json.loads(data)
                
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited - get retry-after header
                retry_after = float(e.headers.get('Retry-After', 60))
                raise RateLimitError(
                    f"Rate limited by {url}",
                    retry_after=retry_after
                )
            elif e.code >= 500:
                raise NetworkError(
                    f"Server error {e.code} from {url}",
                    endpoint=url
                )
            else:
                raise DataError(f"HTTP {e.code}: {e.reason}")
                
        except urllib.error.URLError as e:
            raise HiveConnectionError(
                f"Connection failed to {url}: {e.reason}",
                host=urllib.parse.urlparse(url).netloc
            )
        except TimeoutError:
            raise HiveTimeoutError(
                f"Request timed out: {url}",
                timeout_seconds=self.config.timeout,
                operation='http_request'
            )
            
    def read(self) -> Generator[Dict[str, Any], None, None]:
        """Read from HTTP source with pagination support."""
        if not self._is_open:
            raise RuntimeError("Source not open")
            
        url = self.base_url
        page = 0
        
        retry_policy = RetryPolicy(
            max_retries=self.config.max_retries,
            strategy=BackoffStrategy.EXPONENTIAL
        )
        
        while url:
            # Use circuit breaker
            if not self._circuit.allow_request():
                raise NetworkError(
                    f"Circuit open for {self.base_url}",
                    endpoint=self.base_url
                )
                
            try:
                with RetryContext(retry_policy) as retry:
                    for attempt in retry:
                        try:
                            data = self._make_request(url)
                            retry.success()
                            break
                        except TransientError as e:
                            retry.record_failure(e)
                            
                self._circuit.record_success()
                
                # Handle response
                if isinstance(data, list):
                    for record in data:
                        self._records_read += 1
                        yield record
                elif isinstance(data, dict):
                    # Check for data array field
                    items = data.get('data') or data.get('items') or data.get('results')
                    if isinstance(items, list):
                        for record in items:
                            self._records_read += 1
                            yield record
                    else:
                        self._records_read += 1
                        yield data
                        
                # Get next page URL
                if self.paginator:
                    url = self.paginator(data, page)
                    page += 1
                else:
                    url = None
                    
            except Exception as e:
                self._circuit.record_failure()
                self._errors += 1
                raise


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
        
    def produce(self, key: Optional[str], value: Any, headers: Optional[Dict[str, str]] = None) -> Message:
        """Produce message to topic."""
        partition = self._get_partition(key)
        
        with self._locks[partition]:
            offset = len(self._partitions[partition])
            message = Message(
                key=key,
                value=value,
                partition=partition,
                offset=offset,
                headers=headers or {}
            )
            self._partitions[partition].append(message)
            return message
            
    def consume(
        self,
        consumer_group: str,
        partition: int,
        max_messages: int = 100,
        timeout: float = 1.0
    ) -> List[Message]:
        """Consume messages from a partition."""
        if partition >= self.num_partitions:
            raise ValueError(f"Partition {partition} does not exist")
            
        # Get current offset for this consumer group
        if consumer_group not in self._offsets:
            self._offsets[consumer_group] = {i: 0 for i in range(self.num_partitions)}
            
        current_offset = self._offsets[consumer_group].get(partition, 0)
        
        with self._locks[partition]:
            messages = self._partitions[partition][current_offset:current_offset + max_messages]
            
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
        poll_timeout: float = 1.0
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
                messages = self.topic.consume(
                    self.consumer_group,
                    partition,
                    max_messages=100,
                    timeout=self.poll_timeout
                )
                
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
    
    def __init__(
        self,
        topic: Topic,
        key_field: Optional[str] = None
    ):
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


# ============================================================================
# Data Generator for Testing
# ============================================================================

class DataGenerator(DataSource[Dict[str, Any]]):
    """
    Synthetic data generator for testing and benchmarking.
    
    Generates realistic-looking data with configurable:
    - Schema
    - Rate
    - Error injection
    - Data patterns
    """
    
    def __init__(
        self,
        count: int = 1000,
        rate: Optional[float] = None,  # Records per second, None = as fast as possible
        error_rate: float = 0.0,  # Probability of error per record
        schema: Optional[Dict[str, str]] = None,  # field -> type
        delay_variance: float = 0.1  # Variance in record timing
    ):
        super().__init__("generator")
        self.count = count
        self.rate = rate
        self.error_rate = error_rate
        self.schema = schema or {
            'id': 'uuid',
            'timestamp': 'timestamp',
            'value': 'float',
            'category': 'category',
            'message': 'text'
        }
        self.delay_variance = delay_variance
        self._generated = 0
        
    def open(self) -> None:
        self._generated = 0
        self._start_time = time.time()
        self._is_open = True
        
    def close(self) -> None:
        self._is_open = False
        
    def _generate_value(self, field_type: str, index: int) -> Any:
        """Generate value for field type."""
        import random
        import string
        
        if field_type == 'uuid':
            return hashlib.md5(f"{index}-{time.time()}".encode()).hexdigest()
        elif field_type == 'timestamp':
            return time.time()
        elif field_type == 'int':
            return random.randint(0, 1000000)
        elif field_type == 'float':
            return random.random() * 1000
        elif field_type == 'bool':
            return random.choice([True, False])
        elif field_type == 'category':
            return random.choice(['A', 'B', 'C', 'D', 'E'])
        elif field_type == 'text':
            length = random.randint(10, 100)
            return ''.join(random.choices(string.ascii_letters + ' ', k=length))
        elif field_type.startswith('enum:'):
            values = field_type[5:].split(',')
            return random.choice(values)
        else:
            return f"value_{index}"
            
    def read(self) -> Generator[Dict[str, Any], None, None]:
        """Generate synthetic records."""
        if not self._is_open:
            raise RuntimeError("Source not open")
            
        import random
        
        for i in range(self.count):
            # Rate limiting
            if self.rate:
                expected_time = i / self.rate
                actual_time = time.time() - self._start_time
                
                if actual_time < expected_time:
                    # Add variance to timing
                    sleep_time = expected_time - actual_time
                    variance = sleep_time * self.delay_variance
                    sleep_time += random.uniform(-variance, variance)
                    time.sleep(max(0, sleep_time))
                    
            # Error injection
            if self.error_rate > 0 and random.random() < self.error_rate:
                self._errors += 1
                raise TransientError(f"Simulated error at record {i}")
                
            # Generate record
            record = {
                field: self._generate_value(field_type, i)
                for field, field_type in self.schema.items()
            }
            
            self._records_read += 1
            self._generated += 1
            yield record
