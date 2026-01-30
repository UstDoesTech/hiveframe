"""
HiveFrame Data Sources
======================
Source implementations for reading data from various sources.
"""

import csv
import hashlib
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Generic, List, Optional, TextIO, TypeVar, Union

from ..exceptions import (
    ConfigurationError,
    DataError,
    NetworkError,
    ParseError,
    RateLimitError,
    TransientError,
)
from ..exceptions import (
    ConnectionError as HiveConnectionError,
)
from ..resilience import (
    BackoffStrategy,
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryContext,
    RetryPolicy,
)
from ..utils import ManagedResource

T = TypeVar("T")


# ============================================================================
# Base Connector Interface
# ============================================================================


class DataSource(ManagedResource, Generic[T]):
    """
    Abstract base for all data sources.

    Provides a uniform interface for reading data from various sources
    with built-in support for:
    - Lazy iteration (generators)
    - Error handling and recovery
    - Metrics collection
    - Resource cleanup

    Extends ManagedResource for lifecycle management.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._records_read = 0

    @abstractmethod
    def read(self) -> Generator[T, None, None]:
        """Read records from the source as a generator."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get source statistics (alias for get_metrics)."""
        stats = super().get_stats()
        stats["records_read"] = self._records_read
        return stats

    # Alias for backward compatibility
    def get_metrics(self) -> Dict[str, Any]:
        """Get source metrics."""
        return self.get_stats()


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
        delimiter: str = ",",
        quotechar: str = '"',
        has_header: bool = True,
        columns: Optional[List[str]] = None,
        encoding: str = "utf-8",
        skip_malformed: bool = True,
        max_field_size: int = 131072,
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
        self._file: Optional[TextIO] = None
        self._reader: Optional[Any] = None  # csv.reader type

    def _do_open(self) -> None:
        if not self.path.exists():
            raise ConfigurationError(f"CSV file not found: {self.path}")

        csv.field_size_limit(self.max_field_size)
        self._file = open(self.path, "r", encoding=self.encoding, newline="")
        self._reader = csv.reader(self._file, delimiter=self.delimiter, quotechar=self.quotechar)

        # Read header if present
        if self.has_header and self.columns is None:
            try:
                assert self._reader is not None
                self.columns = next(self._reader)
            except StopIteration:
                raise DataError("CSV file is empty")

    def _do_close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
        self._reader = None

    def read(self) -> Generator[Dict[str, str], None, None]:
        if not self._is_open:
            raise RuntimeError("Source not open")

        assert self._reader is not None, "Reader must be initialized"

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
                        position=line_num,
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
        self, path: Union[str, Path], encoding: str = "utf-8", skip_malformed: bool = True
    ):
        super().__init__(f"jsonl:{path}")
        self.path = Path(path)
        self.encoding = encoding
        self.skip_malformed = skip_malformed
        self._file: Optional[TextIO] = None

    def _do_open(self) -> None:
        if not self.path.exists():
            raise ConfigurationError(f"JSONL file not found: {self.path}")

        self._file = open(self.path, "r", encoding=self.encoding)

    def _do_close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def read(self) -> Generator[Dict[str, Any], None, None]:
        if not self._is_open:
            raise RuntimeError("Source not open")

        assert self._file is not None, "File must be open"

        line: str
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
                        format_type="json",
                    )


class JSONSource(DataSource[Dict[str, Any]]):
    """
    Regular JSON file source.

    Expects either a JSON array or object with array field.
    Loads entire file into memory - use JSONLSource for large files.
    """

    def __init__(
        self, path: Union[str, Path], array_field: Optional[str] = None, encoding: str = "utf-8"
    ):
        super().__init__(f"json:{path}")
        self.path = Path(path)
        self.array_field = array_field
        self.encoding = encoding
        self._data: List[Dict[str, Any]] = []

    def _do_open(self) -> None:
        if not self.path.exists():
            raise ConfigurationError(f"JSON file not found: {self.path}")

        with open(self.path, "r", encoding=self.encoding) as f:
            content = json.load(f)

        if self.array_field:
            if not isinstance(content, dict) or self.array_field not in content:
                raise DataError(f"Field '{self.array_field}' not found in JSON")
            self._data = content[self.array_field]
        elif isinstance(content, list):
            self._data = content
        else:
            raise DataError("JSON must be array or object with array field specified")

    def _do_close(self) -> None:
        self._data = []

    def read(self) -> Generator[Dict[str, Any], None, None]:
        if not self._is_open:
            raise RuntimeError("Source not open")

        for record in self._data:
            self._records_read += 1
            yield record


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
        paginator: Optional[Callable[[Dict, int], Optional[str]]] = None,
    ):
        super().__init__(f"http:{base_url}")
        self.base_url = base_url.rstrip("/")
        self.config = config or HTTPConfig()
        self.paginator = paginator  # Returns next URL or None

        self._circuit = CircuitBreaker(
            f"http_{hashlib.md5(base_url.encode()).hexdigest()[:8]}",
            CircuitBreakerConfig(failure_threshold=self.config.circuit_failure_threshold),
        )
        self._last_request_time = 0.0
        self._http_lock = threading.Lock()

    def _do_open(self) -> None:
        pass  # HTTP connections are established per-request

    def _do_close(self) -> None:
        pass  # HTTP connections are closed per-request

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        with self._http_lock:
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
                data = response.read().decode("utf-8")
                result: Dict[str, Any] = json.loads(data)
                return result

        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited - get retry-after header
                retry_after = float(e.headers.get("Retry-After", 60))
                raise RateLimitError(f"Rate limited by {url}", retry_after=retry_after)
            elif e.code >= 500:
                raise NetworkError(f"Server error {e.code} from {url}", endpoint=url)
            else:
                raise DataError(f"HTTP {e.code}: {e.reason}")

        except urllib.error.URLError as e:
            raise HiveConnectionError(
                f"Connection failed to {url}: {e.reason}", host=urllib.parse.urlparse(url).netloc
            )

    def read(self) -> Generator[Dict[str, Any], None, None]:
        """Read from HTTP source with pagination support."""
        if not self._is_open:
            raise RuntimeError("Source not open")

        url: Optional[str] = self.base_url
        page = 0

        retry_policy = RetryPolicy(
            max_retries=self.config.max_retries, strategy=BackoffStrategy.EXPONENTIAL
        )

        while url:
            # Use circuit breaker
            if not self._circuit.allow_request():
                raise NetworkError(f"Circuit open for {self.base_url}", endpoint=self.base_url)

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
                    items = data.get("data") or data.get("items") or data.get("results")
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

            except Exception:
                self._circuit.record_failure()
                self._errors += 1
                raise


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
        delay_variance: float = 0.1,  # Variance in record timing
    ):
        super().__init__("generator")
        self.count = count
        self.rate = rate
        self.error_rate = error_rate
        self.schema = schema or {
            "id": "uuid",
            "timestamp": "timestamp",
            "value": "float",
            "category": "category",
            "message": "text",
        }
        self.delay_variance = delay_variance
        self._generated = 0

    def _do_open(self) -> None:
        self._generated = 0

    def _do_close(self) -> None:
        pass  # No resources to clean up

    def _generate_value(self, field_type: str, index: int) -> Any:
        """Generate value for field type."""
        import random
        import string

        if field_type == "uuid":
            return hashlib.md5(f"{index}-{time.time()}".encode()).hexdigest()
        elif field_type == "timestamp":
            return time.time()
        elif field_type == "int":
            return random.randint(0, 1000000)
        elif field_type == "float":
            return random.random() * 1000
        elif field_type == "bool":
            return random.choice([True, False])
        elif field_type == "category":
            return random.choice(["A", "B", "C", "D", "E"])
        elif field_type == "text":
            length = random.randint(10, 100)
            return "".join(random.choices(string.ascii_letters + " ", k=length))
        elif field_type.startswith("enum:"):
            values = field_type[5:].split(",")
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
            if self.rate and self._start_time is not None:
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
