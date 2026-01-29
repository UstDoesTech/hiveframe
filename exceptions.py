"""
HiveFrame Exception Hierarchy
=============================
Robust error handling with categorized exceptions for retry logic,
dead letter queues, and graceful degradation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from enum import Enum, auto
import time
import traceback


class ErrorSeverity(Enum):
    """Severity levels for error categorization."""
    DEBUG = auto()      # Informational, no action needed
    WARNING = auto()    # Recoverable, may indicate issues
    ERROR = auto()      # Requires attention, retryable
    CRITICAL = auto()   # System-level failure, may not be retryable
    FATAL = auto()      # Unrecoverable, requires shutdown


class ErrorCategory(Enum):
    """Categories for error routing and handling."""
    TRANSIENT = auto()       # Temporary failures (network, timeout)
    DATA_QUALITY = auto()    # Malformed or invalid data
    RESOURCE = auto()        # Resource exhaustion (memory, connections)
    CONFIGURATION = auto()   # Invalid configuration
    DEPENDENCY = auto()      # External dependency failures
    INTERNAL = auto()        # Internal logic errors


class HiveFrameError(Exception):
    """
    Base exception for all HiveFrame errors.
    
    Provides structured error information for:
    - Retry decision logic
    - Dead letter queue routing
    - Monitoring and alerting
    - Root cause analysis
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.retryable = retryable
        self.details = details or {}
        self.cause = cause
        self.timestamp = time.time()
        self.traceback_str = traceback.format_exc()
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize error for logging/storage."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'severity': self.severity.name,
            'category': self.category.name,
            'retryable': self.retryable,
            'details': self.details,
            'timestamp': self.timestamp,
            'cause': str(self.cause) if self.cause else None,
            'traceback': self.traceback_str
        }


# ============================================================================
# Transient Errors (Retryable)
# ============================================================================

class TransientError(HiveFrameError):
    """Base class for temporary, retryable failures."""
    
    def __init__(
        self,
        message: str,
        retry_after: float = 1.0,
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.TRANSIENT,
            retryable=True,
            **kwargs
        )
        self.retry_after = retry_after
        self.max_retries = max_retries


class NetworkError(TransientError):
    """Network-related transient failures."""
    
    def __init__(self, message: str, endpoint: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.details['endpoint'] = endpoint


class TimeoutError(TransientError):
    """Operation exceeded time limit."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float = 0,
        operation: str = "",
        **kwargs
    ):
        super().__init__(message, retry_after=timeout_seconds * 0.5, **kwargs)
        self.details['timeout_seconds'] = timeout_seconds
        self.details['operation'] = operation


class RateLimitError(TransientError):
    """Rate limit exceeded from external service."""
    
    def __init__(
        self,
        message: str,
        retry_after: float = 60.0,
        limit_type: str = "",
        **kwargs
    ):
        super().__init__(message, retry_after=retry_after, **kwargs)
        self.details['limit_type'] = limit_type


class ConnectionError(TransientError):
    """Connection establishment or maintenance failure."""
    
    def __init__(self, message: str, host: str = "", port: int = 0, **kwargs):
        super().__init__(message, **kwargs)
        self.details['host'] = host
        self.details['port'] = port


# ============================================================================
# Data Quality Errors (May or may not be retryable)
# ============================================================================

class DataError(HiveFrameError):
    """Base class for data-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA_QUALITY,
            **kwargs
        )


class ValidationError(DataError):
    """Data failed validation rules."""
    
    def __init__(
        self,
        message: str,
        field: str = "",
        expected: Any = None,
        actual: Any = None,
        **kwargs
    ):
        super().__init__(message, retryable=False, **kwargs)
        self.details['field'] = field
        self.details['expected'] = str(expected)
        self.details['actual'] = str(actual)


class SchemaError(DataError):
    """Schema mismatch or violation."""
    
    def __init__(
        self,
        message: str,
        expected_schema: Optional[Dict] = None,
        actual_schema: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(message, retryable=False, **kwargs)
        self.details['expected_schema'] = expected_schema
        self.details['actual_schema'] = actual_schema


class ParseError(DataError):
    """Failed to parse input data."""
    
    def __init__(
        self,
        message: str,
        raw_data: str = "",
        position: int = -1,
        format_type: str = "",
        **kwargs
    ):
        super().__init__(message, retryable=False, **kwargs)
        self.details['raw_data_preview'] = raw_data[:200] if raw_data else ""
        self.details['position'] = position
        self.details['format_type'] = format_type


class EncodingError(DataError):
    """Character encoding/decoding failure."""
    
    def __init__(
        self,
        message: str,
        expected_encoding: str = "utf-8",
        detected_encoding: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, retryable=False, **kwargs)
        self.details['expected_encoding'] = expected_encoding
        self.details['detected_encoding'] = detected_encoding


class NullValueError(DataError):
    """Unexpected null/None value encountered."""
    
    def __init__(self, message: str, field: str = "", **kwargs):
        super().__init__(message, retryable=False, **kwargs)
        self.details['field'] = field


# ============================================================================
# Resource Errors
# ============================================================================

class ResourceError(HiveFrameError):
    """Base class for resource-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class MemoryError(ResourceError):
    """Memory exhaustion or allocation failure."""
    
    def __init__(
        self,
        message: str,
        required_bytes: int = 0,
        available_bytes: int = 0,
        **kwargs
    ):
        super().__init__(message, retryable=True, **kwargs)
        self.details['required_bytes'] = required_bytes
        self.details['available_bytes'] = available_bytes


class DiskSpaceError(ResourceError):
    """Disk space exhaustion."""
    
    def __init__(
        self,
        message: str,
        path: str = "",
        required_bytes: int = 0,
        **kwargs
    ):
        super().__init__(message, retryable=True, **kwargs)
        self.details['path'] = path
        self.details['required_bytes'] = required_bytes


class ConnectionPoolExhausted(ResourceError):
    """No available connections in pool."""
    
    def __init__(self, message: str, pool_size: int = 0, **kwargs):
        super().__init__(message, retryable=True, **kwargs)
        self.details['pool_size'] = pool_size


class WorkerExhausted(ResourceError):
    """All workers are busy or unavailable."""
    
    def __init__(
        self,
        message: str,
        active_workers: int = 0,
        total_workers: int = 0,
        **kwargs
    ):
        super().__init__(message, retryable=True, **kwargs)
        self.details['active_workers'] = active_workers
        self.details['total_workers'] = total_workers


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(HiveFrameError):
    """Base class for configuration errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.FATAL,
            retryable=False,
            **kwargs
        )


class InvalidParameterError(ConfigurationError):
    """Invalid parameter value."""
    
    def __init__(
        self,
        message: str,
        parameter: str = "",
        value: Any = None,
        allowed_values: Optional[List] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.details['parameter'] = parameter
        self.details['value'] = str(value)
        self.details['allowed_values'] = allowed_values


class MissingConfigError(ConfigurationError):
    """Required configuration is missing."""
    
    def __init__(
        self,
        message: str,
        config_key: str = "",
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.details['config_key'] = config_key


# ============================================================================
# Dependency Errors
# ============================================================================

class DependencyError(HiveFrameError):
    """Base class for external dependency errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DEPENDENCY,
            **kwargs
        )


class ServiceUnavailable(DependencyError):
    """External service is unavailable."""
    
    def __init__(
        self,
        message: str,
        service_name: str = "",
        endpoint: str = "",
        **kwargs
    ):
        super().__init__(message, retryable=True, **kwargs)
        self.details['service_name'] = service_name
        self.details['endpoint'] = endpoint


class CircuitOpenError(DependencyError):
    """Circuit breaker is open due to repeated failures."""
    
    def __init__(
        self,
        message: str,
        service_name: str = "",
        reset_time: float = 0,
        **kwargs
    ):
        super().__init__(message, retryable=True, **kwargs)
        self.details['service_name'] = service_name
        self.details['reset_time'] = reset_time


class UpstreamError(DependencyError):
    """Error propagated from upstream system."""
    
    def __init__(
        self,
        message: str,
        upstream_error_code: str = "",
        upstream_service: str = "",
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.details['upstream_error_code'] = upstream_error_code
        self.details['upstream_service'] = upstream_service


# ============================================================================
# Processing Errors
# ============================================================================

class ProcessingError(HiveFrameError):
    """Error during data processing."""
    
    def __init__(
        self,
        message: str,
        partition_id: str = "",
        record_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.details['partition_id'] = partition_id
        self.details['record_key'] = record_key


class TransformError(ProcessingError):
    """Error in transformation function."""
    pass


class AggregationError(ProcessingError):
    """Error during aggregation operation."""
    pass


class JoinError(ProcessingError):
    """Error during join operation."""
    pass


# ============================================================================
# Dead Letter Queue Support
# ============================================================================

@dataclass
class DeadLetterRecord:
    """
    Record that failed processing and was routed to dead letter queue.
    
    Contains full context for debugging and potential reprocessing.
    """
    original_data: Any
    error: HiveFrameError
    partition_id: str
    worker_id: str
    attempt_count: int
    first_failure: float
    last_failure: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'original_data': str(self.original_data)[:1000],  # Truncate large data
            'error': self.error.to_dict(),
            'partition_id': self.partition_id,
            'worker_id': self.worker_id,
            'attempt_count': self.attempt_count,
            'first_failure': self.first_failure,
            'last_failure': self.last_failure,
            'metadata': self.metadata
        }


class DeadLetterQueue:
    """
    Dead Letter Queue for failed records.
    
    Provides storage, monitoring, and potential reprocessing of failed items.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue: List[DeadLetterRecord] = []
        self._lock = __import__('threading').Lock()
        self._error_counts: Dict[str, int] = {}
        
    def push(self, record: DeadLetterRecord) -> bool:
        """Add failed record to DLQ. Returns False if full."""
        with self._lock:
            if len(self._queue) >= self.max_size:
                return False
                
            self._queue.append(record)
            
            # Track error type counts
            error_type = record.error.__class__.__name__
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
            
            return True
            
    def pop(self) -> Optional[DeadLetterRecord]:
        """Remove and return oldest record."""
        with self._lock:
            if self._queue:
                record = self._queue.pop(0)
                error_type = record.error.__class__.__name__
                self._error_counts[error_type] = max(0, self._error_counts.get(error_type, 0) - 1)
                return record
            return None
            
    def peek(self, n: int = 10) -> List[DeadLetterRecord]:
        """View oldest n records without removing."""
        with self._lock:
            return self._queue[:n]
            
    def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        with self._lock:
            return {
                'size': len(self._queue),
                'max_size': self.max_size,
                'utilization': len(self._queue) / self.max_size if self.max_size > 0 else 0,
                'error_distribution': dict(self._error_counts),
                'oldest_timestamp': self._queue[0].first_failure if self._queue else None,
                'newest_timestamp': self._queue[-1].last_failure if self._queue else None
            }
            
    def clear(self) -> int:
        """Clear all records. Returns count of cleared records."""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            self._error_counts.clear()
            return count
            
    def get_by_error_type(self, error_type: str) -> List[DeadLetterRecord]:
        """Get all records with specific error type."""
        with self._lock:
            return [r for r in self._queue if r.error.__class__.__name__ == error_type]
