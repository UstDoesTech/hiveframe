"""
Tests for HiveFrame exceptions module.

Tests cover:
- Exception hierarchy and inheritance
- Error severity and category classification
- Serialization (to_dict)
- Transient errors (NetworkError, TimeoutError, RateLimitError, ConnectionError)
- Data errors (ValidationError, SchemaError, ParseError, EncodingError, NullValueError)
- Resource errors (MemoryError, DiskSpaceError, ConnectionPoolExhausted, WorkerExhausted)
- Configuration errors (InvalidParameterError, MissingConfigError)
- Dependency errors (ServiceUnavailable, CircuitOpenError, UpstreamError)
- Processing errors (TransformError, AggregationError, JoinError)
- Dead Letter Queue functionality
"""

import time

import pytest

from hiveframe.exceptions import (
    AggregationError,
    CircuitOpenError,
    # Configuration errors
    ConfigurationError,
    ConnectionPoolExhausted,
    # Data errors
    DataError,
    DeadLetterQueue,
    # DLQ
    DeadLetterRecord,
    # Dependency errors
    DependencyError,
    DiskSpaceError,
    EncodingError,
    ErrorCategory,
    ErrorSeverity,
    # Base and enums
    HiveFrameError,
    InvalidParameterError,
    JoinError,
    MissingConfigError,
    NetworkError,
    NullValueError,
    ParseError,
    # Processing errors
    ProcessingError,
    RateLimitError,
    # Resource errors
    ResourceError,
    SchemaError,
    ServiceUnavailable,
    TimeoutError,
    TransformError,
    # Transient errors
    TransientError,
    UpstreamError,
    ValidationError,
    WorkerExhausted,
)
from hiveframe.exceptions import (
    ConnectionError as HiveConnectionError,
)
from hiveframe.exceptions import (
    MemoryError as HiveMemoryError,
)

# ============================================================================
# Error Severity and Category Tests
# ============================================================================


class TestErrorEnums:
    """Tests for error enums."""

    def test_severity_levels(self):
        """Test all severity levels exist."""
        assert ErrorSeverity.DEBUG is not None
        assert ErrorSeverity.WARNING is not None
        assert ErrorSeverity.ERROR is not None
        assert ErrorSeverity.CRITICAL is not None
        assert ErrorSeverity.FATAL is not None

    def test_error_categories(self):
        """Test all error categories exist."""
        assert ErrorCategory.TRANSIENT is not None
        assert ErrorCategory.DATA_QUALITY is not None
        assert ErrorCategory.RESOURCE is not None
        assert ErrorCategory.CONFIGURATION is not None
        assert ErrorCategory.DEPENDENCY is not None
        assert ErrorCategory.INTERNAL is not None


# ============================================================================
# Base HiveFrameError Tests
# ============================================================================


class TestHiveFrameError:
    """Tests for base HiveFrameError."""

    def test_basic_creation(self):
        """Test creating a basic error."""
        error = HiveFrameError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"

    def test_default_values(self):
        """Test default severity, category, and retryable."""
        error = HiveFrameError("Test error")

        assert error.severity == ErrorSeverity.ERROR
        assert error.category == ErrorCategory.INTERNAL
        assert not error.retryable

    def test_custom_values(self):
        """Test custom severity and category."""
        error = HiveFrameError(
            "Critical failure",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.RESOURCE,
            retryable=True,
        )

        assert error.severity == ErrorSeverity.CRITICAL
        assert error.category == ErrorCategory.RESOURCE
        assert error.retryable

    def test_details(self):
        """Test error details dictionary."""
        error = HiveFrameError("Test error", details={"key": "value", "count": 42})

        assert error.details["key"] == "value"
        assert error.details["count"] == 42

    def test_cause_chaining(self):
        """Test exception cause chaining."""
        original = ValueError("Original error")
        error = HiveFrameError("Wrapped error", cause=original)

        assert error.cause == original

    def test_timestamp(self):
        """Test error has timestamp."""
        before = time.time()
        error = HiveFrameError("Test error")
        after = time.time()

        assert before <= error.timestamp <= after

    def test_to_dict(self):
        """Test serialization to dictionary."""
        error = HiveFrameError(
            "Test error",
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.DATA_QUALITY,
            retryable=True,
            details={"field": "name"},
        )

        data = error.to_dict()

        assert data["error_type"] == "HiveFrameError"
        assert data["message"] == "Test error"
        assert data["severity"] == "WARNING"
        assert data["category"] == "DATA_QUALITY"
        assert data["retryable"]
        assert data["details"]["field"] == "name"
        assert "timestamp" in data

    def test_is_exception(self):
        """Test that HiveFrameError is a proper exception."""
        with pytest.raises(HiveFrameError) as exc_info:
            raise HiveFrameError("Test exception")

        assert "Test exception" in str(exc_info.value)


# ============================================================================
# Transient Error Tests
# ============================================================================


class TestTransientErrors:
    """Tests for transient, retryable errors."""

    def test_transient_error_base(self):
        """Test TransientError base class."""
        error = TransientError("Temporary failure")

        assert error.category == ErrorCategory.TRANSIENT
        assert error.retryable
        assert error.retry_after == 1.0
        assert error.max_retries == 3

    def test_transient_custom_retry(self):
        """Test custom retry settings."""
        error = TransientError("Retry me", retry_after=5.0, max_retries=10)

        assert error.retry_after == 5.0
        assert error.max_retries == 10

    def test_network_error(self):
        """Test NetworkError."""
        error = NetworkError("Connection refused", endpoint="https://api.example.com")

        assert error.details["endpoint"] == "https://api.example.com"
        assert error.retryable

    def test_timeout_error(self):
        """Test TimeoutError."""
        error = TimeoutError("Request timed out", timeout_seconds=30.0, operation="fetch_data")

        assert error.details["timeout_seconds"] == 30.0
        assert error.details["operation"] == "fetch_data"
        assert error.retry_after == 15.0  # 50% of timeout

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Too many requests", retry_after=60.0, limit_type="hourly")

        assert error.retry_after == 60.0
        assert error.details["limit_type"] == "hourly"

    def test_connection_error(self):
        """Test ConnectionError."""
        error = HiveConnectionError("Failed to connect", host="localhost", port=5432)

        assert error.details["host"] == "localhost"
        assert error.details["port"] == 5432


# ============================================================================
# Data Error Tests
# ============================================================================


class TestDataErrors:
    """Tests for data quality errors."""

    def test_data_error_base(self):
        """Test DataError base class."""
        error = DataError("Invalid data")

        assert error.category == ErrorCategory.DATA_QUALITY

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError(
            "Age must be positive", field="age", expected="positive integer", actual=-5
        )

        assert error.details["field"] == "age"
        assert error.details["expected"] == "positive integer"
        assert error.details["actual"] == "-5"
        assert not error.retryable

    def test_schema_error(self):
        """Test SchemaError."""
        expected = {"name": "string", "age": "int"}
        actual = {"name": "string", "email": "string"}

        error = SchemaError("Schema mismatch", expected_schema=expected, actual_schema=actual)

        assert error.details["expected_schema"] == expected
        assert error.details["actual_schema"] == actual

    def test_parse_error(self):
        """Test ParseError."""
        error = ParseError("Invalid JSON", raw_data='{"broken": ', position=10, format_type="json")

        assert error.details["position"] == 10
        assert error.details["format_type"] == "json"
        assert '{"broken":' in error.details["raw_data_preview"]

    def test_encoding_error(self):
        """Test EncodingError."""
        error = EncodingError(
            "Cannot decode bytes", expected_encoding="utf-8", detected_encoding="latin-1"
        )

        assert error.details["expected_encoding"] == "utf-8"
        assert error.details["detected_encoding"] == "latin-1"

    def test_null_value_error(self):
        """Test NullValueError."""
        error = NullValueError("Unexpected null", field="user_id")

        assert error.details["field"] == "user_id"


# ============================================================================
# Resource Error Tests
# ============================================================================


class TestResourceErrors:
    """Tests for resource exhaustion errors."""

    def test_resource_error_base(self):
        """Test ResourceError base class."""
        error = ResourceError("Resource exhausted")

        assert error.category == ErrorCategory.RESOURCE
        assert error.severity == ErrorSeverity.CRITICAL

    def test_memory_error(self):
        """Test MemoryError."""
        error = HiveMemoryError(
            "Out of memory", required_bytes=1073741824, available_bytes=536870912  # 1GB  # 512MB
        )

        assert error.details["required_bytes"] == 1073741824
        assert error.details["available_bytes"] == 536870912
        assert error.retryable

    def test_disk_space_error(self):
        """Test DiskSpaceError."""
        error = DiskSpaceError("Disk full", path="/data", required_bytes=1000000000)

        assert error.details["path"] == "/data"
        assert error.details["required_bytes"] == 1000000000

    def test_connection_pool_exhausted(self):
        """Test ConnectionPoolExhausted."""
        error = ConnectionPoolExhausted("No connections available", pool_size=10)

        assert error.details["pool_size"] == 10
        assert error.retryable

    def test_worker_exhausted(self):
        """Test WorkerExhausted."""
        error = WorkerExhausted("All workers busy", active_workers=8, total_workers=8)

        assert error.details["active_workers"] == 8
        assert error.details["total_workers"] == 8


# ============================================================================
# Configuration Error Tests
# ============================================================================


class TestConfigurationErrors:
    """Tests for configuration errors."""

    def test_configuration_error_base(self):
        """Test ConfigurationError base class."""
        error = ConfigurationError("Invalid config")

        assert error.category == ErrorCategory.CONFIGURATION
        assert error.severity == ErrorSeverity.FATAL
        assert not error.retryable

    def test_invalid_parameter(self):
        """Test InvalidParameterError."""
        error = InvalidParameterError(
            "Invalid worker count",
            parameter="num_workers",
            value=-1,
            allowed_values=[1, 2, 4, 8, 16],
        )

        assert error.details["parameter"] == "num_workers"
        assert error.details["value"] == "-1"
        assert error.details["allowed_values"] == [1, 2, 4, 8, 16]

    def test_missing_config(self):
        """Test MissingConfigError."""
        error = MissingConfigError("Required config missing", config_key="database_url")

        assert error.details["config_key"] == "database_url"


# ============================================================================
# Dependency Error Tests
# ============================================================================


class TestDependencyErrors:
    """Tests for external dependency errors."""

    def test_dependency_error_base(self):
        """Test DependencyError base class."""
        error = DependencyError("Dependency failed")

        assert error.category == ErrorCategory.DEPENDENCY

    def test_service_unavailable(self):
        """Test ServiceUnavailable."""
        error = ServiceUnavailable(
            "Service down", service_name="payment-api", endpoint="https://payments.example.com"
        )

        assert error.details["service_name"] == "payment-api"
        assert error.details["endpoint"] == "https://payments.example.com"
        assert error.retryable

    def test_circuit_open(self):
        """Test CircuitOpenError."""
        error = CircuitOpenError("Circuit breaker open", service_name="database", reset_time=30.0)

        assert error.details["service_name"] == "database"
        assert error.details["reset_time"] == 30.0
        assert error.retryable

    def test_upstream_error(self):
        """Test UpstreamError."""
        error = UpstreamError(
            "Upstream service failed", upstream_error_code="ERR_503", upstream_service="inventory"
        )

        assert error.details["upstream_error_code"] == "ERR_503"
        assert error.details["upstream_service"] == "inventory"


# ============================================================================
# Processing Error Tests
# ============================================================================


class TestProcessingErrors:
    """Tests for processing errors."""

    def test_processing_error_base(self):
        """Test ProcessingError base class."""
        error = ProcessingError(
            "Processing failed", partition_id="partition-3", record_key="user-123"
        )

        assert error.details["partition_id"] == "partition-3"
        assert error.details["record_key"] == "user-123"

    def test_transform_error(self):
        """Test TransformError."""
        error = TransformError("Transform function failed", partition_id="p1")

        assert isinstance(error, ProcessingError)

    def test_aggregation_error(self):
        """Test AggregationError."""
        error = AggregationError("Failed to aggregate", partition_id="p2")

        assert isinstance(error, ProcessingError)

    def test_join_error(self):
        """Test JoinError."""
        error = JoinError("Join failed", partition_id="p3")

        assert isinstance(error, ProcessingError)


# ============================================================================
# Dead Letter Queue Tests
# ============================================================================


class TestDeadLetterRecord:
    """Tests for DeadLetterRecord."""

    def test_record_creation(self):
        """Test creating a dead letter record."""
        error = ValidationError("Invalid data", field="email")

        record = DeadLetterRecord(
            original_data={"email": "invalid"},
            error=error,
            partition_id="partition-1",
            worker_id="bee-42",
            attempt_count=3,
            first_failure=time.time() - 60,
        )

        assert record.original_data == {"email": "invalid"}
        assert record.attempt_count == 3
        assert record.partition_id == "partition-1"

    def test_record_to_dict(self):
        """Test serializing dead letter record."""
        error = ParseError("Bad JSON", raw_data='{"broken')

        record = DeadLetterRecord(
            original_data='{"broken',
            error=error,
            partition_id="p1",
            worker_id="w1",
            attempt_count=1,
            first_failure=time.time(),
        )

        data = record.to_dict()

        assert "original_data" in data
        assert "error" in data
        assert data["partition_id"] == "p1"
        assert data["attempt_count"] == 1

    def test_record_metadata(self):
        """Test record metadata."""
        error = DataError("Error")

        record = DeadLetterRecord(
            original_data={},
            error=error,
            partition_id="p1",
            worker_id="w1",
            attempt_count=1,
            first_failure=time.time(),
            metadata={"source": "kafka", "topic": "events"},
        )

        assert record.metadata["source"] == "kafka"


class TestDeadLetterQueue:
    """Tests for DeadLetterQueue."""

    def test_queue_creation(self):
        """Test creating a DLQ."""
        dlq = DeadLetterQueue(max_size=100)

        assert dlq.max_size == 100

    def test_push_record(self):
        """Test pushing records to DLQ."""
        dlq = DeadLetterQueue()
        error = DataError("Failed")

        record = DeadLetterRecord(
            original_data={"id": 1},
            error=error,
            partition_id="p1",
            worker_id="w1",
            attempt_count=1,
            first_failure=time.time(),
        )

        result = dlq.push(record)

        assert result
        assert dlq.get_stats()["size"] == 1

    def test_push_full_queue(self):
        """Test pushing to full DLQ."""
        dlq = DeadLetterQueue(max_size=2)

        for i in range(3):
            error = DataError(f"Error {i}")
            record = DeadLetterRecord(
                original_data={"id": i},
                error=error,
                partition_id="p1",
                worker_id="w1",
                attempt_count=1,
                first_failure=time.time(),
            )
            result = dlq.push(record)

            if i < 2:
                assert result
            else:
                assert not result

    def test_pop_record(self):
        """Test popping records from DLQ."""
        dlq = DeadLetterQueue()

        # Push two records
        for i in range(2):
            error = DataError(f"Error {i}")
            record = DeadLetterRecord(
                original_data={"id": i},
                error=error,
                partition_id="p1",
                worker_id="w1",
                attempt_count=1,
                first_failure=time.time(),
            )
            dlq.push(record)

        # Pop should return oldest first
        popped = dlq.pop()

        assert popped.original_data["id"] == 0
        assert dlq.get_stats()["size"] == 1

    def test_pop_empty_queue(self):
        """Test popping from empty DLQ."""
        dlq = DeadLetterQueue()

        result = dlq.pop()

        assert result is None

    def test_peek_records(self):
        """Test peeking at records without removing."""
        dlq = DeadLetterQueue()

        for i in range(5):
            error = DataError(f"Error {i}")
            record = DeadLetterRecord(
                original_data={"id": i},
                error=error,
                partition_id="p1",
                worker_id="w1",
                attempt_count=1,
                first_failure=time.time(),
            )
            dlq.push(record)

        peeked = dlq.peek(n=3)

        assert len(peeked) == 3
        assert dlq.get_stats()["size"] == 5  # None removed

    def test_get_stats(self):
        """Test DLQ statistics."""
        dlq = DeadLetterQueue(max_size=100)

        # Add different error types
        error1 = ValidationError("Bad data", field="x")
        error2 = ParseError("Bad json")
        error3 = ValidationError("Also bad", field="y")

        for error in [error1, error2, error3]:
            record = DeadLetterRecord(
                original_data={},
                error=error,
                partition_id="p1",
                worker_id="w1",
                attempt_count=1,
                first_failure=time.time(),
            )
            dlq.push(record)

        stats = dlq.get_stats()

        assert stats["size"] == 3
        assert stats["max_size"] == 100
        assert stats["error_distribution"]["ValidationError"] == 2
        assert stats["error_distribution"]["ParseError"] == 1

    def test_clear_queue(self):
        """Test clearing the DLQ."""
        dlq = DeadLetterQueue()

        for i in range(5):
            error = DataError(f"Error {i}")
            record = DeadLetterRecord(
                original_data={"id": i},
                error=error,
                partition_id="p1",
                worker_id="w1",
                attempt_count=1,
                first_failure=time.time(),
            )
            dlq.push(record)

        cleared = dlq.clear()

        assert cleared == 5
        assert dlq.get_stats()["size"] == 0

    def test_get_by_error_type(self):
        """Test filtering records by error type."""
        dlq = DeadLetterQueue()

        # Add different error types
        types = [ValidationError, ParseError, ValidationError, SchemaError]

        for i, error_class in enumerate(types):
            if error_class == ValidationError:
                error = error_class(f"Error {i}", field="test")
            else:
                error = error_class(f"Error {i}")

            record = DeadLetterRecord(
                original_data={"id": i},
                error=error,
                partition_id="p1",
                worker_id="w1",
                attempt_count=1,
                first_failure=time.time(),
            )
            dlq.push(record)

        validation_errors = dlq.get_by_error_type("ValidationError")

        assert len(validation_errors) == 2


# ============================================================================
# Exception Inheritance Tests
# ============================================================================


class TestExceptionHierarchy:
    """Tests for exception inheritance structure."""

    def test_all_inherit_from_base(self):
        """Test all custom exceptions inherit from HiveFrameError."""
        exceptions = [
            TransientError("test"),
            NetworkError("test"),
            TimeoutError("test"),
            RateLimitError("test"),
            HiveConnectionError("test"),
            DataError("test"),
            ValidationError("test", field="x"),
            SchemaError("test"),
            ParseError("test"),
            EncodingError("test"),
            NullValueError("test"),
            ResourceError("test"),
            HiveMemoryError("test"),
            DiskSpaceError("test"),
            ConnectionPoolExhausted("test"),
            WorkerExhausted("test"),
            ConfigurationError("test"),
            InvalidParameterError("test"),
            MissingConfigError("test"),
            DependencyError("test"),
            ServiceUnavailable("test"),
            CircuitOpenError("test"),
            UpstreamError("test"),
            ProcessingError("test"),
            TransformError("test"),
            AggregationError("test"),
            JoinError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, HiveFrameError)
            assert isinstance(exc, Exception)

    def test_transient_errors_are_retryable(self):
        """Test all transient errors are marked retryable."""
        transient_errors = [
            TransientError("test"),
            NetworkError("test"),
            TimeoutError("test"),
            RateLimitError("test"),
            HiveConnectionError("test"),
        ]

        for exc in transient_errors:
            assert exc.retryable
            assert exc.category == ErrorCategory.TRANSIENT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
