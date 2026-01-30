---
sidebar_position: 12
---

# Exceptions Reference

All HiveFrame exception types.

```python
from hiveframe.exceptions import (
    HiveFrameError,
    ValidationError,
    ConfigurationError,
    # ... see below
)
```

## Exception Hierarchy

```
Exception
└── HiveFrameError (base class)
    ├── ConfigurationError
    │   └── SchemaError
    ├── ValidationError
    │   ├── DataValidationError
    │   └── SchemaValidationError
    ├── ExecutionError
    │   ├── TaskError
    │   ├── TimeoutError
    │   └── CancellationError
    ├── ColonyError
    │   ├── WorkerError
    │   └── CoordinationError
    ├── ResilienceError
    │   ├── CircuitOpenError
    │   ├── BulkheadFullError
    │   └── RetryExhaustedError
    ├── StorageError
    │   ├── FileNotFoundError
    │   ├── PermissionError
    │   └── CorruptedDataError
    ├── ConnectorError
    │   ├── ConnectionError
    │   └── SerializationError
    └── SQLError
        ├── SQLParseError
        └── SQLExecutionError
```

---

## Base Exception

### HiveFrameError

```python
class HiveFrameError(Exception):
    """
    Base exception for all HiveFrame errors.
    
    Attributes:
        message: Error description
        code: Error code for programmatic handling
        details: Additional context
        cause: Original exception if wrapped
    """
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ) -> None:
        """
        Create HiveFrame error.
        
        Args:
            message: Human-readable message
            code: Error code (e.g., "VALIDATION_001")
            details: Additional context
            cause: Wrapped exception
        """
```

#### Example

```python
from hiveframe.exceptions import HiveFrameError

try:
    process_data()
except HiveFrameError as e:
    print(f"Error: {e.message}")
    print(f"Code: {e.code}")
    print(f"Details: {e.details}")
    if e.cause:
        print(f"Caused by: {e.cause}")
```

---

## Configuration Errors

### ConfigurationError

```python
class ConfigurationError(HiveFrameError):
    """
    Invalid configuration.
    
    Raised when:
    - Missing required config
    - Invalid config value
    - Incompatible options
    """
```

#### Example

```python
from hiveframe.exceptions import ConfigurationError

try:
    colony = Colony(config={"workers": -1})
except ConfigurationError as e:
    print(f"Config error: {e}")
    # Config error: Invalid value for 'workers': must be positive
```

### SchemaError

```python
class SchemaError(ConfigurationError):
    """
    Schema definition error.
    
    Raised when:
    - Invalid data type
    - Duplicate column names
    - Invalid constraints
    """
```

---

## Validation Errors

### ValidationError

```python
class ValidationError(HiveFrameError):
    """
    Data validation failure.
    
    Attributes:
        field: Field that failed validation
        value: Invalid value
        constraint: Violated constraint
    """
```

### DataValidationError

```python
class DataValidationError(ValidationError):
    """
    Data content validation failure.
    
    Raised when:
    - Null in non-nullable field
    - Value out of range
    - Invalid format
    """
```

#### Example

```python
from hiveframe.exceptions import DataValidationError

try:
    df = hf.DataFrame([{"age": -5}], schema=schema)
except DataValidationError as e:
    print(f"Validation failed: {e.field} = {e.value}")
    print(f"Constraint: {e.constraint}")
```

### SchemaValidationError

```python
class SchemaValidationError(ValidationError):
    """
    Schema mismatch error.
    
    Raised when:
    - Data doesn't match schema
    - Missing required columns
    - Type mismatch
    """
```

---

## Execution Errors

### ExecutionError

```python
class ExecutionError(HiveFrameError):
    """
    Task or operation execution failure.
    """
```

### TaskError

```python
class TaskError(ExecutionError):
    """
    Task execution failure.
    
    Attributes:
        task_id: Failed task ID
        worker_id: Worker that ran the task
        duration_ms: Execution time before failure
    """
```

#### Example

```python
from hiveframe.exceptions import TaskError

try:
    future = colony.submit(process_batch, data)
    result = future.result()
except TaskError as e:
    print(f"Task {e.task_id} failed on {e.worker_id}")
    print(f"After {e.duration_ms}ms: {e.message}")
```

### TimeoutError

```python
class TimeoutError(ExecutionError):
    """
    Operation timed out.
    
    Attributes:
        timeout_ms: Configured timeout
        elapsed_ms: Actual elapsed time
        operation: Operation that timed out
    """
```

### CancellationError

```python
class CancellationError(ExecutionError):
    """
    Operation was cancelled.
    
    Raised when:
    - Manual cancellation
    - Colony shutdown
    - Resource limits
    """
```

---

## Colony Errors

### ColonyError

```python
class ColonyError(HiveFrameError):
    """
    Colony-level failure.
    
    Raised when:
    - Colony fails to start
    - Coordination failure
    - Resource exhaustion
    """
```

### WorkerError

```python
class WorkerError(ColonyError):
    """
    Worker-specific failure.
    
    Attributes:
        worker_id: Failed worker
        reason: Failure reason
    """
```

### CoordinationError

```python
class CoordinationError(ColonyError):
    """
    Distributed coordination failure.
    
    Raised when:
    - Leader election fails
    - Network partition detected
    - Consensus timeout
    """
```

---

## Resilience Errors

### ResilienceError

```python
class ResilienceError(HiveFrameError):
    """
    Resilience pattern error.
    """
```

### CircuitOpenError

```python
class CircuitOpenError(ResilienceError):
    """
    Circuit breaker is open.
    
    Attributes:
        circuit_name: Name of the circuit
        state: Current circuit state
        open_since: When circuit opened
        recovery_time: When recovery starts
    """
```

#### Example

```python
from hiveframe.exceptions import CircuitOpenError

try:
    result = circuit_breaker.execute(call_service)
except CircuitOpenError as e:
    print(f"Circuit {e.circuit_name} is open")
    print(f"Opened at: {e.open_since}")
    print(f"Will try recovery at: {e.recovery_time}")
    # Use fallback
    result = get_cached_result()
```

### BulkheadFullError

```python
class BulkheadFullError(ResilienceError):
    """
    Bulkhead capacity exhausted.
    
    Attributes:
        bulkhead_name: Name of the bulkhead
        max_concurrent: Maximum allowed
        active_calls: Current active calls
        wait_time_ms: Time waited before rejection
    """
```

### RetryExhaustedError

```python
class RetryExhaustedError(ResilienceError):
    """
    All retry attempts failed.
    
    Attributes:
        attempts: Number of attempts made
        last_exception: Final exception
        total_delay_ms: Total retry delay
    """
```

---

## Storage Errors

### StorageError

```python
class StorageError(HiveFrameError):
    """
    Storage operation failure.
    """
```

### FileNotFoundError

```python
class FileNotFoundError(StorageError):
    """
    File or path not found.
    
    Attributes:
        path: Missing path
    """
```

### PermissionError

```python
class PermissionError(StorageError):
    """
    Access denied.
    
    Attributes:
        path: Inaccessible path
        operation: Denied operation
    """
```

### CorruptedDataError

```python
class CorruptedDataError(StorageError):
    """
    Data corruption detected.
    
    Attributes:
        path: Corrupted file
        expected_checksum: Expected value
        actual_checksum: Actual value
    """
```

---

## Connector Errors

### ConnectorError

```python
class ConnectorError(HiveFrameError):
    """
    External system connector failure.
    """
```

### ConnectionError

```python
class ConnectionError(ConnectorError):
    """
    Failed to connect to external system.
    
    Attributes:
        host: Target host
        port: Target port
        reason: Connection failure reason
    """
```

### SerializationError

```python
class SerializationError(ConnectorError):
    """
    Serialization/deserialization failure.
    
    Attributes:
        format: Expected format (json, avro, etc.)
        data: Problematic data
    """
```

---

## SQL Errors

### SQLError

```python
class SQLError(HiveFrameError):
    """
    SQL operation failure.
    """
```

### SQLParseError

```python
class SQLParseError(SQLError):
    """
    SQL syntax error.
    
    Attributes:
        query: Failed query
        position: Error position
        expected: Expected token
    """
```

#### Example

```python
from hiveframe.exceptions import SQLParseError

try:
    result = sql.execute("SELEC * FROM users")  # typo
except SQLParseError as e:
    print(f"Parse error at position {e.position}")
    print(f"Query: {e.query}")
    print(f"Expected: {e.expected}")
```

### SQLExecutionError

```python
class SQLExecutionError(SQLError):
    """
    SQL execution failure.
    
    Attributes:
        query: Failed query
        stage: Execution stage (plan, optimize, execute)
    """
```

---

## Error Handling Patterns

### Basic Pattern

```python
from hiveframe.exceptions import HiveFrameError, ValidationError

try:
    result = process()
except ValidationError as e:
    # Handle validation specifically
    log_validation_error(e)
    return error_response(e)
except HiveFrameError as e:
    # Handle all HiveFrame errors
    log_error(e)
    raise
except Exception as e:
    # Handle unexpected errors
    log_unexpected(e)
    raise HiveFrameError("Unexpected error", cause=e)
```

### With DLQ

```python
from hiveframe.exceptions import HiveFrameError
from hiveframe.dlq import DeadLetterQueue

dlq = DeadLetterQueue(storage_path="dlq/")

def process_record(record):
    try:
        return transform(record)
    except ValidationError as e:
        # Send to DLQ for later review
        dlq.send(record, error=e, retryable=True)
        return None
    except HiveFrameError as e:
        # Non-retryable error
        dlq.send(record, error=e, retryable=False)
        return None
```

### Resilience Combination

```python
from hiveframe.exceptions import (
    CircuitOpenError,
    BulkheadFullError,
    TimeoutError,
    RetryExhaustedError
)

def resilient_call(data):
    try:
        return executor.execute(process, data)
    except CircuitOpenError:
        return cached_response()
    except BulkheadFullError:
        raise ServiceOverloadedError()
    except TimeoutError:
        metrics.increment("timeouts")
        raise
    except RetryExhaustedError as e:
        logger.error("All retries failed", exc=e.last_exception)
        raise
```

## See Also

- [Error Handling Guide](/docs/how-to/handle-errors-dlq) - Error handling patterns
- [Resilience Reference](./resilience) - Resilience patterns
- [Core Reference](./core) - Core module
