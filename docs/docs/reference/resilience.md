---
sidebar_position: 7
---

# Resilience Module

Fault tolerance patterns: retry, circuit breaker, timeout, and bulkhead.

```python
from hiveframe.resilience import (
    RetryPolicy,
    CircuitBreaker,
    Timeout,
    Bulkhead,
    ResilientExecutor
)
```

## RetryPolicy

Automatic retry with configurable backoff.

### Class Definition

```python
class RetryPolicy:
    """
    Configure automatic retry behavior.
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_ms: int = 1000,
        max_delay_ms: int = 30000,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ) -> None:
        """
        Create retry policy.
        
        Args:
            max_attempts: Maximum retry attempts (including first)
            base_delay_ms: Initial delay between retries
            max_delay_ms: Maximum delay cap
            exponential_base: Exponential backoff multiplier
            jitter: Add randomness to prevent thundering herd
            retryable_exceptions: Only retry these exceptions
                                 (None = retry all)
        """
```

### Methods

```python
def execute(
    self,
    func: Callable[[], T],
    *args: Any,
    **kwargs: Any
) -> T:
    """
    Execute function with retry.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries exhausted
        
    Example:
        policy = RetryPolicy(max_attempts=3)
        result = policy.execute(fetch_data, url)
    """

async def execute_async(
    self,
    func: Callable[[], Awaitable[T]],
    *args: Any,
    **kwargs: Any
) -> T:
    """
    Execute async function with retry.
    
    Example:
        result = await policy.execute_async(
            fetch_data_async, url
        )
    """
```

### Decorator Usage

```python
from hiveframe.resilience import retry

@retry(max_attempts=3, base_delay_ms=1000)
def fetch_data(url: str) -> dict:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Async
@retry(max_attempts=3)
async def fetch_data_async(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### Example

```python
from hiveframe.resilience import RetryPolicy
import requests

# Configure policy
policy = RetryPolicy(
    max_attempts=5,
    base_delay_ms=500,
    max_delay_ms=10000,
    exponential_base=2.0,
    jitter=True,
    retryable_exceptions=(
        requests.ConnectionError,
        requests.Timeout,
    )
)

# Execute with retry
def call_api():
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()

try:
    result = policy.execute(call_api)
except requests.HTTPError as e:
    print(f"Failed after all retries: {e}")
```

---

## CircuitBreaker

Prevent cascade failures by stopping calls to failing services.

### Class Definition

```python
class CircuitBreaker:
    """
    Circuit breaker for fault isolation.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, requests rejected immediately
    - HALF_OPEN: Testing recovery, limited requests
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout_ms: int = 30000,
        half_open_max_calls: int = 3
    ) -> None:
        """
        Create circuit breaker.
        
        Args:
            failure_threshold: Failures to trip circuit
            success_threshold: Successes to close circuit
            timeout_ms: Time before testing recovery
            half_open_max_calls: Calls allowed in half-open
        """
```

### Methods

```python
def execute(
    self,
    func: Callable[[], T],
    *args: Any,
    **kwargs: Any
) -> T:
    """
    Execute function through circuit breaker.
    
    Args:
        func: Function to execute
        
    Returns:
        Function result
        
    Raises:
        CircuitOpenError: Circuit is open
        Original exception if function fails
    """

def allow_request(self) -> bool:
    """Check if request should be allowed."""

def record_success(self) -> None:
    """Record successful call."""

def record_failure(self) -> None:
    """Record failed call."""

def reset(self) -> None:
    """Reset to closed state."""
```

### Properties

```python
@property
def state(self) -> CircuitState:
    """Current state (CLOSED, OPEN, HALF_OPEN)."""

@property
def failure_count(self) -> int:
    """Current failure count."""

@property
def metrics(self) -> CircuitMetrics:
    """Circuit breaker metrics."""
```

### Decorator Usage

```python
from hiveframe.resilience import circuit_breaker

@circuit_breaker(
    failure_threshold=5,
    timeout_ms=30000
)
def call_external_service():
    return requests.get("https://api.example.com")
```

### Example

```python
from hiveframe.resilience import CircuitBreaker, CircuitOpenError

# Create circuit breaker
cb = CircuitBreaker(
    failure_threshold=5,
    success_threshold=3,
    timeout_ms=30000
)

def process_request(data):
    try:
        result = cb.execute(call_external_api, data)
        return result
    except CircuitOpenError:
        # Circuit is open - use fallback
        return get_cached_response(data)
    except Exception as e:
        # Request failed
        return handle_error(e)

# Check state
if cb.state == CircuitState.OPEN:
    print("Service unavailable, circuit open")
```

---

## Timeout

Limit operation duration.

### Class Definition

```python
class Timeout:
    """
    Enforce time limits on operations.
    """
    
    def __init__(
        self,
        timeout_ms: int,
        cancel_on_timeout: bool = True
    ) -> None:
        """
        Create timeout wrapper.
        
        Args:
            timeout_ms: Maximum execution time
            cancel_on_timeout: Attempt to cancel operation
        """
```

### Methods

```python
def execute(
    self,
    func: Callable[[], T],
    *args: Any,
    **kwargs: Any
) -> T:
    """
    Execute with timeout.
    
    Returns:
        Function result
        
    Raises:
        TimeoutError: Operation exceeded time limit
    """

async def execute_async(
    self,
    func: Callable[[], Awaitable[T]],
    *args: Any,
    **kwargs: Any
) -> T:
    """Execute async function with timeout."""
```

### Decorator Usage

```python
from hiveframe.resilience import timeout

@timeout(timeout_ms=5000)
def slow_operation():
    # Must complete within 5 seconds
    return compute_result()

# Async
@timeout(timeout_ms=5000)
async def slow_async_operation():
    return await fetch_data()
```

### Example

```python
from hiveframe.resilience import Timeout, TimeoutError

timeout_policy = Timeout(timeout_ms=10000)

try:
    result = timeout_policy.execute(long_running_query, params)
except TimeoutError:
    print("Query timed out after 10 seconds")
    result = get_default_result()
```

---

## Bulkhead

Isolate failures by limiting concurrent operations.

### Class Definition

```python
class Bulkhead:
    """
    Limit concurrent operations for isolation.
    """
    
    def __init__(
        self,
        max_concurrent: int,
        max_wait_ms: int = 0,
        name: Optional[str] = None
    ) -> None:
        """
        Create bulkhead.
        
        Args:
            max_concurrent: Maximum concurrent calls
            max_wait_ms: Time to wait for slot (0 = no wait)
            name: Name for metrics/logging
        """
```

### Methods

```python
def execute(
    self,
    func: Callable[[], T],
    *args: Any,
    **kwargs: Any
) -> T:
    """
    Execute within bulkhead.
    
    Raises:
        BulkheadFullError: No capacity available
    """

async def execute_async(
    self,
    func: Callable[[], Awaitable[T]]
) -> T:
    """Execute async function within bulkhead."""

def acquire(self, timeout_ms: Optional[int] = None) -> bool:
    """Acquire a slot. Returns False if unavailable."""

def release(self) -> None:
    """Release a slot."""
```

### Properties

```python
@property
def available_slots(self) -> int:
    """Available capacity."""

@property
def active_calls(self) -> int:
    """Current concurrent calls."""
```

### Example

```python
from hiveframe.resilience import Bulkhead, BulkheadFullError

# Limit database connections
db_bulkhead = Bulkhead(
    max_concurrent=10,
    max_wait_ms=1000,
    name="database"
)

# Limit external API calls
api_bulkhead = Bulkhead(
    max_concurrent=5,
    name="external-api"
)

def query_database(query):
    try:
        return db_bulkhead.execute(run_query, query)
    except BulkheadFullError:
        raise ServiceOverloadedError("Database pool exhausted")

def call_api(request):
    try:
        return api_bulkhead.execute(make_request, request)
    except BulkheadFullError:
        return get_cached_response(request)
```

---

## ResilientExecutor

Combine multiple resilience patterns.

### Class Definition

```python
class ResilientExecutor:
    """
    Combine retry, circuit breaker, timeout, and bulkhead.
    """
    
    def __init__(
        self,
        retry: Optional[RetryPolicy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        timeout: Optional[Timeout] = None,
        bulkhead: Optional[Bulkhead] = None
    ) -> None:
        """
        Create resilient executor.
        
        Execution order:
        1. Bulkhead (capacity check)
        2. Circuit breaker (health check)
        3. Timeout (time limit)
        4. Retry (failure recovery)
        """
```

### Methods

```python
def execute(
    self,
    func: Callable[[], T],
    *args: Any,
    **kwargs: Any
) -> T:
    """
    Execute with all configured patterns.
    
    Raises:
        BulkheadFullError: No capacity
        CircuitOpenError: Circuit open
        TimeoutError: Exceeded time limit
        Original exception after retries exhausted
    """

@staticmethod
def builder() -> "ResilientExecutorBuilder":
    """Create builder for fluent configuration."""
```

### Builder Pattern

```python
executor = ResilientExecutor.builder() \
    .with_retry(
        max_attempts=3,
        base_delay_ms=1000
    ) \
    .with_circuit_breaker(
        failure_threshold=5,
        timeout_ms=30000
    ) \
    .with_timeout(timeout_ms=10000) \
    .with_bulkhead(max_concurrent=10) \
    .build()
```

### Complete Example

```python
from hiveframe.resilience import (
    ResilientExecutor,
    RetryPolicy,
    CircuitBreaker,
    Timeout,
    Bulkhead
)

# Configure individual policies
retry = RetryPolicy(
    max_attempts=3,
    base_delay_ms=500,
    exponential_base=2.0
)

circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    success_threshold=3,
    timeout_ms=60000
)

timeout = Timeout(timeout_ms=5000)

bulkhead = Bulkhead(
    max_concurrent=20,
    max_wait_ms=1000
)

# Create executor
executor = ResilientExecutor(
    retry=retry,
    circuit_breaker=circuit_breaker,
    timeout=timeout,
    bulkhead=bulkhead
)

# Use executor
def fetch_user_data(user_id: int) -> dict:
    def call():
        response = requests.get(f"https://api.example.com/users/{user_id}")
        response.raise_for_status()
        return response.json()
    
    try:
        return executor.execute(call)
    except CircuitOpenError:
        return {"user_id": user_id, "status": "unavailable"}
    except TimeoutError:
        return {"user_id": user_id, "status": "timeout"}
    except BulkheadFullError:
        raise ServiceOverloadedError("Too many concurrent requests")
```

---

## Exception Types

```python
from hiveframe.resilience import (
    CircuitOpenError,    # Circuit breaker is open
    BulkheadFullError,   # No capacity available
    TimeoutError,        # Operation timed out
    RetryExhaustedError  # All retries failed
)
```

## See Also

- [Configure Retry](/docs/how-to/configure-retry) - Retry configuration
- [Circuit Breaker Guide](/docs/how-to/use-circuit-breaker) - Circuit breaker patterns
- [Error Handling](/docs/how-to/handle-errors-dlq) - Error handling with DLQ
