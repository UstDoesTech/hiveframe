---
sidebar_position: 2
---

# Core Module

The core module provides the fundamental building blocks for HiveFrame's distributed processing.

```python
from hiveframe import Colony, Cell, create_colony
from hiveframe.core import DataType
```

## Colony

The `Colony` class is the central coordinator for distributed processing, managing workers and task distribution.

### Class Definition

```python
class Colony:
    """
    Central coordinator for distributed data processing.
    
    The Colony manages the lifecycle of all worker bees,
    coordinates task distribution via the waggle dance protocol,
    and provides fault tolerance through scout bee recovery.
    """
    
    def __init__(
        self,
        name: str = "default",
        config: Optional[Dict[str, Any]] = None,
        num_workers: int = 4,
        max_memory: str = "4g"
    ) -> None:
        """
        Initialize a new Colony.
        
        Args:
            name: Colony identifier for logging and metrics
            config: Configuration dictionary for fine-tuning
            num_workers: Number of worker bees to spawn
            max_memory: Maximum memory allocation (e.g., "4g", "512m")
        """
```

### Configuration Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `worker.threads` | int | 4 | Threads per worker |
| `worker.memory` | str | "1g" | Memory per worker |
| `scheduler.strategy` | str | "waggle" | Scheduling strategy |
| `recovery.enabled` | bool | True | Enable scout recovery |
| `recovery.interval_ms` | int | 5000 | Recovery check interval |

### Methods

#### `start()`

Start the colony and all workers.

```python
def start(self) -> "Colony":
    """
    Start the colony, initializing all workers.
    
    Returns:
        Self for method chaining
        
    Raises:
        ColonyError: If colony fails to start
        
    Example:
        colony = Colony("my-colony").start()
    """
```

#### `stop()`

Gracefully stop the colony.

```python
def stop(self, timeout: float = 30.0) -> None:
    """
    Stop the colony and all workers.
    
    Args:
        timeout: Maximum seconds to wait for graceful shutdown
        
    Example:
        colony.stop(timeout=60.0)
    """
```

#### `submit()`

Submit a task for distributed execution.

```python
def submit(
    self,
    task: Callable[..., T],
    *args: Any,
    priority: int = 0,
    **kwargs: Any
) -> Future[T]:
    """
    Submit a task for execution.
    
    Args:
        task: Function to execute
        *args: Positional arguments
        priority: Task priority (higher = more important)
        **kwargs: Keyword arguments
        
    Returns:
        Future containing the eventual result
        
    Example:
        future = colony.submit(process_data, df, batch_size=1000)
        result = future.result()
    """
```

#### `distribute()`

Distribute data across workers.

```python
def distribute(
    self,
    data: Iterable[T],
    partitions: Optional[int] = None
) -> DistributedDataset[T]:
    """
    Distribute data across workers.
    
    Args:
        data: Iterable of items to distribute
        partitions: Number of partitions (default: num_workers)
        
    Returns:
        DistributedDataset for parallel operations
        
    Example:
        items = range(1000000)
        distributed = colony.distribute(items, partitions=8)
        results = distributed.map(process).collect()
    """
```

### Properties

```python
@property
def status(self) -> ColonyStatus:
    """Current colony status (STARTING, RUNNING, STOPPING, STOPPED)."""

@property
def workers(self) -> List[WorkerInfo]:
    """Information about all workers."""

@property
def metrics(self) -> ColonyMetrics:
    """Current colony metrics."""
```

### Example Usage

```python
from hiveframe import Colony

# Create and start a colony
colony = Colony(
    name="data-processing",
    num_workers=8,
    config={
        "worker.memory": "2g",
        "scheduler.strategy": "waggle"
    }
)
colony.start()

# Submit tasks
futures = []
for batch in data_batches:
    future = colony.submit(process_batch, batch, priority=1)
    futures.append(future)

# Wait for results
results = [f.result() for f in futures]

# Cleanup
colony.stop()
```

### Context Manager

```python
# Automatic start/stop with context manager
with Colony("my-colony", num_workers=4) as colony:
    result = colony.submit(heavy_computation, data).result()
# Colony automatically stopped
```

---

## Cell

A `Cell` represents a unit of data storage within the colony.

### Class Definition

```python
class Cell:
    """
    Unit of data storage in the colony.
    
    Cells are the fundamental data containers, analogous to
    honeycomb cells in a beehive. They store data and metadata
    for efficient distributed access.
    """
    
    def __init__(
        self,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        schema: Optional[Schema] = None
    ) -> None:
        """
        Create a new Cell.
        
        Args:
            data: The data to store
            metadata: Optional metadata dictionary
            schema: Optional schema for validation
        """
```

### Methods

#### `get()`

```python
def get(self) -> Any:
    """
    Retrieve the cell's data.
    
    Returns:
        The stored data
        
    Example:
        cell = Cell([1, 2, 3])
        data = cell.get()  # [1, 2, 3]
    """
```

#### `transform()`

```python
def transform(
    self,
    func: Callable[[T], U],
    preserve_metadata: bool = True
) -> "Cell[U]":
    """
    Apply a transformation to the cell's data.
    
    Args:
        func: Transformation function
        preserve_metadata: Keep original metadata
        
    Returns:
        New Cell with transformed data
        
    Example:
        cell = Cell([1, 2, 3])
        doubled = cell.transform(lambda x: [i * 2 for i in x])
    """
```

#### `validate()`

```python
def validate(self, schema: Schema) -> bool:
    """
    Validate cell data against a schema.
    
    Args:
        schema: Schema to validate against
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
```

---

## create_colony()

Factory function for creating colonies with common configurations.

```python
def create_colony(
    name: str = "default",
    profile: str = "development",
    **overrides: Any
) -> Colony:
    """
    Create a colony with a predefined profile.
    
    Args:
        name: Colony name
        profile: Configuration profile
            - "development": 2 workers, debug logging
            - "production": 8 workers, optimized settings
            - "testing": 1 worker, deterministic
        **overrides: Override specific settings
        
    Returns:
        Configured Colony instance
        
    Example:
        # Development colony
        colony = create_colony("dev", profile="development")
        
        # Production with custom workers
        colony = create_colony(
            "prod",
            profile="production",
            num_workers=16
        )
    """
```

### Profile Defaults

| Profile | Workers | Memory | Logging |
|---------|---------|--------|---------|
| development | 2 | 512m | DEBUG |
| production | 8 | 2g | INFO |
| testing | 1 | 256m | WARNING |

---

## DataType

Enumeration of supported data types.

```python
class DataType(Enum):
    """Supported data types for schema definitions."""
    
    STRING = "string"
    INTEGER = "integer"
    LONG = "long"
    FLOAT = "float"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    DATE = "date"
    BINARY = "binary"
    ARRAY = "array"
    MAP = "map"
    STRUCT = "struct"
```

### Type Mapping

| DataType | Python Type | Size |
|----------|-------------|------|
| STRING | str | Variable |
| INTEGER | int | 32-bit |
| LONG | int | 64-bit |
| FLOAT | float | 32-bit |
| DOUBLE | float | 64-bit |
| BOOLEAN | bool | 1-bit |
| TIMESTAMP | datetime | 64-bit |
| DATE | date | 32-bit |

---

## Exceptions

Core module exceptions:

```python
from hiveframe.exceptions import (
    ColonyError,      # Colony operation failures
    WorkerError,      # Worker-specific errors
    TaskError,        # Task execution failures
    ValidationError,  # Data validation errors
)
```

See [Exceptions Reference](./exceptions) for details.

## See Also

- [DataFrame](./dataframe) - DataFrame operations
- [Architecture](/docs/explanation/architecture-overview) - How Colony works
- [Getting Started](/docs/tutorials/getting-started) - Tutorial
