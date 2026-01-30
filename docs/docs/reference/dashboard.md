---
sidebar_position: 11
---

# Dashboard Module

Real-time monitoring dashboard.

```python
from hiveframe.dashboard import (
    DashboardServer,
    DashboardConfig,
    Widget
)
```

## DashboardServer

Serve the monitoring dashboard.

### Class Definition

```python
class DashboardServer:
    """
    HTTP server for the monitoring dashboard.
    """
    
    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        colony: Optional[Colony] = None
    ) -> None:
        """
        Create dashboard server.
        
        Args:
            config: Dashboard configuration
            colony: Colony to monitor
        """
```

### Methods

```python
def start(
    self,
    host: str = "0.0.0.0",
    port: int = 8080,
    background: bool = True
) -> None:
    """
    Start dashboard server.
    
    Args:
        host: Bind address
        port: HTTP port
        background: Run in background thread
        
    Example:
        dashboard = DashboardServer(colony=colony)
        dashboard.start(port=8080)
        print(f"Dashboard at http://localhost:8080")
    """

def stop(self) -> None:
    """Stop dashboard server."""

def add_widget(
    self,
    widget: Widget,
    position: Optional[Tuple[int, int]] = None
) -> None:
    """
    Add custom widget.
    
    Args:
        widget: Widget to add
        position: Grid position (row, col)
    """

def add_page(
    self,
    name: str,
    title: str,
    widgets: List[Widget]
) -> None:
    """
    Add custom dashboard page.
    
    Args:
        name: URL path
        title: Page title
        widgets: Page widgets
    """
```

### Properties

```python
@property
def url(self) -> str:
    """Dashboard URL."""

@property
def is_running(self) -> bool:
    """Server running status."""
```

---

## DashboardConfig

Configure dashboard appearance and behavior.

```python
class DashboardConfig:
    """
    Dashboard configuration.
    """
    
    def __init__(
        self,
        title: str = "HiveFrame Dashboard",
        theme: str = "dark",
        refresh_interval_ms: int = 5000,
        auth_enabled: bool = False,
        auth_users: Optional[Dict[str, str]] = None,
        readonly: bool = False,
        custom_css: Optional[str] = None,
        logo_url: Optional[str] = None
    ) -> None:
        """
        Configure dashboard.
        
        Args:
            title: Browser title
            theme: "dark" or "light"
            refresh_interval_ms: Auto-refresh interval
            auth_enabled: Require authentication
            auth_users: Username/password pairs
            readonly: Disable controls
            custom_css: Custom stylesheet URL
            logo_url: Custom logo URL
        """
```

### Example

```python
config = DashboardConfig(
    title="Production Monitor",
    theme="dark",
    refresh_interval_ms=2000,
    auth_enabled=True,
    auth_users={
        "admin": "secure_password",
        "viewer": "readonly_password"
    }
)

dashboard = DashboardServer(config=config, colony=colony)
```

---

## Built-in Pages

### Overview Page (`/`)

- Cluster health status
- Worker count and state
- Tasks in progress
- Memory/CPU utilization
- Recent errors

### Workers Page (`/workers`)

- Per-worker metrics
- Task assignment
- Resource usage
- Health status

### Tasks Page (`/tasks`)

- Active tasks
- Task queue depth
- Execution history
- Failed tasks

### Streaming Page (`/streaming`)

- Stream throughput
- Processing lag
- Window status
- Checkpoint progress

### Metrics Page (`/metrics`)

- Time-series graphs
- Custom metric queries
- Export options

---

## Widgets

### Built-in Widgets

```python
from hiveframe.dashboard import (
    MetricWidget,
    ChartWidget,
    TableWidget,
    GaugeWidget,
    StatusWidget,
    LogWidget
)
```

#### MetricWidget

```python
widget = MetricWidget(
    title="Tasks/Second",
    metric="hiveframe_tasks_total",
    aggregation="rate",
    format="{value:.1f}/s"
)
```

#### ChartWidget

```python
widget = ChartWidget(
    title="Throughput",
    metrics=["records_in", "records_out"],
    chart_type="line",  # line, bar, area
    time_range="1h",
    refresh_interval_ms=5000
)
```

#### TableWidget

```python
widget = TableWidget(
    title="Active Workers",
    columns=["name", "status", "tasks", "memory"],
    data_source="workers"
)
```

#### GaugeWidget

```python
widget = GaugeWidget(
    title="CPU Usage",
    metric="cpu_percent",
    min_value=0,
    max_value=100,
    thresholds={
        "green": 70,
        "yellow": 85,
        "red": 100
    }
)
```

#### StatusWidget

```python
widget = StatusWidget(
    title="Services",
    checks=["database", "kafka", "redis"]
)
```

#### LogWidget

```python
widget = LogWidget(
    title="Recent Logs",
    max_lines=100,
    filter_level="WARNING",
    sources=["queen", "worker-*"]
)
```

### Custom Widgets

```python
from hiveframe.dashboard import Widget

class CustomWidget(Widget):
    """Custom dashboard widget."""
    
    def __init__(self, title: str, **kwargs):
        super().__init__(title)
        self.config = kwargs
    
    def render(self, context: dict) -> str:
        """Render widget HTML."""
        return f"""
        <div class="widget custom-widget">
            <h3>{self.title}</h3>
            <div class="content">
                {self.generate_content(context)}
            </div>
        </div>
        """
    
    def get_data(self) -> dict:
        """Fetch widget data."""
        return {"value": fetch_custom_metric()}
```

---

## API Endpoints

The dashboard exposes REST endpoints:

### Cluster Status

```http
GET /api/cluster
```

```json
{
  "name": "production",
  "status": "healthy",
  "workers": {
    "total": 10,
    "ready": 10,
    "busy": 7
  },
  "tasks": {
    "active": 42,
    "queued": 15,
    "completed": 12345
  }
}
```

### Workers

```http
GET /api/workers
```

```json
[
  {
    "id": "worker-1",
    "status": "running",
    "tasks_active": 3,
    "cpu_percent": 45.2,
    "memory_mb": 1024
  }
]
```

### Metrics

```http
GET /api/metrics?names=tasks_total,latency_p99&range=1h
```

```json
{
  "tasks_total": [
    {"timestamp": "2024-01-15T10:00:00Z", "value": 1000},
    {"timestamp": "2024-01-15T10:01:00Z", "value": 1050}
  ],
  "latency_p99": [...]
}
```

### Control (if not readonly)

```http
POST /api/workers/{id}/restart
POST /api/cluster/scale
{
  "workers": 15
}
```

---

## Complete Example

```python
from hiveframe import Colony
from hiveframe.dashboard import (
    DashboardServer,
    DashboardConfig,
    ChartWidget,
    GaugeWidget,
    TableWidget
)

# Create colony
colony = Colony("production", num_workers=10)
colony.start()

# Configure dashboard
config = DashboardConfig(
    title="Production Monitor",
    theme="dark",
    refresh_interval_ms=2000
)

# Create dashboard
dashboard = DashboardServer(config=config, colony=colony)

# Add custom widgets
dashboard.add_widget(
    ChartWidget(
        title="Processing Rate",
        metrics=["tasks_completed"],
        chart_type="area",
        time_range="30m"
    ),
    position=(0, 0)
)

dashboard.add_widget(
    GaugeWidget(
        title="Memory Usage",
        metric="memory_percent",
        thresholds={"green": 70, "yellow": 85, "red": 95}
    ),
    position=(0, 1)
)

# Add custom page
dashboard.add_page(
    name="custom",
    title="Custom Metrics",
    widgets=[
        ChartWidget(title="Custom 1", metrics=["metric_a"]),
        TableWidget(title="Details", columns=["name", "value"])
    ]
)

# Start dashboard
dashboard.start(port=8080)
print(f"Dashboard: http://localhost:8080")

# Keep running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    dashboard.stop()
    colony.stop()
```

## See Also

- [Setup Monitoring](/docs/how-to/setup-monitoring) - Monitoring guide
- [Metrics Reference](./monitoring) - Metrics API
- [Colony Reference](./core) - Colony API
