"""
HiveFrame Colony Dashboard
==========================

Web UI for monitoring and managing HiveFrame clusters.

Key Features:
- Real-time colony metrics visualization
- Worker status and role distribution
- Waggle dance activity monitor
- Pheromone signal display
- Query execution history
- Interactive SQL console

Usage:
    from hiveframe.dashboard import Dashboard

    # Start dashboard server
    dashboard = Dashboard(port=8080)
    dashboard.start()

    # Or run from command line
    # python -m hiveframe.dashboard --port 8080
"""

from .server import Dashboard, DashboardConfig
from .api import DashboardAPI
from .components import (
    ColonyMetricsPanel,
    WorkerStatusPanel,
    DanceFloorPanel,
    QueryHistoryPanel,
)

__all__ = [
    "Dashboard",
    "DashboardConfig",
    "DashboardAPI",
    "ColonyMetricsPanel",
    "WorkerStatusPanel",
    "DanceFloorPanel",
    "QueryHistoryPanel",
]
