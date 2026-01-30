"""
Dashboard API
=============

REST API endpoints for the Colony Dashboard.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import json
import uuid

from .components import (
    ColonyMetricsPanel,
    WorkerStatusPanel,
    DanceFloorPanel,
    QueryHistoryPanel,
    WorkerInfo,
    DanceInfo,
    QueryInfo,
)


@dataclass
class APIResponse:
    """Standard API response wrapper."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "success": self.success,
                "data": self.data,
                "error": self.error,
            }
        )


class DashboardAPI:
    """
    Dashboard REST API
    ------------------

    Provides endpoints for the Colony Dashboard web interface.

    Endpoints:
        GET  /api/metrics         - Colony metrics
        GET  /api/workers         - Worker status
        GET  /api/dances          - Dance floor activity
        GET  /api/queries         - Query history
        POST /api/queries         - Execute SQL query
        GET  /api/queries/{id}    - Get query details
        GET  /api/health          - Health check
        POST /api/config          - Update configuration
    """

    def __init__(self, hive=None, sql_context=None):
        """
        Initialize API.

        Args:
            hive: HiveFrame instance
            sql_context: SwarmQLContext instance
        """
        self.hive = hive
        self.sql_context = sql_context

        # Panels
        self.metrics_panel = ColonyMetricsPanel()
        self.workers_panel = WorkerStatusPanel()
        self.dance_floor_panel = DanceFloorPanel()
        self.query_history_panel = QueryHistoryPanel()

        # Query execution callback
        self._query_callback: Optional[Callable] = None

    def set_query_callback(self, callback: Callable) -> None:
        """Set callback for query execution."""
        self._query_callback = callback

    # === Metrics Endpoints ===

    def get_metrics(self) -> APIResponse:
        """Get colony metrics."""
        # Update from hive if available
        if self.hive and hasattr(self.hive, "colony"):
            colony = self.hive.colony
            if colony:
                self.metrics_panel.update(
                    {
                        "temperature": colony.get_colony_temperature(),
                        "active_food_sources": len(colony.food_sources),
                        "pheromone_levels": {
                            "throttle": colony.sense_pheromone("throttle"),
                            "alarm": colony.sense_pheromone("alarm"),
                            "recruit": colony.sense_pheromone("recruit"),
                        },
                    }
                )

        return APIResponse(success=True, data=self.metrics_panel.to_dict())

    def update_metrics(self, metrics: Dict[str, Any]) -> APIResponse:
        """Update colony metrics."""
        self.metrics_panel.update(metrics)
        return APIResponse(success=True)

    # === Worker Endpoints ===

    def get_workers(self) -> APIResponse:
        """Get worker status."""
        # Update from hive if available
        if self.hive and hasattr(self.hive, "bees"):
            self.workers_panel.workers = []
            for bee in self.hive.bees:
                info = WorkerInfo(
                    worker_id=bee.worker_id,
                    role=bee.role.name,
                    status="ACTIVE" if bee.current_source else "IDLE",
                    current_partition=(
                        bee.current_source.partition_id if bee.current_source else None
                    ),
                    processed_count=len(bee.results),
                    last_active=datetime.now(),
                )
                self.workers_panel.update_worker(info)

        return APIResponse(success=True, data=self.workers_panel.to_dict())

    def get_worker(self, worker_id: str) -> APIResponse:
        """Get specific worker details."""
        for worker in self.workers_panel.workers:
            if worker.worker_id == worker_id:
                return APIResponse(
                    success=True,
                    data={
                        "workerId": worker.worker_id,
                        "role": worker.role,
                        "status": worker.status,
                        "currentPartition": worker.current_partition,
                        "processedCount": worker.processed_count,
                        "averageQuality": worker.average_quality,
                        "lastActive": (
                            worker.last_active.isoformat() if worker.last_active else None
                        ),
                    },
                )
        return APIResponse(success=False, error=f"Worker {worker_id} not found")

    # === Dance Floor Endpoints ===

    def get_dances(self) -> APIResponse:
        """Get dance floor activity."""
        # Update from hive if available
        if self.hive and hasattr(self.hive, "colony"):
            colony = self.hive.colony
            if colony and hasattr(colony, "dance_floor"):
                for partition_id, dances in colony.dance_floor.dances.items():
                    for dance in dances[-5:]:  # Recent dances
                        dance_info = DanceInfo(
                            worker_id=dance.worker_id,
                            partition_id=dance.partition_id,
                            quality_score=dance.quality_score,
                            processing_time=dance.processing_time,
                            result_size=dance.result_size,
                            vigor=dance.vigor,
                            timestamp=datetime.fromtimestamp(dance.timestamp),
                        )
                        self.dance_floor_panel.add_dance(dance_info)

        return APIResponse(success=True, data=self.dance_floor_panel.to_dict())

    # === Query Endpoints ===

    def get_queries(self) -> APIResponse:
        """Get query history."""
        return APIResponse(success=True, data=self.query_history_panel.to_dict())

    def execute_query(self, sql: str) -> APIResponse:
        """
        Execute a SQL query.

        Args:
            sql: SQL query string

        Returns:
            APIResponse with query results or error
        """
        query_id = str(uuid.uuid4())
        query_info = QueryInfo(
            query_id=query_id, sql=sql, status="RUNNING", start_time=datetime.now()
        )
        self.query_history_panel.add_query(query_info)

        try:
            # Execute via SQL context if available
            if self.sql_context:
                result_df = self.sql_context.sql(sql)
                results = result_df.collect()

                query_info.status = "COMPLETED"
                query_info.end_time = datetime.now()
                query_info.rows_returned = len(results)
                self.query_history_panel.add_query(query_info)

                return APIResponse(
                    success=True,
                    data={
                        "queryId": query_id,
                        "columns": list(results[0].keys()) if results else [],
                        "rows": results[:1000],  # Limit to 1000 rows
                        "totalRows": len(results),
                        "truncated": len(results) > 1000,
                    },
                )
            elif self._query_callback:
                result = self._query_callback(sql)
                query_info.status = "COMPLETED"
                query_info.end_time = datetime.now()
                self.query_history_panel.add_query(query_info)
                return APIResponse(success=True, data=result)
            else:
                raise ValueError("No SQL context or query callback configured")

        except Exception as e:
            query_info.status = "FAILED"
            query_info.end_time = datetime.now()
            query_info.error_message = str(e)
            self.query_history_panel.add_query(query_info)
            return APIResponse(success=False, error=str(e))

    def get_query(self, query_id: str) -> APIResponse:
        """Get query details by ID."""
        query = self.query_history_panel.get_query(query_id)
        if query:
            return APIResponse(
                success=True,
                data={
                    "queryId": query.query_id,
                    "sql": query.sql,
                    "status": query.status,
                    "startTime": query.start_time.isoformat(),
                    "endTime": query.end_time.isoformat() if query.end_time else None,
                    "rowsProcessed": query.rows_processed,
                    "rowsReturned": query.rows_returned,
                    "errorMessage": query.error_message,
                    "executionPlan": query.execution_plan,
                },
            )
        return APIResponse(success=False, error=f"Query {query_id} not found")

    def explain_query(self, sql: str) -> APIResponse:
        """Get execution plan for a query."""
        try:
            if self.sql_context:
                plan = self.sql_context.explain(sql)
                return APIResponse(success=True, data={"plan": plan})
            return APIResponse(success=False, error="SQL context not configured")
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    # === Health Endpoint ===

    def get_health(self) -> APIResponse:
        """Health check endpoint."""
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "hive": "connected" if self.hive else "not_configured",
                "sql_context": "connected" if self.sql_context else "not_configured",
            },
        }

        if self.hive and hasattr(self.hive, "colony") and self.hive.colony:
            status["colony"] = {
                "workers": len(self.hive.bees) if hasattr(self.hive, "bees") else 0,
                "food_sources": len(self.hive.colony.food_sources),
                "temperature": self.hive.colony.get_colony_temperature(),
            }

        return APIResponse(success=True, data=status)

    # === Configuration Endpoint ===

    def update_config(self, config: Dict[str, Any]) -> APIResponse:
        """Update dashboard configuration."""
        # Apply configuration changes
        if self.hive:
            if "abandonment_limit" in config:
                self.hive.abandonment_limit = config["abandonment_limit"]
            if "max_cycles" in config:
                self.hive.max_cycles = config["max_cycles"]

        return APIResponse(success=True, data={"message": "Configuration updated"})

    # === Tables Endpoint ===

    def get_tables(self) -> APIResponse:
        """Get list of registered tables."""
        if self.sql_context:
            tables = self.sql_context.tables()
            return APIResponse(success=True, data={"tables": tables})
        return APIResponse(success=True, data={"tables": []})

    def get_table_schema(self, table_name: str) -> APIResponse:
        """Get schema for a table."""
        if self.sql_context:
            df = self.sql_context.table(table_name)
            if df:
                schema = [{"name": name, "type": dtype.name} for name, dtype in df.schema.fields]
                return APIResponse(
                    success=True,
                    data={"table": table_name, "schema": schema, "rowCount": df.count()},
                )
            return APIResponse(success=False, error=f"Table {table_name} not found")
        return APIResponse(success=False, error="SQL context not configured")
