"""
Dashboard Components
====================

UI components for the Colony Dashboard.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class ColonyMetricsPanel:
    """
    Colony Metrics Panel
    --------------------
    Displays real-time colony health metrics.

    Metrics shown:
    - Colony temperature (aggregate load)
    - Average fitness score
    - Active food sources
    - Pheromone levels
    - Processing throughput
    """

    temperature: float = 0.5
    average_fitness: float = 0.0
    active_food_sources: int = 0
    pheromone_levels: Dict[str, float] = field(default_factory=dict)
    throughput: float = 0.0

    # Historical data for charts
    temperature_history: List[MetricPoint] = field(default_factory=list)
    fitness_history: List[MetricPoint] = field(default_factory=list)
    throughput_history: List[MetricPoint] = field(default_factory=list)

    def update(self, metrics: Dict[str, Any]) -> None:
        """Update panel with new metrics."""
        now = datetime.now()

        if "temperature" in metrics:
            self.temperature = metrics["temperature"]
            self.temperature_history.append(MetricPoint(now, self.temperature))

        if "average_fitness" in metrics:
            self.average_fitness = metrics["average_fitness"]
            self.fitness_history.append(MetricPoint(now, self.average_fitness))

        if "active_food_sources" in metrics:
            self.active_food_sources = metrics["active_food_sources"]

        if "pheromone_levels" in metrics:
            self.pheromone_levels = metrics["pheromone_levels"]

        if "throughput" in metrics:
            self.throughput = metrics["throughput"]
            self.throughput_history.append(MetricPoint(now, self.throughput))

        # Keep only recent history (last hour)
        cutoff = datetime.now().timestamp() - 3600
        self.temperature_history = [
            p for p in self.temperature_history if p.timestamp.timestamp() > cutoff
        ]
        self.fitness_history = [p for p in self.fitness_history if p.timestamp.timestamp() > cutoff]
        self.throughput_history = [
            p for p in self.throughput_history if p.timestamp.timestamp() > cutoff
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "temperature": self.temperature,
            "averageFitness": self.average_fitness,
            "activeFoodSources": self.active_food_sources,
            "pheromone_levels": self.pheromone_levels,
            "throughput": self.throughput,
            "temperatureHistory": [
                {"timestamp": p.timestamp.isoformat(), "value": p.value}
                for p in self.temperature_history[-60:]  # Last 60 points
            ],
            "fitnessHistory": [
                {"timestamp": p.timestamp.isoformat(), "value": p.value}
                for p in self.fitness_history[-60:]
            ],
            "throughputHistory": [
                {"timestamp": p.timestamp.isoformat(), "value": p.value}
                for p in self.throughput_history[-60:]
            ],
        }


@dataclass
class WorkerInfo:
    """Information about a worker bee."""

    worker_id: str
    role: str  # EMPLOYED, ONLOOKER, SCOUT
    status: str  # ACTIVE, IDLE, FORAGING
    current_partition: Optional[str] = None
    processed_count: int = 0
    average_quality: float = 0.0
    last_active: Optional[datetime] = None


@dataclass
class WorkerStatusPanel:
    """
    Worker Status Panel
    -------------------
    Displays status of all worker bees in the colony.

    Shows:
    - Worker count by role
    - Active/idle workers
    - Current assignments
    - Performance metrics
    """

    workers: List[WorkerInfo] = field(default_factory=list)

    @property
    def employed_count(self) -> int:
        return sum(1 for w in self.workers if w.role == "EMPLOYED")

    @property
    def onlooker_count(self) -> int:
        return sum(1 for w in self.workers if w.role == "ONLOOKER")

    @property
    def scout_count(self) -> int:
        return sum(1 for w in self.workers if w.role == "SCOUT")

    @property
    def active_count(self) -> int:
        return sum(1 for w in self.workers if w.status == "ACTIVE")

    @property
    def idle_count(self) -> int:
        return sum(1 for w in self.workers if w.status == "IDLE")

    def update_worker(self, info: WorkerInfo) -> None:
        """Update or add worker info."""
        for i, w in enumerate(self.workers):
            if w.worker_id == info.worker_id:
                self.workers[i] = info
                return
        self.workers.append(info)

    def remove_worker(self, worker_id: str) -> None:
        """Remove a worker from the panel."""
        self.workers = [w for w in self.workers if w.worker_id != worker_id]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "totalWorkers": len(self.workers),
            "roleDistribution": {
                "employed": self.employed_count,
                "onlooker": self.onlooker_count,
                "scout": self.scout_count,
            },
            "statusDistribution": {
                "active": self.active_count,
                "idle": self.idle_count,
            },
            "workers": [
                {
                    "workerId": w.worker_id,
                    "role": w.role,
                    "status": w.status,
                    "currentPartition": w.current_partition,
                    "processedCount": w.processed_count,
                    "averageQuality": w.average_quality,
                    "lastActive": w.last_active.isoformat() if w.last_active else None,
                }
                for w in self.workers
            ],
        }


@dataclass
class DanceInfo:
    """Information about a waggle dance."""

    worker_id: str
    partition_id: str
    quality_score: float
    processing_time: float
    result_size: int
    vigor: float
    timestamp: datetime


@dataclass
class DanceFloorPanel:
    """
    Dance Floor Panel
    -----------------
    Visualizes waggle dance activity in real-time.

    Shows:
    - Recent dances with quality scores
    - Most advertised partitions
    - Dance vigor distribution
    - Partition selection patterns
    """

    recent_dances: List[DanceInfo] = field(default_factory=list)
    max_dances: int = 100

    @property
    def partition_dance_counts(self) -> Dict[str, int]:
        """Count dances per partition."""
        counts: Dict[str, int] = {}
        for dance in self.recent_dances:
            counts[dance.partition_id] = counts.get(dance.partition_id, 0) + 1
        return counts

    @property
    def average_vigor(self) -> float:
        """Calculate average dance vigor."""
        if not self.recent_dances:
            return 0.0
        return sum(d.vigor for d in self.recent_dances) / len(self.recent_dances)

    @property
    def top_partitions(self) -> List[tuple]:
        """Get top partitions by dance count."""
        counts = self.partition_dance_counts
        sorted_parts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_parts[:10]

    def add_dance(self, dance: DanceInfo) -> None:
        """Add a new dance to the floor."""
        self.recent_dances.append(dance)
        # Keep only recent dances
        if len(self.recent_dances) > self.max_dances:
            self.recent_dances = self.recent_dances[-self.max_dances :]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "totalDances": len(self.recent_dances),
            "averageVigor": self.average_vigor,
            "topPartitions": [{"partitionId": p, "count": c} for p, c in self.top_partitions],
            "recentDances": [
                {
                    "workerId": d.worker_id,
                    "partitionId": d.partition_id,
                    "qualityScore": d.quality_score,
                    "processingTime": d.processing_time,
                    "resultSize": d.result_size,
                    "vigor": d.vigor,
                    "timestamp": d.timestamp.isoformat(),
                }
                for d in self.recent_dances[-20:]  # Last 20 dances
            ],
            "vigorDistribution": self._vigor_distribution(),
        }

    def _vigor_distribution(self) -> Dict[str, int]:
        """Calculate vigor distribution buckets."""
        buckets = {"0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for dance in self.recent_dances:
            if dance.vigor < 0.2:
                buckets["0-0.2"] += 1
            elif dance.vigor < 0.4:
                buckets["0.2-0.4"] += 1
            elif dance.vigor < 0.6:
                buckets["0.4-0.6"] += 1
            elif dance.vigor < 0.8:
                buckets["0.6-0.8"] += 1
            else:
                buckets["0.8-1.0"] += 1
        return buckets


@dataclass
class QueryInfo:
    """Information about a query execution."""

    query_id: str
    sql: str
    status: str  # RUNNING, COMPLETED, FAILED
    start_time: datetime
    end_time: Optional[datetime] = None
    rows_processed: int = 0
    rows_returned: int = 0
    error_message: Optional[str] = None
    execution_plan: Optional[str] = None


@dataclass
class QueryHistoryPanel:
    """
    Query History Panel
    -------------------
    Displays history of SQL query executions.

    Shows:
    - Recent queries with status
    - Execution times
    - Row counts
    - Errors and warnings
    """

    queries: List[QueryInfo] = field(default_factory=list)
    max_queries: int = 100

    @property
    def running_queries(self) -> List[QueryInfo]:
        return [q for q in self.queries if q.status == "RUNNING"]

    @property
    def completed_queries(self) -> List[QueryInfo]:
        return [q for q in self.queries if q.status == "COMPLETED"]

    @property
    def failed_queries(self) -> List[QueryInfo]:
        return [q for q in self.queries if q.status == "FAILED"]

    def add_query(self, query: QueryInfo) -> None:
        """Add a query to history."""
        # Check if exists (update)
        for i, q in enumerate(self.queries):
            if q.query_id == query.query_id:
                self.queries[i] = query
                return
        self.queries.append(query)
        # Keep only recent queries
        if len(self.queries) > self.max_queries:
            self.queries = self.queries[-self.max_queries :]

    def get_query(self, query_id: str) -> Optional[QueryInfo]:
        """Get query by ID."""
        for q in self.queries:
            if q.query_id == query_id:
                return q
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "totalQueries": len(self.queries),
            "runningCount": len(self.running_queries),
            "completedCount": len(self.completed_queries),
            "failedCount": len(self.failed_queries),
            "queries": [
                {
                    "queryId": q.query_id,
                    "sql": q.sql[:100] + "..." if len(q.sql) > 100 else q.sql,
                    "status": q.status,
                    "startTime": q.start_time.isoformat(),
                    "endTime": q.end_time.isoformat() if q.end_time else None,
                    "duration": (q.end_time - q.start_time).total_seconds() if q.end_time else None,
                    "rowsProcessed": q.rows_processed,
                    "rowsReturned": q.rows_returned,
                    "errorMessage": q.error_message,
                }
                for q in sorted(self.queries, key=lambda x: x.start_time, reverse=True)[:20]
            ],
        }
