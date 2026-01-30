"""
Speculative Execution
=====================

Scout bees proactively retry slow tasks to minimize tail latency.
Inspired by how scout bees explore multiple food sources in parallel.

Key Concepts:
- Slow Task Detection: Identify stragglers using statistical analysis
- Scout Task Runners: Background workers that speculatively re-execute
- Racing Tasks: First completion wins, duplicates are cancelled
- Adaptive Speculation: Learn which tasks benefit from speculation
"""

import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import math
import statistics


class TaskState(Enum):
    """State of a tracked task."""

    PENDING = auto()
    RUNNING = auto()
    SPECULATING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class SpeculativeConfig:
    """Configuration for speculative execution."""

    # Enable/disable speculation
    enabled: bool = True

    # Multiplier for median task time to trigger speculation
    slow_task_multiplier: float = 1.5

    # Minimum tasks completed before speculation starts
    min_tasks_for_stats: int = 5

    # Maximum concurrent speculative tasks
    max_speculative_tasks: int = 10

    # Percentage of slow tasks to speculate (0.0 to 1.0)
    speculation_percentage: float = 0.1

    # Minimum time before task can be speculated (seconds)
    min_speculation_delay: float = 1.0

    # Maximum speculation attempts per task
    max_speculation_attempts: int = 2

    # Time window for task statistics (seconds)
    stats_window: float = 300.0


@dataclass
class TaskExecution:
    """Tracks a single task execution attempt."""

    execution_id: str
    task_id: str
    start_time: float
    end_time: Optional[float] = None
    is_speculative: bool = False
    worker_id: Optional[str] = None
    result: Any = None
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        """Execution duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def is_running(self) -> bool:
        """Check if execution is still running."""
        return self.end_time is None


@dataclass
class TrackedTask:
    """A task being tracked for speculative execution."""

    task_id: str
    data: Any
    state: TaskState = TaskState.PENDING
    created_at: float = field(default_factory=time.time)
    executions: List[TaskExecution] = field(default_factory=list)
    speculation_count: int = 0
    priority: float = 0.5

    @property
    def primary_execution(self) -> Optional[TaskExecution]:
        """Get the primary (non-speculative) execution."""
        for ex in self.executions:
            if not ex.is_speculative:
                return ex
        return None

    @property
    def is_slow(self) -> bool:
        """Check if primary execution is considered slow."""
        primary = self.primary_execution
        if primary is None or not primary.is_running:
            return False
        elapsed = time.time() - primary.start_time
        return elapsed > 5.0  # Basic threshold, refined by SlowTaskDetector

    @property
    def any_completed(self) -> bool:
        """Check if any execution completed successfully."""
        return any(ex.end_time is not None and ex.error is None for ex in self.executions)

    def get_winning_execution(self) -> Optional[TaskExecution]:
        """Get the first successful execution."""
        for ex in sorted(self.executions, key=lambda e: e.end_time or float("inf")):
            if ex.end_time is not None and ex.error is None:
                return ex
        return None


class TaskTracker:
    """
    Tracks task execution state and history.

    Maintains statistics for slow task detection and speculation decisions.
    """

    def __init__(self, config: SpeculativeConfig):
        self.config = config
        self._tasks: Dict[str, TrackedTask] = {}
        self._completed_times: List[Tuple[float, float]] = []  # (timestamp, duration)
        self._lock = threading.Lock()

    def register_task(self, task_id: str, data: Any, priority: float = 0.5) -> TrackedTask:
        """Register a new task for tracking."""
        with self._lock:
            task = TrackedTask(task_id=task_id, data=data, priority=priority)
            self._tasks[task_id] = task
            return task

    def start_execution(
        self, task_id: str, worker_id: str, is_speculative: bool = False
    ) -> Optional[TaskExecution]:
        """Record start of task execution."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None

            execution = TaskExecution(
                execution_id=str(uuid.uuid4()),
                task_id=task_id,
                start_time=time.time(),
                is_speculative=is_speculative,
                worker_id=worker_id,
            )
            task.executions.append(execution)

            if is_speculative:
                task.speculation_count += 1
                task.state = TaskState.SPECULATING
            else:
                task.state = TaskState.RUNNING

            return execution

    def complete_execution(
        self, execution_id: str, result: Any = None, error: Optional[str] = None
    ) -> Optional[TrackedTask]:
        """Record completion of task execution."""
        with self._lock:
            for task in self._tasks.values():
                for ex in task.executions:
                    if ex.execution_id == execution_id:
                        ex.end_time = time.time()
                        ex.result = result
                        ex.error = error

                        # Update task state
                        if error:
                            if not task.any_completed:
                                task.state = TaskState.FAILED
                        else:
                            task.state = TaskState.COMPLETED

                            # Record completion time for statistics
                            if ex.duration:
                                self._completed_times.append((time.time(), ex.duration))
                                # Keep only recent times
                                cutoff = time.time() - self.config.stats_window
                                self._completed_times = [
                                    (ts, dur) for ts, dur in self._completed_times if ts > cutoff
                                ]

                        return task
            return None

    def get_task(self, task_id: str) -> Optional[TrackedTask]:
        """Get task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_running_tasks(self) -> List[TrackedTask]:
        """Get all currently running tasks."""
        with self._lock:
            return [
                t
                for t in self._tasks.values()
                if t.state in (TaskState.RUNNING, TaskState.SPECULATING)
            ]

    def get_completion_stats(self) -> Dict[str, float]:
        """Get task completion time statistics."""
        with self._lock:
            if not self._completed_times:
                return {"median": 0, "mean": 0, "stddev": 0, "p95": 0}

            durations = [dur for _, dur in self._completed_times]

            return {
                "median": statistics.median(durations),
                "mean": statistics.mean(durations),
                "stddev": statistics.stdev(durations) if len(durations) > 1 else 0,
                "p95": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
                "count": len(durations),
            }

    def cleanup_completed(self, max_age: float = 3600) -> int:
        """Remove old completed tasks."""
        with self._lock:
            cutoff = time.time() - max_age
            old_count = len(self._tasks)

            self._tasks = {
                tid: task
                for tid, task in self._tasks.items()
                if task.state not in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED)
                or task.created_at > cutoff
            }

            return old_count - len(self._tasks)


class SlowTaskDetector:
    """
    Detects slow (straggler) tasks for speculation.

    Uses statistical analysis of task completion times to identify
    tasks that are taking significantly longer than expected.
    """

    def __init__(self, config: SpeculativeConfig, tracker: TaskTracker):
        self.config = config
        self.tracker = tracker

    def detect_slow_tasks(self) -> List[TrackedTask]:
        """
        Identify tasks that should be speculated.

        A task is considered slow if it's running longer than
        slow_task_multiplier * median_completion_time.
        """
        stats = self.tracker.get_completion_stats()

        # Need enough data points for reliable detection
        if stats.get("count", 0) < self.config.min_tasks_for_stats:
            return []

        threshold = stats["median"] * self.config.slow_task_multiplier
        current_time = time.time()

        slow_tasks = []
        for task in self.tracker.get_running_tasks():
            primary = task.primary_execution
            if primary is None or not primary.is_running:
                continue

            elapsed = current_time - primary.start_time

            # Check if task qualifies for speculation
            if (
                elapsed > threshold
                and elapsed > self.config.min_speculation_delay
                and task.speculation_count < self.config.max_speculation_attempts
                and task.state != TaskState.SPECULATING
            ):
                slow_tasks.append(task)

        # Sort by elapsed time (longest running first) and priority
        slow_tasks.sort(
            key=lambda t: (
                -t.priority,
                -(current_time - (t.primary_execution.start_time if t.primary_execution else 0)),
            )
        )

        # Limit to speculation percentage
        max_to_speculate = int(len(slow_tasks) * self.config.speculation_percentage)
        max_to_speculate = max(max_to_speculate, 1) if slow_tasks else 0
        max_to_speculate = min(max_to_speculate, self.config.max_speculative_tasks)

        return slow_tasks[:max_to_speculate]

    def get_speculation_priority(self, task: TrackedTask) -> float:
        """
        Calculate speculation priority for a task.

        Higher priority = more likely to benefit from speculation.
        """
        if task.primary_execution is None:
            return 0.0

        stats = self.tracker.get_completion_stats()
        elapsed = time.time() - task.primary_execution.start_time

        if stats["median"] == 0:
            return task.priority

        # How many times slower than median
        slowness_factor = elapsed / stats["median"]

        # Combine with task priority
        return task.priority * min(slowness_factor, 5.0) / 5.0


class ScoutTaskRunner:
    """
    Scout bee that speculatively executes tasks.

    Runs in the background, picking up slow tasks and racing
    against the primary execution.
    """

    def __init__(self, worker_id: str, tracker: TaskTracker, process_fn: Callable[[Any], Any]):
        self.worker_id = worker_id
        self.tracker = tracker
        self.process_fn = process_fn
        self._current_execution: Optional[TaskExecution] = None
        self._cancelled = False

    def run_speculative(self, task: TrackedTask) -> Optional[Any]:
        """
        Speculatively execute a task.

        Returns result if this execution wins, None if cancelled or fails.
        """
        # Start execution
        execution = self.tracker.start_execution(
            task_id=task.task_id, worker_id=self.worker_id, is_speculative=True
        )

        if execution is None:
            return None

        self._current_execution = execution
        self._cancelled = False

        try:
            # Process the task
            result = self.process_fn(task.data)

            # Check if we were cancelled during processing
            if self._cancelled:
                return None

            # Record completion
            self.tracker.complete_execution(execution_id=execution.execution_id, result=result)

            return result

        except Exception as e:
            self.tracker.complete_execution(execution_id=execution.execution_id, error=str(e))
            return None
        finally:
            self._current_execution = None

    def cancel(self) -> None:
        """Cancel current speculative execution."""
        self._cancelled = True


class SpeculativeExecutor:
    """
    Main interface for speculative execution.

    Manages task tracking, slow task detection, and speculative
    execution using scout bee workers.

    Example:
        executor = SpeculativeExecutor(
            process_fn=lambda x: heavy_computation(x),
            num_scouts=4
        )

        # Start the executor
        executor.start()

        # Submit tasks
        for item in data:
            executor.submit(item)

        # Collect results
        results = executor.collect()

        # Stop when done
        executor.stop()
    """

    def __init__(
        self,
        process_fn: Callable[[Any], Any],
        num_scouts: int = 2,
        num_workers: int = 4,
        config: Optional[SpeculativeConfig] = None,
    ):
        self.process_fn = process_fn
        self.num_scouts = num_scouts
        self.num_workers = num_workers
        self.config = config or SpeculativeConfig()

        self.tracker = TaskTracker(self.config)
        self.detector = SlowTaskDetector(self.config, self.tracker)

        self._scouts: List[ScoutTaskRunner] = []
        self._worker_pool: Optional[ThreadPoolExecutor] = None
        self._scout_pool: Optional[ThreadPoolExecutor] = None
        self._speculation_thread: Optional[threading.Thread] = None
        self._running = False
        self._results: Dict[str, Any] = {}
        self._result_lock = threading.Lock()

    def start(self) -> None:
        """Start the executor with worker and scout pools."""
        self._running = True

        # Create worker pool for primary execution
        self._worker_pool = ThreadPoolExecutor(max_workers=self.num_workers)

        # Create scout pool for speculative execution
        self._scout_pool = ThreadPoolExecutor(max_workers=self.num_scouts)

        # Create scout runners
        for i in range(self.num_scouts):
            scout = ScoutTaskRunner(
                worker_id=f"scout_{i}", tracker=self.tracker, process_fn=self.process_fn
            )
            self._scouts.append(scout)

        # Start speculation monitoring thread
        if self.config.enabled:
            self._speculation_thread = threading.Thread(target=self._speculation_loop, daemon=True)
            self._speculation_thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the executor and wait for completion."""
        self._running = False

        # Cancel any ongoing speculation
        for scout in self._scouts:
            scout.cancel()

        # Shutdown pools
        if self._worker_pool:
            self._worker_pool.shutdown(wait=True)
            self._worker_pool = None

        if self._scout_pool:
            self._scout_pool.shutdown(wait=True)
            self._scout_pool = None

        # Wait for speculation thread
        if self._speculation_thread:
            self._speculation_thread.join(timeout=timeout)
            self._speculation_thread = None

        self._scouts.clear()

    def submit(self, data: Any, task_id: Optional[str] = None, priority: float = 0.5) -> str:
        """
        Submit a task for execution.

        Returns task ID for tracking.
        """
        if not self._running or self._worker_pool is None:
            raise RuntimeError("Executor not started")

        task_id = task_id or str(uuid.uuid4())

        # Register task
        task = self.tracker.register_task(task_id, data, priority)

        # Submit to worker pool
        future = self._worker_pool.submit(self._execute_task, task)

        return task_id

    def _execute_task(self, task: TrackedTask) -> Any:
        """Execute task with primary worker."""
        execution = self.tracker.start_execution(
            task_id=task.task_id, worker_id="primary", is_speculative=False
        )

        if execution is None:
            return None

        try:
            result = self.process_fn(task.data)

            # Store result
            with self._result_lock:
                if task.task_id not in self._results:
                    self._results[task.task_id] = result

            self.tracker.complete_execution(execution_id=execution.execution_id, result=result)

            return result

        except Exception as e:
            self.tracker.complete_execution(execution_id=execution.execution_id, error=str(e))
            raise

    def _speculation_loop(self) -> None:
        """Background loop for speculation detection and execution."""
        while self._running:
            try:
                # Detect slow tasks
                slow_tasks = self.detector.detect_slow_tasks()

                # Submit speculative executions
                for task in slow_tasks:
                    if not self._running:
                        break

                    # Find available scout
                    for scout in self._scouts:
                        if scout._current_execution is None:
                            if self._scout_pool:
                                self._scout_pool.submit(self._run_speculation, scout, task)
                            break

            except Exception:
                pass  # Continue on errors

            time.sleep(0.5)  # Check every 500ms

    def _run_speculation(self, scout: ScoutTaskRunner, task: TrackedTask) -> None:
        """Run speculative execution with a scout."""
        # Check if task already completed
        if task.any_completed:
            return

        result = scout.run_speculative(task)

        if result is not None:
            with self._result_lock:
                if task.task_id not in self._results:
                    self._results[task.task_id] = result

    def get_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """
        Get result for a specific task.

        Waits up to timeout seconds if task is still running.
        """
        start = time.time()

        while True:
            with self._result_lock:
                if task_id in self._results:
                    return self._results[task_id]

            task = self.tracker.get_task(task_id)
            if task and task.state in (TaskState.FAILED, TaskState.CANCELLED):
                return None

            if timeout is not None and time.time() - start > timeout:
                return None

            time.sleep(0.1)

    def collect(self, timeout: float = 60.0) -> Dict[str, Any]:
        """
        Collect all results.

        Waits for all submitted tasks to complete.
        """
        start = time.time()

        while time.time() - start < timeout:
            running = self.tracker.get_running_tasks()
            if not running:
                break
            time.sleep(0.1)

        with self._result_lock:
            return dict(self._results)

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        completion_stats = self.tracker.get_completion_stats()

        # Count speculative wins
        speculative_wins = 0
        total_completed = 0

        for task in self.tracker._tasks.values():
            if task.state == TaskState.COMPLETED:
                total_completed += 1
                winner = task.get_winning_execution()
                if winner and winner.is_speculative:
                    speculative_wins += 1

        return {
            "total_tasks": len(self.tracker._tasks),
            "completed_tasks": total_completed,
            "speculative_wins": speculative_wins,
            "speculation_success_rate": (
                speculative_wins / total_completed if total_completed > 0 else 0
            ),
            "completion_stats": completion_stats,
        }
