"""
Challenge Scenario: Error Handling & Recovery
=============================================
Tests HiveFrame's ability to handle errors gracefully and recover.

Scenarios:
1. Transient errors with recovery
2. Permanent failures with dead letter queue
3. Error cascades and circuit breaker activation
4. Mixed error rates with quality degradation
5. Poison pill detection and isolation
"""

import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..exceptions import (
    DeadLetterQueue,
    DeadLetterRecord,
    HiveFrameError,
    ProcessingError,
    TransientError,
    ValidationError,
)
from ..monitoring import get_logger, get_profiler, get_registry
from ..resilience import (
    BackoffStrategy,
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryPolicy,
    with_retry,
)

logger = get_logger("challenge.errors")
metrics = get_registry()
profiler = get_profiler()


@dataclass
class ErrorScenarioConfig:
    """Configuration for error injection scenarios."""

    transient_error_rate: float = 0.1  # 10% transient errors
    permanent_error_rate: float = 0.02  # 2% permanent errors
    cascade_probability: float = 0.05  # 5% chance of cascade
    poison_pill_rate: float = 0.01  # 1% poison pills
    max_retries: int = 3
    timeout_seconds: float = 5.0


@dataclass
class ScenarioResult:
    """Results from a challenge scenario."""

    scenario_name: str
    total_records: int
    successful: int
    failed_transient: int
    failed_permanent: int
    retries_used: int
    dead_letter_count: int
    circuit_trips: int
    elapsed_seconds: float
    throughput: float
    error_rate: float
    recovery_rate: float  # Transient errors that recovered

    def summary(self) -> str:
        return f"""
=== {self.scenario_name} ===
Total Records:     {self.total_records}
Successful:        {self.successful} ({100*self.successful/max(1,self.total_records):.1f}%)
Failed Transient:  {self.failed_transient}
Failed Permanent:  {self.failed_permanent}
Retries Used:      {self.retries_used}
Dead Letters:      {self.dead_letter_count}
Circuit Trips:     {self.circuit_trips}
Elapsed Time:      {self.elapsed_seconds:.2f}s
Throughput:        {self.throughput:.1f} rec/s
Error Rate:        {100*self.error_rate:.2f}%
Recovery Rate:     {100*self.recovery_rate:.1f}%
"""


class ErrorInjector:
    """
    Injects various types of errors into processing.

    Simulates real-world failure modes:
    - Network timeouts (transient)
    - Data corruption (permanent)
    - Downstream service failures (cascading)
    - Resource exhaustion (transient)
    """

    def __init__(self, config: ErrorScenarioConfig):
        self.config = config
        self._error_counts = {"transient": 0, "permanent": 0, "cascade": 0}
        self._lock = threading.Lock()

    def maybe_inject_error(self, record_id: str) -> None:
        """Potentially inject an error based on configuration."""
        r = random.random()

        # Check for poison pill (specific bad records)
        if self.config.poison_pill_rate > 0:
            if hash(record_id) % 100 < self.config.poison_pill_rate * 100:
                with self._lock:
                    self._error_counts["permanent"] += 1
                raise ValidationError(
                    f"Poison pill detected: {record_id}",
                    field="record_id",
                    expected="valid",
                    actual="poison",
                )

        # Check for transient error
        if r < self.config.transient_error_rate:
            with self._lock:
                self._error_counts["transient"] += 1
            raise TransientError(
                f"Simulated transient error for {record_id}",
                retry_after=0.1,
                max_retries=self.config.max_retries,
            )

        # Check for permanent error
        elif r < self.config.transient_error_rate + self.config.permanent_error_rate:
            with self._lock:
                self._error_counts["permanent"] += 1
            raise ValidationError(
                f"Simulated permanent error for {record_id}",
                field="data",
                expected="valid",
                actual="corrupted",
            )

        # Check for cascade
        elif r < (
            self.config.transient_error_rate
            + self.config.permanent_error_rate
            + self.config.cascade_probability
        ):
            with self._lock:
                self._error_counts["cascade"] += 1
            raise ProcessingError(
                f"Simulated cascade failure for {record_id}", partition_id=record_id
            )

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._error_counts)


def run_transient_recovery_scenario(
    num_records: int = 1000, error_rate: float = 0.2, max_retries: int = 3
) -> ScenarioResult:
    """
    Scenario 1: Transient Error Recovery

    Tests the system's ability to recover from temporary failures
    using retry logic with exponential backoff.

    Expected behavior:
    - Most transient errors should recover within max_retries
    - Throughput should degrade gracefully under errors
    - No data loss for recoverable errors
    """
    logger.info(
        "Starting transient recovery scenario", num_records=num_records, error_rate=error_rate
    )

    config = ErrorScenarioConfig(
        transient_error_rate=error_rate, permanent_error_rate=0, max_retries=max_retries
    )
    injector = ErrorInjector(config)

    retry_policy = RetryPolicy(
        max_retries=max_retries,
        base_delay=0.01,
        max_delay=0.5,
        strategy=BackoffStrategy.EXPONENTIAL,
    )

    successful = 0
    failed = 0
    records = [{"id": f"rec_{i}", "value": i} for i in range(num_records)]

    start_time = time.time()

    @with_retry(retry_policy)
    def process_with_retry(record: Dict) -> Dict:
        injector.maybe_inject_error(record["id"])
        return {"id": record["id"], "result": record["value"] * 2}

    for record in records:
        try:
            with profiler.profile("transient_recovery_process"):
                process_with_retry(record)
            successful += 1
        except HiveFrameError:
            failed += 1
            logger.debug("Record failed after retries", record_id=record["id"])

    elapsed = time.time() - start_time
    stats = injector.get_stats()

    # Recovery rate = (transient errors - permanent failures) / transient errors
    transient_recovered = stats["transient"] - failed
    recovery_rate = transient_recovered / max(1, stats["transient"])

    return ScenarioResult(
        scenario_name="Transient Error Recovery",
        total_records=num_records,
        successful=successful,
        failed_transient=failed,
        failed_permanent=0,
        retries_used=stats["transient"],  # Each transient error triggers retry
        dead_letter_count=0,
        circuit_trips=0,
        elapsed_seconds=elapsed,
        throughput=num_records / elapsed,
        error_rate=failed / num_records,
        recovery_rate=recovery_rate,
    )


def run_dead_letter_scenario(
    num_records: int = 1000, permanent_error_rate: float = 0.05
) -> ScenarioResult:
    """
    Scenario 2: Dead Letter Queue Handling

    Tests the system's ability to route permanently failed
    records to a dead letter queue for later inspection.

    Expected behavior:
    - Permanent failures should be captured in DLQ
    - Processing should continue for other records
    - DLQ should contain full error context
    """
    logger.info(
        "Starting dead letter scenario", num_records=num_records, error_rate=permanent_error_rate
    )

    config = ErrorScenarioConfig(
        transient_error_rate=0, permanent_error_rate=permanent_error_rate, poison_pill_rate=0.01
    )
    injector = ErrorInjector(config)
    dlq = DeadLetterQueue(max_size=1000)

    successful = 0
    failed = 0
    records = [{"id": f"rec_{i}", "value": i} for i in range(num_records)]

    start_time = time.time()

    for record in records:
        try:
            with profiler.profile("dlq_process"):
                injector.maybe_inject_error(str(record["id"]))
                {"id": record["id"], "result": record["value"] * 2}  # type: ignore
            successful += 1
        except HiveFrameError as e:
            failed += 1
            # Route to dead letter queue
            dlq_record = DeadLetterRecord(
                original_data=record,
                error=e,
                partition_id=str(record["id"]),
                worker_id="test_worker",
                attempt_count=1,
                first_failure=time.time(),
            )
            dlq.push(dlq_record)

    elapsed = time.time() - start_time
    dlq_stats = dlq.get_stats()

    logger.info(
        "Dead letter queue stats",
        size=dlq_stats["size"],
        error_distribution=dlq_stats["error_distribution"],
    )

    return ScenarioResult(
        scenario_name="Dead Letter Queue Handling",
        total_records=num_records,
        successful=successful,
        failed_transient=0,
        failed_permanent=failed,
        retries_used=0,
        dead_letter_count=dlq_stats["size"],
        circuit_trips=0,
        elapsed_seconds=elapsed,
        throughput=num_records / elapsed,
        error_rate=failed / num_records,
        recovery_rate=0,  # No recovery expected for permanent errors
    )


def run_circuit_breaker_scenario(
    num_records: int = 1000, failure_burst_size: int = 20, failure_threshold: int = 5
) -> ScenarioResult:
    """
    Scenario 3: Circuit Breaker Activation

    Tests the circuit breaker pattern under sustained failures.

    Expected behavior:
    - Circuit should open after failure_threshold failures
    - Requests should be rejected while circuit is open
    - Circuit should half-open after timeout and test recovery
    """
    logger.info(
        "Starting circuit breaker scenario",
        num_records=num_records,
        failure_threshold=failure_threshold,
    )

    circuit = CircuitBreaker(
        "test_service",
        CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=2,
            timeout=1.0,  # Short timeout for testing
        ),
    )

    # Track circuit state changes
    circuit_trips = [0]

    def on_state_change(old_state, new_state):
        if new_state.name == "OPEN":
            circuit_trips[0] += 1
            logger.warning("Circuit opened!", old_state=old_state.name)

    circuit.add_state_listener(on_state_change)

    successful = 0
    failed_by_error = 0
    failed_by_circuit = 0

    # Create records with burst of failures in the middle
    records = []
    failure_start = num_records // 3
    failure_end = failure_start + failure_burst_size

    for i in range(num_records):
        should_fail = failure_start <= i < failure_end
        records.append({"id": f"rec_{i}", "value": i, "should_fail": should_fail})

    start_time = time.time()

    def risky_operation(record: Dict) -> Dict:
        if record["should_fail"]:
            raise TransientError(f"Service unavailable for {record['id']}")
        return {"id": record["id"], "result": record["value"] * 2}

    for record in records:
        try:
            with profiler.profile("circuit_breaker_process"):
                circuit.call(risky_operation, record)
            successful += 1
        except TransientError:
            failed_by_error += 1
        except Exception as e:
            if "Circuit" in str(e):
                failed_by_circuit += 1
            else:
                failed_by_error += 1

        # Small delay to allow circuit state transitions
        time.sleep(0.001)

    elapsed = time.time() - start_time

    logger.info("Circuit breaker stats", trips=circuit_trips[0], final_state=circuit.state.name)

    return ScenarioResult(
        scenario_name="Circuit Breaker Activation",
        total_records=num_records,
        successful=successful,
        failed_transient=failed_by_error,
        failed_permanent=failed_by_circuit,
        retries_used=0,
        dead_letter_count=0,
        circuit_trips=circuit_trips[0],
        elapsed_seconds=elapsed,
        throughput=num_records / elapsed,
        error_rate=(failed_by_error + failed_by_circuit) / num_records,
        recovery_rate=successful / max(1, num_records - failure_burst_size),
    )


def run_mixed_error_scenario(num_records: int = 1000) -> ScenarioResult:
    """
    Scenario 4: Mixed Error Rates

    Tests the system under realistic mixed error conditions.

    Expected behavior:
    - System should handle multiple error types simultaneously
    - Quality metrics should degrade proportionally
    - Recovery mechanisms should work together
    """
    logger.info("Starting mixed error scenario", num_records=num_records)

    config = ErrorScenarioConfig(
        transient_error_rate=0.15,
        permanent_error_rate=0.03,
        cascade_probability=0.02,
        poison_pill_rate=0.005,
        max_retries=3,
    )
    injector = ErrorInjector(config)
    dlq = DeadLetterQueue(max_size=500)

    retry_policy = RetryPolicy(
        max_retries=config.max_retries, base_delay=0.01, strategy=BackoffStrategy.EXPONENTIAL
    )

    circuit = CircuitBreaker(
        "mixed_service", CircuitBreakerConfig(failure_threshold=10, timeout=0.5)
    )

    successful = 0
    failed_transient = 0
    failed_permanent = 0
    retries_used = 0
    circuit_trips = [0]

    def on_trip(old, new):
        if new.name == "OPEN":
            circuit_trips[0] += 1

    circuit.add_state_listener(on_trip)

    records = [{"id": f"rec_{i}", "value": i} for i in range(num_records)]
    start_time = time.time()

    for record in records:
        attempts = 0
        success = False
        last_error: Optional[Exception] = None

        while attempts < config.max_retries and not success:
            attempts += 1
            try:
                # Check circuit
                if not circuit.allow_request():
                    raise ProcessingError("Circuit open")

                # Try processing
                with profiler.profile("mixed_error_process"):
                    injector.maybe_inject_error(str(record["id"]))
                    {"id": record["id"], "result": record["value"] * 2}  # type: ignore

                circuit.record_success()
                success = True
                successful += 1

            except TransientError as e:
                last_error = e
                retries_used += 1
                circuit.record_failure()
                time.sleep(retry_policy.calculate_delay(attempts))

            except (ValidationError, ProcessingError) as e:
                last_error = e
                circuit.record_failure()
                break  # No retry for permanent errors

        if not success:
            if isinstance(last_error, TransientError):
                failed_transient += 1
            else:
                failed_permanent += 1

            # Add to DLQ
            if last_error:
                # Cast to HiveFrameError for type checking
                error_to_log = (
                    last_error
                    if isinstance(last_error, HiveFrameError)
                    else ProcessingError(str(last_error))
                )
                dlq.push(
                    DeadLetterRecord(
                        original_data=record,
                        error=error_to_log,
                        partition_id=str(record["id"]),
                        worker_id="test_worker",
                        attempt_count=attempts,
                        first_failure=time.time(),
                    )
                )

    elapsed = time.time() - start_time
    stats = injector.get_stats()

    transient_recovered = stats["transient"] - failed_transient
    recovery_rate = transient_recovered / max(1, stats["transient"])

    return ScenarioResult(
        scenario_name="Mixed Error Conditions",
        total_records=num_records,
        successful=successful,
        failed_transient=failed_transient,
        failed_permanent=failed_permanent,
        retries_used=retries_used,
        dead_letter_count=dlq.get_stats()["size"],
        circuit_trips=circuit_trips[0],
        elapsed_seconds=elapsed,
        throughput=num_records / elapsed,
        error_rate=(failed_transient + failed_permanent) / num_records,
        recovery_rate=recovery_rate,
    )


def run_poison_pill_scenario(num_records: int = 1000, poison_rate: float = 0.02) -> ScenarioResult:
    """
    Scenario 5: Poison Pill Detection

    Tests the system's ability to detect and isolate bad records
    that consistently cause failures.

    Expected behavior:
    - Poison pills should be quickly identified
    - Processing should continue for good records
    - Poison pills should be routed to DLQ with clear labeling
    """
    logger.info("Starting poison pill scenario", num_records=num_records, poison_rate=poison_rate)

    # Create records with some poison pills
    records = []
    poison_ids = set()

    for i in range(num_records):
        record = {"id": f"rec_{i}", "value": i}
        # Deterministically mark some as poison
        if hash(f"rec_{i}") % 100 < poison_rate * 100:
            record["poison"] = True
            poison_ids.add(f"rec_{i}")
        records.append(record)

    dlq = DeadLetterQueue(max_size=500)
    successful = 0
    poisoned = 0

    start_time = time.time()

    for record in records:
        try:
            with profiler.profile("poison_pill_process"):
                if record.get("poison"):
                    raise ValidationError(
                        f"Poison pill: {record['id']}", field="poison", expected=False, actual=True
                    )
                {"id": record["id"], "result": record["value"] * 2}  # type: ignore
            successful += 1

        except ValidationError as e:
            poisoned += 1
            dlq.push(
                DeadLetterRecord(
                    original_data=record,
                    error=e,
                    partition_id=str(record["id"]),
                    worker_id="test_worker",
                    attempt_count=1,
                    first_failure=time.time(),
                    metadata={"poison_pill": True},
                )
            )

    elapsed = time.time() - start_time

    # Verify all poison pills were caught
    dlq_records = dlq.peek(n=poisoned)
    caught_poisons = sum(1 for r in dlq_records if r.metadata.get("poison_pill"))

    logger.info(
        "Poison pill detection complete",
        expected_poisons=len(poison_ids),
        caught_poisons=caught_poisons,
    )

    return ScenarioResult(
        scenario_name="Poison Pill Detection",
        total_records=num_records,
        successful=successful,
        failed_transient=0,
        failed_permanent=poisoned,
        retries_used=0,
        dead_letter_count=poisoned,
        circuit_trips=0,
        elapsed_seconds=elapsed,
        throughput=num_records / elapsed,
        error_rate=poisoned / num_records,
        recovery_rate=0,  # Poison pills can't recover
    )


def run_all_error_scenarios() -> List[ScenarioResult]:
    """Run all error handling scenarios and return results."""
    results = []

    print("\n" + "=" * 60)
    print("HiveFrame Error Handling Challenge Suite")
    print("=" * 60)

    scenarios = [
        ("Transient Recovery", lambda: run_transient_recovery_scenario(1000, 0.2)),
        ("Dead Letter Queue", lambda: run_dead_letter_scenario(1000, 0.05)),
        ("Circuit Breaker", lambda: run_circuit_breaker_scenario(500, 30, 5)),
        ("Mixed Errors", lambda: run_mixed_error_scenario(1000)),
        ("Poison Pills", lambda: run_poison_pill_scenario(1000, 0.02)),
    ]

    for name, scenario_fn in scenarios:
        print(f"\nRunning: {name}...")
        try:
            result = scenario_fn()
            results.append(result)
            print(result.summary())
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_records = sum(r.total_records for r in results)
    total_success = sum(r.successful for r in results)
    avg_recovery = sum(r.recovery_rate for r in results) / len(results) if results else 0

    print(f"Total Records Processed: {total_records}")
    print(
        f"Total Successful:        {total_success} ({100*total_success/max(1,total_records):.1f}%)"
    )
    print(f"Average Recovery Rate:   {100*avg_recovery:.1f}%")

    return results


class ErrorHandlingChallenger:
    """
    Wrapper class for running error handling challenges.
    
    Provides a standardized interface for the CI system.
    """
    
    def run_all_challenges(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all error handling challenges and return results in CI format.
        
        Returns:
            Dict mapping challenge name to result dict with 'passed' key.
        """
        results = {}
        
        scenarios = [
            ("transient_recovery", lambda: run_transient_recovery_scenario(1000, 0.2)),
            ("dead_letter_queue", lambda: run_dead_letter_scenario(1000, 0.05)),
            ("circuit_breaker", lambda: run_circuit_breaker_scenario(500, 30, 5)),
            ("mixed_errors", lambda: run_mixed_error_scenario(1000)),
            ("poison_pills", lambda: run_poison_pill_scenario(1000, 0.02)),
        ]
        
        for name, scenario_fn in scenarios:
            try:
                result = scenario_fn()
                # Calculate success rate from the result
                if result.total_records > 0:
                    success_rate = result.successful / result.total_records
                else:
                    success_rate = 0.0
                # Consider a challenge passed if success rate is above 70% or recovery rate is good
                passed = success_rate >= 0.7 or result.recovery_rate >= 0.8
                results[name] = {
                    "passed": passed,
                    "success_rate": success_rate,
                    "recovery_rate": result.recovery_rate,
                    "total_records": result.total_records,
                    "elapsed_seconds": result.elapsed_seconds,
                }
            except Exception as e:
                logger.error(f"Challenge {name} failed with exception", error=str(e))
                results[name] = {
                    "passed": False,
                    "error": str(e),
                }
        
        return results


if __name__ == "__main__":
    run_all_error_scenarios()
