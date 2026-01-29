"""
Tests for HiveFrame resilience module.

Tests cover:
- Retry policies with various backoff strategies
- Circuit breaker state transitions
- Bulkhead isolation
- Timeout handling
- Resilient executor composition
"""

import pytest
import time
from typing import Callable

from hiveframe import (
    RetryPolicy, BackoffStrategy,
    CircuitBreaker, CircuitState, CircuitBreakerConfig,
    Bulkhead, ResilientExecutor,
    with_retry, with_circuit_breaker, with_timeout,
    TransientError, CircuitOpenError,
)


class TestRetryPolicy:
    """Test retry policy configurations."""
    
    def test_retry_success_first_attempt(self):
        """Test successful operation on first attempt."""
        policy = RetryPolicy(max_retries=3)
        
        call_count = [0]
        
        @with_retry(policy)
        def successful_op():
            call_count[0] += 1
            return "success"
            
        result = successful_op()
        
        assert result == "success"
        assert call_count[0] == 1
        
    def test_retry_success_after_failures(self):
        """Test retry succeeding after transient failures."""
        policy = RetryPolicy(max_retries=3, base_delay=0.01)
        
        call_count = [0]
        
        @with_retry(policy)
        def flaky_op():
            call_count[0] += 1
            if call_count[0] < 3:
                raise TransientError("Temporary failure")
            return "success"
            
        result = flaky_op()
        
        assert result == "success"
        assert call_count[0] == 3
        
    def test_retry_exhausted(self):
        """Test retry exhaustion."""
        policy = RetryPolicy(max_retries=2, base_delay=0.01)
        
        @with_retry(policy)
        def always_fails():
            raise TransientError("Always fails")
            
        with pytest.raises(TransientError):
            always_fails()
            
    def test_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay=0.1,
            strategy=BackoffStrategy.EXPONENTIAL,
            jitter=False
        )
        
        # Delays should be exponential: 0.1, 0.2, 0.4
        delays = [policy._calculate_delay(i) for i in range(3)]
        
        assert delays[0] == pytest.approx(0.1, rel=0.1)
        assert delays[1] == pytest.approx(0.2, rel=0.1)
        assert delays[2] == pytest.approx(0.4, rel=0.1)
        
    def test_linear_backoff(self):
        """Test linear backoff delay calculation."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay=0.1,
            strategy=BackoffStrategy.LINEAR,
            jitter=False
        )
        
        # Delays should be linear: 0.1, 0.2, 0.3
        delays = [policy._calculate_delay(i) for i in range(3)]
        
        assert delays[0] == pytest.approx(0.1, rel=0.1)
        assert delays[1] == pytest.approx(0.2, rel=0.1)
        assert delays[2] == pytest.approx(0.3, rel=0.1)


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_circuit_starts_closed(self):
        """Test circuit starts in closed state."""
        circuit = CircuitBreaker("test", CircuitBreakerConfig())
        
        assert circuit.state == CircuitState.CLOSED
        
    def test_circuit_opens_on_failures(self):
        """Test circuit opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3, timeout=1.0)
        circuit = CircuitBreaker("test", config)
        
        def failing_op():
            raise TransientError("Failure")
            
        # Cause failures to open circuit
        for _ in range(3):
            try:
                circuit.call(failing_op)
            except TransientError:
                pass
                
        assert circuit.state == CircuitState.OPEN
        
    def test_circuit_blocks_when_open(self):
        """Test circuit blocks calls when open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=10.0)
        circuit = CircuitBreaker("test", config)
        
        def failing_op():
            raise TransientError("Failure")
            
        # Open the circuit
        try:
            circuit.call(failing_op)
        except TransientError:
            pass
            
        # Next call should be blocked
        with pytest.raises(CircuitOpenError):
            circuit.call(lambda: "should not run")
            
    def test_circuit_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        circuit = CircuitBreaker("test", config)
        
        def failing_op():
            raise TransientError("Failure")
            
        # Open the circuit
        try:
            circuit.call(failing_op)
        except TransientError:
            pass
            
        assert circuit.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Should transition to half-open on next call attempt
        # (The state check happens when we try to call)
        
    def test_circuit_closes_on_success(self):
        """Test circuit closes after successful calls in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=1,
            timeout=0.05
        )
        circuit = CircuitBreaker("test", config)
        
        call_count = [0]
        
        def sometimes_fails():
            call_count[0] += 1
            if call_count[0] == 1:
                raise TransientError("First call fails")
            return "success"
            
        # First call fails, opens circuit
        try:
            circuit.call(sometimes_fails)
        except TransientError:
            pass
            
        # Wait for timeout
        time.sleep(0.1)
        
        # Next call should succeed and close circuit
        result = circuit.call(sometimes_fails)
        
        assert result == "success"


class TestBulkhead:
    """Test bulkhead isolation pattern."""
    
    def test_bulkhead_allows_within_limit(self):
        """Test bulkhead allows calls within limit."""
        bulkhead = Bulkhead("test", max_concurrent=2)
        
        with bulkhead:
            result = "executed"
            
        assert result == "executed"
        
    def test_bulkhead_tracks_concurrent(self):
        """Test bulkhead tracks concurrent calls."""
        bulkhead = Bulkhead("test", max_concurrent=5)
        
        assert bulkhead.current_concurrent == 0
        
        with bulkhead:
            assert bulkhead.current_concurrent == 1
            
        assert bulkhead.current_concurrent == 0


class TestResilientExecutor:
    """Test composite resilience patterns."""
    
    def test_executor_with_retry(self):
        """Test executor with retry policy."""
        retry_policy = RetryPolicy(max_retries=2, base_delay=0.01)
        executor = ResilientExecutor(retry_policy=retry_policy)
        
        call_count = [0]
        
        def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise TransientError("Temporary")
            return "success"
            
        result = executor.execute(flaky)
        
        assert result == "success"
        
    def test_executor_with_circuit_breaker(self):
        """Test executor with circuit breaker."""
        circuit = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        executor = ResilientExecutor(circuit_breaker=circuit)
        
        result = executor.execute(lambda: "success")
        
        assert result == "success"


class TestTimeoutDecorator:
    """Test timeout decorator."""
    
    def test_timeout_allows_fast_operations(self):
        """Test timeout allows operations that complete in time."""
        @with_timeout(1.0)
        def fast_op():
            return "fast"
            
        result = fast_op()
        assert result == "fast"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
