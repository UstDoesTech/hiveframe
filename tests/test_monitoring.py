"""
Tests for HiveFrame monitoring module.

Tests cover:
- Prometheus-style metrics (Counter, Gauge, Histogram, Summary)
- MetricLabels and label handling
- MetricsRegistry management
- Structured logging (Logger, LogRecord, LogLevel)
- Distributed tracing (Tracer, TraceSpan)
- Performance profiler
- Colony health monitor
"""

import pytest
import time
import threading
from typing import Dict, Any
from io import StringIO

from hiveframe.monitoring import (
    # Metrics
    MetricType, MetricLabels, Metric,
    Counter, Gauge, Histogram, Summary,
    MetricsRegistry, get_registry,
    # Logging
    LogLevel, LogRecord, LogHandler,
    ConsoleHandler, BufferedHandler, Logger, get_logger,
    # Tracing
    TraceSpan, Tracer, get_tracer,
    # Health monitoring
    WorkerHealthSnapshot, ColonyHealthReport, ColonyHealthMonitor,
    # Profiler
    PerformanceProfiler, get_profiler,
)


# ============================================================================
# MetricLabels Tests
# ============================================================================

class TestMetricLabels:
    """Tests for MetricLabels class."""
    
    def test_empty_labels(self):
        """Test creating empty labels."""
        labels = MetricLabels()
        assert labels.labels == {}
        
    def test_labels_with_values(self):
        """Test creating labels with values."""
        labels = MetricLabels({"method": "GET", "path": "/api"})
        assert labels.labels["method"] == "GET"
        
    def test_labels_hashable(self):
        """Test that labels are hashable for dict keys."""
        labels1 = MetricLabels({"a": "1"})
        labels2 = MetricLabels({"a": "1"})
        
        d = {labels1: "value"}
        assert d[labels2] == "value"
        
    def test_labels_equality(self):
        """Test label equality comparison."""
        labels1 = MetricLabels({"a": "1", "b": "2"})
        labels2 = MetricLabels({"b": "2", "a": "1"})  # Different order
        
        assert labels1 == labels2
        
    def test_prometheus_format_empty(self):
        """Test Prometheus formatting with no labels."""
        labels = MetricLabels()
        assert labels.to_prometheus() == ""
        
    def test_prometheus_format_with_labels(self):
        """Test Prometheus formatting with labels."""
        labels = MetricLabels({"method": "GET", "status": "200"})
        result = labels.to_prometheus()
        
        assert 'method="GET"' in result
        assert 'status="200"' in result
        assert result.startswith("{")
        assert result.endswith("}")


# ============================================================================
# Counter Tests
# ============================================================================

class TestCounter:
    """Tests for Counter metric."""
    
    def test_counter_creation(self):
        """Test creating a counter."""
        counter = Counter("requests_total", "Total requests")
        assert counter.name == "requests_total"
        assert counter.metric_type == MetricType.COUNTER
        
    def test_counter_increment(self):
        """Test incrementing counter."""
        counter = Counter("test_counter")
        
        counter.inc()
        assert counter.get() == 1.0
        
        counter.inc(5)
        assert counter.get() == 6.0
        
    def test_counter_with_labels(self):
        """Test counter with labels."""
        counter = Counter("requests_total")
        
        counter.inc(labels={"method": "GET"})
        counter.inc(labels={"method": "POST"})
        counter.inc(labels={"method": "GET"})
        
        assert counter.get(labels={"method": "GET"}) == 2.0
        assert counter.get(labels={"method": "POST"}) == 1.0
        
    def test_counter_negative_raises(self):
        """Test that negative increment raises error."""
        counter = Counter("test_counter")
        
        with pytest.raises(ValueError):
            counter.inc(-1)
            
    def test_counter_collect(self):
        """Test collecting counter values."""
        counter = Counter("test_counter")
        counter.inc(labels={"a": "1"})
        counter.inc(2, labels={"a": "2"})
        
        collected = counter.collect()
        assert len(collected) == 2
        
    def test_counter_prometheus_format(self):
        """Test Prometheus exposition format."""
        counter = Counter("http_requests_total", "Total HTTP requests")
        counter.inc(10, labels={"method": "GET"})
        
        output = counter.to_prometheus()
        
        assert "# HELP http_requests_total" in output
        assert "# TYPE http_requests_total counter" in output
        assert "http_requests_total" in output
        
    def test_counter_thread_safety(self):
        """Test counter is thread-safe."""
        counter = Counter("threaded_counter")
        
        def increment():
            for _ in range(100):
                counter.inc()
                
        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        assert counter.get() == 1000


# ============================================================================
# Gauge Tests
# ============================================================================

class TestGauge:
    """Tests for Gauge metric."""
    
    def test_gauge_creation(self):
        """Test creating a gauge."""
        gauge = Gauge("queue_size", "Current queue size")
        assert gauge.name == "queue_size"
        assert gauge.metric_type == MetricType.GAUGE
        
    def test_gauge_set(self):
        """Test setting gauge value."""
        gauge = Gauge("test_gauge")
        
        gauge.set(42)
        assert gauge.get() == 42
        
        gauge.set(10)
        assert gauge.get() == 10
        
    def test_gauge_inc_dec(self):
        """Test incrementing and decrementing gauge."""
        gauge = Gauge("test_gauge")
        
        gauge.set(10)
        gauge.inc(5)
        assert gauge.get() == 15
        
        gauge.dec(3)
        assert gauge.get() == 12
        
    def test_gauge_with_labels(self):
        """Test gauge with labels."""
        gauge = Gauge("active_workers")
        
        gauge.set(5, labels={"role": "employed"})
        gauge.set(3, labels={"role": "scout"})
        
        assert gauge.get(labels={"role": "employed"}) == 5
        assert gauge.get(labels={"role": "scout"}) == 3
        
    def test_gauge_negative_value(self):
        """Test gauge can have negative values."""
        gauge = Gauge("temperature")
        gauge.set(-10)
        assert gauge.get() == -10
        
    def test_gauge_prometheus_format(self):
        """Test Prometheus exposition format."""
        gauge = Gauge("memory_bytes", "Memory usage in bytes")
        gauge.set(1024000)
        
        output = gauge.to_prometheus()
        
        assert "# TYPE memory_bytes gauge" in output


# ============================================================================
# Histogram Tests
# ============================================================================

class TestHistogram:
    """Tests for Histogram metric."""
    
    def test_histogram_creation(self):
        """Test creating a histogram."""
        hist = Histogram("request_duration_seconds", "Request latency")
        assert hist.name == "request_duration_seconds"
        assert hist.metric_type == MetricType.HISTOGRAM
        
    def test_histogram_observe(self):
        """Test observing values."""
        hist = Histogram("test_hist")
        
        hist.observe(0.1)
        hist.observe(0.5)
        hist.observe(1.5)
        
        collected = hist.collect()
        assert len(collected) == 1
        assert collected[0]['count'] == 3
        
    def test_histogram_custom_buckets(self):
        """Test histogram with custom buckets."""
        hist = Histogram(
            "custom_hist",
            buckets=(0.1, 0.5, 1.0, float('inf'))
        )
        
        hist.observe(0.05)  # bucket 0.1
        hist.observe(0.3)   # bucket 0.5
        hist.observe(0.7)   # bucket 1.0
        hist.observe(2.0)   # bucket inf
        
        collected = hist.collect()[0]
        assert collected['count'] == 4
        
    def test_histogram_time_context(self):
        """Test histogram time() context manager."""
        hist = Histogram("operation_seconds")
        
        with hist.time():
            time.sleep(0.05)  # 50ms
            
        collected = hist.collect()[0]
        assert collected['count'] == 1
        assert collected['sum'] >= 0.05
        
    def test_histogram_with_labels(self):
        """Test histogram with labels."""
        hist = Histogram("request_duration")
        
        hist.observe(0.1, labels={"method": "GET"})
        hist.observe(0.2, labels={"method": "POST"})
        
        collected = hist.collect()
        assert len(collected) == 2
        
    def test_histogram_prometheus_format(self):
        """Test Prometheus exposition format."""
        hist = Histogram("http_duration_seconds")
        hist.observe(0.1)
        
        output = hist.to_prometheus()
        
        assert "# TYPE http_duration_seconds histogram" in output
        assert "_bucket" in output
        assert "_sum" in output
        assert "_count" in output


# ============================================================================
# Summary Tests
# ============================================================================

class TestSummary:
    """Tests for Summary metric."""
    
    def test_summary_creation(self):
        """Test creating a summary."""
        summary = Summary("request_latency", "Request latency summary")
        assert summary.name == "request_latency"
        assert summary.metric_type == MetricType.SUMMARY
        
    def test_summary_observe(self):
        """Test observing values."""
        summary = Summary("test_summary")
        
        for i in range(100):
            summary.observe(i / 100.0)
            
        collected = summary.collect()[0]
        assert collected['count'] == 100
        
    def test_summary_with_labels(self):
        """Test summary with labels."""
        summary = Summary("operation_duration")
        
        summary.observe(0.1, labels={"op": "read"})
        summary.observe(0.2, labels={"op": "write"})
        
        collected = summary.collect()
        assert len(collected) == 2


# ============================================================================
# MetricsRegistry Tests
# ============================================================================

class TestMetricsRegistry:
    """Tests for MetricsRegistry."""
    
    def test_registry_creation(self):
        """Test creating a registry."""
        registry = MetricsRegistry(prefix="myapp")
        assert registry.prefix == "myapp"
        
    def test_registry_counter(self):
        """Test getting counter from registry."""
        registry = MetricsRegistry(prefix="test")
        
        counter = registry.counter("requests", "Total requests")
        counter.inc()
        
        # Getting same counter again should return same instance
        same_counter = registry.counter("requests")
        assert same_counter.get() == 1
        
    def test_registry_gauge(self):
        """Test getting gauge from registry."""
        registry = MetricsRegistry(prefix="test")
        
        gauge = registry.gauge("queue_size")
        gauge.set(42)
        
        assert registry.gauge("queue_size").get() == 42
        
    def test_registry_histogram(self):
        """Test getting histogram from registry."""
        registry = MetricsRegistry(prefix="test")
        
        hist = registry.histogram("latency")
        hist.observe(0.1)
        
        collected = registry.histogram("latency").collect()
        assert collected[0]['count'] == 1
        
    def test_registry_prefix(self):
        """Test that registry applies prefix to metric names."""
        registry = MetricsRegistry(prefix="hiveframe")
        
        counter = registry.counter("requests")
        assert counter.name == "hiveframe_requests"
        
    def test_registry_prometheus_export(self):
        """Test exporting all metrics in Prometheus format."""
        registry = MetricsRegistry(prefix="export_test")
        
        registry.counter("requests").inc(10)
        registry.gauge("active").set(5)
        
        output = registry.to_prometheus()
        
        assert "export_test_requests" in output
        assert "export_test_active" in output
        
    def test_registry_json_export(self):
        """Test exporting all metrics as JSON."""
        registry = MetricsRegistry(prefix="json_test")
        
        registry.counter("ops").inc(100)
        
        data = registry.to_json()
        
        assert "json_test_ops" in data
        assert data["json_test_ops"]["type"] == "COUNTER"
        
    def test_get_registry(self):
        """Test getting the default registry."""
        reg = get_registry()
        assert reg is not None
        assert isinstance(reg, MetricsRegistry)


# ============================================================================
# Logging Tests
# ============================================================================

class TestLogLevel:
    """Tests for LogLevel enum."""
    
    def test_log_level_ordering(self):
        """Test log levels are properly ordered."""
        assert LogLevel.DEBUG.value < LogLevel.INFO.value
        assert LogLevel.INFO.value < LogLevel.WARNING.value
        assert LogLevel.WARNING.value < LogLevel.ERROR.value
        assert LogLevel.ERROR.value < LogLevel.CRITICAL.value


class TestLogRecord:
    """Tests for LogRecord."""
    
    def test_log_record_creation(self):
        """Test creating a log record."""
        record = LogRecord(
            timestamp=time.time(),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test"
        )
        
        assert record.level == LogLevel.INFO
        assert record.message == "Test message"
        
    def test_log_record_to_json(self):
        """Test log record JSON serialization."""
        record = LogRecord(
            timestamp=1234567890.0,
            level=LogLevel.ERROR,
            message="Error occurred",
            logger_name="processor",
            extra={"partition": 3}
        )
        
        json_str = record.to_json()
        
        assert '"level": "ERROR"' in json_str
        assert '"message": "Error occurred"' in json_str
        assert '"partition": 3' in json_str
        
    def test_log_record_to_text(self):
        """Test log record text formatting."""
        record = LogRecord(
            timestamp=time.time(),
            level=LogLevel.WARNING,
            message="Warning message",
            logger_name="test",
            extra={"key": "value"}
        )
        
        text = record.to_text()
        
        assert "WARNING" in text
        assert "Warning message" in text
        assert "key=value" in text


class TestBufferedHandler:
    """Tests for BufferedHandler."""
    
    def test_buffered_handler_stores_logs(self):
        """Test buffered handler stores log records."""
        handler = BufferedHandler(max_size=10)
        
        record = LogRecord(
            timestamp=time.time(),
            level=LogLevel.INFO,
            message="Test",
            logger_name="test"
        )
        handler.handle(record)
        
        logs = handler.get_logs()
        assert len(logs) == 1
        
    def test_buffered_handler_max_size(self):
        """Test buffered handler respects max size."""
        handler = BufferedHandler(max_size=5)
        
        for i in range(10):
            record = LogRecord(
                timestamp=time.time(),
                level=LogLevel.INFO,
                message=f"Message {i}",
                logger_name="test"
            )
            handler.handle(record)
            
        logs = handler.get_logs()
        assert len(logs) == 5
        
    def test_buffered_handler_level_filter(self):
        """Test filtering logs by level."""
        handler = BufferedHandler()
        
        for level in [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]:
            record = LogRecord(
                timestamp=time.time(),
                level=level,
                message=f"{level.name} message",
                logger_name="test"
            )
            handler.handle(record)
            
        warning_and_above = handler.get_logs(level=LogLevel.WARNING)
        assert len(warning_and_above) == 2  # WARNING and ERROR
        
    def test_buffered_handler_clear(self):
        """Test clearing the buffer."""
        handler = BufferedHandler()
        
        record = LogRecord(
            timestamp=time.time(),
            level=LogLevel.INFO,
            message="Test",
            logger_name="test"
        )
        handler.handle(record)
        
        handler.clear()
        assert len(handler.get_logs()) == 0


class TestLogger:
    """Tests for Logger."""
    
    def test_logger_creation(self):
        """Test creating a logger."""
        logger = Logger("test-component")
        assert logger.name == "test-component"
        
    def test_logger_with_handler(self):
        """Test logger with custom handler."""
        handler = BufferedHandler()
        logger = Logger("test", handlers=[handler])
        
        logger.info("Test message")
        
        logs = handler.get_logs()
        assert len(logs) == 1
        assert logs[0].message == "Test message"
        
    def test_logger_level_filtering(self):
        """Test log level filtering."""
        handler = BufferedHandler()
        logger = Logger("test", level=LogLevel.WARNING, handlers=[handler])
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        logs = handler.get_logs()
        assert len(logs) == 2  # Only WARNING and ERROR
        
    def test_logger_with_context(self):
        """Test logger context propagation."""
        handler = BufferedHandler()
        logger = Logger("test", handlers=[handler])
        
        child = logger.with_context(request_id="123", user="alice")
        child.info("Request processed")
        
        logs = handler.get_logs()
        assert logs[0].extra["request_id"] == "123"
        assert logs[0].extra["user"] == "alice"
        
    def test_logger_extra_kwargs(self):
        """Test passing extra kwargs to log methods."""
        handler = BufferedHandler()
        logger = Logger("test", handlers=[handler])
        
        logger.info("Processing record", record_id=42, partition=3)
        
        logs = handler.get_logs()
        assert logs[0].extra["record_id"] == 42
        assert logs[0].extra["partition"] == 3
        
    def test_get_logger(self):
        """Test get_logger helper."""
        logger = get_logger("my-component")
        assert logger is not None
        assert logger.name == "my-component"


# ============================================================================
# Tracing Tests
# ============================================================================

class TestTraceSpan:
    """Tests for TraceSpan."""
    
    def test_span_creation(self):
        """Test creating a trace span."""
        span = TraceSpan(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id=None,
            operation="process_record",
            start_time=time.time()
        )
        
        assert span.trace_id == "trace-123"
        assert span.operation == "process_record"
        
    def test_span_duration(self):
        """Test span duration calculation."""
        start = time.time()
        span = TraceSpan(
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            operation="test",
            start_time=start,
            end_time=start + 0.1  # 100ms
        )
        
        assert 99 <= span.duration_ms <= 101
        
    def test_span_tags(self):
        """Test span tags."""
        span = TraceSpan(
            trace_id="t1",
            span_id="s1",
            parent_span_id=None,
            operation="test",
            start_time=time.time(),
            tags={"worker_id": "bee-1", "partition": "3"}
        )
        
        assert span.tags["worker_id"] == "bee-1"
        assert span.tags["partition"] == "3"


class TestTracer:
    """Tests for Tracer."""
    
    def test_tracer_creation(self):
        """Test creating a tracer."""
        tracer = Tracer()
        assert tracer is not None
        
    def test_start_trace(self):
        """Test starting a new trace."""
        tracer = Tracer()
        
        span = tracer.start_trace("process_batch", worker_id="bee-1")
        
        assert span is not None
        assert span.trace_id is not None
        assert span.operation == "process_batch"
        assert span.tags["worker_id"] == "bee-1"
        
    def test_start_child_span(self):
        """Test starting a child span."""
        tracer = Tracer()
        
        parent = tracer.start_trace("parent_op")
        child = tracer.start_span(
            parent.trace_id,
            "child_op",
            parent_span_id=parent.span_id
        )
        
        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id
        
    def test_end_span(self):
        """Test ending a span."""
        tracer = Tracer()
        
        span = tracer.start_trace("test_op")
        time.sleep(0.01)
        tracer.end_span(span)
        
        assert span.end_time is not None
        assert span.duration_ms >= 10
        assert span.status == 'ok'
        
    def test_end_span_with_error(self):
        """Test ending a span with error status."""
        tracer = Tracer()
        
        span = tracer.start_trace("failing_op")
        tracer.end_span(span, status='error')
        
        assert span.status == 'error'
        
    def test_log_to_span(self):
        """Test adding logs to a span."""
        tracer = Tracer()
        
        span = tracer.start_trace("test_op")
        tracer.log_to_span(span, "Processing started", count=10)
        tracer.log_to_span(span, "Processing complete")
        
        assert len(span.logs) == 2
        assert span.logs[0]["message"] == "Processing started"
        assert span.logs[0]["count"] == 10
        
    def test_get_trace(self):
        """Test getting all spans for a trace."""
        tracer = Tracer()
        
        parent = tracer.start_trace("parent")
        child1 = tracer.start_span(parent.trace_id, "child1", parent.span_id)
        child2 = tracer.start_span(parent.trace_id, "child2", parent.span_id)
        
        spans = tracer.get_trace(parent.trace_id)
        
        assert len(spans) == 3
        
    def test_trace_operation_context_manager(self):
        """Test trace_operation context manager."""
        tracer = Tracer()
        
        with tracer.trace_operation("context_op", key="value") as span:
            time.sleep(0.01)
            
        assert span.end_time is not None
        assert span.status == 'ok'
        assert span.tags["key"] == "value"
        
    def test_trace_operation_with_exception(self):
        """Test trace_operation captures exceptions."""
        tracer = Tracer()
        
        with pytest.raises(ValueError):
            with tracer.trace_operation("failing_op") as span:
                raise ValueError("Test error")
                
        assert span.status == 'error'
        assert any("Test error" in log["message"] for log in span.logs)
        
    def test_get_tracer(self):
        """Test getting the default tracer."""
        tracer = get_tracer()
        assert tracer is not None
        assert isinstance(tracer, Tracer)


# ============================================================================
# Performance Profiler Tests
# ============================================================================

class TestPerformanceProfiler:
    """Tests for PerformanceProfiler."""
    
    def test_profiler_creation(self):
        """Test creating a profiler."""
        profiler = PerformanceProfiler()
        assert profiler is not None
        
    def test_profile_operation(self):
        """Test profiling an operation."""
        profiler = PerformanceProfiler()
        
        with profiler.profile("test_operation"):
            time.sleep(0.05)
            
        stats = profiler.get_stats("test_operation")
        
        assert stats['count'] == 1
        assert stats['mean'] >= 0.05
        
    def test_profile_multiple_calls(self):
        """Test profiling multiple calls."""
        profiler = PerformanceProfiler()
        
        for _ in range(5):
            with profiler.profile("repeated_op"):
                time.sleep(0.01)
                
        stats = profiler.get_stats("repeated_op")
        
        assert stats['count'] == 5
        assert stats['mean'] >= 0.01
        
    def test_stats_percentiles(self):
        """Test profiler calculates percentiles."""
        profiler = PerformanceProfiler()
        
        for i in range(25):
            with profiler.profile("varied_op"):
                time.sleep(i * 0.001)  # 0-24ms
                
        stats = profiler.get_stats("varied_op")
        
        assert 'p50' in stats
        assert 'p95' in stats
        assert 'p99' in stats
        
    def test_stats_empty_operation(self):
        """Test getting stats for non-existent operation."""
        profiler = PerformanceProfiler()
        
        stats = profiler.get_stats("nonexistent")
        
        assert stats['count'] == 0
        assert stats['mean'] == 0
        
    def test_get_all_stats(self):
        """Test getting stats for all operations."""
        profiler = PerformanceProfiler()
        
        with profiler.profile("op1"):
            pass
        with profiler.profile("op2"):
            pass
            
        all_stats = profiler.get_all_stats()
        
        assert "op1" in all_stats
        assert "op2" in all_stats
        
    def test_report_generation(self):
        """Test generating human-readable report."""
        profiler = PerformanceProfiler()
        
        with profiler.profile("operation"):
            time.sleep(0.01)
            
        report = profiler.report()
        
        assert "Performance Report" in report
        assert "operation" in report
        
    def test_get_profiler(self):
        """Test getting the default profiler."""
        profiler = get_profiler()
        assert profiler is not None
        assert isinstance(profiler, PerformanceProfiler)


# ============================================================================
# Colony Health Monitor Tests
# ============================================================================

class TestColonyHealthDataClasses:
    """Tests for colony health data classes."""
    
    def test_worker_health_snapshot(self):
        """Test WorkerHealthSnapshot creation."""
        snapshot = WorkerHealthSnapshot(
            worker_id="bee-1",
            role="employed",
            processed_count=100,
            error_count=2,
            last_activity=time.time(),
            current_load=0.7,
            avg_latency=15.5,
            status="healthy"
        )
        
        assert snapshot.worker_id == "bee-1"
        assert snapshot.status == "healthy"
        
    def test_colony_health_report(self):
        """Test ColonyHealthReport creation."""
        report = ColonyHealthReport(
            timestamp=time.time(),
            total_workers=10,
            healthy_workers=8,
            degraded_workers=1,
            unhealthy_workers=1,
            dead_workers=0,
            overall_status="healthy",
            temperature=0.6,
            throughput=1000.0,
            error_rate=0.02,
            worker_snapshots=[],
            alerts=["WARNING: High load"]
        )
        
        assert report.total_workers == 10
        assert report.overall_status == "healthy"
        assert len(report.alerts) == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestMonitoringIntegration:
    """Integration tests for monitoring components."""
    
    def test_metrics_with_logging(self):
        """Test metrics and logging together."""
        registry = MetricsRegistry(prefix="integration")
        handler = BufferedHandler()
        logger = Logger("test", handlers=[handler])
        
        counter = registry.counter("processed")
        
        for i in range(5):
            counter.inc()
            logger.info(f"Processed item {i}", count=counter.get())
            
        assert counter.get() == 5
        logs = handler.get_logs()
        assert len(logs) == 5
        
    def test_tracing_with_profiler(self):
        """Test tracing with performance profiling."""
        tracer = Tracer()
        profiler = PerformanceProfiler()
        
        with tracer.trace_operation("complex_operation") as span:
            with profiler.profile("inner_work"):
                time.sleep(0.01)
                
        assert span.duration_ms >= 10
        assert profiler.get_stats("inner_work")['count'] == 1
        
    def test_full_observability_stack(self):
        """Test all observability components together."""
        # Set up components
        registry = MetricsRegistry(prefix="full_test")
        handler = BufferedHandler()
        logger = Logger("worker", level=LogLevel.DEBUG, handlers=[handler])
        tracer = Tracer()
        profiler = PerformanceProfiler()
        
        # Simulate a processing operation
        counter = registry.counter("records")
        latency = registry.histogram("latency")
        
        with tracer.trace_operation("process_batch") as span:
            for i in range(3):
                with profiler.profile("process_record"):
                    with latency.time():
                        time.sleep(0.005)
                        counter.inc()
                        logger.debug(f"Processed record {i}", record_id=i)
                        
        # Verify all components captured data
        assert counter.get() == 3
        assert latency.collect()[0]['count'] == 3
        assert len(handler.get_logs()) == 3
        assert span.duration_ms >= 15
        assert profiler.get_stats("process_record")['count'] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
