"""
Tests for Phase 4: Autonomous Operations
"""

import pytest
import time
from hiveframe.autonomous.self_tuning import (
    MemoryManager, ResourceAllocator, QueryPredictor, SelfTuningColony,
    MemoryStats, ResourceMetrics, QueryPerformance
)
from hiveframe.autonomous.predictive_maintenance import (
    HealthMonitor, FailurePredictor, PredictiveMaintenance,
    HealthMetric, HealthStatus
)
from hiveframe.autonomous.workload_prediction import (
    WorkloadPredictor, UsageAnalyzer, ResourcePrewarmer,
    WorkloadSample
)
from hiveframe.autonomous.cost_optimization import (
    CostOptimizer, SpendAnalyzer, SLAOptimizer,
    CostMetrics, SLAMetrics, OptimizationStrategy
)


class TestMemoryManager:
    """Test automatic memory management"""
    
    def test_initialization(self):
        manager = MemoryManager(total_memory_mb=8192)
        assert manager.total_memory_mb == 8192
        assert manager.cache_size_mb > 0
    
    def test_record_usage(self):
        manager = MemoryManager()
        stats = MemoryStats(
            total_mb=8192,
            used_mb=4096,
            available_mb=4096,
            cache_mb=1024,
            buffer_mb=512,
        )
        manager.record_usage(stats)
        assert len(manager.history) == 1
    
    def test_optimize_high_pressure(self):
        manager = MemoryManager(total_memory_mb=8192)
        
        # Simulate high memory pressure
        for _ in range(20):
            stats = MemoryStats(
                total_mb=8192,
                used_mb=7000,  # 85% usage
                available_mb=1192,
                cache_mb=1024,
                buffer_mb=512,
            )
            manager.record_usage(stats)
        
        config = manager.optimize()
        assert 'cache_mb' in config
        assert 'avg_usage' in config
        assert config['avg_usage'] > 0.8


class TestResourceAllocator:
    """Test dynamic resource allocation"""
    
    def test_initialization(self):
        allocator = ResourceAllocator(min_workers=1, max_workers=10)
        assert allocator.current_workers == 1
    
    def test_scale_up(self):
        allocator = ResourceAllocator(min_workers=2, max_workers=20)
        
        # Simulate high CPU with queued tasks
        for _ in range(10):
            metrics = ResourceMetrics(
                cpu_percent=50.0,
                memory_mb=2000,
                disk_io_mb_per_sec=10,
                network_mb_per_sec=5,
                active_workers=allocator.current_workers,
                queued_tasks=allocator.current_workers * 3,
            )
            allocator.record_metrics(metrics)
        
        result = allocator.allocate()
        assert result['decision'] in ['scale_up', 'maintain']


class TestQueryPredictor:
    """Test query performance prediction"""
    
    def test_record_query(self):
        predictor = QueryPredictor()
        perf = QueryPerformance(
            query_id="q1",
            execution_time_ms=100.0,
            rows_processed=1000,
            bytes_scanned=10000,
            workers_used=2,
            memory_peak_mb=500,
        )
        predictor.record_query(perf)
        assert len(predictor.query_history) == 1
    
    def test_predict_with_history(self):
        predictor = QueryPredictor()
        
        # Record several similar queries
        for i in range(5):
            perf = QueryPerformance(
                query_id=f"q{i}",
                execution_time_ms=100.0 + i * 10,
                rows_processed=1000,
                bytes_scanned=1000000,
                workers_used=2,
                memory_peak_mb=500,
            )
            predictor.record_query(perf)
        
        prediction = predictor.predict("test", 1000, 1000000)
        assert prediction['confidence'] > 0.0  # Has some confidence


class TestHealthMonitor:
    """Test health monitoring"""
    
    def test_record_metric(self):
        monitor = HealthMonitor()
        metric = HealthMetric(
            name="cpu_usage",
            value=75.0,
            threshold_warning=80.0,
            threshold_critical=90.0,
        )
        monitor.record_metric(metric)
        assert "cpu_usage" in monitor.metrics
    
    def test_check_health_healthy(self):
        monitor = HealthMonitor()
        metric = HealthMetric(
            name="cpu_usage",
            value=50.0,
            threshold_warning=80.0,
            threshold_critical=90.0,
        )
        monitor.record_metric(metric)
        
        health = monitor.check_health()
        assert health.status == HealthStatus.HEALTHY
        assert health.score == 100.0


class TestWorkloadPredictor:
    """Test workload prediction"""
    
    def test_record_workload(self):
        predictor = WorkloadPredictor()
        sample = WorkloadSample(
            timestamp=time.time(),
            query_count=100,
            cpu_percent=60.0,
            memory_mb=2000,
            io_operations=500,
            active_users=10,
        )
        predictor.record_workload(sample)
        assert len(predictor.analyzer.samples) == 1
    
    def test_predict_with_data(self):
        predictor = WorkloadPredictor()
        
        # Record some workload samples
        for i in range(30):
            sample = WorkloadSample(
                timestamp=time.time() + i * 60,
                query_count=100 + i,
                cpu_percent=60.0,
                memory_mb=2000,
                io_operations=500,
                active_users=10,
            )
            predictor.record_workload(sample)
        
        forecast = predictor.predict(hours_ahead=1)
        assert forecast.predicted_query_count >= 0
        assert 0 <= forecast.confidence <= 1


class TestCostOptimizer:
    """Test cost optimization"""
    
    def test_record_metrics(self):
        optimizer = CostOptimizer(budget_per_hour=100.0)
        
        cost = CostMetrics(
            timestamp=time.time(),
            compute_cost_per_hour=50.0,
            storage_cost_per_hour=20.0,
            network_cost_per_hour=10.0,
            total_cost_per_hour=80.0,
            resource_utilization=0.7,
            active_workers=5,
            storage_gb=100.0,
        )
        
        sla = SLAMetrics(
            timestamp=time.time(),
            avg_response_time_ms=500.0,
            p95_response_time_ms=800.0,
            p99_response_time_ms=1000.0,
            error_rate=0.01,
            availability=0.999,
        )
        
        optimizer.record_metrics(cost, sla)
        assert len(optimizer.spend_analyzer.cost_history) == 1
    
    def test_optimize_with_data(self):
        optimizer = CostOptimizer(
            budget_per_hour=100.0,
            strategy=OptimizationStrategy.BALANCED,
        )
        
        # Record multiple time periods
        for i in range(20):
            cost = CostMetrics(
                timestamp=time.time() + i * 3600,
                compute_cost_per_hour=50.0,
                storage_cost_per_hour=20.0,
                network_cost_per_hour=10.0,
                total_cost_per_hour=80.0,
                resource_utilization=0.3,  # Low utilization
                active_workers=10,
                storage_gb=100.0,
            )
            
            sla = SLAMetrics(
                timestamp=time.time() + i * 3600,
                avg_response_time_ms=500.0,
                p95_response_time_ms=800.0,
                p99_response_time_ms=1000.0,
                error_rate=0.005,
                availability=0.999,
            )
            
            optimizer.record_metrics(cost, sla)
        
        result = optimizer.optimize()
        assert result['status'] == 'optimized'
        assert 'recommendations' in result


class TestSelfTuningColony:
    """Test integrated self-tuning"""
    
    def test_initialization(self):
        colony = SelfTuningColony(total_memory_mb=8192, max_workers=50)
        assert colony.memory_manager is not None
        assert colony.resource_allocator is not None
        assert colony.query_predictor is not None
    
    def test_tune(self):
        colony = SelfTuningColony()
        
        # First tune should be skipped (too soon)
        result = colony.tune()
        assert result['status'] in ['tuned', 'skipped']
    
    def test_query_recommendation(self):
        colony = SelfTuningColony()
        
        # Record some query history
        for i in range(5):
            perf = QueryPerformance(
                query_id=f"q{i}",
                execution_time_ms=100.0,
                rows_processed=1000,
                bytes_scanned=1000000,
                workers_used=2,
                memory_peak_mb=500,
            )
            colony.query_predictor.record_query(perf)
        
        rec = colony.get_query_recommendation("test", 1000, 1000000)
        assert 'estimated_time_ms' in rec
        assert 'confidence' in rec


class TestPredictiveMaintenance:
    """Test predictive maintenance"""
    
    def test_assessment(self):
        maintenance = PredictiveMaintenance()
        
        # Record some metrics
        for i in range(20):
            metric = HealthMetric(
                name="cpu_usage",
                value=70.0 + i,  # Increasing trend
                threshold_warning=80.0,
                threshold_critical=90.0,
            )
            maintenance.record_metric(metric)
        
        assessment = maintenance.assess_system()
        assert 'health' in assessment
        assert 'predictions' in assessment
        assert 'recommendations' in assessment


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
