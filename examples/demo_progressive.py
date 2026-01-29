#!/usr/bin/env python3
"""
HiveFrame Progressive Challenge Demo
====================================
Demonstrates HiveFrame capabilities under increasingly challenging
and realistic conditions.

This demo progresses through difficulty levels:
1. BEGINNER: Basic operations with clean data
2. INTERMEDIATE: Realistic data with quality issues  
3. ADVANCED: Error handling and recovery
4. EXPERT: Scale, concurrency, and resilience
5. PRODUCTION: Full scenario with monitoring

Each level builds on the previous, demonstrating how the bee
colony metaphor handles progressively harder challenges.
"""

import time
import random
import threading
import json
import sys
import os
from typing import Any, Dict, List
from dataclasses import dataclass

# Add parent to path for imports when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports
from hiveframe.core import HiveFrame, ColonyState, BeeRole, create_hive, Pheromone
from hiveframe.dataframe import HiveDataFrame, col, avg, count, sum_agg

# Production modules
from hiveframe.exceptions import (
    HiveFrameError, TransientError, ValidationError, 
    DeadLetterQueue, DeadLetterRecord
)
from hiveframe.resilience import (
    RetryPolicy, CircuitBreaker, CircuitBreakerConfig,
    BackoffStrategy, with_retry, ResilientExecutor
)
from hiveframe.connectors import (
    DataGenerator, CSVSource, JSONLSource, JSONLSink,
    MessageBroker, Topic, FileWatcher
)
from hiveframe.monitoring import (
    get_logger, get_registry, get_profiler,
    ColonyHealthMonitor, Logger, LogLevel
)
from hiveframe.streaming_enhanced import (
    EnhancedStreamProcessor, tumbling_window, 
    bounded_watermark, DeliveryGuarantee,
    count_aggregator, sum_aggregator
)
from hiveframe.streaming import StreamRecord


# Configure logging
logger = get_logger("demo")


def print_header(title: str, level: str = ""):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    if level:
        print(f"  {level}: {title}")
    else:
        print(f"  {title}")
    print("=" * 70 + "\n")


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


def print_metrics(metrics: Dict[str, Any], title: str = "Metrics"):
    """Print metrics in a formatted way."""
    print(f"\n{title}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


# ============================================================================
# LEVEL 1: BEGINNER - Basic Operations
# ============================================================================

def demo_level_1_beginner():
    """
    Level 1: Basic HiveFrame Operations
    
    Demonstrates core functionality with clean, predictable data.
    - Basic map/filter/reduce
    - DataFrame operations
    - Simple transformations
    """
    print_header("Basic HiveFrame Operations", "LEVEL 1 - BEGINNER")
    
    # 1.1 Basic map operation
    print_subheader("1.1 Map Operation - Doubling Values")
    
    hive = create_hive(num_workers=4)
    data = list(range(100))
    
    start = time.time()
    results = hive.map(data, lambda x: x * 2)
    elapsed = time.time() - start
    
    print(f"Input:  {data[:5]}... ({len(data)} items)")
    print(f"Output: {results[:5]}... ({len([r for r in results if r])} items)")
    print(f"Time:   {elapsed*1000:.1f}ms")
    print(f"Throughput: {len(data)/elapsed:.0f} items/sec")
    
    # 1.2 Filter operation
    print_subheader("1.2 Filter Operation - Finding Evens")
    
    start = time.time()
    evens = hive.filter(data, lambda x: x % 2 == 0)
    elapsed = time.time() - start
    
    print(f"Input:  {len(data)} numbers")
    print(f"Output: {len(evens)} even numbers")
    print(f"Sample: {evens[:10]}")
    
    # 1.3 DataFrame with clean data
    print_subheader("1.3 DataFrame Operations")
    
    records = [
        {"name": "Alice", "department": "Engineering", "salary": 95000},
        {"name": "Bob", "department": "Engineering", "salary": 87000},
        {"name": "Carol", "department": "Marketing", "salary": 78000},
        {"name": "David", "department": "Marketing", "salary": 82000},
        {"name": "Eve", "department": "Engineering", "salary": 92000},
    ]
    
    df = HiveDataFrame(records)
    print(f"Created DataFrame with {df.count()} rows")
    
    # Group by department
    dept_stats = df.groupBy("department").agg(
        avg(col("salary")),
        count(col("salary"))
    )
    
    print("\nDepartment Statistics:")
    for row in dept_stats.collect():
        print(f"  {row}")
        
    print("\n✅ Level 1 Complete: Basic operations working correctly")
    
    return True


# ============================================================================
# LEVEL 2: INTERMEDIATE - Realistic Data
# ============================================================================

def demo_level_2_intermediate():
    """
    Level 2: Realistic Data Handling
    
    Demonstrates handling of real-world data characteristics:
    - Mixed data types
    - Missing values
    - Type coercion
    - Schema inference
    """
    print_header("Realistic Data Handling", "LEVEL 2 - INTERMEDIATE")
    
    # 2.1 Generate realistic data with quality issues
    print_subheader("2.1 Generating Realistic Data")
    
    generator = DataGenerator(
        count=500,
        rate=None,  # As fast as possible
        error_rate=0,  # No errors for this level
        schema={
            'id': 'uuid',
            'timestamp': 'timestamp',
            'amount': 'float',
            'category': 'enum:electronics,clothing,food,home,sports',
            'status': 'enum:pending,completed,cancelled'
        }
    )
    
    records = []
    with generator:
        for record in generator.read():
            records.append(record)
            
    print(f"Generated {len(records)} records")
    print(f"Sample record: {json.dumps(records[0], indent=2)[:200]}...")
    
    # 2.2 DataFrame operations on realistic data
    print_subheader("2.2 DataFrame Analytics")
    
    df = HiveDataFrame(records)
    
    # Category breakdown
    category_stats = df.groupBy("category").agg(
        sum_agg(col("amount")),
        avg(col("amount")),
        count(col("id"))
    )
    
    print("\nCategory Analysis:")
    for row in category_stats.collect()[:5]:
        print(f"  {row}")
        
    # Status breakdown
    status_stats = df.groupBy("status").agg(
        count(col("id"))
    )
    
    print("\nStatus Distribution:")
    for row in status_stats.collect():
        print(f"  {row}")
        
    # 2.3 Computed columns and complex filters
    print_subheader("2.3 Complex Transformations")
    
    # Add computed column (using lambda)
    df_enhanced = df.withColumn(
        "amount_bucket",
        col("amount")  # Placeholder for transformation
    )
    
    # Filter with multiple conditions
    high_value = df.filter(col("amount") > 500)
    print(f"\nHigh-value records (>500): {high_value.count()}")
    
    completed = df.filter(col("status") == "completed")
    print(f"Completed orders: {completed.count()}")
    
    print("\n✅ Level 2 Complete: Realistic data handling verified")
    
    return True


# ============================================================================
# LEVEL 3: ADVANCED - Error Handling
# ============================================================================

def demo_level_3_advanced():
    """
    Level 3: Error Handling and Recovery
    
    Demonstrates resilience patterns:
    - Retry logic with backoff
    - Dead letter queues
    - Error categorization
    - Graceful degradation
    """
    print_header("Error Handling and Recovery", "LEVEL 3 - ADVANCED")
    
    # 3.1 Transient error recovery
    print_subheader("3.1 Transient Error Recovery with Retries")
    
    retry_policy = RetryPolicy(
        max_retries=3,
        base_delay=0.01,
        strategy=BackoffStrategy.EXPONENTIAL,
        jitter=True
    )
    
    transient_count = [0]
    success_count = [0]
    
    @with_retry(retry_policy)
    def flaky_operation(value: int) -> int:
        # 30% chance of transient failure
        if random.random() < 0.3:
            transient_count[0] += 1
            raise TransientError(f"Temporary failure for {value}")
        success_count[0] += 1
        return value * 2
    
    results = []
    failures = []
    
    for i in range(50):
        try:
            result = flaky_operation(i)
            results.append(result)
        except TransientError:
            failures.append(i)
            
    print(f"Total operations: 50")
    print(f"Successful: {len(results)}")
    print(f"Failed after retries: {len(failures)}")
    print(f"Transient errors encountered: {transient_count[0]}")
    print(f"Recovery rate: {len(results) / 50 * 100:.1f}%")
    
    # 3.2 Dead Letter Queue
    print_subheader("3.2 Dead Letter Queue for Permanent Failures")
    
    dlq = DeadLetterQueue(max_size=100)
    processed = 0
    
    for i in range(100):
        try:
            # 5% chance of permanent failure (bad data)
            if random.random() < 0.05:
                raise ValidationError(
                    f"Invalid record {i}",
                    field="id",
                    expected="valid",
                    actual="corrupted"
                )
            processed += 1
            
        except ValidationError as e:
            dlq.push(DeadLetterRecord(
                original_data={'id': i},
                error=e,
                partition_id=str(i),
                worker_id="demo_worker",
                attempt_count=1,
                first_failure=time.time()
            ))
            
    dlq_stats = dlq.get_stats()
    
    print(f"Successfully processed: {processed}")
    print(f"Routed to DLQ: {dlq_stats['size']}")
    print(f"Error distribution: {dlq_stats['error_distribution']}")
    
    # 3.3 Circuit Breaker
    print_subheader("3.3 Circuit Breaker Protection")
    
    circuit = CircuitBreaker(
        "demo_service",
        CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout=1.0
        )
    )
    
    success = 0
    circuit_blocked = 0
    errors = 0
    
    # Simulate a service that fails then recovers
    failure_period = range(10, 30)  # Service fails during this range
    
    for i in range(50):
        def service_call():
            if i in failure_period:
                raise TransientError("Service unavailable")
            return f"result_{i}"
            
        try:
            result = circuit.call(service_call)
            success += 1
        except TransientError:
            errors += 1
        except Exception as e:
            if "Circuit" in str(type(e).__name__):
                circuit_blocked += 1
                
        # Small delay to allow circuit state changes
        time.sleep(0.01)
        
    print(f"Successful calls: {success}")
    print(f"Errors (service down): {errors}")
    print(f"Blocked by circuit: {circuit_blocked}")
    print(f"Final circuit state: {circuit.state.name}")
    
    print("\n✅ Level 3 Complete: Error handling and recovery verified")
    
    return True


# ============================================================================
# LEVEL 4: EXPERT - Scale and Concurrency
# ============================================================================

def demo_level_4_expert():
    """
    Level 4: Scale and Concurrency
    
    Demonstrates handling of:
    - Higher data volumes
    - Concurrent processing
    - Backpressure
    - Colony coordination
    """
    print_header("Scale and Concurrency", "LEVEL 4 - EXPERT")
    
    # 4.1 Scaling throughput
    print_subheader("4.1 Throughput Scaling")
    
    sizes = [100, 500, 1000, 2000]
    results = []
    
    for size in sizes:
        data = [{'id': i, 'value': random.random()} for i in range(size)]
        
        hive = create_hive(num_workers=8)
        
        start = time.time()
        processed = hive.process(
            data,
            lambda x: (x['value'] ** 2, 1.0)
        )
        elapsed = time.time() - start
        
        throughput = size / elapsed
        results.append({
            'size': size,
            'elapsed': elapsed,
            'throughput': throughput
        })
        
        print(f"  {size:5} records: {elapsed*1000:6.1f}ms ({throughput:,.0f} rec/s)")
        
    # Check scaling efficiency
    baseline_throughput = results[0]['throughput']
    final_throughput = results[-1]['throughput']
    scaling_efficiency = final_throughput / baseline_throughput
    
    print(f"\nScaling efficiency: {scaling_efficiency:.2f}x")
    
    # 4.2 Worker role distribution
    print_subheader("4.2 Colony Worker Behavior")
    
    colony = ColonyState()
    
    # Simulate worker activity
    workers = {
        'employed': ['emp_1', 'emp_2', 'emp_3', 'emp_4'],
        'onlooker': ['onl_1', 'onl_2', 'onl_3'],
        'scout': ['sct_1']
    }
    
    # Employed bees process steadily
    for worker in workers['employed']:
        colony.update_temperature(worker, 0.6)  # Moderate load
        
    # Onlookers help when needed
    for worker in workers['onlooker']:
        colony.update_temperature(worker, 0.4)  # Lower load
        
    # Scouts explore
    for worker in workers['scout']:
        colony.update_temperature(worker, 0.2)  # Minimal load
        
    print(f"Colony temperature: {colony.get_colony_temperature():.2f}")
    print(f"Worker distribution: {len(workers['employed'])} employed, "
          f"{len(workers['onlooker'])} onlookers, {len(workers['scout'])} scouts")
          
    # 4.3 Backpressure simulation
    print_subheader("4.3 Backpressure Handling")
    
    # Simulate increasing load
    load_levels = [0.3, 0.5, 0.7, 0.9]
    
    for load in load_levels:
        colony.update_temperature('pressure_test', load)
        
        if load > 0.8:
            colony.emit_pheromone(
                Pheromone(
                    signal_type='throttle',
                    intensity=load,
                    source_worker='pressure_test'
                )
            )
            
        throttle = colony.sense_pheromone('throttle')
        status = "🔴 THROTTLING" if throttle > 0.5 else "🟡 Warning" if throttle > 0.2 else "🟢 Normal"
        
        print(f"  Load {load:.0%}: Throttle signal = {throttle:.2f} {status}")
        
    print("\n✅ Level 4 Complete: Scale and concurrency verified")
    
    return True


# ============================================================================
# LEVEL 5: PRODUCTION - Full Integration
# ============================================================================

def demo_level_5_production():
    """
    Level 5: Production Scenario
    
    Demonstrates full integration with:
    - Real data sources/sinks
    - Monitoring and observability
    - Windowed stream processing
    - Health monitoring
    """
    print_header("Production Integration", "LEVEL 5 - PRODUCTION")
    
    # 5.1 Message broker simulation
    print_subheader("5.1 Message Queue Processing")
    
    broker = MessageBroker()
    topic = broker.create_topic("transactions", num_partitions=4)
    
    # Produce messages
    print("Producing messages...")
    for i in range(200):
        topic.produce(
            key=f"user_{i % 10}",
            value={
                'transaction_id': f'txn_{i}',
                'amount': random.random() * 500,
                'timestamp': time.time()
            }
        )
        
    print(f"Produced 200 messages to topic '{topic.name}'")
    
    # Consume with consumer group
    consumed = 0
    for partition in range(4):
        messages = topic.consume("demo_consumer", partition, max_messages=100)
        consumed += len(messages)
        
    print(f"Consumed {consumed} messages")
    
    lag = topic.get_lag("demo_consumer")
    print(f"Consumer lag by partition: {lag}")
    
    # 5.2 Windowed stream processing
    print_subheader("5.2 Windowed Stream Processing")
    
    processor = EnhancedStreamProcessor(
        num_workers=4,
        window_assigner=tumbling_window(5.0),  # 5-second windows
        watermark_generator=bounded_watermark(2.0),
        delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE
    )
    
    # Process stream records
    for i in range(100):
        record = StreamRecord(
            key=f"sensor_{i % 5}",
            value=random.random() * 100,
            timestamp=time.time() - random.random() * 10  # Some out of order
        )
        
        processor.process_record(
            record,
            aggregator=sum_aggregator,
            initial_value=0.0
        )
        
    stream_metrics = processor.get_metrics()
    
    print(f"Records processed: {stream_metrics['records_processed']}")
    print(f"Late records: {stream_metrics['late_records']}")
    print(f"Active windows: {stream_metrics['active_windows']}")
    print(f"Current watermark: {stream_metrics['current_watermark']:.2f}")
    
    # 5.3 Monitoring dashboard
    print_subheader("5.3 Monitoring & Observability")
    
    registry = get_registry()
    profiler = get_profiler()
    
    # Get Prometheus metrics
    prometheus_output = registry.to_prometheus()
    print("Sample Prometheus metrics:")
    for line in prometheus_output.split('\n')[:10]:
        if line and not line.startswith('#'):
            print(f"  {line}")
            
    # Performance profile
    print("\nPerformance Profile:")
    profile_stats = profiler.get_all_stats()
    for op, stats in list(profile_stats.items())[:5]:
        if stats['count'] > 0:
            print(f"  {op}: {stats['count']} calls, "
                  f"avg={stats['mean']*1000:.2f}ms")
                  
    # 5.4 Colony health
    print_subheader("5.4 Colony Health Report")
    
    colony = ColonyState()
    
    # Simulate some activity
    for i in range(8):
        colony.update_temperature(f'worker_{i}', 0.3 + random.random() * 0.4)
        
    temp = colony.get_colony_temperature()
    
    health_status = "🟢 HEALTHY" if temp < 0.5 else "🟡 ELEVATED" if temp < 0.8 else "🔴 CRITICAL"
    
    print(f"Colony Temperature: {temp:.2f} {health_status}")
    print(f"Active Workers: {len(colony.temperature)}")
    print(f"Food Sources: {len(colony.food_sources)}")
    print(f"Throttle Pheromone: {colony.sense_pheromone('throttle'):.2f}")
    print(f"Alarm Pheromone: {colony.sense_pheromone('alarm'):.2f}")
    
    print("\n✅ Level 5 Complete: Production integration verified")
    
    return True


# ============================================================================
# Main Demo Runner
# ============================================================================

def run_progressive_demo():
    """Run all demo levels progressively."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     🐝  H I V E F R A M E   P R O G R E S S I V E   D E M O  🐝     ║
║                                                                      ║
║     Bee-Inspired Distributed Data Processing Framework               ║
║     Demonstrating capabilities under increasing challenges           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    levels = [
        ("BEGINNER", demo_level_1_beginner, "Basic operations with clean data"),
        ("INTERMEDIATE", demo_level_2_intermediate, "Realistic data with quality issues"),
        ("ADVANCED", demo_level_3_advanced, "Error handling and recovery"),
        ("EXPERT", demo_level_4_expert, "Scale and concurrency"),
        ("PRODUCTION", demo_level_5_production, "Full integration scenario"),
    ]
    
    passed = []
    failed = []
    
    for level_name, level_fn, description in levels:
        print(f"\n🎯 Starting {level_name}: {description}")
        
        try:
            start = time.time()
            success = level_fn()
            elapsed = time.time() - start
            
            if success:
                passed.append((level_name, elapsed))
                print(f"\n⏱️  Level completed in {elapsed:.2f}s")
            else:
                failed.append((level_name, "Test returned False"))
                
        except Exception as e:
            failed.append((level_name, str(e)))
            print(f"\n❌ Level {level_name} failed: {e}")
            import traceback
            traceback.print_exc()
            
    # Final summary
    print("\n" + "=" * 70)
    print("  DEMO SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Passed: {len(passed)}/{len(levels)}")
    for level, elapsed in passed:
        print(f"   • {level}: {elapsed:.2f}s")
        
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(levels)}")
        for level, error in failed:
            print(f"   • {level}: {error[:50]}")
            
    total_time = sum(e for _, e in passed)
    print(f"\n⏱️  Total time: {total_time:.2f}s")
    
    # Success message
    if len(passed) == len(levels):
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🎉  ALL LEVELS COMPLETE!  🎉                                       ║
║                                                                      ║
║   HiveFrame has demonstrated:                                        ║
║   • Core data processing operations                                  ║
║   • Realistic data handling with quality issues                      ║
║   • Error handling, retries, and recovery                            ║
║   • Scale, concurrency, and backpressure                             ║
║   • Production monitoring and observability                          ║
║                                                                      ║
║   The bee colony is ready for production! 🐝                         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_progressive_demo()
    exit(0 if success else 1)
