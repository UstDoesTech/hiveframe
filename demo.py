#!/usr/bin/env python3
"""
HiveFrame Demo & Benchmarks
===========================

Demonstrates the bee-inspired data processing framework
and compares performance to traditional approaches.

Run: python demo.py
"""

import sys
import os
import time
import random
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hiveframe import (
    HiveFrame, HiveDataFrame, HiveStream,
    col, lit, sum_agg, avg, count, max_agg, min_agg,
    create_hive, BeeRole
)


def generate_sample_data(n: int = 10000) -> List[Dict[str, Any]]:
    """Generate sample transaction data."""
    categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home', 'Sports']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    return [
        {
            'id': i,
            'amount': round(random.uniform(10, 1000), 2),
            'category': random.choice(categories),
            'region': random.choice(regions),
            'quantity': random.randint(1, 20),
            'customer_id': f"CUST_{random.randint(1, 1000):04d}",
            'timestamp': time.time() - random.randint(0, 86400 * 30)
        }
        for i in range(n)
    ]


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n--- {title} ---\n")


# =============================================================================
# DEMO 1: Core HiveFrame API (RDD-style)
# =============================================================================

def demo_core_api():
    """Demonstrate the core HiveFrame RDD-like API."""
    print_header("DEMO 1: Core HiveFrame API (RDD-style)")
    
    # Create a hive with 8 workers
    hive = create_hive(num_workers=8)
    print(f"Created HiveFrame with 8 workers")
    print(f"  - Employed bees: 50% (exploit assigned partitions)")
    print(f"  - Onlooker bees: 40% (reinforce quality solutions)")
    print(f"  - Scout bees: 10% (explore new territory)")
    
    # Sample data
    data = list(range(100))
    
    print_subheader("Map Operation")
    print("Input: [0, 1, 2, ..., 99]")
    print("Transform: x -> x * 2")
    
    start = time.time()
    results = hive.map(data, lambda x: x * 2)
    elapsed = time.time() - start
    
    print(f"Output (first 10): {results[:10]}")
    print(f"Time: {elapsed:.4f}s")
    
    print_subheader("Filter Operation")
    print("Filter: x > 150")
    
    start = time.time()
    filtered = hive.filter(results, lambda x: x > 150)
    elapsed = time.time() - start
    
    print(f"Output (first 10): {filtered[:10]}")
    print(f"Count: {len(filtered)} (expected: 24)")
    print(f"Time: {elapsed:.4f}s")
    
    print_subheader("Reduce Operation")
    print("Reduce: sum all values")
    
    start = time.time()
    total = hive.reduce(filtered, lambda a, b: a + b)
    elapsed = time.time() - start
    
    print(f"Sum: {total}")
    print(f"Time: {elapsed:.4f}s")


# =============================================================================
# DEMO 2: DataFrame API
# =============================================================================

def demo_dataframe_api():
    """Demonstrate the Spark-like DataFrame API."""
    print_header("DEMO 2: DataFrame API")
    
    # Generate sample data
    print("Generating 5000 transaction records...")
    data = generate_sample_data(5000)
    
    # Create DataFrame
    hive = create_hive(num_workers=8)
    df = HiveDataFrame.from_records(data, hive=hive)
    
    print_subheader("Schema")
    df.printSchema()
    
    print_subheader("Sample Data")
    df.limit(5).show()
    
    print_subheader("Filter & Select")
    print("Query: Electronics transactions > $500")
    
    start = time.time()
    result = (df
              .filter((col('category') == 'Electronics') & (col('amount') > 500))
              .select('id', 'amount', 'region', 'customer_id'))
    elapsed = time.time() - start
    
    print(f"Found {result.count()} matching transactions")
    result.limit(5).show()
    print(f"Time: {elapsed:.4f}s")
    
    print_subheader("Aggregation by Category")
    
    start = time.time()
    summary = (df
               .groupBy('category')
               .agg(
                   count(col('id')),
                   sum_agg(col('amount')),
                   avg(col('amount')),
                   max_agg(col('amount'))
               ))
    elapsed = time.time() - start
    
    summary.show()
    print(f"Time: {elapsed:.4f}s")
    
    print_subheader("Multi-level Grouping")
    print("Query: Sum by category and region")
    
    start = time.time()
    regional = (df
                .groupBy('category', 'region')
                .agg(
                    count(col('id')),
                    sum_agg(col('amount'))
                )
                .orderBy('category'))
    elapsed = time.time() - start
    
    regional.limit(10).show()
    print(f"Time: {elapsed:.4f}s")
    
    print_subheader("Computed Columns")
    print("Adding total_value = amount * quantity")
    
    start = time.time()
    enriched = (df
                .withColumn('total_value', col('amount') * col('quantity'))
                .select('id', 'category', 'amount', 'quantity', 'total_value')
                .orderBy('total_value', ascending=False))
    elapsed = time.time() - start
    
    enriched.limit(5).show()
    print(f"Time: {elapsed:.4f}s")
    
    print_subheader("Statistics")
    df.select('amount', 'quantity').describe().show()


# =============================================================================
# DEMO 3: Stream Processing
# =============================================================================

def demo_streaming():
    """Demonstrate the bee-inspired streaming API."""
    print_header("DEMO 3: Stream Processing")
    
    print("Creating HiveStream with 6 workers...")
    stream = HiveStream(
        num_workers=6,
        buffer_size=1000,
        employed_ratio=0.5,
        onlooker_ratio=0.3,
        scout_ratio=0.2
    )
    
    # Processing function
    def process_record(value: Dict) -> Dict:
        """Simulate processing with variable latency."""
        time.sleep(random.uniform(0.001, 0.01))
        return {
            'processed_id': value['id'],
            'doubled_amount': value['amount'] * 2,
            'processed_at': time.time()
        }
    
    print("\nStarting stream processing...")
    stream.start(process_record)
    
    # Submit records
    print("Submitting 500 records...")
    submitted = 0
    start = time.time()
    
    for i in range(500):
        record = {
            'id': i,
            'amount': random.uniform(10, 100),
            'category': random.choice(['A', 'B', 'C'])
        }
        if stream.submit(f"key_{i}", record):
            submitted += 1
            
    submit_time = time.time() - start
    print(f"Submitted {submitted} records in {submit_time:.3f}s")
    
    # Collect results
    print("\nCollecting results...")
    results = []
    collect_start = time.time()
    timeout = 10.0
    
    while time.time() - collect_start < timeout:
        result = stream.get_result(timeout=0.5)
        if result:
            results.append(result)
        if len(results) >= submitted:
            break
            
    collect_time = time.time() - collect_start
    
    print(f"Collected {len(results)} results in {collect_time:.3f}s")
    
    # Show metrics
    print_subheader("Stream Metrics")
    metrics = stream.get_metrics()
    print(f"Buffer fill level: {metrics['buffer_fill']:.1%}")
    print(f"Colony temperature: {metrics['colony_temperature']:.3f}")
    print(f"Throttle level: {metrics['throttle_level']:.3f}")
    print(f"Alarm level: {metrics['alarm_level']:.3f}")
    
    print("\nPartition Health:")
    for partition, health in metrics['partition_health'].items():
        bar = "█" * int(health * 20) + "░" * (20 - int(health * 20))
        print(f"  Partition {partition}: [{bar}] {health:.3f}")
        
    print("\nWorker Stats:")
    for worker_id, stats in sorted(metrics['worker_stats'].items()):
        print(f"  {worker_id}: processed={stats['processed']}, errors={stats['errors']}")
        
    stream.stop()
    print("\nStream stopped.")


# =============================================================================
# DEMO 4: Benchmark Comparison
# =============================================================================

def traditional_map(data: List[Any], fn, num_workers: int = 8) -> List[Any]:
    """Traditional thread pool map (like Spark's executor model)."""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(fn, data))


def traditional_filter(data: List[Any], predicate, num_workers: int = 8) -> List[Any]:
    """Traditional thread pool filter."""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(lambda x: x if predicate(x) else None, data))
    return [r for r in results if r is not None]


def benchmark_comparison():
    """Compare HiveFrame to traditional approaches."""
    print_header("DEMO 4: Benchmark Comparison")
    
    sizes = [1000, 5000, 10000]
    num_workers = 8
    
    hive = create_hive(num_workers=num_workers)
    
    results_table = []
    
    for size in sizes:
        print(f"\n{'='*50}")
        print(f"Data size: {size:,} records")
        print('='*50)
        
        # Generate data
        data = list(range(size))
        
        # CPU-bound transformation
        def heavy_transform(x):
            # Simulate CPU work
            result = x
            for _ in range(100):
                result = (result * 17 + 31) % 10000
            return result
        
        print_subheader("MAP Operation (CPU-bound)")
        
        # Traditional
        start = time.time()
        trad_result = traditional_map(data, heavy_transform, num_workers)
        trad_time = time.time() - start
        print(f"  Traditional ThreadPool: {trad_time:.4f}s")
        
        # HiveFrame
        start = time.time()
        hive_result = hive.map(data, heavy_transform)
        hive_time = time.time() - start
        print(f"  HiveFrame:              {hive_time:.4f}s")
        
        speedup = trad_time / hive_time if hive_time > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")
        
        results_table.append({
            'size': size,
            'operation': 'map',
            'traditional': trad_time,
            'hiveframe': hive_time,
            'speedup': speedup
        })
        
        print_subheader("FILTER Operation")
        
        predicate = lambda x: x % 3 == 0
        
        # Traditional
        start = time.time()
        trad_filtered = traditional_filter(data, predicate, num_workers)
        trad_time = time.time() - start
        print(f"  Traditional ThreadPool: {trad_time:.4f}s ({len(trad_filtered)} results)")
        
        # HiveFrame
        start = time.time()
        hive_filtered = hive.filter(data, predicate)
        hive_time = time.time() - start
        print(f"  HiveFrame:              {hive_time:.4f}s ({len(hive_filtered)} results)")
        
        speedup = trad_time / hive_time if hive_time > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")
        
        results_table.append({
            'size': size,
            'operation': 'filter',
            'traditional': trad_time,
            'hiveframe': hive_time,
            'speedup': speedup
        })
    
    # Summary
    print_header("Benchmark Summary")
    
    print(f"{'Size':>10} | {'Operation':>10} | {'Traditional':>12} | {'HiveFrame':>12} | {'Speedup':>8}")
    print("-" * 60)
    
    for r in results_table:
        print(f"{r['size']:>10,} | {r['operation']:>10} | {r['traditional']:>12.4f}s | {r['hiveframe']:>12.4f}s | {r['speedup']:>7.2f}x")
    
    print("\nNote: HiveFrame's advantage increases with:")
    print("  - Variable workload (adaptive task allocation)")
    print("  - Uneven partition quality (quality-weighted selection)")
    print("  - Need for fault tolerance (abandonment mechanism)")


# =============================================================================
# DEMO 5: Observability & Colony Behavior
# =============================================================================

def demo_colony_behavior():
    """Demonstrate the bee colony behavior in action."""
    print_header("DEMO 5: Colony Behavior Visualization")
    
    from hiveframe.core import ColonyState, Bee, BeeRole, DanceFloor
    
    # Create a colony
    colony = ColonyState(abandonment_limit=5)
    dance_floor = colony.dance_floor
    
    print("Creating colony with food sources (data partitions)...")
    
    # Add food sources with varying quality
    for i in range(10):
        quality = random.uniform(0.3, 1.0)
        colony.add_food_source(f"partition_{i}", {'quality': quality, 'data': list(range(100))})
        print(f"  Partition {i}: initial fitness = {quality:.3f}")
    
    # Simulate foraging cycles
    print("\nSimulating 5 foraging cycles...")
    
    def process_fn(data):
        time.sleep(random.uniform(0.01, 0.05))
        quality = data.get('quality', 0.5)
        return {'processed': True}, quality + random.uniform(-0.1, 0.1)
    
    # Create bees
    bees = []
    for i in range(4):
        bees.append(Bee(f"employed_{i}", BeeRole.EMPLOYED, colony, process_fn))
    for i in range(3):
        bees.append(Bee(f"onlooker_{i}", BeeRole.ONLOOKER, colony, process_fn))
    for i in range(1):
        bees.append(Bee(f"scout_{i}", BeeRole.SCOUT, colony, process_fn))
    
    print(f"\nColony composition: {len(bees)} bees")
    print(f"  - Employed: 4 (exploit)")
    print(f"  - Onlooker: 3 (reinforce)")
    print(f"  - Scout: 1 (explore)")
    
    for cycle in range(5):
        print(f"\n--- Cycle {cycle + 1} ---")
        
        # Each bee forages
        dances = []
        for bee in bees:
            dance = bee.forage()
            if dance:
                dances.append(dance)
        
        # Report dances
        print(f"  Waggle dances performed: {len(dances)}")
        if dances:
            avg_quality = sum(d.quality_score for d in dances) / len(dances)
            avg_vigor = sum(d.vigor for d in dances) / len(dances)
            print(f"  Average quality score: {avg_quality:.3f}")
            print(f"  Average dance vigor: {avg_vigor:.3f}")
        
        # Show fitness evolution
        print("  Partition fitness:")
        for pid, source in sorted(colony.food_sources.items()):
            bar = "█" * int(source.fitness * 20) + "░" * (20 - int(source.fitness * 20))
            abandoned = " (ABANDONED)" if source.trials >= 5 else ""
            print(f"    {pid}: [{bar}] {source.fitness:.3f}{abandoned}")
        
        # Colony temperature
        temp = colony.get_colony_temperature()
        print(f"  Colony temperature: {temp:.3f}")
    
    print("\n" + "=" * 50)
    print("Key Observations:")
    print("=" * 50)
    print("- Higher-fitness partitions attract more processing")
    print("- Waggle dances spread information about quality")
    print("- Abandoned partitions (low fitness) get re-explored")
    print("- Colony self-organizes without central coordination")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("         🐝 HIVEFRAME: BEE-INSPIRED DATA PROCESSING 🐝")
    print("=" * 60)
    print("\nBiomimicry concepts in action:")
    print("  • Waggle Dance Protocol - quality-based task advertisement")
    print("  • Three-Tier Workers - employed, onlooker, scout bees")
    print("  • Stigmergic Coordination - indirect communication")
    print("  • Adaptive Load Balancing - self-organizing workers")
    print("  • ABC Algorithm - abandonment-based exploration")
    
    demos = [
        ("Core API", demo_core_api),
        ("DataFrame API", demo_dataframe_api),
        ("Streaming", demo_streaming),
        ("Benchmarks", benchmark_comparison),
        ("Colony Behavior", demo_colony_behavior),
    ]
    
    for name, demo_fn in demos:
        try:
            demo_fn()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("                    DEMOS COMPLETE")
    print("=" * 60)
    print("\nHiveFrame demonstrates how bee colony intelligence")
    print("patterns can replace traditional centralized processing")
    print("with decentralized, self-organizing, adaptive systems.")
    print("\nKey advantages over Spark-style processing:")
    print("  ✓ No single-point-of-failure driver")
    print("  ✓ Quality-weighted work distribution")
    print("  ✓ Self-healing through abandonment mechanism")
    print("  ✓ Adaptive backpressure via pheromone signals")
    print("  ✓ Emergent load balancing without central scheduler")


if __name__ == '__main__':
    main()
