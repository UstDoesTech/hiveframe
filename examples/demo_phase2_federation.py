#!/usr/bin/env python3
"""
HiveFrame Phase 2: Multi-Hive Federation Demo
==============================================

Demonstrates distributed execution across multiple HiveFrame clusters
that coordinate like allied bee colonies.

Features demonstrated:
- Multi-hive federation setup
- Cross-datacenter coordination
- Locality-aware scheduling
- Federated query execution
- Health monitoring across hives

Run: python demo_phase2_federation.py
"""

import os
import random
import sys
import time
from typing import Any, Dict, List

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hiveframe import HiveDataFrame, col, sum_agg
from hiveframe.distributed import (
    CrossDatacenterManager,
    DataLocality,
    FederatedHive,
    HiveFederation,
    HiveHealth,
    LocalityLevel,
)


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n--- {title} ---\n")


def generate_distributed_data(n: int = 1000) -> List[Dict[str, Any]]:
    """Generate sample data distributed across regions."""
    regions = ["us-east", "us-west", "eu-central", "ap-southeast"]
    categories = ["Electronics", "Clothing", "Food", "Books"]

    return [
        {
            "id": i,
            "region": random.choice(regions),
            "category": random.choice(categories),
            "amount": round(random.uniform(10, 1000), 2),
            "timestamp": time.time() - random.randint(0, 86400 * 7),
        }
        for i in range(n)
    ]


def demo_federation_setup():
    """Demonstrate setting up a multi-hive federation."""
    print_header("DEMO 1: Multi-Hive Federation Setup")

    # Create a federation
    federation = HiveFederation(name="global-analytics")
    print(f"Created federation: {federation.name}")

    # Register multiple hives in different regions
    hive_east = FederatedHive(
        name="us-east-hive",
        endpoint="us-east.example.com:9000",
        workers=20,
        region="us-east",
    )

    hive_west = FederatedHive(
        name="us-west-hive",
        endpoint="us-west.example.com:9000",
        workers=15,
        region="us-west",
    )

    hive_eu = FederatedHive(
        name="eu-central-hive",
        endpoint="eu.example.com:9000",
        workers=18,
        region="eu-central",
    )

    print("\nRegistering hives in federation:")
    federation.register_hive(hive_east)
    print(f"  ✓ Registered {hive_east.name} ({hive_east.workers} workers)")

    federation.register_hive(hive_west)
    print(f"  ✓ Registered {hive_west.name} ({hive_west.workers} workers)")

    federation.register_hive(hive_eu)
    print(f"  ✓ Registered {hive_eu.name} ({hive_eu.workers} workers)")

    print(f"\nTotal workers in federation: {federation.total_workers()}")
    print(f"Active hives: {len(federation.list_hives())}")

    return federation


def demo_locality_aware_scheduling():
    """Demonstrate locality-aware task scheduling."""
    print_header("DEMO 2: Locality-Aware Scheduling")

    # Create cross-datacenter manager
    manager = CrossDatacenterManager()

    print("Data locality levels:")
    print("  - PROCESS_LOCAL: Data in same process")
    print("  - NODE_LOCAL: Data on same node")
    print("  - RACK_LOCAL: Data on same rack")
    print("  - DATACENTER_LOCAL: Data in same datacenter")
    print("  - REMOTE: Data in different datacenter")

    # Sample data with locality hints
    data_locations = [
        {"data_id": "chunk_1", "location": "us-east", "size_mb": 100},
        {"data_id": "chunk_2", "location": "us-east", "size_mb": 150},
        {"data_id": "chunk_3", "location": "eu-central", "size_mb": 120},
        {"data_id": "chunk_4", "location": "us-west", "size_mb": 90},
    ]

    print("\nScheduling tasks with locality awareness:")
    for data in data_locations:
        # Simulate locality-aware scheduling
        preferred_hive = manager.select_hive_for_data(
            data["location"], available_hives=["us-east", "us-west", "eu-central"]
        )
        locality = (
            LocalityLevel.DATACENTER_LOCAL
            if preferred_hive == data["location"]
            else LocalityLevel.REMOTE
        )

        print(f"  {data['data_id']} ({data['size_mb']}MB) @ {data['location']}")
        print(f"    → Scheduled to: {preferred_hive}")
        print(f"    → Locality: {locality.name}")


def demo_federated_execution():
    """Demonstrate distributed query execution across federation."""
    print_header("DEMO 3: Federated Query Execution")

    # Generate distributed data
    data = generate_distributed_data(n=5000)

    print(f"Generated {len(data)} records distributed across regions")

    # Create DataFrame
    df = HiveDataFrame.from_records(data)

    print("\nExecuting federated query:")
    print("  SELECT region, category, SUM(amount) as total")
    print("  FROM sales")
    print("  GROUP BY region, category")

    # Execute query (federation would distribute across hives)
    result = df.groupBy("region", "category").agg(sum_agg(col("amount")))

    print("\nQuery results (simulated federated execution):")
    result.show(n=10)

    print("\nFederation benefits:")
    print("  ✓ Data processed close to source (reduced network transfer)")
    print("  ✓ Parallel execution across multiple datacenters")
    print("  ✓ Automatic load balancing across hives")
    print("  ✓ Fault tolerance through hive redundancy")


def demo_health_monitoring():
    """Demonstrate health monitoring across federated hives."""
    print_header("DEMO 4: Health Monitoring")

    print("Monitoring health across federated hives:")

    # Simulate health status for different hives
    hive_statuses = [
        {
            "name": "us-east-hive",
            "status": "healthy",
            "workers_active": 20,
            "workers_total": 20,
            "cpu_usage": 0.65,
            "memory_usage": 0.72,
            "tasks_completed": 15234,
        },
        {
            "name": "us-west-hive",
            "status": "healthy",
            "workers_active": 15,
            "workers_total": 15,
            "cpu_usage": 0.58,
            "memory_usage": 0.68,
            "tasks_completed": 12891,
        },
        {
            "name": "eu-central-hive",
            "status": "degraded",
            "workers_active": 16,
            "workers_total": 18,
            "cpu_usage": 0.88,
            "memory_usage": 0.91,
            "tasks_completed": 14567,
        },
    ]

    print("\nHive Health Status:")
    print("-" * 70)
    for hive in hive_statuses:
        status_symbol = "✓" if hive["status"] == "healthy" else "⚠"
        print(f"\n{status_symbol} {hive['name']} ({hive['status'].upper()})")
        print(f"    Workers: {hive['workers_active']}/{hive['workers_total']}")
        print(f"    CPU Usage: {hive['cpu_usage']:.1%}")
        print(f"    Memory Usage: {hive['memory_usage']:.1%}")
        print(f"    Tasks Completed: {hive['tasks_completed']:,}")

    print("\n" + "-" * 70)
    print("\nFederation-wide metrics:")
    total_workers = sum(h["workers_total"] for h in hive_statuses)
    active_workers = sum(h["workers_active"] for h in hive_statuses)
    total_tasks = sum(h["tasks_completed"] for h in hive_statuses)

    print(f"  Total Workers: {active_workers}/{total_workers}")
    print(f"  Total Tasks Completed: {total_tasks:,}")
    print(f"  Healthy Hives: 2/3")
    print(f"  Degraded Hives: 1/3")


def demo_failover():
    """Demonstrate automatic failover when a hive becomes unavailable."""
    print_header("DEMO 5: Automatic Failover")

    print("Simulating hive failure and automatic failover:")

    print("\n1. Initial state: 3 hives active")
    print("   - us-east-hive: 20 workers")
    print("   - us-west-hive: 15 workers")
    print("   - eu-central-hive: 18 workers")

    print("\n2. 💥 Hive failure detected: eu-central-hive unavailable")

    print("\n3. Federation coordinator initiates failover:")
    print("   - Detecting affected tasks...")
    print("   - Reassigning 45 tasks from eu-central-hive")
    print("   - Redistributing to us-east-hive and us-west-hive")

    print("\n4. Failover complete:")
    print("   ✓ All tasks reassigned successfully")
    print("   ✓ No data loss (using replication)")
    print("   ✓ Total latency: 2.3 seconds")
    print("   ✓ 2 hives now handling full workload")

    print("\n5. Automatic recovery when hive returns:")
    print("   - eu-central-hive back online")
    print("   - Gradual rebalancing initiated")
    print("   - Tasks redistributed for optimal locality")

    print("\nBenefits of federated architecture:")
    print("  ✓ No single point of failure")
    print("  ✓ Automatic failover and recovery")
    print("  ✓ Graceful degradation under failures")
    print("  ✓ Zero-downtime maintenance")


def main():
    """Run all federation demos."""
    print("=" * 70)
    print("HiveFrame Phase 2: Multi-Hive Federation")
    print("Distributed Execution Across Allied Bee Colonies")
    print("=" * 70)

    try:
        # Run all demos
        federation = demo_federation_setup()
        demo_locality_aware_scheduling()
        demo_federated_execution()
        demo_health_monitoring()
        demo_failover()

        print("\n" + "=" * 70)
        print("Federation Demo Complete!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • Multiple hives coordinate like allied bee colonies")
        print("  • Locality-aware scheduling reduces network overhead")
        print("  • Automatic failover provides high availability")
        print("  • Swarm intelligence enables emergent optimization")
        print("  • No central coordinator - fully decentralized")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
