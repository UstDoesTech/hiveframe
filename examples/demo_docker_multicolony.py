#!/usr/bin/env python3
"""
HiveFrame Docker Multi-Colony Demo
===================================

Demonstrates running HiveFrame in Docker containers with multi-colony coordination.
This script is designed to be run inside Docker containers.

Run with docker-compose:
    docker-compose exec colony-us-east python3 /app/examples/demo_docker_multicolony.py
"""

import os
import sys
import time
from typing import List

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hiveframe import HiveDataFrame, col, create_hive


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def get_colony_info() -> dict:
    """Get colony information from environment variables."""
    return {
        "name": os.getenv("HIVE_COLONY_NAME", "unknown"),
        "region": os.getenv("HIVE_COLONY_REGION", "unknown"),
        "workers": int(os.getenv("HIVE_NUM_WORKERS", "8")),
        "hostname": os.getenv("HOSTNAME", "unknown"),
    }


def demo_basic_processing():
    """Demonstrate basic data processing in a Docker container."""
    print_header("Basic HiveFrame Processing in Docker")

    colony_info = get_colony_info()
    print(f"Colony Name: {colony_info['name']}")
    print(f"Region: {colony_info['region']}")
    print(f"Workers: {colony_info['workers']}")
    print(f"Hostname: {colony_info['hostname']}")
    print()

    # Create hive
    print("Creating hive...")
    hive = create_hive(num_workers=colony_info["workers"])
    print(f"✓ Hive created with {colony_info['workers']} workers")
    print()

    # Simple map operation
    print("Running map operation...")
    data = list(range(100))
    start = time.time()
    results = hive.map(data, lambda x: x * 2)
    elapsed = time.time() - start
    print(f"✓ Processed {len(data)} items in {elapsed:.3f}s")
    print(f"  Sample results: {results[:5]}...")
    print()


def demo_dataframe_api():
    """Demonstrate DataFrame API in Docker."""
    print_header("DataFrame API Demo")

    # Create sample data
    data = [
        {"id": i, "value": i * 2, "region": f"region-{i % 3}"}
        for i in range(100)
    ]

    # Create DataFrame
    df = HiveDataFrame.from_records(data)
    print("Created DataFrame with 100 rows")
    print()

    # Filter and show
    print("Filtering rows where value > 50...")
    filtered = df.filter(col("value") > 50)
    print(f"✓ Found {filtered.count()} matching rows")
    print("\nFirst 5 rows:")
    filtered.limit(5).show()


def demo_distributed_workload():
    """Simulate a distributed workload across colonies."""
    print_header("Distributed Workload Simulation")

    colony_info = get_colony_info()

    # Create hive
    hive = create_hive(num_workers=colony_info["workers"])

    # Simulate processing regional data
    print(f"Processing data for region: {colony_info['region']}")
    print()

    # Generate region-specific data
    num_items = 1000
    start = time.time()

    def process_item(x):
        """Simulate some processing work."""
        result = x
        for _ in range(100):
            result = (result * 1.5 + 1) % 1000
        return result

    results = hive.map(list(range(num_items)), process_item)
    elapsed = time.time() - start

    print(f"✓ Processed {num_items} items in {elapsed:.3f}s")
    print(f"  Throughput: {num_items / elapsed:.0f} items/sec")
    print(f"  Average time per item: {elapsed / num_items * 1000:.2f}ms")
    print()


def main():
    """Run all demos."""
    print("\n" + "🐝" * 35)
    print("  HiveFrame Docker Multi-Colony Demo")
    print("🐝" * 35)

    try:
        demo_basic_processing()
        demo_dataframe_api()
        demo_distributed_workload()

        print_header("Demo Complete!")
        print("✓ All demos completed successfully")
        print("\nThis container is ready to process distributed workloads.")
        print("Each colony can independently process data while coordinating")
        print("with other colonies in the federation.")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
