#!/usr/bin/env python3
"""
HiveFrame Storage Demo
======================

Demonstrates HiveFrame's storage layer capabilities including
Parquet file support and Delta Lake integration.

Features demonstrated:
- Parquet read/write operations
- Delta Lake tables with ACID transactions
- Time travel (version history)
- Schema evolution

Run: python demo_storage.py
"""

import sys
import os
import time
import random
import tempfile
import shutil
from typing import List, Dict, Any
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hiveframe import HiveDataFrame, col
from hiveframe.storage import (
    read_parquet,
    write_parquet,
    DeltaTable,
    read_delta,
    write_delta,
    StorageOptions,
)


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n--- {title} ---\n")


# =============================================================================
# Sample Data Generation
# =============================================================================


def generate_transactions(n: int = 1000) -> List[Dict[str, Any]]:
    """Generate sample transaction data."""
    categories = ["Electronics", "Clothing", "Food", "Books", "Home", "Sports"]
    regions = ["North", "South", "East", "West"]

    return [
        {
            "transaction_id": f"TXN_{i:06d}",
            "customer_id": f"CUST_{random.randint(1, 100):04d}",
            "amount": round(random.uniform(10, 500), 2),
            "category": random.choice(categories),
            "region": random.choice(regions),
            "timestamp": datetime(2025, random.randint(1, 12), random.randint(1, 28)).isoformat(),
            "is_refunded": random.random() < 0.05,
        }
        for i in range(n)
    ]


def generate_events(n: int = 500) -> List[Dict[str, Any]]:
    """Generate sample event data."""
    event_types = ["click", "view", "purchase", "signup", "logout"]

    return [
        {
            "event_id": i,
            "user_id": random.randint(1, 50),
            "event_type": random.choice(event_types),
            "page": f"/page/{random.randint(1, 20)}",
            "duration_ms": random.randint(100, 10000),
            "event_date": f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        }
        for i in range(n)
    ]


# =============================================================================
# Demo Functions
# =============================================================================


def demo_parquet_operations():
    """Demonstrate Parquet file operations."""
    print_header("Parquet File Operations")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="hiveframe_parquet_")

    try:
        # Generate sample data
        transactions = generate_transactions(1000)
        df = HiveDataFrame(transactions)

        print(f"Created DataFrame with {df.count()} rows")
        df.limit(3).show()

        # Write to Parquet
        print_subheader("1. Writing Parquet Files")

        parquet_path = os.path.join(temp_dir, "transactions.parquet")

        start = time.time()
        write_parquet(df, parquet_path)
        write_time = time.time() - start

        print(f"Wrote {df.count()} records to: {parquet_path}")
        print(f"Write time: {write_time*1000:.2f}ms")
        print(f"File size: {os.path.getsize(parquet_path) / 1024:.2f} KB")

        # Read from Parquet
        print_subheader("2. Reading Parquet Files")

        start = time.time()
        df_read = read_parquet(parquet_path)
        read_time = time.time() - start

        print(f"Read {df_read.count()} records")
        print(f"Read time: {read_time*1000:.2f}ms")
        df_read.limit(3).show()

        # Write multiple files
        print_subheader("3. Multiple Parquet Files")

        for i in range(3):
            chunk = generate_transactions(500)
            chunk_df = HiveDataFrame(chunk)
            chunk_path = os.path.join(temp_dir, f"chunk_{i}.parquet")
            write_parquet(chunk_df, chunk_path)
            size_kb = os.path.getsize(chunk_path) / 1024
            print(f"  Chunk {i}: {size_kb:.2f} KB")

    finally:
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory")


def demo_delta_table_basics():
    """Demonstrate basic Delta Lake operations."""
    print_header("Delta Lake Basics")

    temp_dir = tempfile.mkdtemp(prefix="hiveframe_delta_")

    try:
        delta_path = os.path.join(temp_dir, "events_delta")

        # Create initial data
        events = generate_events(200)
        df = HiveDataFrame(events)

        print(f"Initial data: {df.count()} events")

        # Write to Delta
        print_subheader("1. Creating Delta Table")

        start = time.time()
        write_delta(df, delta_path, mode="overwrite")
        elapsed = time.time() - start

        print(f"Created Delta table at: {delta_path}")
        print(f"Write time: {elapsed*1000:.2f}ms")

        # Read from Delta
        print_subheader("2. Reading Delta Table")

        start = time.time()
        df_read = read_delta(delta_path)
        elapsed = time.time() - start

        print(f"Read {df_read.count()} records")
        print(f"Read time: {elapsed*1000:.2f}ms")
        df_read.limit(3).show()

        # Append more data
        print_subheader("3. Appending Data")

        new_events = generate_events(50)
        new_df = HiveDataFrame(new_events)

        start = time.time()
        write_delta(new_df, delta_path, mode="append")
        elapsed = time.time() - start

        df_updated = read_delta(delta_path)
        print(f"Appended 50 events")
        print(f"Total records now: {df_updated.count()}")
        print(f"Append time: {elapsed*1000:.2f}ms")

    finally:
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory")


def demo_delta_crud():
    """Demonstrate Delta Lake with DeltaTable class."""
    print_header("Delta Lake Table Operations")

    temp_dir = tempfile.mkdtemp(prefix="hiveframe_delta_crud_")

    try:
        delta_path = os.path.join(temp_dir, "customers")

        # Initial data
        customers = [
            {"id": 1, "name": "Alice", "status": "active", "balance": 1000},
            {"id": 2, "name": "Bob", "status": "active", "balance": 500},
            {"id": 3, "name": "Carol", "status": "active", "balance": 750},
            {"id": 4, "name": "David", "status": "inactive", "balance": 200},
            {"id": 5, "name": "Eve", "status": "active", "balance": 1500},
        ]

        df = HiveDataFrame(customers)

        # Create table using DeltaTable class
        delta_table = DeltaTable(delta_path)
        delta_table.write(df, mode="overwrite")

        print("Initial table:")
        delta_table.to_dataframe().show()

        # Append more data
        print_subheader("1. Appending Data")

        new_customers = [
            {"id": 6, "name": "Frank", "status": "active", "balance": 800},
            {"id": 7, "name": "Grace", "status": "active", "balance": 1200},
        ]

        delta_table.write(HiveDataFrame(new_customers), mode="append")

        print("After append:")
        delta_table.to_dataframe().show()

        # Overwrite all data
        print_subheader("2. Overwriting Data")

        fresh_data = [
            {"id": 100, "name": "NewUser1", "status": "active", "balance": 999},
            {"id": 101, "name": "NewUser2", "status": "pending", "balance": 500},
        ]

        delta_table.write(HiveDataFrame(fresh_data), mode="overwrite")

        print("After overwrite:")
        delta_table.to_dataframe().show()

    finally:
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory")


def demo_time_travel():
    """Demonstrate Delta Lake time travel capabilities."""
    print_header("Delta Lake Time Travel")

    temp_dir = tempfile.mkdtemp(prefix="hiveframe_delta_tt_")

    try:
        delta_path = os.path.join(temp_dir, "inventory")
        delta_table = DeltaTable(delta_path)

        # Version 0: Initial load
        print_subheader("Version 0: Initial Load")

        inventory_v0 = [
            {"product": "Widget A", "quantity": 100, "price": 10.00},
            {"product": "Widget B", "quantity": 50, "price": 25.00},
            {"product": "Widget C", "quantity": 200, "price": 5.00},
        ]

        delta_table.write(HiveDataFrame(inventory_v0), mode="overwrite")
        print("Initial inventory:")
        delta_table.to_dataframe().show()

        time.sleep(0.1)  # Small delay between versions

        # Version 1: Add more products
        print_subheader("Version 1: Add Products")

        more_products = [
            {"product": "Widget D", "quantity": 75, "price": 15.00},
            {"product": "Widget E", "quantity": 30, "price": 50.00},
        ]

        delta_table.write(HiveDataFrame(more_products), mode="append")
        print("After adding new products:")
        delta_table.to_dataframe().show()

        time.sleep(0.1)

        # Version 2: Full update
        print_subheader("Version 2: Inventory Reset")

        fresh_inventory = [
            {"product": "New Widget", "quantity": 500, "price": 8.00},
        ]

        delta_table.write(HiveDataFrame(fresh_inventory), mode="overwrite")
        print("After reset:")
        delta_table.to_dataframe().show()

        # View history
        print_subheader("Transaction History")

        history = delta_table.history(limit=10)
        print("Recent transactions:")
        for entry in history:
            print(f"  Version {entry['version']}: {entry['operation']} at {entry['timestamp']}")

        # Time travel: Read version 0
        print_subheader("Time Travel: Version 0")

        df_v0 = delta_table.to_dataframe(version=0)
        print("Reading data as of version 0:")
        df_v0.show()

        # Time travel: Read version 1
        print_subheader("Time Travel: Version 1")

        df_v1 = delta_table.to_dataframe(version=1)
        print("Reading data as of version 1:")
        df_v1.show()

        # Current version
        print_subheader("Current Version")

        print("Current data (latest version):")
        delta_table.to_dataframe().show()

    finally:
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory")


def demo_transactions():
    """Demonstrate transaction-based writes."""
    print_header("Transaction-Based Writes")

    temp_dir = tempfile.mkdtemp(prefix="hiveframe_delta_txn_")

    try:
        delta_path = os.path.join(temp_dir, "accounts")
        delta_table = DeltaTable(delta_path)

        # Initial accounts
        accounts = [
            {"account_id": "A001", "owner": "Alice", "balance": 1000},
            {"account_id": "A002", "owner": "Bob", "balance": 500},
        ]

        delta_table.write(HiveDataFrame(accounts), mode="overwrite")

        print("Initial account balances:")
        delta_table.to_dataframe().show()

        # Demonstrate multiple writes creating versions
        print_subheader("Sequential Writes")

        # First update
        print("  Step 1: Adding new account...")
        delta_table.write(
            HiveDataFrame([{"account_id": "A003", "owner": "Carol", "balance": 750}]), mode="append"
        )

        # Second update
        print("  Step 2: Adding another account...")
        delta_table.write(
            HiveDataFrame([{"account_id": "A004", "owner": "David", "balance": 1200}]),
            mode="append",
        )

        print("\nFinal account balances:")
        delta_table.to_dataframe().show()

        # Show that all writes are versioned
        print_subheader("Version History")
        history = delta_table.history()
        for entry in history:
            print(f"  Version {entry['version']}: {entry['operation']}")

    finally:
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory")


def demo_performance():
    """Demonstrate storage performance."""
    print_header("Storage Performance Comparison")

    temp_dir = tempfile.mkdtemp(prefix="hiveframe_perf_")

    try:
        sizes = [1000, 5000, 10000]

        for size in sizes:
            print_subheader(f"Dataset Size: {size:,} records")

            # Generate data
            data = generate_transactions(size)
            df = HiveDataFrame(data)

            # Parquet performance
            parquet_path = os.path.join(temp_dir, f"txn_{size}.parquet")

            start = time.time()
            write_parquet(df, parquet_path)
            parquet_write = time.time() - start

            start = time.time()
            _ = read_parquet(parquet_path).collect()
            parquet_read = time.time() - start

            parquet_size = os.path.getsize(parquet_path) / 1024

            # Delta performance
            delta_path = os.path.join(temp_dir, f"txn_{size}_delta")

            start = time.time()
            write_delta(df, delta_path, mode="overwrite")
            delta_write = time.time() - start

            start = time.time()
            _ = read_delta(delta_path).collect()
            delta_read = time.time() - start

            # Calculate Delta table size (sum all files)
            delta_size = (
                sum(
                    os.path.getsize(os.path.join(root, f))
                    for root, dirs, files in os.walk(delta_path)
                    for f in files
                )
                / 1024
            )

            print(f"  Format      | Write (ms) | Read (ms) | Size (KB)")
            print(f"  ------------|------------|-----------|----------")
            print(
                f"  Parquet     | {parquet_write*1000:>10.2f} | {parquet_read*1000:>9.2f} | {parquet_size:>8.2f}"
            )
            print(
                f"  Delta Lake  | {delta_write*1000:>10.2f} | {delta_read*1000:>9.2f} | {delta_size:>8.2f}"
            )

    finally:
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run all storage demos."""
    print("\n" + "=" * 60)
    print("    🐝 HIVEFRAME STORAGE: PARQUET & DELTA LAKE 🐝")
    print("=" * 60)
    print("\nHiveFrame's storage layer provides:")
    print("  • Columnar storage with Parquet")
    print("  • ACID transactions with Delta Lake")
    print("  • Time travel and version history")
    print("  • Schema evolution support")
    print("  • Partition pruning for fast queries")

    demos = [
        ("Parquet Operations", demo_parquet_operations),
        ("Delta Lake Basics", demo_delta_table_basics),
        ("Delta CRUD", demo_delta_crud),
        ("Time Travel", demo_time_travel),
        ("ACID Transactions", demo_transactions),
        ("Performance Comparison", demo_performance),
    ]

    for name, demo_fn in demos:
        try:
            demo_fn()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("           STORAGE DEMOS COMPLETE")
    print("=" * 60)
    print("\nHiveFrame storage demonstrates how bee-inspired patterns")
    print("enable efficient, reliable data persistence with ACID guarantees.")


if __name__ == "__main__":
    main()
