#!/usr/bin/env python3
"""
HiveFrame SwarmQL Demo
======================

Demonstrates the SwarmQL SQL engine - HiveFrame's bee-inspired
SQL query processor that translates SQL into distributed operations.

Features demonstrated:
- Table registration and catalog management
- SELECT, WHERE, GROUP BY queries
- Aggregation functions
- Query execution plans
- Join operations via SQL

Run: python demo_sql.py
"""

import os
import random
import sys
import time
from typing import Any, Dict, List

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hiveframe import HiveDataFrame
from hiveframe.sql import SwarmQLContext


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


def generate_users(n: int = 100) -> List[Dict[str, Any]]:
    """Generate sample user data."""
    cities = ["New York", "San Francisco", "Chicago", "Los Angeles", "Seattle", "Boston"]

    return [
        {
            "user_id": i,
            "name": f"User_{i}",
            "city": random.choice(cities),
            "age": random.randint(18, 65),
            "signup_year": random.randint(2018, 2025),
            "is_premium": random.random() > 0.7,
        }
        for i in range(1, n + 1)
    ]


def generate_orders(n: int = 500, user_count: int = 100) -> List[Dict[str, Any]]:
    """Generate sample order data."""
    categories = ["Electronics", "Clothing", "Food", "Books", "Home"]
    statuses = ["completed", "pending", "cancelled", "refunded"]

    return [
        {
            "order_id": 1000 + i,
            "user_id": random.randint(1, user_count),
            "amount": round(random.uniform(10, 500), 2),
            "category": random.choice(categories),
            "status": random.choice(statuses),
            "order_date": f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        }
        for i in range(n)
    ]


def generate_products(n: int = 50) -> List[Dict[str, Any]]:
    """Generate sample product data."""
    categories = ["Electronics", "Clothing", "Food", "Books", "Home"]

    return [
        {
            "product_id": i,
            "product_name": f"Product_{i}",
            "category": random.choice(categories),
            "price": round(random.uniform(5, 200), 2),
            "stock": random.randint(0, 100),
            "rating": round(random.uniform(1, 5), 1),
        }
        for i in range(1, n + 1)
    ]


# =============================================================================
# Demo Functions
# =============================================================================


def demo_basic_queries():
    """Demonstrate basic SQL queries."""
    print_header("Basic SQL Queries")

    # Create context and register tables
    ctx = SwarmQLContext(num_workers=4)

    users = generate_users(20)
    ctx.register_table("users", HiveDataFrame(users))

    print(f"Registered table 'users' with {len(users)} rows")
    print(f"Available tables: {ctx.tables()}")

    # Simple SELECT
    print_subheader("1. SELECT All Columns")
    print("Query: SELECT * FROM users LIMIT 5")

    start = time.time()
    result = ctx.sql("SELECT * FROM users LIMIT 5")
    elapsed = time.time() - start

    result.show()
    print(f"Execution time: {elapsed*1000:.2f}ms")

    # SELECT with specific columns
    print_subheader("2. SELECT Specific Columns")
    print("Query: SELECT name, city, age FROM users LIMIT 5")

    result = ctx.sql("SELECT name, city, age FROM users LIMIT 5")
    result.show()

    # WHERE clause
    print_subheader("3. WHERE Clause Filtering")
    print("Query: SELECT name, age, city FROM users WHERE age > 30")

    result = ctx.sql("SELECT name, age, city FROM users WHERE age > 30")
    result.show()

    # Multiple conditions
    print_subheader("4. Multiple Conditions")
    print("Query: SELECT * FROM users WHERE age > 25 AND is_premium = true")

    result = ctx.sql("SELECT name, age, city FROM users WHERE age > 25 AND is_premium = true")
    result.show()


def demo_aggregations():
    """Demonstrate SQL aggregation queries."""
    print_header("SQL Aggregations")

    ctx = SwarmQLContext(num_workers=4)

    users = generate_users(100)
    orders = generate_orders(500, 100)

    ctx.register_table("users", HiveDataFrame(users))
    ctx.register_table("orders", HiveDataFrame(orders))

    print(f"Registered 'users' ({len(users)} rows) and 'orders' ({len(orders)} rows)")

    # COUNT with alias
    print_subheader("1. COUNT with Alias")
    print("Query: SELECT city, COUNT(*) as user_count FROM users GROUP BY city\n")

    result = ctx.sql("""
        SELECT city, COUNT(*) as user_count
        FROM users
        GROUP BY city
    """)
    result.show()

    # SUM aggregation with alias
    print_subheader("2. SUM with Alias")
    print("Query: SELECT category, SUM(amount) as total_revenue FROM orders GROUP BY category\n")

    result = ctx.sql("""
        SELECT category, SUM(amount) as total_revenue
        FROM orders
        GROUP BY category
    """)
    result.show()

    # AVG aggregation with alias
    print_subheader("3. AVG with Alias")
    print("Query: SELECT status, AVG(amount) as avg_order_value FROM orders GROUP BY status\n")

    result = ctx.sql("""
        SELECT status, AVG(amount) as avg_order_value
        FROM orders
        GROUP BY status
    """)
    result.show()

    # Multiple aggregations with aliases
    print_subheader("4. Multiple Aggregations with Aliases")
    print(
        "Query: SELECT category, COUNT(*) as order_count, SUM(amount) as total, AVG(amount) as average FROM orders GROUP BY category\n"
    )

    result = ctx.sql("""
        SELECT category, COUNT(*) as order_count, SUM(amount) as total, AVG(amount) as average
        FROM orders
        GROUP BY category
    """)
    result.show()

    # Filtered aggregation
    print_subheader("5. Filtered Aggregation")
    print(
        "Query: SELECT category, COUNT(*) as high_value_orders FROM orders WHERE amount > 200 GROUP BY category\n"
    )

    result = ctx.sql("""
        SELECT category, COUNT(*) as high_value_orders
        FROM orders
        WHERE amount > 200
        GROUP BY category
    """)
    result.show()


def demo_query_plans():
    """Demonstrate query execution plans."""
    print_header("Query Execution Plans")

    ctx = SwarmQLContext(num_workers=4)

    orders = generate_orders(100, 20)
    ctx.register_table("orders", HiveDataFrame(orders))

    print("The EXPLAIN command shows how SwarmQL optimizes and executes queries")
    print("using bee-inspired distributed processing.\n")

    # Simple query plan
    print_subheader("1. Simple SELECT Plan")
    query = "SELECT * FROM orders WHERE amount > 100"
    print(f"Query: {query}\n")

    plan = ctx.explain(query)
    print(plan)

    # Aggregation query plan
    print_subheader("2. Aggregation Plan")
    query = "SELECT category, COUNT(*), AVG(amount) FROM orders GROUP BY category"
    print(f"Query: {query}\n")

    plan = ctx.explain(query)
    print(plan)

    # Complex query plan
    print_subheader("3. Complex Query Plan")
    query = """
        SELECT category, status, SUM(amount)
        FROM orders
        WHERE amount > 50
        GROUP BY category, status
    """
    print(f"Query: {query}\n")

    plan = ctx.explain(query)
    print(plan)


def demo_catalog_management():
    """Demonstrate catalog management."""
    print_header("Catalog Management")

    ctx = SwarmQLContext(num_workers=4)

    print_subheader("1. Registering Tables")

    # Register multiple tables
    ctx.register_table("users", HiveDataFrame(generate_users(50)))
    ctx.register_table("orders", HiveDataFrame(generate_orders(200, 50)))
    ctx.register_table("products", HiveDataFrame(generate_products(30)))

    print(f"Registered tables: {ctx.tables()}")

    print_subheader("2. Accessing Table Info")

    for table_name in ctx.tables():
        table = ctx.table(table_name)
        print(f"  {table_name}: {table.count()} rows")

    print_subheader("3. Dropping Tables")

    print(f"Before drop: {ctx.tables()}")
    ctx.drop_table("products")
    print(f"After dropping 'products': {ctx.tables()}")

    print_subheader("4. Creating DataFrames from Context")

    # Create a DataFrame directly through context
    new_data = [
        {"id": 1, "value": "A"},
        {"id": 2, "value": "B"},
        {"id": 3, "value": "C"},
    ]

    df = ctx.create_dataframe(new_data)
    ctx.register_table("temp_data", df)

    print(f"Created and registered 'temp_data': {ctx.tables()}")
    ctx.sql("SELECT * FROM temp_data").show()


def demo_performance():
    """Demonstrate SQL query performance."""
    print_header("SQL Performance Benchmark")

    ctx = SwarmQLContext(num_workers=8)

    sizes = [1000, 5000, 10000]

    for size in sizes:
        print_subheader(f"Dataset Size: {size:,} rows")

        orders = generate_orders(size, size // 10)
        ctx.register_table("orders", HiveDataFrame(orders))

        # Simple query
        start = time.time()
        result = ctx.sql("SELECT * FROM orders WHERE amount > 100")
        count = result.count()
        simple_time = time.time() - start
        print(f"  Simple filter: {simple_time*1000:.2f}ms ({count} results)")

        # Aggregation query
        start = time.time()
        result = ctx.sql("""
            SELECT category, COUNT(*), AVG(amount), SUM(amount)
            FROM orders
            GROUP BY category
        """)
        _ = result.collect()
        agg_time = time.time() - start
        print(f"  Aggregation: {agg_time*1000:.2f}ms")

        # Complex query
        start = time.time()
        result = ctx.sql("""
            SELECT category, status, COUNT(*), AVG(amount)
            FROM orders
            WHERE amount > 50
            GROUP BY category, status
        """)
        _ = result.collect()
        complex_time = time.time() - start
        print(f"  Complex query: {complex_time*1000:.2f}ms")

        ctx.drop_table("orders")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run all SQL demos."""
    print("\n" + "=" * 60)
    print("      🐝 SWARMQL: BEE-INSPIRED SQL ENGINE 🐝")
    print("=" * 60)
    print("\nSwarmQL translates SQL queries into distributed HiveFrame")
    print("operations, using bee colony intelligence for optimization.")
    print("\nFeatures:")
    print("  • Standard SQL syntax (SELECT, WHERE, GROUP BY)")
    print("  • Bee-inspired query optimization")
    print("  • Swarm-based distributed execution")
    print("  • Table catalog management")

    demos = [
        ("Basic Queries", demo_basic_queries),
        ("Aggregations", demo_aggregations),
        ("Query Plans", demo_query_plans),
        ("Catalog Management", demo_catalog_management),
        ("Performance", demo_performance),
    ]

    for name, demo_fn in demos:
        try:
            demo_fn()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("              SWARMQL DEMOS COMPLETE")
    print("=" * 60)
    print("\nSwarmQL demonstrates how bee colony patterns enable")
    print("efficient SQL query execution without a centralized optimizer.")


if __name__ == "__main__":
    main()
