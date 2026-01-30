# SwarmQL 2.0 - Full ANSI SQL with Bee-Inspired Extensions

SwarmQL 2.0 represents a major milestone in HiveFrame's query capabilities, combining full ANSI SQL compliance with innovative bee-inspired extensions that leverage swarm intelligence principles for query optimization and execution.

## Table of Contents

- [Overview](#overview)
- [ANSI SQL Features](#ansi-sql-features)
- [Bee-Inspired Extensions](#bee-inspired-extensions)
- [Examples](#examples)
- [Performance Considerations](#performance-considerations)

## Overview

SwarmQL 2.0 extends the original SwarmQL engine with:

1. **Complete ANSI SQL Support**: CTEs, set operations, subqueries, window functions, and comprehensive function library
2. **Bee-Inspired Query Hints**: Novel execution strategies inspired by bee colony behavior
3. **Backward Compatibility**: All SwarmQL 1.x queries continue to work

## ANSI SQL Features

### Common Table Expressions (CTEs)

CTEs allow you to define temporary result sets that can be referenced within a SELECT statement.

```sql
-- Simple CTE
WITH engineering AS (
    SELECT * FROM employees WHERE department = 'Engineering'
)
SELECT name, salary FROM engineering WHERE salary > 100000;

-- Multiple CTEs
WITH
    high_performers AS (
        SELECT employee_id, performance_score 
        FROM reviews 
        WHERE performance_score > 4.0
    ),
    engineers AS (
        SELECT id, name FROM employees WHERE department = 'Engineering'
    )
SELECT e.name, hp.performance_score
FROM engineers e
JOIN high_performers hp ON e.id = hp.employee_id;
```

### Set Operations

Combine results from multiple SELECT statements using set operations.

#### UNION

```sql
-- UNION removes duplicates
SELECT customer_id FROM orders_2023
UNION
SELECT customer_id FROM orders_2024;

-- UNION ALL keeps duplicates
SELECT product_id FROM cart_items
UNION ALL
SELECT product_id FROM wishlist_items;
```

#### INTERSECT

```sql
-- Find customers who placed orders in both years
SELECT customer_id FROM orders_2023
INTERSECT
SELECT customer_id FROM orders_2024;
```

#### EXCEPT

```sql
-- Find products in inventory but not in active orders
SELECT product_id FROM inventory
EXCEPT
SELECT product_id FROM order_items WHERE status = 'active';
```

### Subqueries

SwarmQL 2.0 supports subqueries in various contexts.

#### IN Subqueries

```sql
-- Find customers with high-value orders
SELECT name 
FROM customers 
WHERE id IN (
    SELECT customer_id 
    FROM orders 
    WHERE total_amount > 1000
);
```

#### Scalar Subqueries

```sql
-- Show order total as percentage of all orders
SELECT 
    order_id,
    total_amount,
    (SELECT SUM(total_amount) FROM orders) as grand_total
FROM orders;
```

#### Subqueries in FROM

```sql
-- Query a derived table
SELECT dept, avg_salary
FROM (
    SELECT department as dept, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
) dept_stats
WHERE avg_salary > 75000;
```

### Window Functions

Perform calculations across a set of rows related to the current row.

#### ROW_NUMBER

```sql
SELECT 
    name,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank
FROM employees;
```

#### RANK and DENSE_RANK

```sql
SELECT 
    product_name,
    sales,
    RANK() OVER (ORDER BY sales DESC) as rank,
    DENSE_RANK() OVER (ORDER BY sales DESC) as dense_rank
FROM product_sales;
```

#### LAG and LEAD

```sql
SELECT 
    date,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY date) as prev_day_revenue,
    LEAD(revenue, 1) OVER (ORDER BY date) as next_day_revenue
FROM daily_sales;
```

### String Functions

Comprehensive string manipulation functions.

```sql
-- UPPER / LOWER
SELECT UPPER(name) as upper_name, LOWER(email) as lower_email
FROM users;

-- TRIM
SELECT TRIM(description) as clean_description
FROM products;

-- LENGTH
SELECT name, LENGTH(name) as name_length
FROM categories;

-- CONCAT
SELECT CONCAT(first_name, ' ', last_name) as full_name
FROM employees;

-- SUBSTRING
SELECT SUBSTRING(product_code, 0, 3) as category_code
FROM products;
```

### Date/Time Functions

Work with temporal data.

```sql
-- Current date and timestamp
SELECT CURRENT_DATE() as today, CURRENT_TIMESTAMP() as now;

-- Date arithmetic
SELECT 
    order_date,
    DATE_ADD(order_date, 7) as expected_delivery,
    DATE_SUB(order_date, 1) as processing_date
FROM orders;

-- Date differences
SELECT 
    DATE_DIFF(completed_date, created_date) as days_to_complete
FROM tasks;

-- Extract date parts
SELECT 
    EXTRACT(YEAR FROM order_date) as year,
    EXTRACT(MONTH FROM order_date) as month,
    EXTRACT(DAY FROM order_date) as day
FROM orders;
```

### Other Functions

#### COALESCE

Returns the first non-null value.

```sql
SELECT 
    name,
    COALESCE(mobile_phone, work_phone, 'No phone') as contact_number
FROM contacts;
```

#### NULLIF

Returns NULL if two values are equal.

```sql
SELECT 
    product_id,
    NULLIF(discount_price, regular_price) as actual_discount
FROM products;
```

## Bee-Inspired Extensions

SwarmQL 2.0 introduces novel query hints inspired by bee colony behavior. These extensions enable the query optimizer to leverage swarm intelligence principles for enhanced execution strategies.

### WAGGLE JOIN

The WAGGLE JOIN hint instructs the executor to use quality-weighted join execution strategies, inspired by the waggle dance that bees use to communicate food source quality.

```sql
-- Standard join with quality-weighted execution
SELECT o.order_id, c.customer_name
FROM orders o
WAGGLE JOIN customers c ON o.customer_id = c.id;
```

**How it works:**
- The executor evaluates multiple join strategies in parallel
- Each strategy reports its "fitness" (throughput, latency, resource usage)
- The best-performing strategy is selected dynamically
- Similar to how bees choose food sources based on waggle dance intensity

**Use cases:**
- Large joins with uncertain cardinality
- Joins on skewed data
- Multi-table joins with complex predicates

### SWARM PARTITION BY (Future)

Adaptive partitioning hints for distributed execution.

```sql
-- Hint for dynamic partition adjustment
SELECT /*+ SWARM PARTITION BY customer_region */ 
    customer_id, 
    SUM(order_total)
FROM orders
GROUP BY customer_id;
```

**How it works:**
- Initial partitioning based on data distribution
- Worker bees monitor partition load and report to the swarm
- Partitions dynamically split or merge based on load
- Inspired by how bee colonies adjust forager allocation

### PHEROMONE CACHE (Future)

Intelligent result caching based on query patterns.

```sql
-- Enable pheromone-based caching
SELECT /*+ PHEROMONE CACHE */ 
    product_category, 
    COUNT(*) as total_sales
FROM sales
WHERE sale_date >= CURRENT_DATE() - 7
GROUP BY product_category;
```

**How it works:**
- Frequently accessed query patterns leave "pheromone trails"
- Cache priority based on trail strength (access frequency + recency)
- Automatic cache invalidation when source data changes
- Similar to how ants use pheromones to mark profitable paths

### SCOUT HINT (Future)

Enable speculative execution for query exploration.

```sql
-- Allow scout bees to explore alternative execution paths
SELECT /*+ SCOUT HINT */ 
    customer_id, 
    AVG(order_value) as avg_order
FROM orders
WHERE order_date >= '2024-01-01'
GROUP BY customer_id;
```

**How it works:**
- Main query executes with best-known plan
- "Scout bees" explore alternative execution strategies
- If a scout finds a better path, traffic redirects
- Future queries benefit from discovered optimizations
- Inspired by scout bees that search for new hive locations

## Examples

### Complete ETL Pipeline with SwarmQL 2.0

```sql
-- Complex ETL using CTEs, set operations, and bee-inspired hints
WITH 
    -- Extract recent orders
    recent_orders AS (
        SELECT 
            order_id,
            customer_id,
            order_date,
            total_amount,
            ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) as order_rank
        FROM orders
        WHERE order_date >= CURRENT_DATE() - 30
    ),
    -- Extract high-value customers
    high_value_customers AS (
        SELECT customer_id, SUM(total_amount) as total_spent
        FROM orders
        WHERE order_date >= CURRENT_DATE() - 365
        GROUP BY customer_id
        HAVING SUM(total_amount) > 10000
    )
-- Join and aggregate with quality-weighted execution
SELECT 
    c.customer_name,
    c.customer_segment,
    ro.total_amount as last_order_amount,
    hvc.total_spent as annual_total,
    COALESCE(c.preferred_contact, c.email, 'N/A') as contact
FROM recent_orders ro
WAGGLE JOIN customers c ON ro.customer_id = c.id
INNER JOIN high_value_customers hvc ON c.id = hvc.customer_id
WHERE ro.order_rank = 1
ORDER BY hvc.total_spent DESC;
```

### Data Quality Analysis

```sql
-- Find data quality issues using advanced SQL
WITH 
    duplicate_records AS (
        SELECT email, COUNT(*) as count
        FROM customers
        GROUP BY email
        HAVING COUNT(*) > 1
    ),
    missing_data AS (
        SELECT id FROM customers
        WHERE phone IS NULL AND email IS NULL
    ),
    invalid_dates AS (
        SELECT id FROM orders
        WHERE order_date > CURRENT_DATE()
    )
-- Combine all quality issues
SELECT 'Duplicate Email' as issue_type, email as identifier
FROM duplicate_records
UNION ALL
SELECT 'Missing Contact' as issue_type, CAST(id AS STRING) as identifier
FROM missing_data
UNION ALL
SELECT 'Future Date' as issue_type, CAST(id AS STRING) as identifier
FROM invalid_dates;
```

## Performance Considerations

### When to Use Bee-Inspired Hints

1. **WAGGLE JOIN**: 
   - Large tables (>1M rows)
   - Uncertain join cardinality
   - Multiple join strategies possible

2. **Query Optimization**:
   - CTEs are materialized once and reused
   - Set operations can be more efficient than UNION + DISTINCT
   - Window functions avoid self-joins

### Best Practices

1. **Use CTEs for readability**: Break complex queries into logical steps
2. **Prefer UNION ALL when possible**: Avoid unnecessary DISTINCT operations
3. **Window functions over self-joins**: More efficient and easier to read
4. **Test with EXPLAIN**: Use EXPLAIN to understand query plans

### Limitations

Current implementation notes:

1. **Window Functions**: Basic support for ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD
2. **Correlated Subqueries**: Limited support - use joins where possible
3. **Bee-Inspired Hints**: WAGGLE JOIN fully implemented; others planned for future releases

## Migration from SwarmQL 1.x

All SwarmQL 1.x queries continue to work without modification. New features are additive:

```sql
-- SwarmQL 1.x query (still works)
SELECT name, age FROM users WHERE age > 21;

-- SwarmQL 2.0 enhanced version
WITH eligible_users AS (
    SELECT name, age FROM users WHERE age > 21
)
SELECT 
    name,
    age,
    ROW_NUMBER() OVER (ORDER BY age DESC) as age_rank
FROM eligible_users;
```

## Future Enhancements

Planned for SwarmQL 3.0:

- **Recursive CTEs**: Support for hierarchical queries
- **Full correlated subquery support**: Complex subquery predicates
- **More window functions**: FIRST_VALUE, LAST_VALUE, NTH_VALUE
- **Complete bee-inspired hint implementation**: SWARM PARTITION, PHEROMONE CACHE, SCOUT HINT
- **Query federation**: Queries across multiple data sources

## Conclusion

SwarmQL 2.0 represents a significant step forward in combining traditional SQL power with innovative swarm intelligence concepts. The bee-inspired extensions enable novel optimization strategies that adapt to data and workload patterns, while maintaining full backward compatibility with existing queries.

For more examples and detailed API documentation, see the [API Reference](api-reference.md) and [Examples](../examples/swarmql/).
