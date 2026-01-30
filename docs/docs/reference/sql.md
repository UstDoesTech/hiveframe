---
sidebar_position: 4
---

# SQL Module

SQL interface for querying DataFrames using familiar SQL syntax.

```python
from hiveframe.sql import SQLContext, SQLExecutor
from hiveframe.sql.types import SQLType
```

## SQLContext

Main entry point for SQL operations.

### Class Definition

```python
class SQLContext:
    """
    Context for executing SQL queries against DataFrames.
    
    Provides a SQL interface familiar to Spark SQL users,
    with support for standard SQL operations, window functions,
    and HiveFrame-specific extensions.
    """
    
    def __init__(
        self,
        colony: Optional[Colony] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a SQL context.
        
        Args:
            colony: Colony for distributed execution.
                    If None, uses local execution.
            config: SQL configuration options
        """
```

### Configuration Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sql.case_sensitive` | bool | False | Case-sensitive identifiers |
| `sql.ansi_mode` | bool | True | ANSI SQL compliance |
| `sql.timezone` | str | "UTC" | Default timezone |
| `optimizer.enabled` | bool | True | ABC query optimizer |
| `optimizer.max_iterations` | int | 100 | Max optimization cycles |

### Methods

#### `register_table()`

```python
def register_table(
    self,
    name: str,
    df: DataFrame
) -> None:
    """
    Register a DataFrame as a table.
    
    Args:
        name: Table name for SQL queries
        df: DataFrame to register
        
    Example:
        sql = SQLContext()
        sql.register_table("users", users_df)
        sql.register_table("orders", orders_df)
    """
```

#### `register_view()`

```python
def register_view(
    self,
    name: str,
    query: str
) -> None:
    """
    Register a view from a SQL query.
    
    Args:
        name: View name
        query: SQL query defining the view
        
    Example:
        sql.register_view(
            "active_users",
            "SELECT * FROM users WHERE status = 'active'"
        )
    """
```

#### `execute()`

```python
def execute(self, query: str) -> DataFrame:
    """
    Execute a SQL query.
    
    Args:
        query: SQL query string
        
    Returns:
        DataFrame with query results
        
    Raises:
        SQLParseError: Invalid SQL syntax
        SQLExecutionError: Query execution failed
        
    Example:
        result = sql.execute('''
            SELECT 
                u.name,
                COUNT(o.id) as order_count,
                SUM(o.amount) as total_spent
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.status = 'active'
            GROUP BY u.name
            ORDER BY total_spent DESC
            LIMIT 10
        ''')
    """
```

#### `explain()`

```python
def explain(
    self,
    query: str,
    extended: bool = False
) -> str:
    """
    Show query execution plan.
    
    Args:
        query: SQL query
        extended: Show detailed plan
        
    Returns:
        Execution plan as string
        
    Example:
        plan = sql.explain(
            "SELECT * FROM users WHERE age > 21",
            extended=True
        )
        print(plan)
    """
```

#### `tables()`

```python
def tables(self) -> List[str]:
    """
    List registered tables.
    
    Returns:
        List of table names
        
    Example:
        for table in sql.tables():
            print(table)
    """
```

#### `describe()`

```python
def describe(self, table: str) -> DataFrame:
    """
    Describe a table's schema.
    
    Args:
        table: Table name
        
    Returns:
        DataFrame with column info
        
    Example:
        sql.describe("users").show()
        # +--------+--------+--------+
        # |col_name|data_type|nullable|
        # +--------+--------+--------+
        # |id      |long    |false   |
        # |name    |string  |true    |
        # +--------+--------+--------+
    """
```

### Supported SQL Features

#### SELECT Statements

```sql
-- Basic SELECT
SELECT name, age FROM users;

-- With expressions
SELECT name, age * 12 AS age_months FROM users;

-- DISTINCT
SELECT DISTINCT city FROM users;

-- Aliases
SELECT u.name AS user_name FROM users u;
```

#### WHERE Clause

```sql
-- Comparison
SELECT * FROM users WHERE age > 21;

-- Multiple conditions
SELECT * FROM users 
WHERE age > 21 AND status = 'active';

-- IN operator
SELECT * FROM users WHERE city IN ('NYC', 'LA', 'Chicago');

-- LIKE pattern
SELECT * FROM users WHERE name LIKE 'J%';

-- NULL checks
SELECT * FROM users WHERE email IS NOT NULL;

-- BETWEEN
SELECT * FROM orders WHERE amount BETWEEN 100 AND 500;
```

#### JOINs

```sql
-- INNER JOIN
SELECT u.name, o.amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- LEFT JOIN
SELECT u.name, o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- Multiple joins
SELECT u.name, o.amount, p.name AS product
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN products p ON o.product_id = p.id;
```

#### Aggregations

```sql
-- GROUP BY
SELECT city, COUNT(*) as count, AVG(age) as avg_age
FROM users
GROUP BY city;

-- HAVING
SELECT city, COUNT(*) as count
FROM users
GROUP BY city
HAVING COUNT(*) > 100;

-- Multiple aggregations
SELECT 
    department,
    COUNT(*) as total,
    SUM(salary) as total_salary,
    AVG(salary) as avg_salary,
    MIN(salary) as min_salary,
    MAX(salary) as max_salary
FROM employees
GROUP BY department;
```

#### Window Functions

```sql
-- ROW_NUMBER
SELECT 
    name,
    salary,
    ROW_NUMBER() OVER (
        PARTITION BY department 
        ORDER BY salary DESC
    ) as rank
FROM employees;

-- Running total
SELECT 
    date,
    amount,
    SUM(amount) OVER (
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as running_total
FROM transactions;

-- LAG/LEAD
SELECT 
    date,
    price,
    LAG(price, 1) OVER (ORDER BY date) as prev_price,
    price - LAG(price, 1) OVER (ORDER BY date) as change
FROM stock_prices;
```

#### Subqueries

```sql
-- Scalar subquery
SELECT name, 
       (SELECT AVG(salary) FROM employees) as company_avg
FROM employees;

-- IN subquery
SELECT * FROM users
WHERE id IN (
    SELECT user_id FROM orders WHERE amount > 1000
);

-- EXISTS
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.id
);

-- Derived table
SELECT city, avg_age
FROM (
    SELECT city, AVG(age) as avg_age
    FROM users
    GROUP BY city
) subq
WHERE avg_age > 30;
```

#### Set Operations

```sql
-- UNION
SELECT name FROM customers
UNION
SELECT name FROM suppliers;

-- UNION ALL (keep duplicates)
SELECT name FROM customers
UNION ALL
SELECT name FROM suppliers;

-- INTERSECT
SELECT name FROM customers
INTERSECT
SELECT name FROM vip_list;

-- EXCEPT
SELECT name FROM customers
EXCEPT
SELECT name FROM blacklist;
```

#### Common Table Expressions (CTEs)

```sql
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', date) as month,
        SUM(amount) as total
    FROM orders
    GROUP BY DATE_TRUNC('month', date)
),
growth AS (
    SELECT 
        month,
        total,
        LAG(total) OVER (ORDER BY month) as prev_total
    FROM monthly_sales
)
SELECT 
    month,
    total,
    (total - prev_total) / prev_total * 100 as growth_pct
FROM growth
WHERE prev_total IS NOT NULL;
```

### Example Usage

```python
from hiveframe.sql import SQLContext
import hiveframe as hf

# Create context
sql = SQLContext()

# Load and register data
users = hf.read.parquet("data/users.parquet")
orders = hf.read.parquet("data/orders.parquet")

sql.register_table("users", users)
sql.register_table("orders", orders)

# Run queries
result = sql.execute("""
    WITH user_stats AS (
        SELECT 
            u.id,
            u.name,
            u.signup_date,
            COUNT(o.id) as order_count,
            COALESCE(SUM(o.amount), 0) as total_spent
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.status = 'active'
        GROUP BY u.id, u.name, u.signup_date
    )
    SELECT 
        name,
        order_count,
        total_spent,
        total_spent / NULLIF(order_count, 0) as avg_order_value,
        DATEDIFF(CURRENT_DATE, signup_date) as days_since_signup
    FROM user_stats
    ORDER BY total_spent DESC
    LIMIT 100
""")

result.show()
```

---

## SQL Functions

### String Functions

| Function | Description | Example |
|----------|-------------|---------|
| `UPPER(s)` | Uppercase | `UPPER(name)` |
| `LOWER(s)` | Lowercase | `LOWER(email)` |
| `LENGTH(s)` | String length | `LENGTH(name)` |
| `TRIM(s)` | Remove whitespace | `TRIM(name)` |
| `CONCAT(...)` | Concatenate | `CONCAT(first, ' ', last)` |
| `SUBSTRING(s, start, len)` | Extract substring | `SUBSTRING(code, 1, 3)` |
| `REPLACE(s, from, to)` | Replace text | `REPLACE(phone, '-', '')` |
| `SPLIT(s, delim)` | Split string | `SPLIT(tags, ',')` |

### Date Functions

| Function | Description | Example |
|----------|-------------|---------|
| `CURRENT_DATE` | Today's date | `CURRENT_DATE` |
| `CURRENT_TIMESTAMP` | Current time | `CURRENT_TIMESTAMP` |
| `DATE_TRUNC(unit, date)` | Truncate date | `DATE_TRUNC('month', date)` |
| `DATE_ADD(date, days)` | Add days | `DATE_ADD(date, 7)` |
| `DATEDIFF(end, start)` | Days between | `DATEDIFF(end_date, start_date)` |
| `YEAR(date)` | Extract year | `YEAR(created_at)` |
| `MONTH(date)` | Extract month | `MONTH(created_at)` |
| `DAY(date)` | Extract day | `DAY(created_at)` |

### Numeric Functions

| Function | Description | Example |
|----------|-------------|---------|
| `ABS(n)` | Absolute value | `ABS(difference)` |
| `ROUND(n, d)` | Round | `ROUND(price, 2)` |
| `FLOOR(n)` | Floor | `FLOOR(score)` |
| `CEIL(n)` | Ceiling | `CEIL(rating)` |
| `MOD(n, d)` | Modulo | `MOD(id, 10)` |
| `POWER(n, p)` | Power | `POWER(base, 2)` |
| `SQRT(n)` | Square root | `SQRT(variance)` |

### Conditional Functions

```sql
-- CASE expression
CASE 
    WHEN age < 18 THEN 'minor'
    WHEN age < 65 THEN 'adult'
    ELSE 'senior'
END as age_group

-- COALESCE (first non-null)
COALESCE(nickname, name, 'Unknown')

-- NULLIF (return null if equal)
total / NULLIF(count, 0)

-- IF
IF(status = 'active', 1, 0)
```

## See Also

- [DataFrame](./dataframe) - DataFrame API
- [SQL Tutorial](/docs/tutorials/sql-analytics) - SQL tutorial
- [ABC Optimization](/docs/explanation/abc-optimization) - Query optimizer
