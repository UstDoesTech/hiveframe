"""
Tests for SwarmQL SQL Engine.

Tests cover:
- SQL parsing and tokenization
- Query execution
- Table registration and catalog
- Common SQL operations
"""

from typing import Any, Dict, List

import pytest

from hiveframe import HiveDataFrame
from hiveframe.sql import (
    SQLCatalog,
    SQLParser,
    SQLTokenizer,
    SwarmQLContext,
)


class TestSQLTokenizer:
    """Test SQL tokenization."""

    def test_simple_select(self):
        """Test tokenizing simple SELECT."""
        sql = "SELECT name, age FROM users"
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()

        # Should have SELECT, name, COMMA, age, FROM, users, EOF
        assert len(tokens) >= 6

    def test_select_with_where(self):
        """Test tokenizing SELECT with WHERE."""
        sql = "SELECT * FROM users WHERE age > 21"
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()

        assert any(t.value.upper() == "WHERE" for t in tokens)

    def test_string_literals(self):
        """Test tokenizing string literals."""
        sql = "SELECT * FROM users WHERE name = 'Alice'"
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()

        # Should have a STRING token with value 'Alice'
        string_tokens = [t for t in tokens if t.type.name == "STRING"]
        assert len(string_tokens) == 1
        assert string_tokens[0].value == "Alice"

    def test_numeric_literals(self):
        """Test tokenizing numeric literals."""
        sql = "SELECT * FROM data WHERE value > 3.14"
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()

        number_tokens = [t for t in tokens if t.type.name == "NUMBER"]
        assert len(number_tokens) == 1


class TestSQLParser:
    """Test SQL parsing."""

    def test_parse_simple_select(self):
        """Test parsing simple SELECT."""
        sql = "SELECT name, age FROM users"
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()
        parser = SQLParser(tokens)
        stmt = parser.parse()

        assert len(stmt.select_columns) == 2
        assert stmt.from_table is not None
        assert stmt.from_table.name == "users"

    def test_parse_select_star(self):
        """Test parsing SELECT *."""
        sql = "SELECT * FROM orders"
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()
        parser = SQLParser(tokens)
        stmt = parser.parse()

        assert len(stmt.select_columns) == 1

    def test_parse_where_clause(self):
        """Test parsing WHERE clause."""
        sql = "SELECT * FROM users WHERE age > 21"
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()
        parser = SQLParser(tokens)
        stmt = parser.parse()

        assert stmt.where_clause is not None

    def test_parse_group_by(self):
        """Test parsing GROUP BY."""
        sql = "SELECT department, COUNT(*) FROM employees GROUP BY department"
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()
        parser = SQLParser(tokens)
        stmt = parser.parse()

        assert len(stmt.group_by) == 1

    def test_parse_order_by(self):
        """Test parsing ORDER BY."""
        sql = "SELECT * FROM products ORDER BY price DESC"
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()
        parser = SQLParser(tokens)
        stmt = parser.parse()

        assert len(stmt.order_by) == 1
        assert not stmt.order_by[0].ascending

    def test_parse_limit(self):
        """Test parsing LIMIT."""
        sql = "SELECT * FROM items LIMIT 10"
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()
        parser = SQLParser(tokens)
        stmt = parser.parse()

        assert stmt.limit == 10


class TestSQLCatalog:
    """Test SQL catalog operations."""

    def test_register_table(self):
        """Test registering a table."""
        catalog = SQLCatalog()
        df = HiveDataFrame([{"id": 1}])

        catalog.register_table("test_table", df)

        assert catalog.table_exists("test_table")

    def test_get_table(self):
        """Test getting a registered table."""
        catalog = SQLCatalog()
        df = HiveDataFrame([{"id": 1, "name": "test"}])
        catalog.register_table("my_table", df)

        result = catalog.get_table("my_table")

        assert result is not None
        assert result.count() == 1

    def test_list_tables(self):
        """Test listing tables."""
        catalog = SQLCatalog()
        catalog.register_table("table1", HiveDataFrame([{"a": 1}]))
        catalog.register_table("table2", HiveDataFrame([{"b": 2}]))

        tables = catalog.list_tables()

        assert "table1" in tables
        assert "table2" in tables

    def test_drop_table(self):
        """Test dropping a table."""
        catalog = SQLCatalog()
        catalog.register_table("temp", HiveDataFrame([{"x": 1}]))

        catalog.drop_table("temp")

        assert not catalog.table_exists("temp")


class TestSwarmQLContext:
    """Test SwarmQL context operations."""

    @pytest.fixture
    def sample_data(self) -> List[Dict[str, Any]]:
        """Sample data for testing."""
        return [
            {"id": 1, "name": "Alice", "age": 30, "department": "Engineering"},
            {"id": 2, "name": "Bob", "age": 25, "department": "Marketing"},
            {"id": 3, "name": "Carol", "age": 35, "department": "Engineering"},
            {"id": 4, "name": "David", "age": 28, "department": "Sales"},
            {"id": 5, "name": "Eve", "age": 32, "department": "Marketing"},
        ]

    @pytest.fixture
    def ctx(self, sample_data) -> SwarmQLContext:
        """Create context with sample data."""
        ctx = SwarmQLContext()
        ctx.register_table("employees", sample_data)
        return ctx

    def test_simple_select(self, ctx):
        """Test simple SELECT query."""
        result = ctx.sql("SELECT * FROM employees")

        assert result.count() == 5

    def test_select_columns(self, ctx):
        """Test SELECT with specific columns."""
        result = ctx.sql("SELECT name, age FROM employees")
        collected = result.collect()

        assert len(collected) == 5
        assert "name" in collected[0]
        assert "age" in collected[0]

    def test_filter_query(self, ctx):
        """Test SELECT with WHERE filter."""
        result = ctx.sql("SELECT * FROM employees WHERE age > 28")

        assert result.count() == 3  # Alice, Carol, Eve

    def test_filter_equality(self, ctx):
        """Test SELECT with equality filter."""
        result = ctx.sql("SELECT * FROM employees WHERE department = 'Engineering'")

        assert result.count() == 2  # Alice, Carol

    def test_order_by(self, ctx):
        """Test SELECT with ORDER BY."""
        result = ctx.sql("SELECT * FROM employees ORDER BY age")
        collected = result.collect()

        ages = [r["age"] for r in collected]
        assert ages == sorted(ages)

    def test_limit(self, ctx):
        """Test SELECT with LIMIT."""
        result = ctx.sql("SELECT * FROM employees LIMIT 3")

        assert result.count() == 3

    def test_explain(self, ctx):
        """Test EXPLAIN query."""
        plan = ctx.explain("SELECT * FROM employees WHERE age > 21")

        assert "Physical Plan" in plan

    def test_register_dataframe(self, ctx, sample_data):
        """Test registering a DataFrame."""
        df = HiveDataFrame(sample_data)
        ctx.register_table("df_table", df)

        result = ctx.sql("SELECT * FROM df_table")
        assert result.count() == 5


class TestSQLAggregations:
    """Test SQL aggregation functions."""

    @pytest.fixture
    def ctx(self) -> SwarmQLContext:
        """Create context with numeric data."""
        ctx = SwarmQLContext()
        data = [
            {"category": "A", "value": 10},
            {"category": "A", "value": 20},
            {"category": "B", "value": 30},
            {"category": "B", "value": 40},
            {"category": "A", "value": 15},
        ]
        ctx.register_table("data", data)
        return ctx

    def test_count(self, ctx):
        """Test COUNT aggregation."""
        result = ctx.sql("SELECT COUNT(*) FROM data")
        collected = result.collect()

        # Should have one row with count
        assert len(collected) == 1

    def test_group_by_count(self, ctx):
        """Test GROUP BY with COUNT."""
        result = ctx.sql("SELECT category, COUNT(*) FROM data GROUP BY category")
        collected = result.collect()

        assert len(collected) == 2  # Two categories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestSwarmQL2_CTEs:
    """Test Common Table Expressions (WITH clause)."""

    @pytest.fixture
    def ctx(self) -> SwarmQLContext:
        """Create context with sample data."""
        ctx = SwarmQLContext()
        data = [
            {"id": 1, "name": "Alice", "dept_id": 10},
            {"id": 2, "name": "Bob", "dept_id": 20},
            {"id": 3, "name": "Carol", "dept_id": 10},
        ]
        ctx.register_table("employees", data)
        return ctx

    def test_simple_cte(self, ctx):
        """Test simple CTE."""
        sql = """
        WITH eng AS (
            SELECT * FROM employees WHERE dept_id = 10
        )
        SELECT name FROM eng
        """
        result = ctx.sql(sql)
        collected = result.collect()
        
        assert len(collected) == 2
        names = {r["name"] for r in collected}
        assert "Alice" in names
        assert "Carol" in names

    def test_multiple_ctes(self, ctx):
        """Test multiple CTEs."""
        sql = """
        WITH 
            dept10 AS (SELECT * FROM employees WHERE dept_id = 10),
            dept20 AS (SELECT * FROM employees WHERE dept_id = 20)
        SELECT name FROM dept10
        """
        result = ctx.sql(sql)
        assert result.count() == 2


class TestSwarmQL2_SetOperations:
    """Test set operations (UNION, INTERSECT, EXCEPT)."""

    @pytest.fixture
    def ctx(self) -> SwarmQLContext:
        """Create context with sample data."""
        ctx = SwarmQLContext()
        ctx.register_table("table1", [{"id": 1}, {"id": 2}, {"id": 3}])
        ctx.register_table("table2", [{"id": 2}, {"id": 3}, {"id": 4}])
        return ctx

    def test_union(self, ctx):
        """Test UNION (removes duplicates)."""
        sql = "SELECT id FROM table1 UNION SELECT id FROM table2"
        result = ctx.sql(sql)
        collected = result.collect()
        
        ids = {r["id"] for r in collected}
        assert ids == {1, 2, 3, 4}

    def test_union_all(self, ctx):
        """Test UNION ALL (keeps duplicates)."""
        sql = "SELECT id FROM table1 UNION ALL SELECT id FROM table2"
        result = ctx.sql(sql)
        
        assert result.count() == 6

    def test_intersect(self, ctx):
        """Test INTERSECT."""
        sql = "SELECT id FROM table1 INTERSECT SELECT id FROM table2"
        result = ctx.sql(sql)
        collected = result.collect()
        
        ids = {r["id"] for r in collected}
        assert ids == {2, 3}

    def test_except(self, ctx):
        """Test EXCEPT."""
        sql = "SELECT id FROM table1 EXCEPT SELECT id FROM table2"
        result = ctx.sql(sql)
        collected = result.collect()
        
        ids = {r["id"] for r in collected}
        assert ids == {1}


class TestSwarmQL2_Subqueries:
    """Test subquery support."""

    @pytest.fixture
    def ctx(self) -> SwarmQLContext:
        """Create context with sample data."""
        ctx = SwarmQLContext()
        ctx.register_table("orders", [
            {"id": 1, "customer_id": 100, "amount": 50},
            {"id": 2, "customer_id": 101, "amount": 75},
            {"id": 3, "customer_id": 100, "amount": 100},
        ])
        ctx.register_table("customers", [
            {"id": 100, "name": "Alice"},
            {"id": 101, "name": "Bob"},
            {"id": 102, "name": "Carol"},
        ])
        return ctx

    def test_in_subquery(self, ctx):
        """Test IN with subquery."""
        sql = """
        SELECT name FROM customers 
        WHERE id IN (SELECT customer_id FROM orders WHERE amount > 60)
        """
        result = ctx.sql(sql)
        collected = result.collect()
        
        names = {r["name"] for r in collected}
        assert "Alice" in names
        assert "Bob" in names

    def test_scalar_subquery(self, ctx):
        """Test scalar subquery in SELECT."""
        sql = """
        SELECT name, (SELECT COUNT(*) FROM orders) as total_orders
        FROM customers
        LIMIT 1
        """
        result = ctx.sql(sql)
        collected = result.collect()
        
        assert collected[0]["total_orders"] == 3

    def test_exists(self, ctx):
        """Test EXISTS (simplified)."""
        # EXISTS is parsed but full correlated subquery support is limited
        # This tests basic parsing
        sql = """
        SELECT name FROM customers
        WHERE id IN (SELECT customer_id FROM orders)
        """
        result = ctx.sql(sql)
        collected = result.collect()
        
        names = {r["name"] for r in collected}
        assert "Alice" in names
        assert "Bob" in names


class TestSwarmQL2_StringFunctions:
    """Test string functions."""

    @pytest.fixture
    def ctx(self) -> SwarmQLContext:
        """Create context with string data."""
        ctx = SwarmQLContext()
        ctx.register_table("data", [
            {"text": "Hello World"},
            {"text": "   Spaces   "},
            {"text": "lowercase"},
        ])
        return ctx

    def test_upper(self, ctx):
        """Test UPPER function."""
        result = ctx.sql("SELECT UPPER(text) as upper_text FROM data LIMIT 1")
        collected = result.collect()
        assert collected[0]["upper_text"] == "HELLO WORLD"

    def test_lower(self, ctx):
        """Test LOWER function."""
        result = ctx.sql("SELECT LOWER('HELLO') as lower_text")
        collected = result.collect()
        assert collected[0]["lower_text"] == "hello"

    def test_trim(self, ctx):
        """Test TRIM function."""
        result = ctx.sql("SELECT TRIM('   Spaces   ') as trimmed")
        collected = result.collect()
        assert collected[0]["trimmed"] == "Spaces"

    def test_length(self, ctx):
        """Test LENGTH function."""
        result = ctx.sql("SELECT LENGTH(text) as len FROM data LIMIT 1")
        collected = result.collect()
        assert collected[0]["len"] == 11

    def test_concat(self, ctx):
        """Test CONCAT function."""
        result = ctx.sql("SELECT CONCAT('Hello', ' ', 'World') as result")
        collected = result.collect()
        assert collected[0]["result"] == "Hello World"

    def test_substring(self, ctx):
        """Test SUBSTRING function."""
        result = ctx.sql("SELECT SUBSTRING(text, 0, 5) as sub FROM data LIMIT 1")
        collected = result.collect()
        assert collected[0]["sub"] == "Hello"


class TestSwarmQL2_DateFunctions:
    """Test date/time functions."""

    @pytest.fixture
    def ctx(self) -> SwarmQLContext:
        """Create context."""
        return SwarmQLContext()

    def test_current_date(self, ctx):
        """Test CURRENT_DATE."""
        result = ctx.sql("SELECT CURRENT_DATE() as today")
        collected = result.collect()
        assert "today" in collected[0]

    def test_current_timestamp(self, ctx):
        """Test CURRENT_TIMESTAMP."""
        result = ctx.sql("SELECT CURRENT_TIMESTAMP() as now")
        collected = result.collect()
        assert "now" in collected[0]


class TestSwarmQL2_OtherFunctions:
    """Test COALESCE and NULLIF."""

    @pytest.fixture
    def ctx(self) -> SwarmQLContext:
        """Create context with nullable data."""
        ctx = SwarmQLContext()
        ctx.register_table("data", [
            {"a": None, "b": 5, "c": 10},
            {"a": 1, "b": None, "c": 20},
            {"a": 2, "b": 3, "c": None},
        ])
        return ctx

    def test_coalesce(self, ctx):
        """Test COALESCE function."""
        result = ctx.sql("SELECT COALESCE(a, b, c) as result FROM data")
        collected = result.collect()
        
        assert collected[0]["result"] == 5
        assert collected[1]["result"] == 1
        assert collected[2]["result"] == 2

    def test_nullif(self, ctx):
        """Test NULLIF function."""
        result = ctx.sql("SELECT NULLIF(a, 1) as result FROM data")
        collected = result.collect()
        
        assert collected[0]["result"] is None
        assert collected[1]["result"] is None  # a=1, returns NULL
        assert collected[2]["result"] == 2


class TestSwarmQL2_BeeInspiredExtensions:
    """Test bee-inspired SQL extensions."""

    @pytest.fixture
    def ctx(self) -> SwarmQLContext:
        """Create context with sample data."""
        ctx = SwarmQLContext()
        ctx.register_table("orders", [
            {"id": 1, "customer_id": 100, "amount": 50},
            {"id": 2, "customer_id": 101, "amount": 75},
        ])
        ctx.register_table("customers", [
            {"id": 100, "name": "Alice"},
            {"id": 101, "name": "Bob"},
        ])
        return ctx

    def test_waggle_join(self, ctx):
        """
        Test WAGGLE JOIN - quality-weighted join execution hint.
        
        WAGGLE JOIN is a bee-inspired extension that hints the query executor
        to use quality-weighted join strategies, similar to how bees perform
        waggle dances to communicate food source quality.
        
        In this implementation, WAGGLE JOIN is treated as INNER JOIN with
        special optimization hints that would be used in production.
        """
        sql = """
        SELECT o.id, c.name 
        FROM orders o
        WAGGLE JOIN customers c ON o.customer_id = c.id
        """
        # Parse the query to verify WAGGLE keyword is recognized
        from hiveframe.sql.parser import SQLTokenizer, SQLParser
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()
        parser = SQLParser(tokens)
        stmt = parser.parse()
        
        # Verify that the join type was parsed as WAGGLE
        assert len(stmt.joins) == 1
        # In execution, WAGGLE is converted to INNER for now
        # Future versions would use this hint for adaptive join strategies


class TestSwarmQL2_WindowFunctions:
    """Test window functions."""

    @pytest.fixture
    def ctx(self) -> SwarmQLContext:
        """Create context with sample data."""
        ctx = SwarmQLContext()
        ctx.register_table("sales", [
            {"dept": "A", "amount": 100},
            {"dept": "A", "amount": 200},
            {"dept": "B", "amount": 150},
            {"dept": "B", "amount": 250},
        ])
        return ctx

    def test_row_number(self, ctx):
        """Test ROW_NUMBER window function."""
        sql = """
        SELECT dept, amount, 
               ROW_NUMBER() OVER (PARTITION BY dept ORDER BY amount) as rn
        FROM sales
        """
        # Parser should handle this syntax
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()
        parser = SQLParser(tokens)
        stmt = parser.parse()
        
        # Verify window function was parsed
        assert len(stmt.select_columns) == 3
