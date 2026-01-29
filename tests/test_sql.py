"""
Tests for SwarmQL SQL Engine.

Tests cover:
- SQL parsing and tokenization
- Query execution
- Table registration and catalog
- Common SQL operations
"""

import pytest
from typing import List, Dict, Any

from hiveframe import HiveDataFrame
from hiveframe.sql import (
    SwarmQLContext,
    SQLParser,
    SQLTokenizer,
    SQLCatalog,
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
        
        assert any(t.value.upper() == 'WHERE' for t in tokens)
        
    def test_string_literals(self):
        """Test tokenizing string literals."""
        sql = "SELECT * FROM users WHERE name = 'Alice'"
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()
        
        # Should have a STRING token with value 'Alice'
        string_tokens = [t for t in tokens if t.type.name == 'STRING']
        assert len(string_tokens) == 1
        assert string_tokens[0].value == 'Alice'
        
    def test_numeric_literals(self):
        """Test tokenizing numeric literals."""
        sql = "SELECT * FROM data WHERE value > 3.14"
        tokenizer = SQLTokenizer(sql)
        tokens = tokenizer.tokenize()
        
        number_tokens = [t for t in tokens if t.type.name == 'NUMBER']
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
        assert stmt.from_table.name == 'users'
        
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
        assert stmt.order_by[0].ascending == False
        
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
        df = HiveDataFrame([{'id': 1}])
        
        catalog.register_table('test_table', df)
        
        assert catalog.table_exists('test_table')
        
    def test_get_table(self):
        """Test getting a registered table."""
        catalog = SQLCatalog()
        df = HiveDataFrame([{'id': 1, 'name': 'test'}])
        catalog.register_table('my_table', df)
        
        result = catalog.get_table('my_table')
        
        assert result is not None
        assert result.count() == 1
        
    def test_list_tables(self):
        """Test listing tables."""
        catalog = SQLCatalog()
        catalog.register_table('table1', HiveDataFrame([{'a': 1}]))
        catalog.register_table('table2', HiveDataFrame([{'b': 2}]))
        
        tables = catalog.list_tables()
        
        assert 'table1' in tables
        assert 'table2' in tables
        
    def test_drop_table(self):
        """Test dropping a table."""
        catalog = SQLCatalog()
        catalog.register_table('temp', HiveDataFrame([{'x': 1}]))
        
        catalog.drop_table('temp')
        
        assert not catalog.table_exists('temp')


class TestSwarmQLContext:
    """Test SwarmQL context operations."""
    
    @pytest.fixture
    def sample_data(self) -> List[Dict[str, Any]]:
        """Sample data for testing."""
        return [
            {'id': 1, 'name': 'Alice', 'age': 30, 'department': 'Engineering'},
            {'id': 2, 'name': 'Bob', 'age': 25, 'department': 'Marketing'},
            {'id': 3, 'name': 'Carol', 'age': 35, 'department': 'Engineering'},
            {'id': 4, 'name': 'David', 'age': 28, 'department': 'Sales'},
            {'id': 5, 'name': 'Eve', 'age': 32, 'department': 'Marketing'},
        ]
        
    @pytest.fixture
    def ctx(self, sample_data) -> SwarmQLContext:
        """Create context with sample data."""
        ctx = SwarmQLContext()
        ctx.register_table('employees', sample_data)
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
        assert 'name' in collected[0]
        assert 'age' in collected[0]
        
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
        
        ages = [r['age'] for r in collected]
        assert ages == sorted(ages)
        
    def test_limit(self, ctx):
        """Test SELECT with LIMIT."""
        result = ctx.sql("SELECT * FROM employees LIMIT 3")
        
        assert result.count() == 3
        
    def test_explain(self, ctx):
        """Test EXPLAIN query."""
        plan = ctx.explain("SELECT * FROM employees WHERE age > 21")
        
        assert 'Physical Plan' in plan
        
    def test_register_dataframe(self, ctx, sample_data):
        """Test registering a DataFrame."""
        df = HiveDataFrame(sample_data)
        ctx.register_table('df_table', df)
        
        result = ctx.sql("SELECT * FROM df_table")
        assert result.count() == 5


class TestSQLAggregations:
    """Test SQL aggregation functions."""
    
    @pytest.fixture
    def ctx(self) -> SwarmQLContext:
        """Create context with numeric data."""
        ctx = SwarmQLContext()
        data = [
            {'category': 'A', 'value': 10},
            {'category': 'A', 'value': 20},
            {'category': 'B', 'value': 30},
            {'category': 'B', 'value': 40},
            {'category': 'A', 'value': 15},
        ]
        ctx.register_table('data', data)
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
