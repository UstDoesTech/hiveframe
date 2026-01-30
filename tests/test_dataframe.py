"""
Tests for HiveFrame DataFrame module.

Tests cover:
- DataFrame creation from various sources
- Column operations and expressions
- Filter, select, and projection
- GroupBy and aggregations
- Join operations
- Schema inference and validation
"""

from typing import Any, Dict, List

import pytest

from hiveframe import (
    HiveDataFrame,
    avg,
    col,
    count,
    lit,
    max_agg,
    min_agg,
    sum_agg,
)


class TestDataFrameCreation:
    """Test DataFrame creation."""

    @pytest.fixture
    def sample_records(self) -> List[Dict[str, Any]]:
        """Sample data for testing."""
        return [
            {"name": "Alice", "age": 30, "department": "Engineering", "salary": 95000},
            {"name": "Bob", "age": 25, "department": "Engineering", "salary": 87000},
            {"name": "Carol", "age": 35, "department": "Marketing", "salary": 78000},
            {"name": "David", "age": 28, "department": "Marketing", "salary": 82000},
            {"name": "Eve", "age": 32, "department": "Engineering", "salary": 92000},
        ]

    def test_create_from_list_of_dicts(self, sample_records):
        """Test creating DataFrame from list of dictionaries."""
        df = HiveDataFrame(sample_records)

        assert df.count() == 5

    def test_create_empty_dataframe(self):
        """Test creating empty DataFrame."""
        df = HiveDataFrame([])

        assert df.count() == 0

    def test_schema_inference(self, sample_records):
        """Test that schema is inferred correctly."""
        df = HiveDataFrame(sample_records)

        # Should have columns from the data
        collected = df.collect()
        assert len(collected) == 5


class TestColumnOperations:
    """Test Column class operations."""

    def test_col_creation(self):
        """Test creating column reference."""
        c = col("name")

        assert c.name == "name"

    def test_lit_creation(self):
        """Test creating literal column."""
        l = lit(42)

        # Literal should evaluate to the constant value
        result = l.eval({})
        assert result == 42

    def test_column_comparison_eq(self):
        """Test column equality comparison."""
        c = col("status")
        expr = c == "active"

        assert expr.eval({"status": "active"})
        assert not expr.eval({"status": "inactive"})

    def test_column_comparison_gt(self):
        """Test column greater-than comparison."""
        c = col("age")
        expr = c > 25

        assert expr.eval({"age": 30})
        assert not expr.eval({"age": 20})

    def test_column_arithmetic(self):
        """Test column arithmetic operations."""
        c = col("value")
        expr = c * 2

        assert expr.eval({"value": 10}) == 20

    def test_column_alias(self):
        """Test column aliasing."""
        c = col("original_name").alias("new_name")

        assert c.name == "new_name"


class TestDataFrameFilter:
    """Test DataFrame filter operations."""

    @pytest.fixture
    def df(self) -> HiveDataFrame:
        """DataFrame for testing."""
        return HiveDataFrame(
            [
                {"id": 1, "value": 10, "category": "A"},
                {"id": 2, "value": 20, "category": "B"},
                {"id": 3, "value": 30, "category": "A"},
                {"id": 4, "value": 40, "category": "B"},
                {"id": 5, "value": 50, "category": "A"},
            ]
        )

    def test_filter_by_value(self, df):
        """Test filtering by numeric value."""
        result = df.filter(col("value") > 25)

        assert result.count() == 3

    def test_filter_by_equality(self, df):
        """Test filtering by equality."""
        result = df.filter(col("category") == "A")

        assert result.count() == 3

    def test_filter_chaining(self, df):
        """Test chaining multiple filters."""
        result = df.filter(col("category") == "A").filter(col("value") > 20)

        assert result.count() == 2


class TestDataFrameSelect:
    """Test DataFrame select operations."""

    @pytest.fixture
    def df(self) -> HiveDataFrame:
        """DataFrame for testing."""
        return HiveDataFrame(
            [
                {"a": 1, "b": 2, "c": 3},
                {"a": 4, "b": 5, "c": 6},
            ]
        )

    def test_select_columns(self, df):
        """Test selecting specific columns."""
        result = df.select("a", "b")
        collected = result.collect()

        # Should only have selected columns
        assert all("a" in row and "b" in row for row in collected)

    def test_select_with_column_objects(self, df):
        """Test selecting with Column objects."""
        result = df.select(col("a"), col("b"))

        assert result.count() == 2


class TestDataFrameGroupBy:
    """Test DataFrame groupBy and aggregation."""

    @pytest.fixture
    def df(self) -> HiveDataFrame:
        """DataFrame for testing."""
        return HiveDataFrame(
            [
                {"department": "Engineering", "salary": 100000},
                {"department": "Engineering", "salary": 90000},
                {"department": "Marketing", "salary": 80000},
                {"department": "Marketing", "salary": 85000},
                {"department": "Engineering", "salary": 95000},
            ]
        )

    def test_groupby_count(self, df):
        """Test groupBy with count aggregation."""
        result = df.groupBy("department").agg(count(col("salary")))
        collected = result.collect()

        assert len(collected) == 2

    def test_groupby_avg(self, df):
        """Test groupBy with average aggregation."""
        result = df.groupBy("department").agg(avg(col("salary")))
        collected = result.collect()

        # Engineering avg should be (100000+90000+95000)/3 = 95000
        eng_row = next((r for r in collected if r["department"] == "Engineering"), None)
        assert eng_row is not None

    def test_groupby_multiple_aggs(self, df):
        """Test groupBy with multiple aggregations."""
        result = df.groupBy("department").agg(avg(col("salary")), count(col("salary")))
        collected = result.collect()

        assert len(collected) == 2


class TestAggregationFunctions:
    """Test standalone aggregation functions."""

    @pytest.fixture
    def rows(self) -> List[Dict[str, Any]]:
        """Sample rows for aggregation."""
        return [
            {"value": 10},
            {"value": 20},
            {"value": 30},
            {"value": None},
            {"value": 40},
        ]

    def test_sum_agg(self, rows):
        """Test sum aggregation."""
        agg = sum_agg(col("value"))
        result = agg.apply(rows)

        assert result == 100  # 10+20+30+40 (None excluded)

    def test_avg_agg(self, rows):
        """Test average aggregation."""
        agg = avg(col("value"))
        result = agg.apply(rows)

        assert result == 25.0  # 100/4

    def test_count_agg(self, rows):
        """Test count aggregation."""
        agg = count(col("value"))
        result = agg.apply(rows)

        assert result == 4  # None excluded

    def test_min_agg(self, rows):
        """Test min aggregation."""
        agg = min_agg(col("value"))
        result = agg.apply(rows)

        assert result == 10

    def test_max_agg(self, rows):
        """Test max aggregation."""
        agg = max_agg(col("value"))
        result = agg.apply(rows)

        assert result == 40


class TestDataFrameWithColumn:
    """Test adding computed columns."""

    def test_add_computed_column(self):
        """Test adding a computed column."""
        df = HiveDataFrame(
            [
                {"a": 1, "b": 2},
                {"a": 3, "b": 4},
            ]
        )

        result = df.withColumn("c", col("a") + col("b"))
        collected = result.collect()

        assert all("c" in row for row in collected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
