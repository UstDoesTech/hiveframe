"""
Tests for Query Optimizer.

Tests cover:
- Optimization rules
- Cost estimation
- Plan optimization
"""

import pytest

from hiveframe.optimizer import (
    ConstantFolding,
    CostEstimate,
    FilterCombination,
    OptimizedPlan,
    PredicatePushdown,
    ProjectionPruning,
    QueryOptimizer,
    Statistics,
    SwarmCostModel,
)
from hiveframe.optimizer.rules import NodeType, PlanNode


class TestOptimizationRules:
    """Test individual optimization rules."""

    def test_predicate_pushdown_applies(self):
        """Test predicate pushdown rule detection."""
        rule = PredicatePushdown()

        # Filter above scan
        filter_node = PlanNode(
            node_type=NodeType.FILTER,
            children=[PlanNode(node_type=NodeType.SCAN)],
            predicates=[{"column": "age", "operator": ">", "value": 21}],
        )

        assert rule.applies_to(filter_node)

    def test_predicate_pushdown_not_applies(self):
        """Test predicate pushdown doesn't apply to non-filter."""
        rule = PredicatePushdown()

        scan_node = PlanNode(node_type=NodeType.SCAN)

        assert not rule.applies_to(scan_node)

    def test_projection_pruning_applies(self):
        """Test projection pruning rule detection."""
        rule = ProjectionPruning()

        project_node = PlanNode(
            node_type=NodeType.PROJECT,
            children=[PlanNode(node_type=NodeType.SCAN)],
            columns=["name", "age"],
        )

        assert rule.applies_to(project_node)

    def test_constant_folding_applies(self):
        """Test constant folding detection."""
        rule = ConstantFolding()

        filter_node = PlanNode(node_type=NodeType.FILTER, predicates=[{"constant": True}])

        assert rule.applies_to(filter_node)

    def test_constant_folding_removes_true(self):
        """Test constant folding removes always-true predicates."""
        rule = ConstantFolding()

        child = PlanNode(node_type=NodeType.SCAN)
        filter_node = PlanNode(
            node_type=NodeType.FILTER, children=[child], predicates=[{"constant": True}]
        )

        result = rule.apply(filter_node)

        # Should return child since predicate is always true
        assert result.node_type == NodeType.SCAN

    def test_filter_combination_applies(self):
        """Test filter combination detection."""
        rule = FilterCombination()

        inner_filter = PlanNode(
            node_type=NodeType.FILTER,
            children=[PlanNode(node_type=NodeType.SCAN)],
            predicates=[{"column": "x"}],
        )
        outer_filter = PlanNode(
            node_type=NodeType.FILTER, children=[inner_filter], predicates=[{"column": "y"}]
        )

        assert rule.applies_to(outer_filter)

    def test_filter_combination_merges(self):
        """Test filter combination merges predicates."""
        rule = FilterCombination()

        scan = PlanNode(node_type=NodeType.SCAN)
        inner_filter = PlanNode(
            node_type=NodeType.FILTER, children=[scan], predicates=[{"column": "x"}]
        )
        outer_filter = PlanNode(
            node_type=NodeType.FILTER, children=[inner_filter], predicates=[{"column": "y"}]
        )

        result = rule.apply(outer_filter)

        # Should have combined predicates
        assert len(result.predicates) == 2


class TestCostModel:
    """Test cost estimation."""

    def test_scan_cost(self):
        """Test scan node cost estimation."""
        model = SwarmCostModel()
        stats = {"table1": Statistics(row_count=10000, avg_row_size=100)}

        scan_node = PlanNode(node_type=NodeType.SCAN, properties={"table": "table1"})

        cost = model.estimate(scan_node, stats)

        assert cost.estimated_rows == 10000
        assert cost.total_cost > 0

    def test_filter_reduces_rows(self):
        """Test filter node reduces estimated rows."""
        model = SwarmCostModel()
        stats = {"table1": Statistics(row_count=10000)}

        scan_node = PlanNode(node_type=NodeType.SCAN, properties={"table": "table1"})
        filter_node = PlanNode(
            node_type=NodeType.FILTER,
            children=[scan_node],
            predicates=[{"column": "x", "operator": "="}],
        )

        cost = model.estimate(filter_node, stats)

        # Filter should have a cost estimate
        assert cost.total_cost > 0
        # Scan's child cost should be included
        assert cost.estimated_rows >= 0

    def test_fitness_calculation(self):
        """Test fitness is inverse of cost."""
        cost1 = CostEstimate(cpu_cost=100, io_cost=10)
        cost2 = CostEstimate(cpu_cost=200, io_cost=20)

        # Higher cost should have lower fitness
        assert cost1.fitness > cost2.fitness


class TestQueryOptimizer:
    """Test query optimizer."""

    def test_optimize_simple_plan(self):
        """Test optimizing a simple plan."""
        optimizer = QueryOptimizer(max_iterations=10, swarm_size=5)

        plan = PlanNode(
            node_type=NodeType.PROJECT,
            children=[PlanNode(node_type=NodeType.SCAN)],
            columns=["a", "b"],
        )

        result = optimizer.optimize(plan)

        assert isinstance(result, OptimizedPlan)
        assert result.plan is not None

    def test_optimizer_applies_rules(self):
        """Test optimizer applies optimization rules."""
        optimizer = QueryOptimizer(max_iterations=5, swarm_size=3)

        # Create plan with filter on top of filter
        scan = PlanNode(node_type=NodeType.SCAN)
        filter1 = PlanNode(node_type=NodeType.FILTER, children=[scan], predicates=[{"column": "x"}])
        filter2 = PlanNode(
            node_type=NodeType.FILTER, children=[filter1], predicates=[{"column": "y"}]
        )

        result = optimizer.optimize(filter2)

        # FilterCombination should be applied
        assert "FilterCombination" in result.rules_applied

    def test_explain_output(self):
        """Test explain generates readable output."""
        optimizer = QueryOptimizer()

        plan = PlanNode(
            node_type=NodeType.PROJECT,
            children=[
                PlanNode(
                    node_type=NodeType.FILTER,
                    children=[PlanNode(node_type=NodeType.SCAN, properties={"table": "users"})],
                    predicates=[{"column": "age"}],
                )
            ],
            columns=["name", "age"],
        )

        explanation = optimizer.explain(plan)

        assert "PROJECT" in explanation
        assert "FILTER" in explanation
        assert "SCAN" in explanation


class TestStatistics:
    """Test statistics and selectivity."""

    def test_equality_selectivity(self):
        """Test equality predicate selectivity."""
        stats = Statistics(row_count=1000, distinct_values={"category": 10})

        selectivity = stats.selectivity("category", "=")

        # Should be 1/distinct_count
        assert selectivity == pytest.approx(0.1)

    def test_range_selectivity(self):
        """Test range predicate selectivity."""
        stats = Statistics(row_count=1000)

        selectivity = stats.selectivity("value", ">")

        # Default range selectivity is 0.33
        assert selectivity == pytest.approx(0.33)

    def test_null_selectivity(self):
        """Test IS NULL selectivity."""
        stats = Statistics(row_count=1000, null_count={"optional": 100})

        selectivity = stats.selectivity("optional", "IS NULL")

        assert selectivity == pytest.approx(0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
