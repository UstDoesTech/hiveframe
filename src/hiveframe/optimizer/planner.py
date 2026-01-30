"""
Query Optimizer
===============

Main query optimizer that combines rule-based and cost-based optimization
using bee-inspired swarm intelligence.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .cost import CostEstimate, CostModel, Statistics, SwarmCostModel
from .rules import (
    NodeType,
    OptimizationRule,
    PlanNode,
    get_default_rules,
)


@dataclass
class PlanCandidate:
    """
    Plan Candidate
    --------------
    A candidate query plan with its fitness score.
    Used in the swarm optimization process.
    """

    plan: PlanNode
    cost: CostEstimate
    fitness: float = 0.0
    trials: int = 0  # For ABC abandonment

    def __post_init__(self):
        self.fitness = self.cost.fitness


@dataclass
class OptimizedPlan:
    """
    Optimized Plan
    --------------
    Result of query optimization.
    """

    plan: PlanNode
    cost: CostEstimate
    rules_applied: List[str] = field(default_factory=list)
    iterations: int = 0
    candidates_explored: int = 0


class QueryOptimizer:
    """
    Query Optimizer
    ---------------

    A bee-inspired query optimizer that combines:
    1. Rule-based optimization (like predicate pushdown)
    2. Cost-based optimization using fitness functions
    3. Swarm search for exploring plan space

    The optimizer uses the ABC (Artificial Bee Colony) algorithm
    to search for optimal query plans, treating each plan as a
    "food source" with a fitness value.

    Usage:
        optimizer = QueryOptimizer()
        optimized = optimizer.optimize(plan, stats)

        # The optimizer explores multiple plans and returns the best
        print(f"Cost reduced by {optimized.cost.total_cost}")
    """

    def __init__(
        self,
        rules: Optional[List[OptimizationRule]] = None,
        cost_model: Optional[CostModel] = None,
        max_iterations: int = 50,
        swarm_size: int = 20,
        abandonment_limit: int = 10,
    ):
        """
        Initialize optimizer.

        Args:
            rules: Optimization rules to apply
            cost_model: Cost estimation model
            max_iterations: Maximum optimization iterations
            swarm_size: Number of candidate plans in swarm
            abandonment_limit: Iterations before abandoning a plan
        """
        self.rules = rules or get_default_rules()
        self.cost_model = cost_model or SwarmCostModel()
        self.max_iterations = max_iterations
        self.swarm_size = swarm_size
        self.abandonment_limit = abandonment_limit

    def optimize(
        self, plan: PlanNode, stats: Optional[Dict[str, Statistics]] = None
    ) -> OptimizedPlan:
        """
        Optimize a query plan.

        Args:
            plan: Input logical plan
            stats: Table statistics for cost estimation

        Returns:
            Optimized plan with cost information
        """
        stats = stats or {}
        rules_applied = []

        # Phase 1: Rule-based optimization
        current_plan = plan
        for rule in self.rules:
            new_plan = self._apply_rule_recursive(current_plan, rule)
            if new_plan != current_plan:
                rules_applied.append(rule.name)
                current_plan = new_plan

        # Phase 2: Cost-based optimization using ABC algorithm
        best_candidate, iterations, explored = self._swarm_optimize(current_plan, stats)

        return OptimizedPlan(
            plan=best_candidate.plan,
            cost=best_candidate.cost,
            rules_applied=rules_applied,
            iterations=iterations,
            candidates_explored=explored,
        )

    def _apply_rule_recursive(self, node: PlanNode, rule: OptimizationRule) -> PlanNode:
        """Apply a rule recursively to plan tree."""
        # Apply to children first (bottom-up)
        new_children = [self._apply_rule_recursive(child, rule) for child in node.children]

        # Create new node with transformed children
        new_node = PlanNode(
            node_type=node.node_type,
            children=new_children,
            columns=node.columns.copy(),
            predicates=node.predicates.copy(),
            properties=node.properties.copy(),
        )

        # Apply rule to this node
        if rule.applies_to(new_node):
            return rule.apply(new_node)

        return new_node

    def _swarm_optimize(self, plan: PlanNode, stats: Dict[str, Statistics]) -> tuple:
        """
        Optimize using ABC (Artificial Bee Colony) algorithm.

        The ABC algorithm has three phases:
        1. Employed Bee Phase: Exploit current solutions
        2. Onlooker Bee Phase: Reinforce good solutions
        3. Scout Bee Phase: Explore new territory

        Returns:
            Tuple of (best_candidate, iterations, candidates_explored)
        """
        # Initialize swarm with variations of the plan
        candidates = self._initialize_swarm(plan, stats)

        best_candidate = max(candidates, key=lambda c: c.fitness)
        iterations = 0
        explored = len(candidates)

        for iteration in range(self.max_iterations):
            # Employed Bee Phase: Local search around each solution
            for i, candidate in enumerate(candidates):
                neighbor = self._generate_neighbor(candidate.plan, stats)
                neighbor_cost = self.cost_model.estimate(neighbor, stats)
                neighbor_candidate = PlanCandidate(neighbor, neighbor_cost)

                if neighbor_candidate.fitness > candidate.fitness:
                    candidates[i] = neighbor_candidate
                    candidate.trials = 0
                else:
                    candidate.trials += 1
                explored += 1

            # Onlooker Bee Phase: Probabilistic selection
            total_fitness = sum(c.fitness for c in candidates)
            if total_fitness > 0:
                for _ in range(len(candidates)):
                    # Roulette wheel selection
                    r = random.uniform(0, total_fitness)
                    cumsum = 0
                    for i, candidate in enumerate(candidates):
                        cumsum += candidate.fitness
                        if cumsum >= r:
                            # Generate neighbor
                            neighbor = self._generate_neighbor(candidate.plan, stats)
                            neighbor_cost = self.cost_model.estimate(neighbor, stats)
                            neighbor_candidate = PlanCandidate(neighbor, neighbor_cost)

                            if neighbor_candidate.fitness > candidate.fitness:
                                candidates[i] = neighbor_candidate
                            explored += 1
                            break

            # Scout Bee Phase: Replace abandoned solutions
            for i, candidate in enumerate(candidates):
                if candidate.trials >= self.abandonment_limit:
                    # Generate new random plan variation
                    new_plan = self._generate_random_variation(plan, stats)
                    new_cost = self.cost_model.estimate(new_plan, stats)
                    candidates[i] = PlanCandidate(new_plan, new_cost)
                    explored += 1

            # Update best
            current_best = max(candidates, key=lambda c: c.fitness)
            if current_best.fitness > best_candidate.fitness:
                best_candidate = current_best

            iterations += 1

            # Early termination if converged
            fitness_variance = self._calculate_variance([c.fitness for c in candidates])
            if fitness_variance < 0.0001:
                break

        return best_candidate, iterations, explored

    def _initialize_swarm(
        self, plan: PlanNode, stats: Dict[str, Statistics]
    ) -> List[PlanCandidate]:
        """Initialize swarm with plan variations."""
        candidates = []

        # Original plan
        cost = self.cost_model.estimate(plan, stats)
        candidates.append(PlanCandidate(plan, cost))

        # Generate variations
        for _ in range(self.swarm_size - 1):
            variation = self._generate_random_variation(plan, stats)
            var_cost = self.cost_model.estimate(variation, stats)
            candidates.append(PlanCandidate(variation, var_cost))

        return candidates

    def _generate_neighbor(self, plan: PlanNode, stats: Dict[str, Statistics]) -> PlanNode:
        """Generate a neighboring plan (local search)."""
        # Apply a random transformation
        transformations = [
            self._swap_join_order,
            self._move_predicate,
            self._change_join_algorithm,
        ]

        transform = random.choice(transformations)
        return transform(plan)

    def _generate_random_variation(self, plan: PlanNode, stats: Dict[str, Statistics]) -> PlanNode:
        """Generate a random plan variation (exploration)."""
        # Apply multiple random transformations
        result = plan.copy()
        num_transforms = random.randint(1, 3)

        for _ in range(num_transforms):
            result = self._generate_neighbor(result, stats)

        return result

    def _swap_join_order(self, plan: PlanNode) -> PlanNode:
        """Swap join order at a random join node."""
        result = plan.copy()

        # Find join nodes
        joins = self._find_nodes(result, NodeType.JOIN)

        if joins:
            join_node = random.choice(joins)
            if len(join_node.children) >= 2:
                # Swap children
                join_node.children[0], join_node.children[1] = (
                    join_node.children[1],
                    join_node.children[0],
                )

        return result

    def _move_predicate(self, plan: PlanNode) -> PlanNode:
        """Move a predicate up or down in the plan."""
        result = plan.copy()

        # Find filter nodes
        filters = self._find_nodes(result, NodeType.FILTER)

        if filters and len(filters) > 1:
            # Swap predicates between two filters
            f1, f2 = random.sample(filters, 2)
            if f1.predicates and f2.predicates:
                # Exchange one predicate
                p1 = f1.predicates.pop()
                p2 = f2.predicates.pop()
                f1.predicates.append(p2)
                f2.predicates.append(p1)

        return result

    def _change_join_algorithm(self, plan: PlanNode) -> PlanNode:
        """Change join algorithm for a join node."""
        result = plan.copy()

        joins = self._find_nodes(result, NodeType.JOIN)

        if joins:
            join_node = random.choice(joins)
            algorithms = ["hash", "sort_merge", "nested_loop"]
            current = join_node.properties.get("join_algorithm", "hash")
            new_alg = random.choice([a for a in algorithms if a != current])
            join_node.properties["join_algorithm"] = new_alg

        return result

    def _find_nodes(self, plan: PlanNode, node_type: NodeType) -> List[PlanNode]:
        """Find all nodes of a specific type."""
        result = []

        if plan.node_type == node_type:
            result.append(plan)

        for child in plan.children:
            result.extend(self._find_nodes(child, node_type))

        return result

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def explain(self, plan: PlanNode, indent: int = 0) -> str:
        """Generate human-readable plan explanation."""
        lines = []
        prefix = "  " * indent

        # Node type and key properties
        node_str = f"{prefix}* {plan.node_type.name}"

        if plan.columns:
            node_str += f" [{', '.join(plan.columns[:5])}{'...' if len(plan.columns) > 5 else ''}]"

        if plan.predicates:
            node_str += f" (filters: {len(plan.predicates)})"

        if "table" in plan.properties:
            node_str += f" table={plan.properties['table']}"

        if "join_algorithm" in plan.properties:
            node_str += f" algorithm={plan.properties['join_algorithm']}"

        lines.append(node_str)

        # Recurse to children
        for child in plan.children:
            lines.append(self.explain(child, indent + 1))

        return "\n".join(lines)
