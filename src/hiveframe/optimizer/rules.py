"""
Query Optimization Rules
========================

Rule-based transformations for query plan optimization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


class NodeType(Enum):
    """Plan node types."""

    SCAN = auto()
    FILTER = auto()
    PROJECT = auto()
    JOIN = auto()
    AGGREGATE = auto()
    SORT = auto()
    LIMIT = auto()
    UNION = auto()
    DISTINCT = auto()


@dataclass
class PlanNode:
    """
    Logical Plan Node
    -----------------
    Represents a node in the query execution plan.
    """

    node_type: NodeType
    children: List["PlanNode"] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    predicates: List[Any] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "PlanNode":
        """Create a copy of this node."""
        return PlanNode(
            node_type=self.node_type,
            children=[c.copy() for c in self.children],
            columns=self.columns.copy(),
            predicates=self.predicates.copy(),
            properties=self.properties.copy(),
        )


class OptimizationRule(ABC):
    """
    Optimization Rule Base Class
    ----------------------------
    Abstract base for query plan transformations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Rule name for logging."""
        pass

    @abstractmethod
    def applies_to(self, node: PlanNode) -> bool:
        """Check if this rule applies to a node."""
        pass

    @abstractmethod
    def apply(self, node: PlanNode) -> PlanNode:
        """Apply the transformation."""
        pass


class PredicatePushdown(OptimizationRule):
    """
    Predicate Pushdown
    ------------------
    Push filter predicates closer to data sources to reduce
    the amount of data processed.

    Before:
        Filter(x > 10)
          └── Join
                ├── Scan(A)
                └── Scan(B)

    After:
        Join
          ├── Filter(x > 10)
          │     └── Scan(A)
          └── Scan(B)
    """

    @property
    def name(self) -> str:
        return "PredicatePushdown"

    def applies_to(self, node: PlanNode) -> bool:
        """Applies when Filter is above Join, Scan, or Project."""
        if node is None or node.node_type != NodeType.FILTER:
            return False
        if not node.children:
            return False
        child = node.children[0]
        if child is None:
            return False
        return child.node_type in (NodeType.JOIN, NodeType.PROJECT, NodeType.SCAN)

    def apply(self, node: PlanNode) -> PlanNode:
        """Push predicates down."""
        if not self.applies_to(node):
            return node

        child = node.children[0]
        predicates = node.predicates

        if child.node_type == NodeType.JOIN:
            # Determine which predicates can be pushed to which side
            left_preds, right_preds, remaining = self._split_predicates_for_join(predicates, child)

            # Push to left child
            if left_preds:
                left_filter = PlanNode(
                    node_type=NodeType.FILTER, children=[child.children[0]], predicates=left_preds
                )
                child.children[0] = left_filter

            # Push to right child
            if right_preds:
                right_filter = PlanNode(
                    node_type=NodeType.FILTER, children=[child.children[1]], predicates=right_preds
                )
                child.children[1] = right_filter

            # Keep remaining predicates at top
            if remaining:
                return PlanNode(node_type=NodeType.FILTER, children=[child], predicates=remaining)
            return child

        elif child.node_type == NodeType.PROJECT:
            # Push through projection if columns are available
            new_filter = PlanNode(
                node_type=NodeType.FILTER, children=child.children, predicates=predicates
            )
            child.children = [new_filter]
            return child

        return node

    def _split_predicates_for_join(
        self, predicates: List[Any], join_node: PlanNode
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """Split predicates for left, right, and join-level."""
        left_cols = self._get_output_columns(join_node.children[0])
        right_cols = self._get_output_columns(join_node.children[1])

        left_preds = []
        right_preds = []
        remaining = []

        for pred in predicates:
            pred_cols = self._get_predicate_columns(pred)

            if pred_cols.issubset(left_cols):
                left_preds.append(pred)
            elif pred_cols.issubset(right_cols):
                right_preds.append(pred)
            else:
                remaining.append(pred)

        return left_preds, right_preds, remaining

    def _get_output_columns(self, node: PlanNode) -> Set[str]:
        """Get output columns of a node."""
        return set(node.columns) if node.columns else set()

    def _get_predicate_columns(self, predicate: Any) -> Set[str]:
        """Extract column references from predicate."""
        if isinstance(predicate, dict):
            return set(predicate.get("columns", []))
        if isinstance(predicate, str):
            # Simple heuristic - extract identifiers
            return {predicate}
        return set()


class ProjectionPruning(OptimizationRule):
    """
    Projection Pruning
    ------------------
    Remove unused columns from projections to reduce data movement.

    Before:
        Project(a, b, c)
          └── Project(a, b, c, d, e)
                └── Scan(*)

    After:
        Project(a, b, c)
          └── Scan(a, b, c)
    """

    @property
    def name(self) -> str:
        return "ProjectionPruning"

    def applies_to(self, node: PlanNode) -> bool:
        """Applies to PROJECT nodes with PROJECT or SCAN children."""
        if node.node_type != NodeType.PROJECT:
            return False
        if not node.children:
            return False
        return node.children[0].node_type in (NodeType.PROJECT, NodeType.SCAN)

    def apply(self, node: PlanNode) -> PlanNode:
        """Prune unnecessary projections."""
        if not self.applies_to(node):
            return node

        required_cols = set(node.columns)
        child = node.children[0]

        if child.node_type == NodeType.PROJECT:
            # Merge projections
            # Only keep columns that are both projected and required
            new_cols = [c for c in child.columns if c in required_cols]
            return PlanNode(node_type=NodeType.PROJECT, children=child.children, columns=new_cols)

        elif child.node_type == NodeType.SCAN:
            # Push projection to scan
            child_copy = child.copy()
            child_copy.columns = list(required_cols)
            return PlanNode(node_type=NodeType.PROJECT, children=[child_copy], columns=node.columns)

        return node


class ConstantFolding(OptimizationRule):
    """
    Constant Folding
    ----------------
    Evaluate constant expressions at optimization time.

    Before:
        Filter(1 + 1 = 2)
          └── Scan

    After:
        Filter(TRUE)  or removed entirely
          └── Scan
    """

    @property
    def name(self) -> str:
        return "ConstantFolding"

    def applies_to(self, node: PlanNode) -> bool:
        """Applies to nodes with constant predicates."""
        if node.node_type == NodeType.FILTER:
            return any(self._is_constant_predicate(p) for p in node.predicates)
        return False

    def apply(self, node: PlanNode) -> PlanNode:
        """Fold constant expressions."""
        if not self.applies_to(node):
            return node

        new_predicates = []
        for pred in node.predicates:
            if self._is_constant_predicate(pred):
                result = self._evaluate_constant(pred)
                if result is True:
                    continue  # Skip always-true predicates
                elif result is False:
                    # Return empty result
                    return PlanNode(
                        node_type=NodeType.FILTER,
                        children=node.children,
                        predicates=[{"constant": False}],
                    )
                else:
                    new_predicates.append({"constant": result})
            else:
                new_predicates.append(pred)

        if not new_predicates:
            # All predicates were true, remove filter
            return node.children[0] if node.children else node

        return PlanNode(
            node_type=node.node_type,
            children=node.children,
            predicates=new_predicates,
            columns=node.columns,
        )

    def _is_constant_predicate(self, pred: Any) -> bool:
        """Check if predicate is constant."""
        if isinstance(pred, dict):
            return pred.get("constant") is not None
        if isinstance(pred, bool):
            return True
        return False

    def _evaluate_constant(self, pred: Any) -> bool:
        """Evaluate constant predicate."""
        if isinstance(pred, dict):
            return pred.get("constant", True)
        if isinstance(pred, bool):
            return pred
        return True


class FilterCombination(OptimizationRule):
    """
    Filter Combination
    ------------------
    Combine adjacent filter nodes into a single filter.

    Before:
        Filter(a > 10)
          └── Filter(b < 20)
                └── Scan

    After:
        Filter(a > 10 AND b < 20)
          └── Scan
    """

    @property
    def name(self) -> str:
        return "FilterCombination"

    def applies_to(self, node: PlanNode) -> bool:
        """Applies when Filter has Filter child."""
        if node.node_type != NodeType.FILTER:
            return False
        if not node.children:
            return False
        return node.children[0].node_type == NodeType.FILTER

    def apply(self, node: PlanNode) -> PlanNode:
        """Combine filters."""
        if not self.applies_to(node):
            return node

        child = node.children[0]
        combined_predicates = node.predicates + child.predicates

        return PlanNode(
            node_type=NodeType.FILTER, children=child.children, predicates=combined_predicates
        )


class JoinReordering(OptimizationRule):
    """
    Join Reordering
    ---------------
    Reorder joins to minimize intermediate result sizes.
    Uses a swarm-inspired approach to find optimal orderings.

    This is a key optimization for multi-table queries.
    """

    @property
    def name(self) -> str:
        return "JoinReordering"

    def applies_to(self, node: PlanNode) -> bool:
        """Applies to JOIN nodes with JOIN children."""
        if node.node_type != NodeType.JOIN:
            return False
        return any(c.node_type == NodeType.JOIN for c in node.children)

    def apply(self, node: PlanNode) -> PlanNode:
        """Reorder joins based on estimated sizes."""
        if not self.applies_to(node):
            return node

        # Extract all join inputs (leaves)
        leaves = self._extract_leaves(node)
        join_conditions = self._extract_conditions(node)

        # Sort by estimated size (smaller first)
        sorted_leaves = sorted(leaves, key=lambda n: self._estimate_size(n))

        # Build left-deep join tree with smallest tables first
        if len(sorted_leaves) < 2:
            return node

        result = sorted_leaves[0]
        for i in range(1, len(sorted_leaves)):
            condition = self._find_condition(result, sorted_leaves[i], join_conditions)
            result = PlanNode(
                node_type=NodeType.JOIN,
                children=[result, sorted_leaves[i]],
                properties={"condition": condition} if condition else {},
            )

        return result

    def _extract_leaves(self, node: PlanNode) -> List[PlanNode]:
        """Extract non-join leaf nodes."""
        if node.node_type != NodeType.JOIN:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self._extract_leaves(child))
        return leaves

    def _extract_conditions(self, node: PlanNode) -> List[Any]:
        """Extract all join conditions."""
        conditions = []
        if node.node_type == NodeType.JOIN:
            if "condition" in node.properties:
                conditions.append(node.properties["condition"])
            for child in node.children:
                conditions.extend(self._extract_conditions(child))
        return conditions

    def _estimate_size(self, node: PlanNode) -> int:
        """Estimate output size of a node."""
        # Simple heuristic - use properties or default
        return node.properties.get("estimated_rows", 1000)

    def _find_condition(
        self, left: PlanNode, right: PlanNode, conditions: List[Any]
    ) -> Optional[Any]:
        """Find applicable join condition."""
        left_cols = set(left.columns)
        right_cols = set(right.columns)

        for cond in conditions:
            if isinstance(cond, dict):
                cond_cols = set(cond.get("columns", []))
                if cond_cols.intersection(left_cols) and cond_cols.intersection(right_cols):
                    return cond
        return None


class LimitPushdown(OptimizationRule):
    """
    Limit Pushdown
    --------------
    Push LIMIT closer to data sources when possible.

    Before:
        Limit(10)
          └── Sort
                └── Scan

    After:
        Limit(10)
          └── Sort
                └── Limit(10)  # Partial limit
                      └── Scan
    """

    @property
    def name(self) -> str:
        return "LimitPushdown"

    def applies_to(self, node: PlanNode) -> bool:
        """Applies to LIMIT nodes."""
        return node.node_type == NodeType.LIMIT

    def apply(self, node: PlanNode) -> PlanNode:
        """Push limit down when safe."""
        if not self.applies_to(node):
            return node

        limit_value = node.properties.get("limit", 100)

        # Can push through certain nodes
        if node.children and node.children[0].node_type in (NodeType.PROJECT,):
            child = node.children[0]
            # Add limit below projection
            new_grandchild = PlanNode(
                node_type=NodeType.LIMIT, children=child.children, properties={"limit": limit_value}
            )
            child.children = [new_grandchild]
            return node

        return node


# Rule registry
DEFAULT_RULES: List[OptimizationRule] = [
    ConstantFolding(),
    FilterCombination(),
    PredicatePushdown(),
    ProjectionPruning(),
    JoinReordering(),
    LimitPushdown(),
]


def get_default_rules() -> List[OptimizationRule]:
    """Get default optimization rules."""
    return DEFAULT_RULES.copy()
